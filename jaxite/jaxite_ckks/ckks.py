"""CKKS Homomorphic Encryption for Approximate Numbers."""

import dataclasses
import functools
import math
import secrets
from typing import Tuple

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import rns
from jaxite.jaxite_ckks import rns_utils

RnsPolynomial = rns.RnsPolynomial
RnsParams = rns.RnsParams
gen_rns_polynomial = rns.gen_rns_polynomial


def _rep(x: int, q: int) -> int:
  """Balanced representative of x mod q."""
  x = x % q
  return x if x <= q // 2 else -(q - x)


@dataclasses.dataclass
class CkksEncoder:
  """A CKKS encoder.

  In CKKS a complex vector z in CC^{N/2} is identified as the image of a real
  polynomial f(X) under mapping rho * tau:
  - tau is the canonical embedding map which takes f(X) to its evaluation
    f(omega_j) under all primitive 2N'th root of unity omega_j;
  - rho maps pairs of conjugate complex numbers (a + bI, a - bI) to a + bI.

  To gain enough precision and represent f(X) as a discrete object, we further
  scale up f(X) and round its coefficients to integers. Specifically:
  - Encode(z) = round(scaling_factor * tau^-1(rho^-1(z)));
  - Decode(f) = 1/scaling_factor * rho(tau(f)).
  The rounded coefficients are represented as a polynomial in R_Q, which is the
  plaintext in CKKS. We call the input complex vector z as the slot values, and
  the CKKS encoding admits slot-wise additive and multiplicative homomorphism.

  Note that Decode is not exactly the inverse of Encode, and precision loss
  will happen due to rounding error and float point arithmetic.
  """

  degree: int
  moduli: list[int]
  scaling_factor: int

  psis_bitrev: list[complex] = dataclasses.field(init=False)
  psis_bitrev_inv: list[complex] = dataclasses.field(init=False)

  def __post_init__(self):
    if not rns_utils.is_power_of_two(self.degree):
      raise ValueError('`degree` must be a power of two.')

    n = self.degree

    # Generate the powers of primitive 2N'th root exp(pi * I / N) and rearrange
    # them in the bit reversed order to run Cooley-Tukey and Gentleman-Sande.
    theta = math.pi / n
    self.psis_bitrev = [
        complex(math.cos(theta * i), math.sin(theta * i)) for i in range(n)
    ]
    rns_utils.bit_reversal_array(self.psis_bitrev)
    self.psis_bitrev_inv = [complex(math.cos(0), -math.sin(0))] + [
        complex(math.cos(theta * i), -math.sin(theta * i))
        for i in range(n - 1, 0, -1)
    ]
    rns_utils.bit_reversal_array(self.psis_bitrev_inv)
    self.psis_bitrev_inv = self.psis_bitrev_inv[::-1]

  def encode(self, values: list[complex]) -> RnsPolynomial:
    """Encode an array of complex values to a CKKS plaintext polynomial.

    Args:
      values: A list of complex numbers to be encoded. The length of `values`
      can be at most N/2 where N is the ring degree of the CKKS scheme.

    Returns:
      An RNS polynomial encoding the given values in its slots.
    """
    num_slots = self.degree >> 1
    if len(values) > num_slots:
      raise ValueError(f'`values` can have at most {num_slots} elements.')

    # Move values to their slot positions.
    coeff_values = [complex(0, 0)] * num_slots
    power = 1
    for j in range(num_slots):
      coeff_values[(power - 1) // 4] = values[j]
      power = (power * 5) % (2 * self.degree)

    # The encoded polynomial is round(scaling_factor * DFT^-1(vs)), where
    # DFT^-1(vs) computes normalized half Gentleman-Sande inverse FFT.
    rns_utils.bit_reversal_array(coeff_values)
    self._half_gentleman_sande(coeff_values)
    # Normalize the transformation and then round to integers.
    factor = self.scaling_factor / num_slots
    coeffs = [0] * self.degree
    for i in range(num_slots):
      coeffs[i] = round(coeff_values[i].real * factor)
      coeffs[i + num_slots] = round(coeff_values[i].imag * factor)

    # Convert to RNS representation.
    coeffs_qi = [[0] * self.degree for _ in range(len(self.moduli))]
    for i, qi in enumerate(self.moduli):
      for j in range(self.degree):
        coeffs_qi[i][j] = coeffs[j] % qi

    return RnsPolynomial(self.degree, self.moduli, coeffs_qi, is_ntt=False)

  def decode(self, plaintext: RnsPolynomial) -> list[complex]:
    """Decode the given plaintext polynomial to its slot values.
    
    Args:
      plaintext: The plaintext polynomial to be decoded.
      
    Returns:
      A list of complex numbers encoded in the plaintext polynomial.
    """
    if plaintext.degree != self.degree:
      raise ValueError(f'`plaintext` must have degree = {self.degree}.')
    if plaintext.is_ntt:
      raise ValueError('`plaintext` must be in the coefficient form.')

    # Convert RNS to integer representation.
    num_slots = self.degree >> 1
    coeffs = self._crt(plaintext.coeffs, plaintext.moduli)
    coeff_values = [
        complex(coeffs[i], coeffs[i + num_slots]) / self.scaling_factor
        for i in range(num_slots)
    ]
    self._half_cooley_tukey(coeff_values)
    rns_utils.bit_reversal_array(coeff_values)

    # Move the slot values to their original positions.
    slots = [complex(0, 0)] * num_slots
    power = 1
    for j in range(num_slots):
      slots[j] = coeff_values[(power - 1) // 4]
      power = (power * 5) % (2 * self.degree)
    return slots

  def _half_cooley_tukey(self, coeffs: list[complex]) -> None:
    """Cooley-Tukey FFT but assume the coeffs are half of the input vector.
    
    This is used to decode a CKKS plaintext polynomial to its slot values.
    """
    num_coeffs = len(coeffs)
    if num_coeffs * 2 != self.degree:
      raise ValueError('`coeffs` must have length degree / 2.')
    log_len = rns_utils.num_bits(num_coeffs)
    for i in range(log_len - 1, -1, -1):
      half_m = 1 << i
      m = half_m << 1
      index_psi = 1 << (log_len - i)
      for k in range(0, num_coeffs, m):
        for j in range(half_m):
          t = coeffs[k + j + half_m] * self.psis_bitrev[index_psi]
          u = coeffs[k + j]
          coeffs[k + j] += t
          coeffs[k + j + half_m] = u - t
        index_psi += 1

  def _half_gentleman_sande(self, coeffs: list[complex]) -> None:
    """Sandie-Gentleman FFT but assume the coeffs are half of the input vector.
    
    This is used to encode a list of complex numbers to a CKKS plaintext
    polynomial.
    """
    num_coeffs = len(coeffs)
    if num_coeffs * 2 != self.degree:
      raise ValueError('`coeffs` must have length degree / 2.')
    log_len = rns_utils.num_bits(num_coeffs)
    index_psi_base = 0
    for i in range(log_len):
      half_m = 1 << i
      m = half_m << 1
      index_psi_inv = index_psi_base
      for k in range(0, num_coeffs, m):
        for j in range(half_m):
          t = coeffs[k + j + half_m]
          u = coeffs[k + j]
          coeffs[k + j] += t
          coeffs[k + j + half_m] = (u - t) * self.psis_bitrev_inv[index_psi_inv]
        index_psi_inv += 1
      index_psi_base += 1 << (log_len - i)

  def _crt(self, coeffs_qs: list[list[int]], qs: list[int]) -> list[int]:
    """CRT interpolation of coeffs_qs mod (qs[i] for all i).
    
    Args:
      coeffs_qs: The coefficients of a polynomial a(X) modulo q_i
      , for all i.
      qs: A list of moduli q_i's whose product is Q.
      
    Returns:
      The coefficients of the polynomial a(X) modulo Q.
    """
    num_moduli = len(qs)
    if num_moduli == 1:
      return [_rep(coeffs_qs[0][i], qs[0]) for i in range(self.degree)]

    q = functools.reduce(lambda x, y: x * y, qs)
    q_hats = [q // q_i for q_i in qs]
    q_hat_invs = [
        rns_utils.inverse_mod(q_hats[i], qs[i]) for i in range(num_moduli)
    ]

    coeffs = [0] * self.degree
    for i in range(num_moduli):
      for j in range(self.degree):
        coeffs[j] += coeffs_qs[i][j] * q_hat_invs[i] * q_hats[i]
    for j in range(self.degree):
      coeffs[j] = _rep(coeffs[j] % q, q)
    return coeffs


@dataclasses.dataclass
class CkksCiphertext:
  """A CKKS ciphertext.

  A CKKS ciphertext of degree k is a list of k+1 polynomials [c0, ..., ck] in
  R_Q such that c0 + c1 * s + ... + ck * s^k = plaintext + error.
  In addition, the level of a ciphertext is L such that Q is a product of L+1
  distinct prime moduli.
  """

  # the RNS moduli whose product is Q
  moduli: list[int]

  # the polynomials.
  components: list[RnsPolynomial]

  @property
  def level(self) -> int:
    return len(self.moduli) - 1

  @property
  def degree(self) -> int:
    return len(self.components) - 1

  def to_ntt_form(self, ntt_params) -> None:
    for c in self.components:
      c.to_ntt_form(ntt_params)

  def to_coeffs_form(self, ntt_params) -> None:
    for c in self.components:
      c.to_coeffs_form(ntt_params)

  def to_jnp_array(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return (
        jnp.array(self.moduli, dtype=jnp.uint64),
        jnp.array([c.to_jnp_array() for c in self.components]),
    )


@dataclasses.dataclass
class CkksSecretKey:
  """A CKKS secret key."""

  # the RNS moduli whose product is Q
  moduli: list[int]

  # the secret key.
  key: RnsPolynomial

  @property
  def level(self) -> int:
    return len(self.moduli) - 1

  @property
  def degree(self) -> int:
    return len(self.key) - 1

  def to_ntt_form(self, ntt_params) -> None:
    self.key.to_ntt_form(ntt_params)

  def to_coeffs_form(self, ntt_params) -> None:
    self.key.to_coeffs_form(ntt_params)


def gen_uniform_polynomial(degree: int, moduli: list[int]) -> RnsPolynomial:
  """Generate a uniformly random RNS polynomial in R_Q = Z[X] / (Q, X^N+1)."""
  coeffs_q = []
  for q in moduli:
    coeffs_q.append([secrets.randbelow(q) for _ in range(degree)])
  return RnsPolynomial(degree, moduli, coeffs_q, is_ntt=False)


def gen_gaussian_polynomial(
    degree: int, moduli: list[int], sigma: float
) -> RnsPolynomial:
  """Generate a random Gaussian polynomial in R_Q = Z[X] / (Q, X^N+1).

  Note: Each coefficient is independently sampled from a rounded Gaussian
        distribution with parameter sigma.

  Args:
    degree: The degree N of the ring R_Q.
    moduli: The list of prime moduli q_i's whose product is Q.
    sigma: The standard deviation of the Gaussian distribution.

  Returns:
    An RNS polynomial with coefficients sampled from a Gaussian distribution.
  """
  prng = secrets.SystemRandom()
  coeffs = [round(prng.normalvariate(0, sigma)) for _ in range(degree)]
  return gen_rns_polynomial(degree, coeffs, moduli)


def gen_secret_key(degree: int, moduli: list[int]) -> 'CkksSecretKey':
  """Generate a CKKS secret key."""
  return CkksSecretKey(
      moduli, gen_gaussian_polynomial(degree, moduli, sigma=3.2)
  )


def gen_ciphertext_from_jnp_array(
    degree: int,
    moduli: list[int],
    components: jax.Array,
    is_ntt: bool = True,
) -> CkksCiphertext:
  """Generate a CKKS ciphertext from its JAX array representation."""
  return CkksCiphertext(
      moduli,
      [
          RnsPolynomial(
              degree,
              moduli,
              coeffs,
              is_ntt=is_ntt,
          )
          for coeffs in components.tolist()
      ],
  )


def encrypt(
    secret_key: CkksSecretKey,
    values: list[complex],
    encoder: CkksEncoder,
    rns_params: RnsParams,
) -> CkksCiphertext:
  """Encrypts a vector of complex values and returns a CKKS ciphertext.

  In CKKS, a complex vector of dimension N/2 is encrypted as (c0, c1) ∈ R_Q^2
  where:
  - c0 = a * s + e + Encode(values)
  - c1 = -a

  Encode(values) performs a scaled FFT on values, rounded to integers, and then
  represented as a polynomial in R_Q. For details see CkksEncoder.

  The polynomials (-a, a * s + e) form a RLWE sample wrt secret s.

  Args:
    secret_key: The CKKS secret key.
    values: A list of complex numbers to be encrypted in the slots.
    encoder: The CKKS encoder.
    rns_params: The RNS parameters.

  Returns:
    A CKKS ciphertext.
  """
  a = gen_uniform_polynomial(rns_params.degree, rns_params.moduli)
  a.to_ntt_form(rns_params.ntt_params)
  secret_key.to_ntt_form(rns_params.ntt_params)
  e = gen_gaussian_polynomial(rns_params.degree, rns_params.moduli, sigma=3.2)
  e.to_ntt_form(rns_params.ntt_params)
  plaintext = encoder.encode(values)
  plaintext.to_ntt_form(rns_params.ntt_params)

  c0 = a * secret_key.key + e + plaintext
  c1 = -a
  return CkksCiphertext(rns_params.moduli, [c0, c1])


def decrypt(
    secret_key: CkksSecretKey,
    ciphertext: CkksCiphertext,
    encoder: CkksEncoder,
    rns_params: RnsParams,
) -> list[complex]:
  """Decrypts a CKKS ciphertext and returns the decrypted complex vector.

  Note: This version of decryption function does not satisfy the IND-CPA-D
        security which is typically required for CKKS. For more details, see
        https://eprint.iacr.org/2020/1533 and https://eprint.iacr.org/2022/816.

  Args:
    secret_key: The CKKS secret key.
    ciphertext: The CKKS ciphertext.
    encoder: The CKKS encoder.
    rns_params: The RNS parameters.

  Returns:
    A list of complex numbers.
  """
  if ciphertext.level != secret_key.level:
    raise ValueError(f'`ciphertext` and `secret_key` must have the same level.')
  if ciphertext.degree < 1:
    raise ValueError('`ciphertext` must have degree >= 1.')

  plaintext = ciphertext.components[0]
  plaintext.to_ntt_form(rns_params.ntt_params)
  secret = secret_key.key
  secret.to_ntt_form(rns_params.ntt_params)
  for i in range(1, ciphertext.degree + 1):
    c = ciphertext.components[i]
    c.to_ntt_form(rns_params.ntt_params)
    plaintext += c * secret
    secret *= secret_key.key

  plaintext.to_coeffs_form(rns_params.ntt_params)
  return encoder.decode(plaintext)
