"""Residue Number System for polynomials in ring Z[X]/(q, X^n+1).

Assume:
- n is a power of two;
- q is a product of NTT-friendly primes q_i (i.e., 2n | q_i - 1);
"""

import dataclasses
from typing import Any

import jax.numpy as jnp
from jaxite.jaxite_ckks import rns_utils


def _mod_exp(x: int, n: int, q: int) -> int:
  """Returns x^n mod q."""
  result = 1
  while n > 0:
    if n & 1:
      result = (result * x) % q
    x = (x * x) % q
    n >>= 1
  return result


def _primitive_root(m: int, q: int):
  """Returns a m'th primitive root of unity mod q.

  Note: q-1 must be divisible by m for the primitive roots to exist. Also assume
        m is a power of two.
  """
  if not rns_utils.is_power_of_two(m):
    raise ValueError('`m` must be a power of two.')
  if (q - 1) % m != 0:
    raise ValueError('`q - 1` must be divisible by m.')

  n = m >> 1
  k = (q - 1) // m
  for t in range(2, q):
    # If q is prime. then t^k mod q is a root of unity since t^(q-1) = 1 mod q.
    candidate = _mod_exp(t, k, q)
    # Since m is a power of two, t^k is a primitive root if (t^k)^n != 1 mod q,
    # i.e. it is -1 mod q.
    if _mod_exp(candidate, n, q) != 1:
      return candidate
  raise ValueError('No primitive root found.')


@dataclasses.dataclass
class Ntt:
  """Number Theoretic Transformations in Z[X]/(q, X^n+1).

  Note: Assume q is prime, n is power of two, and 2n | q-1.
  """

  n: int
  q: int

  n_inv_mod_q: int = dataclasses.field(init=False)

  # psi^i mod q for i in [0, n), in bit-reversal order, where psi is a
  # primitive 2n'th root of unity mod q.
  psis_bitrev: list[int] = dataclasses.field(init=False)

  # psi^(-1) mod q, and -psi^(-i) mod q for i in [1, n), in bit-reversal order.
  psis_inv_bitrev: list[int] = dataclasses.field(init=False)

  def __post_init__(self):
    if not rns_utils.is_power_of_two(self.n):
      raise ValueError('`n` must be a power of two.')
    if self.q % (2 * self.n) != 1:
      raise ValueError('`q - 1` must be divisible by 2N.')

    n = self.n
    q = self.q
    self.n_inv_mod_q = rns_utils.inverse_mod(n, q)

    # Generating the powers of primitive root psi to be used in Cooley-Tukey and
    # Gentleman-Sande.
    psi = _primitive_root(2 * n, q)
    self.psis_bitrev = [_mod_exp(psi, i, q) for i in range(n)]
    self.psis_inv_bitrev = list(self.psis_bitrev)
    rns_utils.bit_reversal_array(self.psis_bitrev)

    self.psis_inv_bitrev = (
        self.psis_inv_bitrev[:1:] + self.psis_inv_bitrev[:0:-1]
    )
    neg_psi_inv = self.psis_inv_bitrev[1]  # psi^(n-1) = -psi^{-1} mod q
    psi_inv = (-neg_psi_inv) % q  # psi^{-1} mod q
    rns_utils.bit_reversal_array(self.psis_inv_bitrev)
    self.psis_inv_bitrev[0] = (self.psis_inv_bitrev[0] * psi_inv) % q
    for i in range(1, n):
      self.psis_inv_bitrev[i] = (self.psis_inv_bitrev[i] * neg_psi_inv) % q

  def forward(self, coeffs: list[int]) -> None:
    """Forward NTT."""
    self._iterative_cooley_tukey(coeffs, rns_utils.num_bits(len(coeffs)))

  def backward(self, coeffs: list[int]) -> None:
    """Backward NTT (normalized)."""
    self._iterative_gentleman_sande(coeffs, rns_utils.num_bits(len(coeffs)))
    for i in range(len(coeffs)):
      coeffs[i] = (coeffs[i] * self.n_inv_mod_q) % self.q

  def _iterative_cooley_tukey(self, coeffs: list[int], log_len: int) -> None:
    """Cooley-Tukey NTT on coeffs in log_len iterations.

    This is used to transform a coefficient-form polynomial (represented by its
    coefficient vector `coeffs`) to the NTT form.
    """
    index_psi = 1
    for i in range(log_len - 1, -1, -1):
      half_m = 1 << i
      m = half_m << 1
      for k in range(0, len(coeffs), m):
        for j in range(0, half_m):
          t = coeffs[k + j + half_m] * self.psis_bitrev[index_psi]
          u = coeffs[k + j]
          coeffs[k + j] += t
          coeffs[k + j] %= self.q
          coeffs[k + j + half_m] = (u - t) % self.q
        index_psi += 1

  def _iterative_gentleman_sande(self, coeffs: list[int], log_len: int) -> None:
    """Gentleman-Sande NTT on coeffs in log_len iterations.

    This is used to transform a NTT-form polynomial (represented by its
    evaluation vector `coeffs`) to the coefficient form.
    """
    index_psi_inv = 0
    for i in range(log_len):
      half_m = 1 << i
      m = half_m << 1
      for k in range(0, len(coeffs), m):
        for j in range(0, half_m):
          t = coeffs[k + j + half_m]
          u = coeffs[k + j]
          coeffs[k + j] += t
          coeffs[k + j] %= self.q
          coeffs[k + j + half_m] = (u - t) * self.psis_inv_bitrev[index_psi_inv]
          coeffs[k + j + half_m] %= self.q
        index_psi_inv += 1


@dataclasses.dataclass
class RnsParams:
  """Parameters for an RNS instance over Z[X]/(Q, X^N+1)."""

  # the degree N of the polynomial.
  degree: int

  # the RNS moduli q_i whose product is Q
  moduli: list[int]

  # the NTT parameters wrt each q_i.
  ntt_params: list[Ntt] = dataclasses.field(init=False)

  def __post_init__(self):
    self.ntt_params = [Ntt(self.degree, modulus) for modulus in self.moduli]


@dataclasses.dataclass
class RnsPolynomial:
  """A polynomial in the quotient ring R_Q = Z[X] / (Q, X^N + 1)."""

  # the degree N of the polynomial.
  degree: int

  # the RNS moduli q_i whose product is Q
  moduli: list[int]

  # the coefficients of the polynomial wrt each RNS modulus q_i.
  coeffs: list[list[int]]

  # is the polynomial in the NTT or the Coefficient form?
  is_ntt: bool = False

  def to_ntt_form(self, ntt_params: list[Ntt]) -> None:
    """Convert the polynomial to the NTT form."""
    if self.is_ntt:
      return
    for i in range(len(self.moduli)):
      ntt_params[i].forward(self.coeffs[i])
    self.is_ntt = True

  def to_coeffs_form(self, ntt_params: list[Ntt]) -> None:
    """Convert the polynomial to the Coefficient form."""
    if not self.is_ntt:
      return
    for i in range(len(self.moduli)):
      ntt_params[i].backward(self.coeffs[i])
    self.is_ntt = False

  def to_jnp_array(self) -> jnp.ndarray:
    """Convert the polynomial to a jax.Array."""
    return jnp.array(self.coeffs, dtype=jnp.uint64)

  def __len__(self) -> int:
    """Returns the length of the polynomial."""
    return self.degree

  def __neg__(self) -> 'RnsPolynomial':
    """Compute the negative of a polynomial."""
    coeffs = [[0] * self.degree for _ in range(len(self.moduli))]
    for i, qi in enumerate(self.moduli):
      for j in range(self.degree):
        coeffs[i][j] = -self.coeffs[i][j] % qi
    return RnsPolynomial(self.degree, self.moduli, coeffs, is_ntt=self.is_ntt)

  def _check_compatible(self, other: 'RnsPolynomial') -> None:
    """Check if two polynomials are compatible."""
    if self.degree != other.degree:
      raise ValueError(
          f'`degree` must be the same: self = {self.degree}, other ='
          f' {other.degree}'
      )
    if len(self.moduli) != len(other.moduli):
      raise ValueError(
          f'`moduli` must have the same length: self = {len(self.moduli)},'
          f' other = {len(other.moduli)}.'
      )
    for i in range(len(self.moduli)):
      if self.moduli[i] != other.moduli[i]:
        raise ValueError(
            f'`moduli` must have the same moduli: self.moduli[{i}] ='
            f' {self.moduli[i]}, other.moduli[{i}] = {other.moduli[i]}'
        )
    if len(self.coeffs) != len(other.coeffs):
      raise ValueError(
          f'`coeffs` must have the same length: self = {len(self.coeffs)},'
          f' other = {len(other.coeffs)}'
      )
    if self.is_ntt != other.is_ntt:
      raise ValueError(
          f'`is_ntt` must be the same: self = {self.is_ntt}, other ='
          f' {other.is_ntt}'
      )

  def __add__(self, other: 'RnsPolynomial') -> 'RnsPolynomial':
    """Add two polynomials in R_Q.

    This function expects both polynomials to be in the same form (NTT / Coeff).
    """
    self._check_compatible(other)
    coeffs = [[0] * self.degree for _ in range(len(self.moduli))]
    for i, qi in enumerate(self.moduli):
      for j in range(self.degree):
        coeffs[i][j] = (self.coeffs[i][j] + other.coeffs[i][j]) % qi
    return RnsPolynomial(self.degree, self.moduli, coeffs, is_ntt=self.is_ntt)

  def __sub__(self, other: 'RnsPolynomial') -> 'RnsPolynomial':
    """Subtract two polynomials in R_Q.

    This function expects both polynomials to be in the same form (NTT / Coeff).
    """
    self._check_compatible(other)
    coeffs = [[0] * self.degree for _ in range(len(self.moduli))]
    for i, qi in enumerate(self.moduli):
      for j in range(self.degree):
        coeffs[i][j] = (self.coeffs[i][j] - other.coeffs[i][j]) % qi
    return RnsPolynomial(self.degree, self.moduli, coeffs, is_ntt=self.is_ntt)

  def __mul__(self, other: 'RnsPolynomial') -> 'RnsPolynomial':
    """Multiply two polynomials in R_Q.

    This function expects both polynomials to be in the NTT form.
    """
    self._check_compatible(other)
    if (not self.is_ntt) or (not other.is_ntt):
      raise ValueError('Both polynomials must be in the NTT form.')

    coeffs = [[0] * self.degree for _ in range(len(self.moduli))]
    for i, qi in enumerate(self.moduli):
      for j in range(self.degree):
        coeffs[i][j] = (self.coeffs[i][j] * other.coeffs[i][j]) % qi
    return RnsPolynomial(self.degree, self.moduli, coeffs, is_ntt=True)


def gen_rns_polynomial(
    degree: int, coeffs: list[int], moduli: list[int]
) -> RnsPolynomial:
  """Generate a RNS polynomial from the given mod-Q coefficients."""
  coeffs_q = []
  for q in moduli:
    coeffs_q.append([coeff % q for coeff in coeffs])
  return RnsPolynomial(degree, moduli, coeffs_q, is_ntt=False)


def gen_rns_polynomial_from_jnp_array(
    degree: int,
    rns_coeffs: jnp.ndarray,
    moduli: list[int],
    is_ntt: bool = False,
) -> RnsPolynomial:
  """Generate a RNS polynomial from the given RNS coefficients."""
  coeffs_q = rns_coeffs.tolist()
  return RnsPolynomial(degree, moduli, coeffs_q, is_ntt=is_ntt)
