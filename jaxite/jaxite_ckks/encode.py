"""Utilities for encoding and decoding plaintexts."""

import math

import jax.numpy as jnp
from jaxite.jaxite_ckks import ntt_cpu
from jaxite.jaxite_ckks import rns_utils
from jaxite.jaxite_ckks import types
import numpy as np

Plaintext = types.Plaintext


def _roots(m: int) -> np.ndarray:
  return np.exp(2j * np.pi * np.arange(m) / m)


def _rot_group(m: int, nh: int, g: int = 5) -> list[int]:
  r = [1]
  for _ in range(1, nh):
    r.append((r[-1] * g) % m)
  return r


def fft_special_inv(vals: np.ndarray, cycl_order: int) -> None:
  """Compute a specialized iFFT for CKKS encoding.

  This is a direct port of OpenFHE's FFTSpecialInv
  https://github.com/openfheorg/openfhe-development/blob/1306d14f8c26bb6150d3e6ad54f28dfe1007689e/src/core/include/math/dftransform.h#L87-L93

  This in-place iFFT algorithm inverts the standard variant of the canonical
  embedding used in CKKS (5th powers of a primitive root of unity).

  Args:
    vals: the input values mutated in place.
    cycl_order: the order of the cyclotomic polynomial, i.e., 2N for x^N + 1.

  Returns:
    Nothing. Mutates `vals` in place to compute the IFFT.
  """
  m = cycl_order
  nh = len(vals)
  roots = _roots(m)
  rg = np.array(_rot_group(m, nh, g=5))
  length = nh
  while length >= 2:
    half = length >> 1
    lenq = length << 2
    step = m // lenq
    mod = rg[:half] % lenq
    idx = ((lenq - mod) % lenq) * step
    current_roots = roots[idx]

    vals_reshaped = vals.reshape(-1, length)
    u = vals_reshaped[:, :half].copy()
    t = vals_reshaped[:, half:].copy()
    vals_reshaped[:, :half] = u + t
    vals_reshaped[:, half:] = (u - t) * current_roots
    length >>= 1

  vals *= 1.0 / nh
  rns_utils.bit_reversal_array(vals)  # type: ignore


def fft_special(vals: np.ndarray, cycl_order: int) -> None:
  """Compute a specialized FFT for CKKS decoding.

  This is a direct port of OpenFHE's FFTSpecial
  https://github.com/openfheorg/openfhe-development/blob/1306d14f8c26bb6150d3e6ad54f28dfe1007689e/src/core/include/math/dftransform.h#L95-L101

  This in-place FFT algorithm computes the standard variant of the canonical
  embedding used in CKKS (5th powers of a primitive root of unity).

  Args:
    vals: the input values mutated in place.
    cycl_order: the order of the cyclotomic polynomial, i.e., 2N for x^N + 1.

  Returns:
    Nothing. Mutates `vals` in place to compute the FFT
  """
  m = cycl_order
  nh = len(vals)
  roots = _roots(m)
  rg = np.array(_rot_group(m, nh, g=5))
  rns_utils.bit_reversal_array(vals)  # type: ignore
  length = 2
  while length <= nh:
    half = length >> 1
    lenq = length << 2
    step = m // lenq
    mod = rg[:half] % lenq
    idx = mod * step
    current_roots = roots[idx]

    vals_reshaped = vals.reshape(-1, length)
    u = vals_reshaped[:, :half].copy()
    v = vals_reshaped[:, half:] * current_roots
    vals_reshaped[:, :half] = u + v
    vals_reshaped[:, half:] = u - v
    length <<= 1


def encode(
    slots: list[complex], degree: int, moduli: list[int], scale: float
) -> Plaintext:
  """Encode a cleartext vector into an RNS-CKKS plaintext.

  Args:
    slots: the length (at most) `degree / 2` list of input values. Zero-padded
      to size `degree / 2` if there are fewer than `degree / 2` values.
    degree: the degree N of the polynomial ring modulus of the plaintext space.
      Must be a power of two.
    moduli: the list of modulus factors q_i, for which Q = product_i(q_i)
      provides the coefficient modulus of the polynomial ring. The output
      representation is an RNS polynomial using these moduli as limbs.
    scale: the scaling factor for the plaintext.

  Returns:
    The encoded plaintext.
  """
  nh = degree // 2
  y = np.array(slots, dtype=complex)
  if len(y) < nh:
    y = np.pad(y, (0, nh - len(y)))
  fft_special_inv(y, degree * 2)
  coeffs = np.concatenate([y.real, y.imag])
  scaled_coeffs = np.round(coeffs * scale)
  moduli_arr = np.array(moduli, dtype=np.uint64)
  poly = (scaled_coeffs % moduli_arr[:, None]).T.astype(np.uint64)
  poly_ntt = ntt_cpu.ntt_negacyclic_poly(poly, moduli)
  return Plaintext(jnp.array(poly_ntt), jnp.array(moduli, dtype=jnp.uint64))


def decode(plaintext: Plaintext, scale: float, num_slots: int) -> list[complex]:
  """Decode an RNS-CKKS plaintext into a cleartext vector.

  Args:
    plaintext: the input plaintext polynomial to decode.
    scale: the scaling factor for the plaintext.
    num_slots: the number of slots to restrict the output to. Though the
      polynomial degree N is tracked by the `plaintext` argument, and hence N/2
      is the actual number of slots, this argument truncates the final result to
      allow the user to specify a cleartext with < N/2 slots.

  Returns:
    The encoded plaintext.
  """
  moduli = plaintext.moduli.tolist()
  degree = plaintext.data.shape[0]
  poly = ntt_cpu.intt_negacyclic_poly(np.array(plaintext.data), moduli)
  combined = np.array(rns_utils.reconstruct_crt(poly.T.tolist(), moduli))
  modulus = math.prod(moduli)
  half_modulus = modulus // 2
  modded = np.where(
      combined > half_modulus, combined - modulus, combined
  ).astype(np.float64)
  coeffs = modded / scale
  nh = degree // 2
  y = coeffs[:nh] + 1j * coeffs[nh:]
  fft_special(y, degree * 2)
  return y[:num_slots].tolist()
