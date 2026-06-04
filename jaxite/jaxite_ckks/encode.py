"""Utilities for encoding and decoding plaintexts."""

import abc
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


ABC = abc.ABC
abstractmethod = abc.abstractmethod


class EncodeBase(ABC):
  """Abstract base class for encoding kernels."""

  @abstractmethod
  def encode(self, slots: list[complex]) -> Plaintext:
    """Encode a cleartext vector into an RNS-CKKS plaintext."""


class DecodeBase(ABC):
  """Abstract base class for decoding kernels."""

  @abstractmethod
  def decode(
      self, plaintext: Plaintext, is_slot_form: bool = False
  ) -> list[complex]:
    """Decode an RNS-CKKS plaintext into a cleartext vector."""


class Encode(EncodeBase):
  """Kernel for CKKS encoding."""

  def __init__(self, degree: int, moduli: list[int], scale: float):
    self.degree = degree
    self.moduli = moduli
    self.scale = scale

  def encode(self, slots: list[complex]) -> Plaintext:
    """Encode a cleartext list of slots into a plaintext.

    If the number of slots is less than the configured self.degree / 2,
    then the remaining slots are filled with zero.

    Args:
        slots: a list of complex values to encode into plaintext slots

    Returns: a Plaintext encoding the given slots.
    """

    nh = self.degree // 2
    y = np.array(slots, dtype=complex)
    if len(y) < nh:
      y = np.pad(y, (0, nh - len(y)))
    fft_special_inv(y, self.degree * 2)
    coeffs = np.concatenate([y.real, y.imag])
    scaled_coeffs = np.round(coeffs * self.scale)
    moduli_arr = np.array(self.moduli, dtype=np.uint32)
    poly = (scaled_coeffs % moduli_arr[:, None]).T.astype(np.uint32)
    poly_ntt = ntt_cpu.ntt_negacyclic_poly(poly, self.moduli)
    return Plaintext(
        jnp.array(poly_ntt, dtype=jnp.uint32),
        jnp.array(self.moduli, dtype=jnp.uint32),
    )


class Decode(DecodeBase):
  """Kernel for CKKS decoding."""

  def __init__(self, scale: float, num_slots: int):
    self.scale = scale
    self.num_slots = num_slots

  def decode(
      self, plaintext: Plaintext, is_slot_form: bool = False
  ) -> list[complex]:
    if self.scale is None or self.num_slots is None:
      raise ValueError(
          "scale and num_slots must be set via precompute_constants first."
      )

    moduli = plaintext.moduli.tolist()
    degree = plaintext.data.shape[0]
    if is_slot_form:
      poly = ntt_cpu.intt_negacyclic_poly(np.array(plaintext.data), moduli)
    else:
      poly = np.array(plaintext.data)
    combined = rns_utils.reconstruct_crt(poly.T.tolist(), moduli)
    modulus = math.prod(moduli)
    half_modulus = modulus // 2
    # Use a list comprehension to avoid NumPy OverflowError when handling
    # large integers.
    modded_divided = [
        (c - modulus if c > half_modulus else c) / self.scale for c in combined
    ]
    coeffs = np.array(modded_divided)
    nh = degree // 2
    y = coeffs[:nh] + 1j * coeffs[nh:]
    fft_special(y, degree * 2)
    return y[: self.num_slots].tolist()
