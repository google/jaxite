"""CPU-only NTT implementation for CKKS encryption and decryption.

This module provides a pure-Python/Numpy implementation of the Number Theoretic
Transform (NTT). It is designed to be used for encryption and decryption tasks
that are typically performed on the CPU and should not depend on JAX. For
high-performance homomorphic operations on accelerators (like TPU), use the
implementations in `ntt.py`.
"""

from jaxite.jaxite_ckks import math as ckks_math
import numpy as np

_PERM_CACHE = {}


def _get_bit_reversal_perm(n):
  """Returns a bit-reversal permutation array of size n."""
  if n not in _PERM_CACHE:
    bits = n.bit_length() - 1
    indices = np.arange(n, dtype=np.uint32)
    perm = np.zeros(n, dtype=np.uint32)
    for _ in range(bits):
      perm = (perm << 1) | (indices & 1)
      indices >>= 1
    _PERM_CACHE[n] = perm
  return _PERM_CACHE[n]


def _get_powers(base, n, q):
  """Returns an array of powers of base modulo q: [base^0, ..., base^{n-1}]."""
  res = np.empty(n, dtype=object)
  curr = 1
  for i in range(n):
    res[i] = curr
    curr = (curr * base) % q
  return res


def ntt_negacyclic_bit_reverse(a, q, psi):
  """Forward negacyclic NTT."""
  n = len(a)
  powers_of_psi = _get_powers(psi, n, q)
  a_twisted = (np.asarray(a, dtype=object) * powers_of_psi) % q
  omega = pow(psi, 2, q)
  return ntt_cyclic_bit_reverse(a_twisted, q, omega)


def intt_negacyclic_bit_reverse(a, q, psi):
  """Inverse negacyclic NTT."""
  n = len(a)
  omega = pow(psi, 2, q)
  a_inv = intt_cyclic_bit_reverse(a, q, omega)
  psi_inv = pow(psi, -1, q)
  powers_of_psi_inv = _get_powers(psi_inv, n, q)
  return (a_inv * powers_of_psi_inv) % q


def ntt_cyclic_bit_reverse(a, q, omega):
  """Iterative forward NTT with bit-reversal."""
  a = np.array(a, dtype=object)
  n = len(a)

  # Bit-reversal (DIT)
  perm = _get_bit_reversal_perm(n)
  a = a[perm]

  length = 2
  while length <= n:
    w_m = pow(omega, n // length, q)
    half = length // 2
    ws = _get_powers(w_m, half, q)

    a = a.reshape(-1, length)
    u = a[:, :half]
    v = (a[:, half:] * ws) % q

    new_a = np.empty_like(a)
    new_a[:, :half] = (u + v) % q
    new_a[:, half:] = (u - v) % q
    a = new_a.ravel()

    length *= 2
  return a


def intt_cyclic_bit_reverse(a, q, omega):
  """Iterative inverse NTT with bit-reversal."""
  a = np.array(a, dtype=object)
  n = len(a)
  inv_root = pow(omega, -1, q)

  length = n
  while length >= 2:
    w_m = pow(inv_root, n // length, q)
    half = length // 2
    ws = _get_powers(w_m, half, q)

    a = a.reshape(-1, length)
    u = a[:, :half]
    v = a[:, half:]

    new_a = np.empty_like(a)
    new_a[:, :half] = (u + v) % q
    new_a[:, half:] = ((u - v) * ws) % q
    a = new_a.ravel()

    length //= 2

  # Bit-reversal (DIF ends with bit-reversal)
  perm = _get_bit_reversal_perm(n)
  a = a[perm]

  inv_n = pow(n, -1, q)
  a = (a * inv_n) % q
  return a


def ntt_negacyclic_poly(poly: np.ndarray, moduli: list[int]) -> np.ndarray:
  """CPU-only NTT of a polynomial in RNS form."""
  degree = poly.shape[0]
  num_moduli = len(moduli)
  res = np.zeros((degree, num_moduli), dtype=np.uint64)
  perm = _get_bit_reversal_perm(degree)
  for i, q in enumerate(moduli):
    psi = ckks_math.root_of_unity(2 * degree, q)
    coeffs = poly[:, i]
    ntt_res = ntt_negacyclic_bit_reverse(coeffs, q, psi)
    res[:, i] = ntt_res[perm].astype(np.uint64)
  return res


def intt_negacyclic_poly(poly: np.ndarray, moduli: list[int]) -> np.ndarray:
  """CPU-only Inverse NTT of a polynomial in RNS form."""
  degree = poly.shape[0]
  num_moduli = len(moduli)
  res = np.zeros((degree, num_moduli), dtype=np.uint64)
  perm = _get_bit_reversal_perm(degree)
  for i, q in enumerate(moduli):
    psi = ckks_math.root_of_unity(2 * degree, q)
    coeffs = poly[:, i]
    coeffs_rev = coeffs[perm]
    intt_res = intt_negacyclic_bit_reverse(coeffs_rev, q, psi)
    res[:, i] = intt_res.astype(np.uint64)
  return res
