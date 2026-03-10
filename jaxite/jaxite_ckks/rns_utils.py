"""Utility functions for the RNS based RLWE schemes."""

from typing import Any


def inverse_mod(x: int, q: int) -> int:
  """Returns the inverse of x mod q."""
  return int(pow(x, -1, q))


def is_power_of_two(x: int) -> bool:
  """Returns True if x is a power of two."""
  return x > 0 and (x & (x - 1)) == 0


def num_bits(x: int) -> int:
  """Returns the number of bits in x."""
  return x.bit_length() - 1


def bit_reversal(x: int, num_bits: int) -> int:
  """Returns the bit-reversal of x with num_bits representation."""
  result = 0
  for _ in range(num_bits):
    result <<= 1
    result |= x & 1
    x >>= 1
  return result


def bit_reversal_array(xs: list[Any]) -> None:
  """Rearrange the given array in bit-reversal order in place."""
  n = num_bits(len(xs))
  for i in range(len(xs)):
    j = bit_reversal(i, n)
    if i < j:
      xs[i], xs[j] = xs[j], xs[i]


def compute_q_hat_inv_mod_q(moduli: list[int]) -> list[int]:
  """Computes QHatInvModq = (Q/qi)^-1 mod qi for each qi in moduli.

  Args:
    moduli: The list of primes (moduli) defining the RNS basis.

  Returns:
    The list of modular inverses.
  """
  q = 1
  for m in moduli:
    q *= m

  q_hat_inv_mod_q = []
  for m in moduli:
    q_hat_i = q // m
    inv = inverse_mod(q_hat_i, m)
    q_hat_inv_mod_q.append(inv)

  return q_hat_inv_mod_q


def compute_q_hat_mod_p(
    original_moduli: list[int], target_moduli: list[int]
) -> list[list[int]]:
  """Computes QHatModp = (Q/qi) mod pj for each qi in original and pj in target.

  Args:
    original_moduli: The list of primes defining the original basis.
    target_moduli: The list of primes defining the target basis.

  Returns:
    A 2D list where result[i][j] = (Q/original_moduli[i]) mod target_moduli[j].
  """
  q = 1
  for m in original_moduli:
    q *= m

  q_hat_mod_p = []
  for m_i in original_moduli:
    q_hat_i = q // m_i
    q_hat_mod_p_i = []
    for m_j in target_moduli:
      q_hat_mod_p_i.append(q_hat_i % m_j)
    q_hat_mod_p.append(q_hat_mod_p_i)

  return q_hat_mod_p


def reconstruct_crt(residues: list[list[int]], moduli: list[int]) -> list[int]:
  """Reconstructs integers from their RNS residues using the Chinese Remainder Theorem.

  Args:
    residues: A 2D list where residues[i][j] is the residue of the j-th integer
      modulo moduli[i]. Shape: (num_moduli, num_integers).
    moduli: A list of moduli q_i.

  Returns:
    A list of reconstructed integers modulo Q = prod(q_i).
  """
  num_moduli = len(moduli)
  num_integers = len(residues[0])

  if num_moduli == 1:
    return [r % moduli[0] for r in residues[0]]

  q = 1
  for m in moduli:
    q *= m

  q_hat_inv_mod_q = compute_q_hat_inv_mod_q(moduli)

  results = [0] * num_integers
  for i in range(num_moduli):
    q_i = moduli[i]
    q_hat_i = q // q_i
    prefactor = q_hat_inv_mod_q[i] * q_hat_i
    for j in range(num_integers):
      results[j] += residues[i][j] * prefactor

  return [r % q for r in results]
