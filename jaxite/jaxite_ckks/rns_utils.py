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
