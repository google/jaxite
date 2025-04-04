"""Utility functions for the Bukect distribution algorithm."""

import numpy as np


def bits_to_numpy_dtype(bits):
  if bits == 8:
    return np.uint8
  elif bits == 16:
    return np.uint16
  elif bits == 32:
    return np.uint32
  elif bits == 64:
    return np.uint64
  else:
    raise ValueError("Unsupported bit size. Use 8, 16, 32, or 64.")


def int_to_array(python_int, base, array_size=None) -> np.ndarray:
  """Converts a Python integer to a JAX array.

  Args:
    python_int: The Python integer to convert.
    base: The number of bits per element in the integer.
    array_size: The desired size of the resulting JAX array. If provided, the
      integer will be padded or trimmed to match this size.

  Returns:
    A JAX array containing the elements of the Python integer.
  """
  chunks = []
  mask = (
      1 << base
  ) - 1  # Mask to extract the lower bits (e.g., 32 bits -> 0xFFFFFFFF)
  dtype = bits_to_numpy_dtype(base)
  # Extract each element from the integer
  while python_int > 0:
    chunks.append(python_int & mask)  # Extract the lower bits
    python_int >>= base  # Shift to remove the extracted bits

  # If array_size is provided, we pad or trim the result to match the
  # desired size
  if array_size is not None:
    assert array_size >= len(chunks)
    chunks = chunks[:array_size] + [0] * (array_size - len(chunks))

  # Convert the list to a JAX array
  return np.array(chunks, dtype=dtype)


def array_to_int(np_array: np.ndarray, base):
  # Initialize the result as a Python integer
  result = 0

  # Iterate over the elements in the array
  for i, elem in enumerate(np_array):
    # Convert each element to an int and shift it by the appropriate
    # number of bits
    result |= int(elem) << (i * base)

  return result


def int_list_to_array(int_list, base, array_size=None) -> np.ndarray:
  chunked_arrays = [
      int_to_array(int_value, base, array_size) for int_value in int_list
  ]
  return np.array(chunked_arrays)
