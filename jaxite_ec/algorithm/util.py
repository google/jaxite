"""Utility functions for the Bukect distribution algorithm."""

from jaxite.jaxite_ec.algorithm import msm_reader as msm_reader_lib
import numpy as np

MSMReader = msm_reader_lib.MSMReader


class BukectDistribution:
  """A class to represent a Bukect distribution."""

  def __init__(self, msm_reader: MSMReader, slice_length, window_num):
    self.msm_reader = msm_reader
    self.slice_length = slice_length
    self.window_num = window_num
    self.bucket_num_in_window = 2**self.slice_length
    self.slice_mask = self.bucket_num_in_window - 1
    self.windows = [
        self.Window(self.bucket_num_in_window) for _ in range(self.window_num)
    ]

  class Window:
    """A class to represent a window in a Bukect distribution."""

    def __init__(self, bukect_num):
      self.bukect_num = bukect_num
      self.buckets = [0] * bukect_num
      self.max = None
      self.min = None

    def get_min_max(self):
      self.max = max(self.buckets[1:])
      self.min = min(self.buckets[1:])
      return self.min, self.max

    def __getitem__(self, index):
      return self.buckets[index]

    def __setitem__(self, index, value):
      self.buckets[index] = value

  def run(self):
    scalar = self.msm_reader.get_next_scalar()
    while scalar != None:
      current_scalar = scalar
      window_id = 0
      while current_scalar != 0:
        bucket_id = current_scalar & self.slice_mask
        self.windows[window_id][bucket_id] += 1
        current_scalar = current_scalar >> self.slice_length
        window_id += 1
      scalar = self.msm_reader.get_next_scalar()

  def print_result(self):
    for window in self.windows:
      bmin, bmax = window.get_min_max()
      print(f"min: {bmin}, max: {bmax}")


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


def int_list_to_2d_array(int_list, base, array_size=None) -> np.ndarray:
  chunked_arrays = [
      int_to_array(int_value, base, array_size) for int_value in int_list
  ]
  return np.array(chunked_arrays)
