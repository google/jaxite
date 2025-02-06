"""This file implements the lazy reduction algorithm.
"""

import random

import numpy as np

randint = random.randint


def byte_len(x: int):
  return (x.bit_length() + 7) // 8


def chunk_decomposition(x, length=-1):
  if length == -1:
    length = byte_len(x)
  return [(x >> (8 * i)) & 255 for i in range(length)]


# Input: a_list: (t+2) byte numbers for a
#        b_list: (t+2) byte numbers for b
#             p: modulus
# Output: c_list: (t_2) byte numbers for (ab % p)
def lazy_reduction_via_matrix(a_list, b_list, p):
  """Performs lazy reduction via matrix multiplication.

  Args:
    a_list: (t+2) byte numbers for a
    b_list: (t+2) byte numbers for b
    p: modulus

  Returns:
    c_list: (t_2) byte numbers for (ab % p)
  """
  t = len(chunk_decomposition(p))
  # Precomputation
  lazy_mat = np.zeros((t + 4, t), dtype=np.uint8)
  for i in range(t + 4):
    val = (256 ** (t + i)) % p
    lazy_mat[i, :] = chunk_decomposition(val, t)

  # Begin computation
  assert len(a_list) == len(b_list)
  batch_size = len(a_list)
  batch_mat = np.zeros((batch_size, t + 4), dtype=np.uint8)
  standard_product = [a_list[i] * b_list[i] for i in range(batch_size)]
  standard_product_low = [s & (256**t - 1) for s in standard_product]
  standard_product_high = [s >> (8 * t) for s in standard_product]
  # Matrix packing
  for i in range(batch_size):
    batch_mat[i] = chunk_decomposition(standard_product_high[i], t + 4)
  # Matrix product
  # Upcast to get proper accumulators
  reduced = np.matmul(batch_mat.astype(np.uint32), lazy_mat.astype(np.uint32))
  c_list = []
  # Recombine into integers
  for i in range(batch_size):
    val = 0
    for j in reversed(range(t)):
      val *= 256
      val += int(reduced[i][j])
    c_list.append(val)
  # Add in standard_product_low; this could be done in u8 form before the
  # carry-add chain above, perhaps from the raw toeplitz output.
  # Since we are using a redundant form the upper part doesn't have to be
  # accurate.
  c_list = [c_list[i] + standard_product_low[i] for i in range(batch_size)]
  return c_list


def main():
  """This test case check the functionality of the lazy reduction algorithm.
  """
  p = randint(2**381, 2**384)
  batch_size = 16
  bound = p * 256 * 256
  # a and b are both < bound
  a_list = [randint(0, bound) for i in range(batch_size)]
  b_list = [randint(0, bound) for i in range(batch_size)]
  c_list = lazy_reduction_via_matrix(a_list, b_list, p)
  for i in range(batch_size):
    # each output is congruent modulo p
    assert c_list[i] % p == (a_list[i] * b_list[i]) % p
    # each output is < bound
    assert c_list[i] < bound
  print(a_list)
  print(b_list)
  print(c_list)


if __name__ == "__main__":
  main()
