"""Utility functions for jaxite_ec.

Note that: All functions that directly take Python int as input cannot be
jitted.
"""

import math
from typing import List

import jax
import jax.numpy as jnp


gcd = math.gcd


BASE = 16
BASE_TYPE = jnp.uint16  # this type must match the BASE, i.e. jnp.uint<BASE>
U16_MASK = 0xFFFF
U32_MASK = 0xFFFFFFFF

U64_CHUNK_NUM = 6
U32_CHUNK_NUM = 12
U16_CHUNK_NUM = 24
U8_CHUNK_NUM = 48
U16_CHUNK_SHIFT_BITS = 16
U32_CHUNK_SHIFT_BITS = 32

MODULUS_377_INT = 0x01AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001
MU_377_INT = 0x98542343310183A5DB0F28160BBD3DCEEEB43799DDAC681ABCB52236169B40B43B5A1DE2710A9647E7F56317936BFF32
BARRETT_SHIFT_U8 = 95  # BARRETT Params for k = 380
CHUNK_MAX = 0xFF
CHUNK_PRECISION = 8

# Lazy Reduction Logics
U16_EXT_CHUNK_NUM = 25
MODULUS_377_S16_INT = MODULUS_377_INT << 16

# Pippenger Logics
COORDINATE_NUM = 4

# RNS Reduction Logics
# Hardware friendly moduli factors are 2**16 - v for v in the following list
RNS_MODULI = [
    0,
    1,
    3,
    5,
    9,
    15,
    17,
    27,
    33,
    39,
    45,
    47,
    57,
    59,
    63,
    77,
    87,
    89,
    99,
    105,
    113,
    117,
    123,
    125,
    129,
    143,
    153,
    155,
    165,
    167,
    173,
    179,
    183,
    189,
    197,
    209,
    213,
    215,
    225,
    227,
    243,
    249,
    14,
    38,
    50,
    54,
    98,
    102,
    110,
    122,
]

MODULI = [
    2**16 if i == 0 else 2**16 - int(i) if i % 2 == 1 else 2**15 - (int(i) // 2)
    for i in RNS_MODULI
]
RNS_PRECISION = 16
NUM_MODULI = len(RNS_MODULI)
MODULI_NP = [m % 2**16 for m in MODULI]
MODULI_SUB = [((256 * 256 * 4 * MODULUS_377_INT) - 2**16) % m for m in MODULI]


def print_hex_values(int_list):
  hex_values = " ".join((hex(value)) for value in int_list)
  print(hex_values)


def int_point_to_jax_point(coordinate_x, coordinate_y, z=None):
  x_array = int_to_array(coordinate_x, BASE, BASE_TYPE, U16_CHUNK_NUM)
  y_array = int_to_array(coordinate_y, BASE, BASE_TYPE, U16_CHUNK_NUM)
  if z:
    z_array = int_to_array(z, BASE, BASE_TYPE, U16_CHUNK_NUM)
    p = jnp.array([x_array, y_array, z_array])
  else:
    p = jnp.array([x_array, y_array])
  return p


def int_point_to_jax_point_pack(
    coordinates: List[int], base=BASE, chunk_num=U16_CHUNK_NUM
):
  result = []
  for i in range(len(coordinates)):
    result.append(int_to_array(coordinates[i], base, array_size=chunk_num))
  return jnp.array(result)


def int_point_coordinate_batch_to_jax_point_pack(
    coordinates: List[List[int]], base=BASE, chunk_num=U16_CHUNK_NUM
):
  result = []
  for i in range(len(coordinates)):
    result.append(int_list_to_2d_array(coordinates[i], base, chunk_num))
  return jnp.array(result)


def int_point_batch_to_jax_point_pack(
    points: List[List[int]], base=BASE, chunk_num=U16_CHUNK_NUM
):
  result = []
  for i in range(len(points)):
    result.append(int_point_to_jax_point_pack(points[i], base, chunk_num))
  return jnp.transpose(jnp.array(result), (1, 0, 2))


def int_point_to_jax_rns_point_pack(coordinates: List[int]):
  result = []
  for i in range(len(coordinates)):
    result.append(int_to_array_rns(coordinates[i]))
  return jnp.array(result)


def int_point_batch_to_jax_rns_point_pack(points: List[List[int]]):
  result = []
  for i in range(len(points)):
    result.append(int_point_to_jax_rns_point_pack(points[i]))
  return jnp.transpose(jnp.array(result), (1, 0, 2))


def jax_point_pack_to_int_point_batch(point_pack: jnp.ndarray, base=BASE):
  points = jnp.transpose(point_pack, (1, 0, 2))
  results = []
  for i in range(len(points)):
    results.append(jax_array_to_int_list(points[i], base))
  return results


def jax_rns_array_to_int_list(jax_array):
  """Converts JAX array to single integer."""
  result_list = []
  for i in range(jax_array.shape[0]):
    value_vector = jax_array[i]
    value_int = array_rns_to_int(value_vector)
    result_list.append(value_int)
  return result_list


def jax_rns_point_pack_to_int_point_batch(point_pack: jnp.ndarray):
  points = jnp.transpose(point_pack, (1, 0, 2))
  results = []
  for i in range(len(points)):
    results.append(jax_rns_array_to_int_list(points[i]))
  return results


def jax_point_pack_to_int_point(point: jax.Array):
  coordinate_num = point.shape[0]
  coordinates = []
  for i in range(coordinate_num):
    c = array_to_int(point[i], BASE)
    coordinates.append(c)
  return coordinates


def array_to_int(jax_array: jax.Array, base) -> int:
  """Converts a JAX array to a single Python integer."""
  result = 0

  for i, elem in enumerate(jax_array):
    result |= int(elem) << (i * base)

  return result


def int_to_array(
    python_int, base=BASE, dtype=jnp.uint16, array_size=U16_CHUNK_NUM
):
  """Chunk decompose a Python integer into a JAX array of fixed dtype and fixed size.

  Args:
    python_int: The Python integer to convert.
    base: The base of the integer representation of the coordinates.
    dtype: The data type of the JAX array. If None, the data type will be
      automatically determined based on the base.
    array_size: The size of the JAX array. If None, the array will have the
      minimum size necessary to store the integer.

  Note that: the default parameter is only for 384-bit data.

  Returns:
    A JAX array representing the integer.
  """
  mask = (1 << base) - 1

  # Chunk Decomposition
  elements = []
  while python_int > 0:
    elements.append(python_int & mask)  # Extract the lower bits
    python_int >>= base  # Shift to remove the extracted bits

  # we pad or trim the result to match the desired size
  if array_size is not None:
    assert array_size >= len(elements)
    elements = elements[:array_size] + [0] * (array_size - len(elements))

  return jnp.array(elements, dtype=dtype)


def int_list_to_jax_array(int_list, base=BASE, array_size=U16_CHUNK_NUM):
  """Converts a list of integers to a JAX array."""
  result = []
  for int_value in int_list:
    result.append(int_to_array(int_value, base, array_size=array_size))
  return jnp.array(result, dtype=jnp.uint16)


def jax_array_to_int_list(jax_array, base):
  """Converts JAX array to single integer."""
  result_list = []
  for i in range(jax_array.shape[0]):
    value_vector = jax_array[i]
    value_int = array_to_int(value_vector, base)
    result_list.append(value_int)
  return result_list


def int_list_to_2d_array(int_list, base, array_size=None) -> jnp.ndarray:
  """Converts a list of integers to a 2D JAX array."""
  chunked_arrays = []
  for int_value in int_list:
    chunked_arrays.append(int_to_array(int_value, base, array_size=array_size))
  return jnp.array(chunked_arrays)


def int_list_to_3d_array(int_list, base, array_size=None) -> jnp.ndarray:
  int_num = len(int_list)
  result_list = []
  i = 0
  for value in int_list:
    value_array = int_to_array(value, base, array_size=array_size).reshape(
        1, -1
    )
    value_array = jnp.pad(value_array, pad_width=((i, int_num - i - 1), (0, 0)))
    result_list.append(value_array)
    i += 1
  return jnp.array(result_list)


# RNS helpers
def total_modulus(moduli):
  modulus = 1  # Compute the big modulus
  for m in moduli:
    modulus *= m
  return modulus


def rns_precompute(moduli):
  modulus = total_modulus(moduli)
  precomputed = []
  for m in moduli:
    rest = modulus // m  # 0 mod all the other moduli
    inverse = pow(rest % m, -1, m)  # factor to make 1 mod this moduli
    icrt_val = (rest * inverse) % modulus  # combine
    precomputed.append(icrt_val)
  return precomputed


def rns_reconstruct(residues, moduli, precomputed):
  assert len(residues) == len(moduli)
  assert len(moduli) == len(precomputed)
  output = 0
  for i, r in enumerate(residues):
    output += precomputed[i] * int(r)
  return output % total_modulus(moduli)


def to_rns(x, moduli):
  assert x < total_modulus(moduli)
  return [x % m for m in moduli]


def ceil_div(x, y):
  return (x + y - 1) // y


def greedy_select(prio_lists, target):
  """Greedily selects moduli to reach a target modulus.

  Args:
    prio_lists: A list of lists of (modulus, bits, priority) tuples. The moduli
      should be relatively prime and in descending order of priority.
    target: The desired modulus.

  Returns:
    A list of (modulus, bits) tuples representing the selected moduli.

  Raises:
    ValueError: If the target modulus cannot be reached.
  """
  modulus = 1
  selected_moduli_list = []
  for prio_list in prio_lists:
    for modulus_bits_prio in prio_list:
      v = modulus_bits_prio[0]
      if gcd(v, modulus) == 1:
        selected_moduli_list.append(modulus_bits_prio)
        modulus *= v
        if modulus >= target:
          return selected_moduli_list
  print(modulus / target)
  raise ValueError("Did not reach target")


def next_odd_smaller(x):
  if x % 2 == 0:
    return x - 1
  return x


def gen_lists(bits_per_word):
  """Generates a list of lists of (modulus, bits, priority) tuples.

  Args:
    bits_per_word: The number of bits per word.

  Returns:
    A list of lists of (modulus, bits, priority) tuples.
  """
  bound = 2 ** (bits_per_word // 2)
  while (bound + 1) ** 2 + bound >= 2**bits_per_word:
    bound -= 1
  duh = [(2**bits_per_word, bits_per_word, 0)]
  small_t = [
      (2**bits_per_word - i, bits_per_word, i)
      for i in range(1, next_odd_smaller(bound), 2)
  ]
  bits_per_word -= 1
  small_t_extra = [
      (2**bits_per_word - i, bits_per_word, i)
      for i in range(1, next_odd_smaller(bound // 2), 2)
  ]
  small_t_pos = [
      (i, bits_per_word, 2**bits_per_word - i)
      for i in range(
          2**bits_per_word + 1,
          next_odd_smaller((bound // 2) + 2**bits_per_word),
          2,
      )
  ]
  prio_lists = [duh, small_t, small_t_extra, small_t_pos]
  return prio_lists


def gen_rns(bits_per_word, target):
  prio_lists = gen_lists(bits_per_word)
  return greedy_select(prio_lists, target)


def word_len(x, word_length):
  return (x.bit_length() + word_length - 1) // word_length


def byte_len(x):
  return word_len(x, 8)


def word_reinterpret(x, word_length, length=-1):
  if length == -1:
    length = word_len(x, word_length)
  assert x < (2**word_length) ** length
  return [
      (x >> (word_length * i)) & (2**word_length - 1) for i in range(length)
  ]


def byte_reinterpret(x, length=-1):
  return word_reinterpret(x, 8, length)


def int_reconstruct(x, word_length):
  return sum([int(d) << (word_length * i) for (i, d) in enumerate(x)])


def rns_precompute(moduli):
  modulus = total_modulus(moduli)
  precomputed = []
  for m in moduli:
    rest = modulus // m  # 0 mod all the other moduli
    inverse = pow(rest % m, -1, m)  # factor to make 1 mod this moduli
    icrt_val = (rest * inverse) % modulus  # combine
    precomputed.append(icrt_val)
  return precomputed


def int_to_array_rns(x):
  return [x % m for m in MODULI]


def array_rns_to_int(residues):
  rns_precompute_values = rns_precompute(MODULI)
  return rns_reconstruct(residues, MODULI, rns_precompute_values)


def int_list_to_jax_array_rns(int_list):
  """Converts a list of integers to a JAX array."""
  result = []
  for int_value in int_list:
    result.append(int_to_array_rns(int_value))
  return jnp.array(result, dtype=jnp.uint16)


def jax_array_rns_to_int_list(jax_array):
  """Converts JAX array to single integer."""
  result_list = []
  for i in range(jax_array.shape[0]):
    value_vector = jax_array[i]
    value_int = array_rns_to_int(value_vector)
    result_list.append(value_int)
  return result_list
