"""Global Configuration for running jaxite_word.

The default input data type is 64 bit integer.
"""

import math
import jax
import jax.numpy as jnp

gcd = math.gcd

# Always Fixed Parameters
BASE = 16
BASE_TYPE = jnp.uint16  # this type must match the BASE, i.e. jnp.uint<BASE>
U16_MASK = 0xFFFF
U32_MASK = 0xFFFFFFFF
U16_CHUNK_SHIFT_BITS = 16
U32_CHUNK_SHIFT_BITS = 3
MODULUS_DEFAULT = 1152921504606748673
MU_DEFAULT = 1152921504606945279

# Workload Dependent Parameters
NUM_ELEMENTS = 2
NUM_TOWERS = 3
NUM_DEGREE = 65536
U32_CHUNK_NUM_DEFAULT = 2
U16_CHUNK_NUM_DEFAULT = 4
U8_CHUNK_NUM_DEFAULT = 8
U32_CHUNK_NUM_U32 = 1
U16_CHUNK_NUM_U32 = 2
U8_CHUNK_NUM_U32 = 4

MODULUS_LIST = (1152921504606748673, 268664833, 557057)
MU_LIST = (1152921504606945279, 268206274, 493446)

MODULUS_ARRAY_ALL = ((32769, 65534, 65535, 4095), (32769, 4099), (32769, 8))
MU_ARRAY_ALL = (
    (32767, 1, 0, 4096),
    (32962, 4092),
    (34694, 7),
)

BARRETT_SHIFT_U8_ALL = (15, 7, 4.75)
BARRETT_SHIFT_U16_ALL = (7.5, 3.5, 2.375)
U32_CHUNK_NUM_ALL = (2, 1, 1)
U16_CHUNK_NUM_ALL = (4, 2, 2)
U8_CHUNK_NUM_ALL = (8, 4, 4)

## modulus set 1
# MODULUS_64 = 1152921504606748673
# MU_64 = 1152921504606945279
# BARRETT_SHIFT_U8 = 15
# BARRETT_SHIFT_U16 = 7.5
# MODULUS_ARRAY = (32769, 65534, 65535, 4095)
# MU_ARRAY = (32767, 1, 0, 4096)
# U32_CHUNK_NUM = 2
# U16_CHUNK_NUM = 4
# U8_CHUNK_NUM = 8

## modulus set 2
MODULUS_32 = 268664833
MU_32 = 268206274
BARRETT_SHIFT_U8 = 7
BARRETT_SHIFT_U16 = 3.5
MODULUS_ARRAY = (32769, 4099)
MU_ARRAY = (32962, 4092)
U32_CHUNK_NUM = 1
U16_CHUNK_NUM = 2
U8_CHUNK_NUM = 4

## modulus set 3
# MODULUS_64 = 557057
# MU_64 = 493446
# BARRETT_SHIFT_U8 = 4.75
# BARRETT_SHIFT_U16 = 2.375
# MODULUS_ARRAY = (32769, 8)
# MU_ARRAY = (34694, 7)

# NTT Test
NTT_BATCH_SIZE = 128
NTT_N1 = 64
NTT_N2 = 128
NTT_DEGREE = 8192

# Lazy Reduction
U16_EXT_CHUNK_NUM = 5


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


def array_to_int(jax_array: jax.Array, base) -> int:
  """Converts a JAX array to a single Python integer."""
  result = 0

  for i, elem in enumerate(jax_array):
    result |= int(elem) << (i * base)

  return result


def compute_barrett_mu(modulus):
  """Compute the Barrett reduction constant mu for a given modulus m.

  Args:
    modulus (int): The modulus.

  Returns:
    tuple: (mu, k) where mu is the precomputed constant and k is the number of
    digits in base b.
  """
  # k is the smallest integer such that m < b^k.
  b = 2 ** (math.floor(math.log(modulus)) + 1)
  # For m < 2^64, k will be 1.
  k_val = math.floor(math.log(modulus, b)) + 1

  # Compute mu = floor(b^(2k) / m)
  barrett_mu = (b ** (2 * k_val)) // modulus
  return barrett_mu, k_val


def int_list_to_jax_array(int_list, base=BASE, array_size=U16_CHUNK_NUM):
  """Converts a (potentially multi-dimensional) list of integers to a JAX array."""

  def recursive_convert(lst):
    if isinstance(lst, list):
      return [recursive_convert(item) for item in lst]
    else:
      return int_to_array(lst, base, array_size=array_size)

  result = recursive_convert(int_list)
  return jnp.array(result, dtype=jnp.uint16)


def jax_array_to_int_list(jax_array, base):
  """Converts a (potentially multi-dimensional) JAX array into a nested list of integers.

  The function recursively traverses the array until it reaches a 1D vector,
  then applies `array_to_int` to convert that vector into an integer.

  Args:
      jax_array: The JAX array to convert.
      base: The base of the integer representation.

  Returns:
      A nested list of integers.
  """
  if jax_array.ndim == 1:
    return array_to_int(jax_array, base)
  else:
    return [jax_array_to_int_list(sub_array, base) for sub_array in jax_array]


def random_list(shape, max_val, dtype=jnp.int32):
  return jax.random.randint(
      jax.random.key(0), shape=shape, minval=0, maxval=max_val, dtype=dtype
  ).tolist()


def random_array(shape, max_val, dtype=jnp.int32):
  return jax.random.randint(
      jax.random.key(0), shape=shape, minval=0, maxval=max_val, dtype=dtype
  )
