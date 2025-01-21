"""library of finite field operations.

# Function Name Terminology
## <func>_<in_datatype> indicates that the function only works for a single
## precision.
## <func> indicates that the function works for general bit precision.

# Terminology
## Chunk Reduction: e.g. u8-chunk -> u16-chunk or u32-chunk
## Chunk Decomposition <-> Chunk Merge:
### Chunk Decomposition: break int into multiple low-precision chunks.
### Chunk Merge: Merge multiple low-precision chunks into an int.
"""

import jax
import jax.numpy as jnp
import jaxite.jaxite_ec.util as utils

jax.config.update("jax_enable_x64", True)


MASK_16 = 0xFFFF
MASK_32 = 0xFFFFFFFF

U64_CHUNK_NUM = 6
U32_CHUNK_NUM = 12
U16_CHUNK_NUM = 24

U16_MAX = 0xFFFF
BASE = 16
BASE_TYPE = jnp.uint16
borrow_low_u16 = [U16_MAX + 1] * (U16_CHUNK_NUM - 1)
borrow_low_u16.append(0)
borrow_low_u16x2 = [U16_MAX + 1] * (2 * U16_CHUNK_NUM - 1)
borrow_low_u16x2.append(0)
borrow_high_u16 = [0]
borrow_high_u16 = borrow_high_u16 + [1] * (U16_CHUNK_NUM - 1)
borrow_high_u16x2 = borrow_high_u16 + [1] * U16_CHUNK_NUM
borrow_high_u16_pad_zero = borrow_high_u16[0 : U16_CHUNK_NUM - 1] + [0]
borrow_high_u16x2_pad_zero = borrow_high_u16x2[0 : 2 * U16_CHUNK_NUM - 1] + [0]

MODULUS_377_INT = 0x01AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001

# BARRETT Params for k = 380
BARRETT_SHIFT_U8 = 95
MU_377_INT = 0x98542343310183A5DB0F28160BBD3DCEEEB43799DDAC681ABCB52236169B40B43B5A1DE2710A9647E7F56317936BFF32


def carry_add_u16(value_c: jax.Array):
  mask = MASK_16
  chunk_bits = 16
  for i in range(U16_CHUNK_NUM):
    low = jnp.bitwise_and(value_c, mask)
    high = jnp.right_shift(value_c, chunk_bits)
    high = jnp.roll(high, 1)
    value_c = jnp.add(low, high)
  return value_c


def carry_add_u16x2(value_c: jax.Array):
  mask = MASK_16
  chunk_bits = 16
  for i in range(U16_CHUNK_NUM * 2):
    low = jnp.bitwise_and(value_c, mask)
    high = jnp.right_shift(value_c, chunk_bits)
    high = jnp.roll(high, 1)
    value_c = jnp.add(low, high)
  return value_c


def carry_add(value_c: jax.Array, iter_num, mask, chunk_shift_bits):
  """The purpose of this API is to enable general-purposed carry add, where the following knobs are generated offline.

  iter_num = math.ceil(total_input_bitwidth / chunk_bitwidth),
  mask = 2**chunk_bitwidth - 1,
  chunk_shift_bits = chunk_bitwidth

  Args:
    value_c: The value to carry add.
    iter_num: The number of iterations to perform.
    mask: The mask to apply to the value.
    chunk_shift_bits: The number of bits to shift the value.

  Returns:
    value_c: The value after carry adding.
  """
  for i in range(iter_num):
    low = jnp.bitwise_and(value_c, mask)
    high = jnp.right_shift(value_c, chunk_shift_bits)
    high = jnp.roll(high, 1)
    value_c = jnp.add(low, high)
  return value_c


def carry_add_while_cond_u16(value_c: jax.Array):
  chunk_shift_bits = 16
  high = jnp.right_shift(value_c, chunk_shift_bits)
  # high = jnp.sum(high)
  cond = jnp.not_equal(high, 0)
  return jnp.any(cond)


def carry_add_while_body_u16(value_c: jax.Array):
  mask = MASK_16
  chunk_shift_bits = 16
  low = jnp.bitwise_and(value_c, mask)
  high = jnp.right_shift(value_c, chunk_shift_bits)
  high = jnp.roll(high, 1)
  value_c = jnp.add(low, high)
  return value_c


def carry_add_while_cond_u32(value_c: jax.Array):
  chunk_shift_bits = 32
  high = jnp.right_shift(value_c, chunk_shift_bits)
  # high = jnp.sum(high)
  cond = jnp.not_equal(high, 0)
  return jnp.any(cond)


def carry_add_while_body_u32(value_c: jax.Array):
  mask = MASK_32
  chunk_shift_bits = 32
  low = jnp.bitwise_and(value_c, mask)
  high = jnp.right_shift(value_c, chunk_shift_bits)
  high = jnp.roll(high, 1)
  value_c = jnp.add(low, high)
  return value_c


def carry_add_while(value_c: jax.Array, cond, body):
  value_c = jax.lax.while_loop(cond, body, value_c)
  return value_c


def conv_1d_2u16xn(value_a: jax.Array, value_b: jax.Array):
  value_a = jax.lax.bitcast_convert_type(value_a, jnp.uint8).reshape(-1)
  value_b = jax.lax.bitcast_convert_type(value_b, jnp.uint8).reshape(-1)
  conv = jnp.convolve(
      value_a.astype(jnp.uint16),
      value_b.astype(jnp.uint16),
      preferred_element_type=jnp.uint32,
  )
  return conv


def chunk_reduction_after_conv(conv: jax.Array):
  chunk_bits = 8
  chunk_bits_2 = 16

  a0 = conv[::2]
  a1 = conv[1::2]
  value_c = jnp.add(a0, jnp.left_shift(a1, chunk_bits))
  b0 = value_c[::2].astype(jnp.uint64)
  b1 = value_c[1::2].astype(jnp.uint64)
  value_c = jnp.add(b0, jnp.left_shift(b1, chunk_bits_2))
  return value_c


def chunk_reduction_after_conv_v2(
    conv: jax.Array, chunk_num_u16, chunk_num_u32
):
  """Chunk reduction after convolution.

  Args:
    conv: The convolution result.
    chunk_num_u16: The number of u16 chunks.
    chunk_num_u32: The number of u32 chunks.

  Returns:
    value_c: The result of the chunk reduction.
  """
  shift_0_8_u16x4 = jnp.array(
      [[0, 8] for _ in range(U16_CHUNK_NUM * 4)], dtype=jnp.uint8
  )
  shift_0_16_u32x4 = jnp.array(
      [[0, 16] for _ in range(U32_CHUNK_NUM * 4)], dtype=jnp.uint8
  )
  value_c = conv.reshape((-1, 2))
  value_c = jnp.left_shift(value_c, shift_0_8_u16x4[:chunk_num_u16])
  value_c = jnp.sum(value_c, axis=1)
  value_c = value_c.reshape((-1, 2)).astype(jnp.uint64)
  value_c = jnp.left_shift(value_c, shift_0_16_u32x4[:chunk_num_u32])
  value_c = jnp.sum(value_c, axis=1)
  return value_c


def compare_u16(value_a: jax.Array, value_b: jax.Array):
  """Compare two u16 values.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.

  Returns:

  cond > 0 -> value_a > value_b
  cond = 0 -> value_a = value_b
  cond < 0 -> value_a < value_b
  """
  sign = jnp.sign(
      jnp.subtract(value_a.astype(jnp.int32), value_b.astype(jnp.int32))
  )
  comp_check_vec_weights_24 = jnp.array(
      [2**i for i in range(24)], dtype=jnp.int32
  )
  weight = jnp.multiply(sign, comp_check_vec_weights_24)
  cond = weight.sum()
  return cond


def add_2u16(value_a: jax.Array, value_b: jax.Array):
  """Add two u16 values.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.

  Returns:
    value_c: The result of the addition.
  """
  value_c = jax.numpy.add(
      value_a.astype(jnp.uint32), value_b.astype(jnp.uint32)
  )
  value_c = carry_add_while(
      value_c, carry_add_while_cond_u16, carry_add_while_body_u16
  )
  return value_c.astype(jnp.uint16)


def add_3u16(value_a: jax.Array, value_b: jax.Array, value_d: jax.Array):
  value_c = jax.numpy.add(
      value_a.astype(jnp.uint32), value_b.astype(jnp.uint32)
  )
  value_c = jax.numpy.add(
      value_c.astype(jnp.uint32), value_d.astype(jnp.uint32)
  )
  value_c = carry_add_while(
      value_c, carry_add_while_cond_u16, carry_add_while_body_u16
  )
  return value_c.astype(jnp.uint16)


def sub_2u16(value_a: jax.Array, value_b: jax.Array):
  """Subtract two u16 values.

  Args:
    value_a: The minuend.
    value_b: The subtrahend.

  Returns:
    value_c: The result of the subtraction.
  """
  borrow_high_u16_pad_zero_array = jnp.array(
      borrow_high_u16_pad_zero, dtype=jnp.uint32
  )
  borrow_low_u16_array = jnp.array(borrow_low_u16, dtype=jnp.uint32)
  value_a = jnp.add(value_a.astype(jnp.uint32), borrow_low_u16_array)
  value_c = jnp.subtract(value_a, value_b)
  value_c = jnp.subtract(value_c, borrow_high_u16_pad_zero_array)

  value_c = carry_add_while(
      value_c, carry_add_while_cond_u16, carry_add_while_body_u16
  )
  value_c = value_c.at[U16_CHUNK_NUM - 1].set(value_c[U16_CHUNK_NUM - 1] - 1)

  value_c = value_c.astype(jnp.uint16)[:U16_CHUNK_NUM]
  return value_c


def sub_2u16x2(value_a: jax.Array, value_b: jax.Array):
  """Subtract two u16 values.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.

  Returns:

  cond > 0 -> value_a > value_b
  cond = 0 -> value_a = value_b
  cond < 0 -> value_a < value_b
  """
  borrow_high_u16x2_pad_zero_array = jnp.array(
      borrow_high_u16x2_pad_zero, dtype=jnp.uint32
  )
  borrow_low_u16x2_array = jnp.array(borrow_low_u16x2, dtype=jnp.uint32)
  value_a = jnp.add(value_a.astype(jnp.uint32), borrow_low_u16x2_array)
  value_c = jnp.subtract(value_a, value_b)
  value_c = jnp.subtract(value_c, borrow_high_u16x2_pad_zero_array)

  # valueC = carry_add_while(valueC, carry_add_while_
  value_c = carry_add_u16x2(value_c)  # While Loop has half worst case
  value_c = value_c.at[U16_CHUNK_NUM * 2 - 1].set(
      value_c[U16_CHUNK_NUM * 2 - 1] - 1
  )

  # result is in U16_CHUNK_NUM
  value_c = value_c.astype(jnp.uint16)[:U16_CHUNK_NUM]
  return value_c


def sub_2u16x2_barrett(value_a: jax.Array, value_b: jax.Array):
  value_a = value_a[:U16_CHUNK_NUM]
  value_b = value_b[:U16_CHUNK_NUM]
  value_c = sub_2u16(value_a, value_b)
  return value_c


def cond_sub_mod_u16(value_a: jax.Array):
  modulus_377_int_array = utils.int_to_array(
      MODULUS_377_INT, 16, jnp.uint16, U16_CHUNK_NUM
  )
  cond = compare_u16(value_a, modulus_377_int_array)
  value_b = sub_2u16(value_a, modulus_377_int_array)

  value_c = jax.lax.select(jax.lax.ge(cond, 0), value_b, value_a)
  return value_c


def cond_sub_2u16(value_a: jax.Array, value_b: jax.Array):
  """Conditional subtraction of two u16 values.

  Args:
    value_a: The minuend.
    value_b: The subtrahend.

  Returns:
    value_c: The result of the conditional subtraction.
  """
  modulus_377_int_array = utils.int_to_array(
      MODULUS_377_INT, 16, jnp.uint16, U16_CHUNK_NUM
  )
  cond = compare_u16(value_a, value_b)

  value_ap = jnp.add(
      value_a.astype(jnp.uint32), modulus_377_int_array.astype(jnp.uint32)
  )

  value_a = jax.lax.select(
      jax.lax.ge(cond, 0), value_a.astype(jnp.uint32), value_ap
  )
  value_c = sub_2u16(value_a, value_b)
  return value_c


def mul_2u16(value_a: jax.Array, value_b: jax.Array):
  """Multiply two u16 values.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.

  Returns:

  cond > 0 -> value_a > value_b
  cond = 0 -> value_a = value_b
  cond < 0 -> value_a < value_b
  """
  conv = conv_1d_2u16xn(value_a, value_b)
  conv = jnp.pad(conv, (0, 1))
  value_c = chunk_reduction_after_conv_v2(
      conv, 2 * U16_CHUNK_NUM, 2 * U32_CHUNK_NUM
  )
  value_c = carry_add_while(
      value_c, carry_add_while_cond_u32, carry_add_while_body_u32
  )

  value_c = jax.lax.bitcast_convert_type(
      value_c.astype(jnp.uint32), jnp.uint16
  ).reshape(-1)[: U16_CHUNK_NUM * 2]
  return value_c


def mul_shift_2u16x2x1(value_a: jax.Array, value_b: jax.Array):
  """Multiply and shift two u16 values.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.

  Returns:

  cond > 0 -> value_a > value_b
  cond = 0 -> value_a = value_b
  cond < 0 -> value_a < value_b
  """
  conv = conv_1d_2u16xn(value_a, value_b)
  conv = jnp.pad(conv, (0, 1))
  value_c = chunk_reduction_after_conv_v2(
      conv, U16_CHUNK_NUM * 3, U32_CHUNK_NUM * 3
  )
  value_c = carry_add_while(
      value_c, carry_add_while_cond_u32, carry_add_while_body_u32
  )

  value_c = jax.lax.bitcast_convert_type(
      value_c.astype(jnp.uint32), jnp.uint8
  ).reshape(-1)[BARRETT_SHIFT_U8:]
  value_c = jax.lax.bitcast_convert_type(
      jnp.pad(value_c, (0, 1)).reshape(-1, 2), jnp.uint16
  )[:U16_CHUNK_NUM]
  return value_c


def mod_mul_barrett_2u16(value_a: jax.Array, value_b: jax.Array):
  """Multiply two u16 values with Barrett reduction.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.

  Returns:
    value_c: The result of the multiplication.
  """
  modulus_377_int_array = utils.int_to_array(
      MODULUS_377_INT, 16, jnp.uint16, U16_CHUNK_NUM
  )
  mu_377_int_array = utils.int_to_array(
      MU_377_INT, 16, jnp.uint16, U16_CHUNK_NUM
  )

  value_x = mul_2u16(value_a, value_b)
  value_d = mul_shift_2u16x2x1(value_x, mu_377_int_array)
  value_e = mul_2u16(value_d, modulus_377_int_array)
  # value_t = sub_2u16x2(value_x, value_e)
  value_t = sub_2u16x2_barrett(value_x, value_e)
  value_c = cond_sub_mod_u16(value_t)
  return value_c


def barrett_reduction_u16x2(value_x: jax.Array):
  """Performs Barrett reduction on a u16x2 value.

  Args:
    value_x: The u16x2 value to perform Barrett reduction on.

  Returns:
    value_c: The result of the Barrett reduction.
  """
  modulus_377_int_array = utils.int_to_array(
      MODULUS_377_INT, 16, jnp.uint16, U16_CHUNK_NUM
  )
  mu_377_int_array = utils.int_to_array(
      MU_377_INT, 16, jnp.uint16, U16_CHUNK_NUM
  )

  value_d = mul_shift_2u16x2x1(value_x, mu_377_int_array)
  value_e = mul_2u16(value_d, modulus_377_int_array)
  value_t = sub_2u16x2(value_x, value_e)
  # value_t = sub_2u16x2_barrett(value_x, value_e)
  value_c = cond_sub_mod_u16(value_t)
  return value_c
