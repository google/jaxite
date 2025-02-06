"""library of finite field operations.

This library is used to implement the finite field operations for the high-precision
elliptic curve.

- Data Representation.
The high-precision elliptic curve coordinate is represented as a vector of
uint16 24-bit integer.
The actual base is increased with the index.
E.g.
    index     [  0,     1,     2, ...]
bit precision [0~7,  8~15, 16~23, ...]

It includes Barrett based modular multiplication, *barrett_reduction_u16x2*


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

import functools

import jax
import jax.numpy as jnp
import jaxite.jaxite_ec.util as utils


jax.config.update("jax_enable_x64", True)


@jax.named_call
@functools.partial(
    jax.jit, static_argnames=("iter_num", "mask", "chunk_shift_bits")
)
def carry_add(
    value_c: jax.Array,
    iter_num=utils.U16_CHUNK_NUM,
    mask=utils.U16_MASK,
    chunk_shift_bits=utils.U16_CHUNK_SHIFT_BITS,
):
  """The purpose of this API is to enable general-purposed carry add, where the following knobs are known before runtime.

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
  for _ in range(iter_num):
    low = jnp.bitwise_and(value_c, mask)
    high = jnp.right_shift(value_c, chunk_shift_bits)
    high = jnp.roll(high, 1)
    value_c = jnp.add(low, high)
  return value_c


@jax.named_call
@functools.partial(jax.jit, static_argnames="chunk_shift_bits")
def check_any_chunk_with_carry(
    value_c: jax.Array,
    chunk_shift_bits=utils.U16_CHUNK_SHIFT_BITS,
) -> jax.Array:
  """This function check whether any chunk of input vector 'value_c' has carry.

  Args:
    value_c: The value to carry add.
    chunk_shift_bits: ideal bit precision of any given chunk. Note that: actual
      bit precision of any given chunk might be higher than chunk_shift_bits
      because it needs to hold the overflow.

  Returns:
    cond: A boolean value indicating whether any chunk of input vector 'value_c'
    has carry.
  """
  high = jnp.right_shift(value_c, chunk_shift_bits)
  cond = jnp.any(jnp.not_equal(high, 0))
  return cond


@jax.named_call
@functools.partial(jax.jit, static_argnames=("mask", "chunk_shift_bits"))
def carry_propagation(
    value_c: jax.Array,
    mask=utils.U16_MASK,
    chunk_shift_bits=utils.U16_CHUNK_SHIFT_BITS,
):
  """The purpose of this API is to enable carry propagation.

  Args:
    value_c: The value to carry propagate.
    mask: 2**chunk_bitwidth - 1,
    chunk_shift_bits: chunk_bitwidth

  This function split each chunk into high and low parts, and high part is left
    roll by 1 to carry the overflowed bits to the next chunk.
  Note that: in a given jax.array, bit range of the chunk within the original
    high precision value is increased from left to the right.

  Returns:
    value_c: The value after carry adding.
  """
  low = jnp.bitwise_and(value_c, mask)
  high = jnp.right_shift(value_c, chunk_shift_bits)
  high = jnp.roll(high, 1)
  value_c = jnp.add(low, high)
  return value_c


def conv_1d_2u16xn(value_a: jax.Array, value_b: jax.Array):
  """This function performs a 1D convolution of two u16 arrays.

  Args:
    value_a: The chunk-decomposition representation of the high-precision int.
    value_b: The chunk-decomposition representation of the high-precision int.

  Returns:
    conv: The convolution results of two input arrays being casted to uint8.
  """
  value_a = jax.lax.bitcast_convert_type(value_a, jnp.uint8).reshape(-1)
  value_b = jax.lax.bitcast_convert_type(value_b, jnp.uint8).reshape(-1)
  conv = jnp.convolve(
      value_a,
      value_b,
      preferred_element_type=jnp.uint32,
  )
  return conv


@jax.named_call
@functools.partial(jax.jit, static_argnames=("chunk_num_u16", "chunk_num_u32"))
def chunk_reduction_after_conv(conv: jax.Array, chunk_num_u16, chunk_num_u32):
  """Given the carry add takes O(C) algorithm complexity, where C is the number of chunks.

  This function performs chunk reduction for ther results of the convolution,
  i.e. merge two consecutive chunks into one chunk with double precision.
  E.g. u8[0, 8, 8, 0] -> u16[8, 2048] 0-> u32[526336]

  Args:
    conv: The convolution result.
    chunk_num_u16: The number of bits in each chunk.
    chunk_num_u32: The number of bits in the second chunk.

  Returns:
    value_c: The result of the chunk reduction.
  """
  shift_0_8_u16x4 = jnp.array(
      [[0, 8] for _ in range(chunk_num_u16 * 4)], dtype=jnp.uint8
  )
  shift_0_16_u32x4 = jnp.array(
      [[0, 16] for _ in range(chunk_num_u32 * 4)], dtype=jnp.uint8
  )
  new_shape = conv.shape[:-1] + (-1, 2) if conv.ndim == 2 else (-1, 2)
  value_c = conv.reshape(new_shape)
  value_c = jnp.left_shift(value_c, shift_0_8_u16x4[:chunk_num_u16])
  value_c = jnp.sum(value_c, axis=-1)
  value_c = value_c.reshape(new_shape).astype(jnp.uint64)
  value_c = jnp.left_shift(value_c, shift_0_16_u32x4[:chunk_num_u32])
  value_c = jnp.sum(value_c, axis=-1)
  return value_c


@jax.named_call
@functools.partial(jax.jit, static_argnames="chunk_num_u16")
def compare_u16(
    value_a: jax.Array, value_b: jax.Array, chunk_num_u16=utils.U16_CHUNK_NUM
):
  """Compare two u16 values.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.
    chunk_num_u16: The number of chunks in the u16 value.

  Returns:
  cond > 0 -> value_a > value_b
  cond = 0 -> value_a = value_b
  cond < 0 -> value_a < value_b
  """
  sign = jnp.sign(
      jnp.subtract(value_a.astype(jnp.int32), value_b.astype(jnp.int32))
  )
  comp_check_vec_weights = jnp.array(
      [2**i for i in range(chunk_num_u16)], dtype=jnp.int32
  )
  weight = jnp.multiply(sign, comp_check_vec_weights)
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
  value_c = jax.lax.while_loop(
      check_any_chunk_with_carry, carry_propagation, value_c
  )
  return value_c.astype(jnp.uint16)


def add_3u16(value_a: jax.Array, value_b: jax.Array, value_d: jax.Array):
  value_c = jax.numpy.add(
      value_a.astype(jnp.uint32), value_b.astype(jnp.uint32)
  )
  value_c = jax.numpy.add(
      value_c.astype(jnp.uint32), value_d.astype(jnp.uint32)
  )
  value_c = jax.lax.while_loop(
      check_any_chunk_with_carry, carry_propagation, value_c
  )
  return value_c.astype(jnp.uint16)


@jax.named_call
@functools.partial(jax.jit, static_argnames=("mask", "chunk_num_u16"))
def sub_2u16(
    value_a: jax.Array,
    value_b: jax.Array,
    mask=utils.U16_MASK,
    chunk_num_u16=utils.U16_CHUNK_NUM,
):
  """Subtract two u16 values.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.
    mask: The mask to apply to the value.
    chunk_num_u16: The number of chunks in the u16 value (default: 24).

  Returns:
    value_c: The result of the subtraction.
  """
  borrow_high_u16_pad_zero_array = jnp.array(
      [0] + [1] * (chunk_num_u16 - 2) + [0], dtype=jnp.uint32
  )
  borrow_low_u16_array = jnp.array(
      [mask + 1] * (chunk_num_u16 - 1) + [0], dtype=jnp.uint32
  )
  value_a = jnp.add(value_a.astype(jnp.uint32), borrow_low_u16_array)
  value_c = jnp.subtract(value_a, value_b)
  value_c = jnp.subtract(value_c, borrow_high_u16_pad_zero_array)

  value_c = jax.lax.while_loop(
      check_any_chunk_with_carry, carry_propagation, value_c
  )
  value_c = value_c.at[chunk_num_u16 - 1].set(value_c[chunk_num_u16 - 1] - 1)

  value_c = value_c.astype(jnp.uint16)
  return value_c


@jax.named_call
@functools.partial(
    jax.jit, static_argnames=("modulus_377_int", "chunk_num_u16")
)
def cond_sub_mod_u16(
    value_a: jax.Array,
    modulus_377_int=utils.MODULUS_377_INT,
    chunk_num_u16=utils.U16_CHUNK_NUM,
):
  """Perform conditional subtraction: value_a - modulus_377_int.

  Args:
    value_a: The minuend.
    modulus_377_int: The modulus 377.
    chunk_num_u16: The number of chunks in the u16 value (default: 24).

  Returns:
    value_c: The result of the conditional subtraction.
  """
  modulus_377_int_array = utils.int_to_array(
      modulus_377_int, 16, jnp.uint16, chunk_num_u16
  )
  cond = compare_u16(value_a, modulus_377_int_array)
  value_b = sub_2u16(value_a, modulus_377_int_array)
  value_c = jax.lax.select(jax.lax.ge(cond, 0), value_b, value_a)
  return value_c


@jax.named_call
@functools.partial(
    jax.jit, static_argnames=("modulus_377_int", "chunk_num_u16")
)
def cond_sub_2u16(
    value_a: jax.Array,
    value_b: jax.Array,
    modulus_377_int=utils.MODULUS_377_INT,
    chunk_num_u16=utils.U16_CHUNK_NUM,
):
  """Perform conditional subtraction: value_a - value_b.

  Args:
    value_a: The minuend.
    value_b: The subtrahend.
    modulus_377_int: The modulus 377.
    chunk_num_u16: The number of chunks in the u16 value (default: 24).

  Returns:
    value_c: The result of the conditional subtraction.
  """
  modulus_377_int_array = utils.int_to_array(
      modulus_377_int, 16, jnp.uint16, chunk_num_u16
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


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=(
        "mask",
        "chunk_num_u16",
        "chunk_shift_bits",
        "output_dtype",
    ),
)
def mul_2u16(
    value_a: jax.Array,
    value_b: jax.Array,
    mask=utils.U32_MASK,
    chunk_num_u16=utils.U16_CHUNK_NUM,
    chunk_shift_bits=utils.U32_CHUNK_SHIFT_BITS,
    output_dtype=jnp.uint16,
):
  """Multiply two u16 values.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.
    mask: The mask to apply to the value.
    chunk_num_u16: The number of chunks in the u16 value (default: 24).
    chunk_shift_bits: The number of bits to shift the value.
    output_dtype: The desired output data type.

  Returns:

  cond > 0 -> value_a > value_b
  cond = 0 -> value_a = value_b
  cond < 0 -> value_a < value_b
  """
  conv = conv_1d_2u16xn(value_a, value_b)
  conv = jnp.pad(conv, (0, 1))
  value_c = chunk_reduction_after_conv(conv, 2 * chunk_num_u16, chunk_num_u16)

  value_c = jax.lax.while_loop(
      functools.partial(
          check_any_chunk_with_carry, chunk_shift_bits=chunk_shift_bits
      ),
      functools.partial(
          carry_propagation, mask=mask, chunk_shift_bits=chunk_shift_bits,
      ),
      value_c,
  )
  ratio = 4 if output_dtype == jnp.uint8 else 2
  value_c = jax.lax.bitcast_convert_type(
      value_c.astype(jnp.uint32), output_dtype
  ).reshape(-1)[: ratio * chunk_num_u16]
  return value_c


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=("barrett_shift_u8", "chunk_num_u16", "chunk_num_u32"),
)
def mul_shift_2u16x2x1(
    value_a: jax.Array,
    value_b: jax.Array,
    barrett_shift_u8=utils.BARRETT_SHIFT_U8,
    chunk_num_u16=utils.U16_CHUNK_NUM,
    chunk_num_u32=utils.U32_CHUNK_NUM,
):
  """Multiply and shift two u16 values.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.
    barrett_shift_u8: The number of bits to shift the value.
    chunk_num_u16: The number of chunks in the u16 value.
    chunk_num_u32: The number of chunks in the u32 value.

  Returns:

  cond > 0 -> value_a > value_b
  cond = 0 -> value_a = value_b
  cond < 0 -> value_a < value_b
  """
  conv = conv_1d_2u16xn(value_a, value_b)
  conv = jnp.pad(conv, (0, 1))
  value_c = chunk_reduction_after_conv(
      conv, chunk_num_u16 * 3, chunk_num_u32 * 3
  )
  value_c = jax.lax.while_loop(
      functools.partial(check_any_chunk_with_carry, chunk_shift_bits=32),
      functools.partial(
          carry_propagation, mask=0xFFFFFFFF, chunk_shift_bits=32
      ),
      value_c,
  )

  value_c = jax.lax.bitcast_convert_type(
      value_c.astype(jnp.uint32), jnp.uint8
  ).reshape(-1)[barrett_shift_u8:]
  value_c = jax.lax.bitcast_convert_type(
      jnp.pad(value_c, (0, 1)).reshape(-1, 2), jnp.uint16
  )[:chunk_num_u16]
  return value_c


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=("mask", "modulus_377_int", "mu_377_int", "chunk_num_u16"),
)
def mod_mul_barrett_2u16(
    value_a: jax.Array,
    value_b: jax.Array,
    mask=utils.U16_MASK,
    modulus_377_int=utils.MODULUS_377_INT,
    mu_377_int=utils.MU_377_INT,
    chunk_num_u16=utils.U16_CHUNK_NUM,
):
  """Multiply two u16 values with Barrett reduction.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.
    mask: The mask to apply to the value.
    modulus_377_int: The modulus 377.
    mu_377_int: The Barrett reduction coefficient.
    chunk_num_u16: The number of chunks in the u16 value (default: 24).

  Returns:
    value_c: The result of the multiplication.
  """
  modulus_377_int_array = utils.int_to_array(
      modulus_377_int, 16, jnp.uint16, chunk_num_u16
  )
  mu_377_int_array = utils.int_to_array(
      mu_377_int, 16, jnp.uint16, chunk_num_u16
  )

  value_x = mul_2u16(value_a, value_b)
  value_d = mul_shift_2u16x2x1(value_x, mu_377_int_array)
  value_e = mul_2u16(value_d, modulus_377_int_array)
  value_t = sub_2u16(value_x, value_e, mask, chunk_num_u16 * 2)
  value_c = cond_sub_mod_u16(value_t[:chunk_num_u16])
  return value_c


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=("mask", "modulus_377_int", "mu_377_int", "chunk_num_u16"),
)
def barrett_reduction_u16x2(
    value_x: jax.Array,
    mask=utils.U16_MASK,
    modulus_377_int=utils.MODULUS_377_INT,
    mu_377_int=utils.MU_377_INT,
    chunk_num_u16=utils.U16_CHUNK_NUM,
):
  """Performs Barrett reduction on a u16x2 value.

  Args:
    value_x: The u16x2 value to perform Barrett reduction on.
    mask: The mask to apply to the value.
    modulus_377_int: The modulus 377.
    mu_377_int: The Barrett reduction coefficient.
    chunk_num_u16: The number of chunks in the u16 value (default: 24).

  Returns:
    value_c: The result of the Barrett reduction.
  """
  modulus_377_int_array = utils.int_to_array(
      modulus_377_int, 16, jnp.uint16, chunk_num_u16
  )
  mu_377_int_array = utils.int_to_array(
      mu_377_int, 16, jnp.uint16, chunk_num_u16
  )

  value_d = mul_shift_2u16x2x1(value_x, mu_377_int_array)
  value_e = mul_2u16(value_d, modulus_377_int_array)
  value_t = sub_2u16(value_x, value_e, mask, chunk_num_u16 * 2)
  value_c = cond_sub_mod_u16(value_t)
  return value_c


def construct_lazy_matrix(
    p, chunk_precision=8, chunk_num_u8=utils.U8_CHUNK_NUM
):
  """Construct the lazy matrix.

  Args:
    p: The modulus.
    chunk_precision: The chunk precision.
    chunk_num_u8: The number of chunks in the u8 value.

  Returns:
    lazy_mat: The lazy matrix.

  Note that: this function runs on CPU of the TPU-VM, which cannot be jitted.
  """
  jax.config.update("jax_enable_x64", True)
  lazy_mat_list = []
  for i in range(chunk_num_u8 + 4):
    val = int(int(256) ** (chunk_num_u8 + i)) % p
    lazy_mat_list.append(
        utils.int_to_array(val, chunk_precision, array_size=chunk_num_u8)
    )
  return jnp.array(lazy_mat_list)


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=("mask", "chunk_num_u8", "chunk_shift_bits"),
)
def mod_mul_lazy_2u16(
    value_a,
    value_b,
    modulus_lazy_mat,
    mask=utils.U32_MASK,
    chunk_num_u8=utils.U8_CHUNK_NUM,
    chunk_shift_bits=32,
):
  """Multiply two u16 values with lazy matrix reduction.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.
    modulus_lazy_mat: The lazy matrix.
    mask: The mask to apply to the value.
    chunk_num_u8: The number of chunks in the u8 value.

  Returns:
    value_c: The result of the multiplication.
  """
  batch_dim = value_a.shape[0]
  mul_2u8 = functools.partial(
      mul_2u16,
      mask=utils.U32_MASK,
      chunk_num_u16=utils.U16_CHUNK_NUM,
      chunk_shift_bits=utils.U32_CHUNK_SHIFT_BITS,
      output_dtype=jnp.uint8,
  )
  value_c = jax.vmap(mul_2u8)(value_a, value_b)
  standard_product_low = value_c[:, :chunk_num_u8]
  standard_product_high = value_c[:, chunk_num_u8:]

  standard_product_high = jnp.pad(standard_product_high, ((0, 0), (0, 4)))
  reduced = jnp.matmul(
      standard_product_high.astype(jnp.uint16),
      modulus_lazy_mat.astype(jnp.uint16),
      preferred_element_type=jnp.uint32,
  )
  value_c_reduced = jnp.add(
      standard_product_low.astype(jnp.uint32), reduced.astype(jnp.uint32)
  )
  value_c_reduced_u32 = chunk_reduction_after_conv(
      value_c_reduced, chunk_num_u8 // 2, chunk_num_u8 // 4
  )
  value_c_reduced_u32 = jnp.pad(value_c_reduced_u32, ((0, 0), (0, 1)))

  value_c_carried = jax.lax.while_loop(
      functools.partial(
          check_any_chunk_with_carry, chunk_shift_bits=chunk_shift_bits
      ),
      functools.partial(
          carry_propagation, mask=mask, chunk_shift_bits=chunk_shift_bits
      ),
      value_c_reduced_u32,
  )

  value_c_u16 = jax.lax.bitcast_convert_type(
      value_c_carried.astype(jnp.uint32), jnp.uint16
  ).reshape(batch_dim, -1)
  return value_c_u16
