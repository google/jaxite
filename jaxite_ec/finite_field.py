"""library of finite field operations.

This library is used to implement the finite field operations for the
high-precision
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
from jaxite.jaxite_ec import util
import numpy as np


total_modulus = util.total_modulus
to_rns = util.to_rns


jax.config.update("jax_enable_x64", True)


@jax.named_call
@functools.partial(
    jax.jit, static_argnames=("iter_num", "mask", "chunk_shift_bits")
)
def carry_add(
    value_c: jax.Array,
    iter_num=util.U16_CHUNK_NUM,
    mask=util.U16_MASK,
    chunk_shift_bits=util.U16_CHUNK_SHIFT_BITS,
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
    chunk_shift_bits=util.U16_CHUNK_SHIFT_BITS,
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
    mask=util.U16_MASK,
    chunk_shift_bits=util.U16_CHUNK_SHIFT_BITS,
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
  precision_dim = value_c.shape[-1]
  roll_mat = jnp.array(
      [0, 1]
      + ([0] * (precision_dim) + [1]) * (precision_dim - 2)
      + [1]
      + [0] * (precision_dim - 1),
      dtype=jnp.uint16,
  ).reshape(precision_dim, precision_dim)
  low = jnp.bitwise_and(value_c, mask)
  high = jnp.right_shift(value_c, chunk_shift_bits).astype(jnp.uint16)
  high = jnp.matmul(high, roll_mat, preferred_element_type=jnp.uint32).astype(
      jnp.uint16
  )
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
def rechunkify(mul_result: jax.Array, chunk_num_u16, chunk_num_u32):
  """Given the carry add takes O(C) algorithm complexity, where C is the number of chunks.

  This function performs chunk reduction for ther results of the convolution,
  i.e. merge two consecutive chunks into one chunk with double precision.
  E.g. u8[0, 8, 8, 0] -> u16[8, 2048] 0-> u32[526336]

  Args:
    mul_result: The chunk-wise multiplication (using convolution) result.
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
  new_shape = (
      mul_result.shape[:-1] + (-1, 2) if mul_result.ndim == 2 else (-1, 2)
  )
  value_c = mul_result.reshape(new_shape)
  value_c = jnp.left_shift(value_c, shift_0_8_u16x4[:chunk_num_u16])
  value_c = jnp.sum(value_c, axis=-1)
  value_c = value_c.reshape(new_shape).astype(jnp.uint64)
  value_c = jnp.left_shift(value_c, shift_0_16_u32x4[:chunk_num_u32])
  value_c = jnp.sum(value_c, axis=-1)
  return value_c


@jax.named_call
@functools.partial(jax.jit, static_argnames="chunk_num_u16")
def compare_u16(
    value_a: jax.Array, value_b: jax.Array, chunk_num_u16=util.U16_CHUNK_NUM
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
  cond = weight.sum(axis=-1)
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
    mask=util.U16_MASK,
    chunk_num_u16=util.U16_CHUNK_NUM,
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
  if value_c.ndim == 1:
    value_c = value_c.at[chunk_num_u16 - 1].set(value_c[chunk_num_u16 - 1] - 1)
  else:
    value_c = value_c.at[:, chunk_num_u16 - 1].set(
        value_c[:, chunk_num_u16 - 1] - 1
    )

  value_c = value_c.astype(jnp.uint16)
  return value_c


@jax.named_call
@functools.partial(
    jax.jit, static_argnames=("modulus_377_int_chunk", "chunk_num_u16")
)
def cond_sub_mod_u16(
    value_a: jax.Array,
    modulus_377_int_chunk=util.MODULUS_377_INT_CHUNK,
    chunk_num_u16=util.U16_CHUNK_NUM,
):
  """Perform conditional subtraction: value_a - modulus_377_int.

  Args:
    value_a: The minuend.
    modulus_377_int_chunk: The modulus 377.
    chunk_num_u16: The number of chunks in the u16 value (default: 24).

  Returns:
    value_c: The result of the conditional subtraction.
  """
  compare_u16_local = functools.partial(
      compare_u16, chunk_num_u16=chunk_num_u16
  )
  sub_2u16_local = functools.partial(sub_2u16, chunk_num_u16=chunk_num_u16)
  modulus_377_int_array = jnp.asarray(modulus_377_int_chunk, jnp.uint16)

  cond = compare_u16_local(value_a, modulus_377_int_array)
  value_b = sub_2u16_local(value_a, modulus_377_int_array)
  cond = jnp.greater_equal(cond, 0).reshape((cond.shape[0], 1))
  value_c = jnp.where(cond, value_b, value_a)
  return value_c


@jax.named_call
@functools.partial(
    jax.jit, static_argnames=("modulus_377_int_chunk", "chunk_num_u16")
)
def cond_sub_2u16(
    value_a: jax.Array,
    value_b: jax.Array,
    modulus_377_int_chunk=util.MODULUS_377_INT_CHUNK,
    chunk_num_u16=util.U16_CHUNK_NUM,
):
  """Perform conditional subtraction: value_a - value_b.

  Args:
    value_a: The minuend.
    value_b: The subtrahend.
    modulus_377_int_chunk: The modulus 377.
    chunk_num_u16: The number of chunks in the u16 value (default: 24).

  Returns:
    value_c: The result of the conditional subtraction.
  """
  modulus_377_int_array = jnp.asarray(modulus_377_int_chunk, jnp.uint16)
  compare_u16_local = functools.partial(
      compare_u16, chunk_num_u16=chunk_num_u16
  )
  sub_2u16_local = functools.partial(sub_2u16, chunk_num_u16=chunk_num_u16)

  cond = compare_u16_local(value_a, value_b)
  cond = jnp.greater_equal(cond, 0).reshape((cond.shape[0], 1))

  value_ap = jnp.add(
      value_a.astype(jnp.uint32), modulus_377_int_array.astype(jnp.uint32)
  )

  value_a = jnp.where(cond, value_a.astype(jnp.uint32), value_ap)
  value_c = sub_2u16_local(value_a, value_b)
  return value_c


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=(
        "mask",
        "chunk_num_u16",
        "chunk_shift_bits",
        "output_dtype",
        "vmap_axes",
    ),
)
def mul_2u16(
    value_a: jax.Array,
    value_b: jax.Array,
    mask=util.U32_MASK,
    chunk_num_u16=util.U16_CHUNK_NUM,
    chunk_shift_bits=util.U32_CHUNK_SHIFT_BITS,
    output_dtype=jnp.uint16,
    vmap_axes=(0, 0),
):
  """Multiply two u16 values.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.
    mask: The mask to apply to the value.
    chunk_num_u16: The number of chunks in the u16 value (default: 24).
    chunk_shift_bits: The number of bits to shift the value.
    output_dtype: The desired output data type.
    vmap_axes: The axes to use for vmap.

  Returns:

  cond > 0 -> value_a > value_b
  cond = 0 -> value_a = value_b
  cond < 0 -> value_a < value_b
  """
  batch_dim = value_a.shape[0]
  mul_result = jax.vmap(conv_1d_2u16xn, in_axes=vmap_axes)(value_a, value_b)
  mul_result = jnp.pad(mul_result, ((0, 0), (0, 1)))
  value_c = rechunkify(mul_result, 2 * chunk_num_u16, chunk_num_u16)

  value_c = jax.lax.while_loop(
      functools.partial(
          check_any_chunk_with_carry, chunk_shift_bits=chunk_shift_bits
      ),
      functools.partial(
          carry_propagation,
          mask=mask,
          chunk_shift_bits=chunk_shift_bits,
      ),
      value_c,
  )
  ratio = 4 if output_dtype == jnp.uint8 else 2
  value_c = jax.lax.bitcast_convert_type(
      value_c.astype(jnp.uint32), output_dtype
  ).reshape(batch_dim, -1)[:, : ratio * chunk_num_u16]
  return value_c


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=(
        "barrett_shift_u8",
        "chunk_num_u16",
        "chunk_num_u32",
        "vmap_axes",
    ),
)
def mul_shift_2u16x2x1(
    value_a: jax.Array,
    value_b: jax.Array,
    mask=util.U32_MASK,
    barrett_shift_u8=util.BARRETT_SHIFT_U8,
    chunk_num_u16=util.U16_CHUNK_NUM,
    chunk_num_u32=util.U32_CHUNK_NUM,
    chunk_shift_bits=util.U32_CHUNK_SHIFT_BITS,
    vmap_axes=(0, None),
):
  """Multiply and shift two u16 values.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.
    mask: The mask to apply to the value.
    barrett_shift_u8: The number of bits to shift the value.
    chunk_num_u16: The number of chunks in the u16 value.
    chunk_num_u32: The number of chunks in the u32 value.
    chunk_shift_bits: The number of bits to shift the value.
    vmap_axes: (0, None) means axis 0 is the mapped access, and The rest is not.

  Returns:

  cond > 0 -> value_a > value_b
  cond = 0 -> value_a = value_b
  cond < 0 -> value_a < value_b
  """
  batch_dim = value_a.shape[0]
  conv = jax.vmap(conv_1d_2u16xn, in_axes=vmap_axes)(value_a, value_b)
  conv = jnp.pad(conv, ((0, 0), (0, 1)))
  value_c = rechunkify(conv, chunk_num_u16 * 3, chunk_num_u32 * 3)
  value_c = jax.lax.while_loop(
      functools.partial(
          check_any_chunk_with_carry, chunk_shift_bits=chunk_shift_bits
      ),
      functools.partial(
          carry_propagation, mask=mask, chunk_shift_bits=chunk_shift_bits
      ),
      value_c,
  )

  value_c = jax.lax.bitcast_convert_type(
      value_c.astype(jnp.uint32), jnp.uint8
  ).reshape(batch_dim, -1)[:, barrett_shift_u8:]
  value_c = jax.lax.bitcast_convert_type(
      jnp.pad(value_c, ((0, 0), (0, 1))).reshape(batch_dim, -1, 2), jnp.uint16
  )[:, :chunk_num_u16]
  return value_c


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=(
        "mask",
        "modulus_377_int_chunk",
        "mu_377_int_chunk",
        "chunk_num_u16",
        "vmap_axes",
    ),
)
def mod_mul_barrett_2u16(
    value_a: jax.Array,
    value_b: jax.Array,
    mask=util.U16_MASK,
    modulus_377_int_chunk=util.MODULUS_377_INT_CHUNK,
    mu_377_int_chunk=util.MU_377_INT_CHUNK,
    chunk_num_u16=util.U16_CHUNK_NUM,
    vmap_axes=(0, None),
):
  """Multiply two u16 values with Barrett reduction.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.
    mask: The mask to apply to the value.
    modulus_377_int_chunk: The modulus 377.
    mu_377_int_chunk: The Barrett reduction coefficient.
    chunk_num_u16: The number of chunks in the u16 value (default: 24).
    vmap_axes: The axes to use for vmap.

  Returns:
    value_c: The result of the multiplication.
  """
  modulus_377_int_array = jnp.asarray(modulus_377_int_chunk, jnp.uint16)
  mu_377_int_array = jnp.asarray(mu_377_int_chunk, jnp.uint16)

  mul_2u16_const = functools.partial(mul_2u16, vmap_axes=vmap_axes)
  sub_2u16_const = functools.partial(
      sub_2u16, mask=mask, chunk_num_u16=chunk_num_u16 * 2
  )
  value_x = mul_2u16(value_a, value_b)
  value_d = mul_shift_2u16x2x1(value_x, mu_377_int_array)
  value_e = mul_2u16_const(value_d, modulus_377_int_array)
  value_t = sub_2u16_const(value_x, value_e)
  value_c = cond_sub_mod_u16(value_t[:, :chunk_num_u16])
  return value_c


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=(
        "mask",
        "modulus_377_int_chunk",
        "mu_377_int_chunk",
        "chunk_num_u16",
        "vmap_axes",
    ),
)
def barrett_reduction_u16x2(
    value_x: jax.Array,
    mask=util.U16_MASK,
    modulus_377_int_chunk=util.MODULUS_377_INT_CHUNK,
    mu_377_int_chunk=util.MU_377_INT_CHUNK,
    chunk_num_u16=util.U16_CHUNK_NUM,
    vmap_axes=(0, None),
):
  """Performs Barrett reduction on a u16x2 value.

  Args:
    value_x: The u16x2 value to perform Barrett reduction on.
    mask: The mask to apply to the value.
    modulus_377_int_chunk: The modulus 377.
    mu_377_int_chunk: The Barrett reduction coefficient.
    chunk_num_u16: The number of chunks in the u16 value (default: 24).
    vmap_axes: The axes to use for vmap.

  Returns:
    value_c: The result of the Barrett reduction.
  """
  modulus_377_int_array = jnp.asarray(modulus_377_int_chunk, jnp.uint16)
  mu_377_int_array = jnp.asarray(mu_377_int_chunk, jnp.uint16)

  mul_2u16_const = functools.partial(mul_2u16, vmap_axes=vmap_axes)
  value_d = mul_shift_2u16x2x1(value_x, mu_377_int_array)
  value_e = mul_2u16_const(value_d, modulus_377_int_array)
  value_t = sub_2u16(value_x, value_e, mask, chunk_num_u16 * 2)
  value_c = cond_sub_mod_u16(value_t[:, :chunk_num_u16])
  return value_c


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=(
        "modulus_lazy_mat",
        "mask",
        "chunk_num_u8",
        "chunk_shift_bits",
    ),
)
def mod_mul_lazy_2u16(
    value_a,
    value_b,
    modulus_lazy_mat=util.MODULUS_377_LAZY_MAT,
    mask=util.U32_MASK,
    chunk_num_u8=util.U8_CHUNK_NUM,
    chunk_shift_bits=util.U32_CHUNK_SHIFT_BITS,
):
  """Multiply two u16 values with lazy matrix reduction.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.
    modulus_lazy_mat: The lazy matrix.
    mask: The mask to apply to the value.
    chunk_num_u8: The number of chunks in the u8 value.
    chunk_shift_bits: The number of bits to shift the value.

  Returns:
    value_c: The result of the multiplication.
  """
  batch_dim = value_a.shape[0]
  modulus_lazy_mat = jnp.asarray(modulus_lazy_mat, dtype=jnp.uint16)
  mul_2u8 = functools.partial(
      mul_2u16,
      mask=util.U32_MASK,
      chunk_num_u16=util.U16_EXT_CHUNK_NUM,
      chunk_shift_bits=util.U32_CHUNK_SHIFT_BITS,
      output_dtype=jnp.uint8,
  )
  value_c = mul_2u8(value_a, value_b)
  standard_product_low = value_c[:, :chunk_num_u8]
  standard_product_high = value_c[:, chunk_num_u8:]

  reduced = jnp.matmul(
      standard_product_high.astype(jnp.uint16),
      modulus_lazy_mat.astype(jnp.uint16),
      preferred_element_type=jnp.uint32,
  )
  value_c_reduced = jnp.add(
      standard_product_low.astype(jnp.uint32), reduced.astype(jnp.uint32)
  )
  value_c_reduced_u32 = rechunkify(
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
  ).reshape(batch_dim, -1)[:, : util.U16_EXT_CHUNK_NUM]
  return value_c_u16


def split_view_32_to_16(a: jnp.ndarray):
  # Interpret each 32-bit element as two 16-bit numbers
  # and reshape to add an extra dimension of size 2.
  v = a.view(jnp.uint16).reshape(a.shape + (2,))
  # Assuming little-endian storage, the lower 16 bits are at index 0
  # and the upper 16 bits are at index 1.
  lower = v[..., 0]
  upper = v[..., 1]
  return upper, lower


def split_view_32_to_16_8(a: jnp.ndarray):
  # First, reshape the 32-bit integers as groups of 4 bytes.
  v8 = a.view(jnp.uint8).reshape(a.shape + (4,))
  # Also, reshape as 16-bit integers (2 per 32-bit element)
  v16 = a.view(jnp.uint16).reshape(a.shape + (2,))
  # For each 32-bit integer:
  # v16[..., 0] gives the lower 16 bits.
  # v8[..., 2] gives the third byte (i.e. the lower 8 bits of the upper 16 bits)
  lower = v16[..., 0]
  upper8 = v8[..., 2]
  return upper8, lower


# Reduce via RNS modulus
@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames="moduli_t",
)
def moduli_rns_red_internal_2u16(vals, moduli_t=util.RNS_MODULI_T):
  """Reduce via RNS modulus.

  Args:
    vals: The values to reduce.
    moduli_t: The moduli for the target.

  Returns:
    The reduced values.
  """
  # See jaxite_ec/advanced_algorithm/rns_red.py for description
  moduli_t = jnp.array(moduli_t, dtype=jnp.uint8)
  u1, l1 = split_view_32_to_16(vals)
  i1 = jnp.add(
      l1.astype(jnp.uint32),
      jnp.multiply(u1.astype(jnp.uint32), moduli_t),
  )
  u2, l2 = split_view_32_to_16_8(i1)
  i2 = jnp.add(
      l2.astype(jnp.uint32),
      jnp.multiply(u2.astype(jnp.uint16), moduli_t).astype(
          jnp.uint32
      ),
  )
  u3, l3 = split_view_32_to_16_8(i2)
  out = jnp.add(l3, jnp.multiply(u3, moduli_t).astype(jnp.uint16))
  return out


# Reduce via prime modulus
@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=("rns_mat", "moduli_t", "num_moduli", "precision"),
)
def mod_red_rns_2u16(
    c_rns_reduced,
    rns_mat=util.RNS_MAT,
    moduli_t=util.RNS_MODULI_T,
    num_moduli=util.NUM_MODULI,
    precision=util.RNS_PRECISION,
):
  """Reduce via RNS modulus.

  Args:
    c_rns_reduced: The values to reduce.
    rns_mat: The RNS precompute.
    moduli_t: The moduli for the target.
    num_moduli: The number of moduli.
    precision: The precision.

  Returns:
    The reduced values.
  """
  rns_stacked_mat = jnp.array(rns_mat[0], jnp.uint8)
  cor_mat = jnp.array(rns_mat[1], jnp.uint16)

  c_target = jnp.matmul(
      c_rns_reduced.view(jnp.uint8),
      rns_stacked_mat,
      preferred_element_type=jnp.uint32,
  )

  mul_res_glb_red_u32 = c_target.reshape(*c_target.shape[:-1], -1, 2)
  mul_res_glb_red_u32 = mul_res_glb_red_u32[..., 0] + (
      mul_res_glb_red_u32[..., 1] << 8
  )
  rns_reduce_u32, qe_u32 = jnp.split(
      mul_res_glb_red_u32, [num_moduli], axis=1
  )

  # obtain the high 32 bits from the quotient estimation results qe_u32
  k = (qe_u32 >> precision).astype(jnp.uint16)
  c_corrected = rns_reduce_u32 + jnp.matmul(
      k, cor_mat, preferred_element_type=jnp.uint32
  )

  return moduli_rns_red_internal_2u16(c_corrected, moduli_t)


# Multiply, without reducing
@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames="moduli_t",
)
def mul_unreduced_rns_2u16(
    value_a,
    value_b,
    moduli_t=util.RNS_MODULI_T,
):
  ab = jnp.multiply(value_a.astype(jnp.uint32), value_b.astype(jnp.uint32))
  return moduli_rns_red_internal_2u16(ab, moduli_t)


# Multiply and reduce
@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=("rns_mat", "moduli_t"),
)
def mod_mul_rns_2u16(
    value_a,
    value_b,
    rns_mat=util.RNS_MAT,
    moduli_t=util.RNS_MODULI_T,
):
  """Multiply two u16 values with RNS reduction.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.
    rns_mat: The RNS precompute.
    moduli_t: The moduli for the target.

  Returns:
    The product of the two u16 values.
  """
  ab = mul_unreduced_rns_2u16(value_a, value_b, moduli_t)
  return mod_red_rns_2u16(ab, rns_mat, moduli_t)


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames="moduli_t",
)
def add_rns_2u16(
    value_a: jax.Array,
    value_b: jax.Array,
    moduli_t=util.RNS_MODULI_T,
):
  """Add two u16 values with RNS reduction.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.
    moduli_t: The moduli for the target.

  Returns:
    The sum of the two u16 values.
  """
  return add_sub_rns_var(value_a, value_b, moduli_t=moduli_t)


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames="moduli_t",
)
def add_rns_3u16(
    value_a: jax.Array,
    value_b: jax.Array,
    value_c: jax.Array,
    moduli_t=util.RNS_MODULI_T,
):
  """Add three u16 values with RNS reduction.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.
    value_c: The third u16 value.
    moduli_t: The moduli for the target.

  Returns:
    The sum of the three u16 values.
  """
  return add_sub_rns_var(value_a, value_b, value_c, moduli_t=moduli_t)


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames="moduli_sub",
)
def negate_rns_for_var_add(
    value_a: jax.Array,
    moduli_sub=util.MODULI_SUB,
):
  """Negate a value for use in subtraction.

  Do not use the output in any function but add_sub_rns_var -- may break
  correctness.

  Args:
    value_a: RNS array to negate
    moduli_sub: Precomputed constants for performing negation, that depend on
      the target modulus

  Returns:
    An intermediate representing the negation of values_a in the target modulus
    in RNS form.

  Note: original data precision is 16 bit, using uint32 to avoid overflow
  """
  moduli_sub = jnp.array(moduli_sub, dtype=jnp.uint32)

  return jnp.add(
      jnp.negative(value_a.astype(jnp.uint16)).astype(jnp.uint32),
      moduli_sub,
  )


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames="moduli_sub",
)
def negate_rns_for_var_add_zero_check(
    value_a: jax.Array,
    moduli_sub=util.MODULI_SUB,
):
  """Negate a value for use in subtraction.

  Do not use the output in any function but add_sub_rns_var -- may break
  correctness.

  Args:
    value_a: RNS array to negate
    moduli_sub: Precomputed constants for performing negation, that depend on
      the target modulus

  Returns:
    An intermediate representing the negation of values_a in the target modulus
    in RNS form.

  Note: original data precision is 16 bit, using uint32 to avoid overflow
  """

  moduli_sub = jnp.array(moduli_sub, dtype=jnp.uint32)
  a = value_a.astype(jnp.uint16)

  # Compute two's complement negation: for nonzero a, jnp.negative(a) computes
  # (2^16 - a).
  neg = jnp.negative(a).astype(jnp.uint32)

  # Build a branchless mask: 0 if a==0, 1 otherwise.
  mask = (a != 0).astype(jnp.uint32)

  # For nonzero a: (2^16 - a) + moduli_sub; for zero: m + 0 multiplied by 0
  # gives 0.
  return (neg + moduli_sub) * mask


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames="moduli_t",
)
def add_sub_rns_var(*values, moduli_t=util.RNS_MODULI_T):
  """Evaluate an static set of additions and subtractions.

  Subtractions are implemented by calling negate_rns_for_var_add on inputs to
  this function. Inputs should be "fresh" or multiplication outputs, and the
  output should be used as a multiplication input. Any other usage produces
  undefined behavior and may break correctness.

  Args:
    *values: A list of RNS values to accumulate
    moduli_t: The moduli for the RNS form.

  Returns:
    The RNS form of the evaluation of the expession.
  """
  assert len(values) > 0
  acc = None
  for v in values:
    if acc != None:
      acc = jnp.add(v.astype(jnp.uint32), acc)
    else:
      acc = v.astype(jnp.uint32)
  assert len(values) < 256
  moduli_t = jnp.array(moduli_t, dtype=jnp.uint8)
  # u1 < 254
  u1, l1 = split_view_32_to_16_8(acc)
  # i1 < 2**16 - 1 + 255t < 2**17 - t for 8 bit t
  i1 = jnp.add(
      jnp.multiply(u1.astype(np.uint16), moduli_t).astype(jnp.uint32),
      l1.astype(jnp.uint32),
  )
  # u2 = 0 or 1, but if u2 = 1 then l < 2**16 - t, so 2**16 - t + t < 2**16
  u2, l2 = split_view_32_to_16_8(i1)
  return jnp.add(jnp.multiply(u2, moduli_t).astype(jnp.uint16), l2)


@functools.partial(jax.jit, static_argnames=("c", "num_moduli"))
def rns_constant(c, num_moduli=util.NUM_MODULI):
  assert c >= 0
  assert c < 2**14  # small constants only please
  return jnp.repeat(jnp.array([c], dtype=jnp.uint16), num_moduli)
