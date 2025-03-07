"""library of finite field operations.

This library is used to implement the finite field operations for the
high-precision arithetic in Fully Homomorphic Encryption (FHE).

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
import jaxite.jaxite_word.util as utils

jax.config.update("jax_enable_x64", True)


def conv_1d(value_a: jax.Array, value_b: jax.Array):
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
@functools.partial(jax.jit, static_argnames="chunk_num_u16")
def chunk_reduction_after_conv(mul_result: jax.Array, chunk_num_u16):
  """Given the carry add takes O(C) algorithm complexity, where C is the number of chunks.

  This function performs chunk reduction for ther results of the convolution,
  i.e. merge two consecutive chunks into one chunk with double precision.
  E.g. u8[0, 8, 8, 0] -> u16[8, 2048]

  Args:
    mul_result: The chunk-wise multiplication (using convolution) result.
    chunk_num_u16: The number of bits in each chunk.

  Returns:
    value_c: The result of the chunk reduction.
  """
  shift_0_8_u16 = jnp.array(
      [[0, 8] for _ in range(chunk_num_u16)], dtype=jnp.uint8
  )
  new_shape = mul_result.shape[:-1] + (-1, 2)
  value_c = mul_result.reshape(new_shape)
  value_c = jnp.left_shift(value_c, shift_0_8_u16)
  value_c = jnp.sum(value_c, axis=-1, dtype=value_c.dtype)
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
  cond = weight.sum(axis=-1)
  return cond


@jax.named_call
@functools.partial(
    jax.jit, static_argnames=("mask", "chunk_num_u16", "chunk_shift_bits")
)
def carry_propagation_u16(
    value_a: jax.Array,
    mask=utils.U16_MASK,
    chunk_num_u16=utils.U16_CHUNK_NUM,
    chunk_shift_bits=utils.U16_CHUNK_SHIFT_BITS,
):
  """Compare two u16 values."""
  initial_carry = jnp.zeros(
      (1,) + ((value_a.shape[0],) * (value_a.ndim - 1)), dtype=jnp.uint32
  )
  value_a_transpose = value_a.transpose(1, 0) if value_a.ndim == 2 else value_a
  print(
      f"value_a_transpose.shape: {value_a_transpose.shape},"
      f" dtype:{value_a_transpose.dtype}"
  )

  def scan_body(carry, xs):
    xs = jnp.add(carry, xs)
    low = jnp.bitwise_and(xs, mask).reshape(-1)
    carry = jnp.right_shift(xs, chunk_shift_bits)
    return (carry, low)

  value_a_carry_propagated = jax.lax.scan(
      scan_body,
      initial_carry,
      value_a_transpose,
      length=chunk_num_u16,
  )[1].transpose(1, 0)
  value_a_carry_propagated = jnp.squeeze(value_a_carry_propagated)
  return value_a_carry_propagated


@jax.named_call
@functools.partial(
    jax.jit, static_argnames=("mask", "chunk_num_u16", "chunk_shift_bits")
)
def add_2u16(
    value_a: jax.Array,
    value_b: jax.Array,
    mask=utils.U16_MASK,
    chunk_num_u16=utils.U16_CHUNK_NUM,
    chunk_shift_bits=utils.U16_CHUNK_SHIFT_BITS,
):
  """Add two u16 values.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.
    mask: The mask to apply to the value.
    chunk_num_u16: The number of chunks in the u16 value (default: 24).
    chunk_shift_bits: The number of bits to shift the value (default: 16).

  Returns:
    value_c: The result of the addition.
  """
  value_c = jax.numpy.add(
      value_a.astype(jnp.uint32), value_b.astype(jnp.uint32)
  )

  carry_propagation_u16_local = functools.partial(
      carry_propagation_u16,
      mask=mask,
      chunk_num_u16=chunk_num_u16,
      chunk_shift_bits=chunk_shift_bits,
  )
  return carry_propagation_u16_local(value_c).astype(jnp.uint16)


@jax.named_call
@functools.partial(
    jax.jit, static_argnames=("mask", "chunk_num_u16", "chunk_shift_bits")
)
def add_3u16(
    value_a: jax.Array,
    value_b: jax.Array,
    value_d: jax.Array,
    mask=utils.U16_MASK,
    chunk_num_u16=utils.U16_CHUNK_NUM,
    chunk_shift_bits=utils.U16_CHUNK_SHIFT_BITS,
):
  """Add three u16 values.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.
    value_d: The third u16 value.
    mask: The mask to apply to the value.
    chunk_num_u16: The number of chunks in the u16 value (default: 24).
    chunk_shift_bits: The number of bits to shift the value (default: 16).

  Returns:
    value_c: The result of the addition.
  """
  value_c = jax.numpy.add(
      value_a.astype(jnp.uint32), value_b.astype(jnp.uint32)
  )
  value_c = jax.numpy.add(
      value_c.astype(jnp.uint32), value_d.astype(jnp.uint32)
  )

  carry_propagation_u16_local = functools.partial(
      carry_propagation_u16,
      mask=mask,
      chunk_num_u16=chunk_num_u16,
      chunk_shift_bits=chunk_shift_bits,
  )
  return carry_propagation_u16_local(value_c).astype(jnp.uint16)


@jax.named_call
@functools.partial(
    jax.jit, static_argnames=("mask", "chunk_num_u16", "chunk_shift_bits")
)
def sub_2u16(
    value_a: jax.Array,
    value_b: jax.Array,
    mask=utils.U16_MASK,
    chunk_num_u16=utils.U16_CHUNK_NUM,
    chunk_shift_bits=utils.U16_CHUNK_SHIFT_BITS,
):
  """Subtract two u16 values.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.
    mask: The mask to apply to the value.
    chunk_num_u16: The number of chunks in the u16 value (default: 24).
    chunk_shift_bits: The number of bits to shift the value (default: 16).

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
  carry_propagation_u16_local = functools.partial(
      carry_propagation_u16,
      mask=mask,
      chunk_num_u16=chunk_num_u16,
      chunk_shift_bits=chunk_shift_bits,
  )
  value_c_u16 = carry_propagation_u16_local(value_c).astype(jnp.uint16)

  value_c_u16 = value_c_u16.at[..., chunk_num_u16 - 1].set(
      value_c_u16[..., chunk_num_u16 - 1] - 1
  )

  value_c_u16 = value_c_u16.astype(jnp.uint16)
  return value_c_u16


@jax.named_call
@functools.partial(jax.jit, static_argnames=("modulus_array", "chunk_num_u16"))
def cond_sub_mod_u16(
    value_a: jax.Array,
    modulus_array=utils.MODULUS_ARRAY,
    chunk_num_u16=utils.U16_CHUNK_NUM,
):
  """Perform conditional subtraction: value_a - modulus.

  Args:
    value_a: The minuend.
    modulus_array: The modulus 377.
    chunk_num_u16: The number of chunks in the u16 value (default: 24).

  Returns:
    value_c: The result of the conditional subtraction.
  """
  modulus_array = jnp.array(modulus_array, dtype=jnp.uint16)
  compare_u16_local = functools.partial(
      compare_u16, chunk_num_u16=chunk_num_u16
  )
  sub_2u16_local = functools.partial(sub_2u16, chunk_num_u16=chunk_num_u16)
  if value_a.shape[0] > 1:
    # Input is batch (Vector, Constant)
    compare_u16_local = jax.vmap(compare_u16_local, in_axes=(0, None))
    sub_2u16_local = jax.vmap(sub_2u16_local, in_axes=(0, None))

  cond = compare_u16_local(value_a, modulus_array)
  value_b = sub_2u16_local(value_a, modulus_array)
  cond = jnp.greater_equal(cond, 0).reshape((cond.shape[0], 1))
  value_c = jnp.where(cond, value_b, value_a)
  return value_c


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=(
        "mask",
        "modulus_array",
        "chunk_num_u16",
        "chunk_shift_bits",
    ),
)
def cond_sub_2u16(
    value_a: jax.Array,
    value_b: jax.Array,
    mask=utils.U16_MASK,
    modulus_array=utils.MODULUS_ARRAY,
    chunk_num_u16=utils.U16_CHUNK_NUM,
    chunk_shift_bits=utils.U16_CHUNK_SHIFT_BITS,
):
  """Perform conditional subtraction: value_a - value_b.

  Args:
    value_a: The minuend.
    value_b: The subtrahend.
    mask: The mask to apply to the value.
    modulus_array: The modulus 377.
    chunk_num_u16: The number of chunks in the u16 value (default: 24).
    chunk_shift_bits: The number of bits to shift the value (default: 16).

  Returns:
    value_c: The result of the conditional subtraction.
  """

  compare_u16_local = functools.partial(
      compare_u16, chunk_num_u16=chunk_num_u16
  )
  sub_2u16_local = functools.partial(
      sub_2u16,
      mask=mask,
      chunk_num_u16=chunk_num_u16,
      chunk_shift_bits=chunk_shift_bits,
  )

  cond = compare_u16_local(value_a, value_b)
  cond = jnp.greater_equal(cond, 0).reshape((cond.shape[0], 1))

  value_ap = jnp.add(
      value_a.astype(jnp.uint32), jnp.array(modulus_array, dtype=jnp.uint32)
  )

  value_a = jnp.where(cond, value_a.astype(jnp.uint32), value_ap)
  return sub_2u16_local(value_a, value_b)


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
    mask=utils.U32_MASK,
    chunk_num_u16=utils.U16_CHUNK_NUM,
    chunk_shift_bits=utils.U16_CHUNK_SHIFT_BITS,
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

  mul_result = jax.vmap(conv_1d, in_axes=vmap_axes)(value_a, value_b)
  mul_result = jnp.pad(mul_result, ((0, 0), (0, 1)))
  value_c = chunk_reduction_after_conv(mul_result, 2 * chunk_num_u16)

  carry_propagation_u16_local = functools.partial(
      carry_propagation_u16,
      mask=mask,
      chunk_num_u16=2 * chunk_num_u16,
      chunk_shift_bits=chunk_shift_bits,
  )
  value_c_u16 = carry_propagation_u16_local(value_c).astype(jnp.uint16)
  ratio = 4 if output_dtype == jnp.uint8 else 2
  return value_c_u16.astype(output_dtype)[:, : ratio * chunk_num_u16]


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=(
        "mask",
        "barrett_shift_u8",
        "chunk_num_u16",
        "chunk_shift_bits",
        "vmap_axes",
    ),
)
def mul_shift_2u16(
    value_a: jax.Array,
    value_b: jax.Array,
    mask=utils.U32_MASK,
    barrett_shift_u8=utils.BARRETT_SHIFT_U8,
    chunk_num_u16=utils.U16_CHUNK_NUM,
    chunk_shift_bits=utils.U16_CHUNK_SHIFT_BITS,
    vmap_axes=(0, None),  # ToDo: Why is this None?
):
  """Multiply and shift two u16 values.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.
    mask: The mask to apply to the value.
    barrett_shift_u8: The number of bits to shift the value.
    chunk_num_u16: The number of chunks in the u16 value.
    chunk_shift_bits: The number of bits to shift the value.
    vmap_axes: The axes to use for vmap.

  Returns:

  cond > 0 -> value_a > value_b
  cond = 0 -> value_a = value_b
  cond < 0 -> value_a < value_b
  """
  chunk_reduction_after_conv_local = functools.partial(
      chunk_reduction_after_conv,
      chunk_num_u16=3 * chunk_num_u16,
  )
  carry_propagation_u16_local = functools.partial(
      carry_propagation_u16,
      mask=mask,
      chunk_num_u16=3 * chunk_num_u16,
      chunk_shift_bits=chunk_shift_bits,
  )
  batch_dim = value_a.shape[0]
  conv = jax.vmap(conv_1d, in_axes=vmap_axes)(value_a, value_b)
  conv = jnp.pad(conv, ((0, 0), (0, 1)))
  value_c = chunk_reduction_after_conv_local(conv)

  value_c_u16 = carry_propagation_u16_local(value_c).astype(jnp.uint16)

  # perform right shifting (shift in terms of multiple 8 bits but not
  # multiple of 16 bits)
  value_c_u8 = jax.lax.bitcast_convert_type(value_c_u16, jnp.uint8).reshape(
      batch_dim, -1
  )
  value_c_u8_shifted = value_c_u8[
      :, barrett_shift_u8 : barrett_shift_u8 + 2 * chunk_num_u16
  ].reshape(batch_dim, -1, 2)
  value_c_u16_shifted = jax.lax.bitcast_convert_type(
      value_c_u8_shifted, jnp.uint16
  )

  return value_c_u16_shifted


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=(
        "mask",
        "modulus_array",
        "mu_array",
        "barrett_shift_u8",
        "chunk_num_u16",
        "chunk_shift_bits",
        "vmap_axes",
    ),
)
def mod_mul_barrett_2u16(
    value_a: jax.Array,
    value_b: jax.Array,
    mask=utils.U16_MASK,
    modulus_array=utils.MODULUS_ARRAY,
    mu_array=utils.MU_ARRAY,
    barrett_shift_u8=utils.BARRETT_SHIFT_U8,
    chunk_num_u16=utils.U16_CHUNK_NUM,
    chunk_shift_bits=utils.U16_CHUNK_SHIFT_BITS,
    vmap_axes=(0, None),
):
  """Multiply two u16 values with Barrett reduction.

  Args:
    value_a: The first u16 value.
    value_b: The second u16 value.
    mask: The mask to apply to the value.
    modulus_array: The modulus 377.
    mu_array: The Barrett reduction coefficient.
    chunk_num_u16: The number of chunks in the u16 value (default: 24).
    vmap_axes: The axes to use for vmap.

  Returns:
    value_c: The result of the multiplication.
  """
  mul_2u16_local = functools.partial(
      mul_2u16,
      mask=mask,
      chunk_num_u16=chunk_num_u16,
      chunk_shift_bits=chunk_shift_bits,
  )
  mul_2u16_const_local = functools.partial(
      mul_2u16,
      mask=mask,
      chunk_num_u16=chunk_num_u16,
      chunk_shift_bits=chunk_shift_bits,
      vmap_axes=vmap_axes,
  )
  mul_shift_2u16_local = functools.partial(
      mul_shift_2u16,
      mask=mask,
      barrett_shift_u8=barrett_shift_u8,
      chunk_num_u16=chunk_num_u16,
      chunk_shift_bits=chunk_shift_bits,
  )
  sub_2u16_const = functools.partial(
      sub_2u16, mask=mask, chunk_num_u16=chunk_num_u16 * 2
  )
  cond_sub_mod_u16_local = functools.partial(
      cond_sub_mod_u16,
      modulus_array=modulus_array,
      chunk_num_u16=chunk_num_u16,
  )
  value_x = mul_2u16_local(value_a, value_b)
  value_d = mul_shift_2u16_local(value_x, jnp.array(mu_array, dtype=jnp.uint16))
  value_e = mul_2u16_const_local(
      value_d, jnp.array(modulus_array, dtype=jnp.uint16)
  )
  value_t = jax.vmap(sub_2u16_const, in_axes=0, out_axes=0)(value_x, value_e)
  value_c = cond_sub_mod_u16_local(value_t[:, :chunk_num_u16])
  return value_c


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=(
        "mask",
        "modulus_array",
        "mu_array",
        "barrett_shift_u8",
        "chunk_num_u16",
        "chunk_shift_bits",
        "vmap_axes",
    ),
)
def mod_reduction_barrett_2u16(
    value_a: jax.Array,
    mask=utils.U16_MASK,
    modulus_array=utils.MODULUS_ARRAY,
    mu_array=utils.MU_ARRAY,
    barrett_shift_u8=utils.BARRETT_SHIFT_U8,
    chunk_num_u16=utils.U16_CHUNK_NUM,
    chunk_shift_bits=utils.U16_CHUNK_SHIFT_BITS,
    vmap_axes=(0, None),
):
  """Multiply two u16 values with Barrett reduction.

  Args:
    value_a: The input value in need of modular reduction.
    mask: The mask to apply to the value.
    modulus_array: The modulus 377.
    mu_array: The Barrett reduction coefficient.
    chunk_num_u16: The number of chunks in the u16 value (default: 24).
    vmap_axes: The axes to use for vmap.

  Returns:
    value_c: The result of the multiplication.
  """
  mul_2u16_const_local = functools.partial(
      mul_2u16,
      mask=mask,
      chunk_num_u16=chunk_num_u16,
      chunk_shift_bits=chunk_shift_bits,
      vmap_axes=vmap_axes,
  )
  mul_shift_2u16_local = functools.partial(
      mul_shift_2u16,
      mask=mask,
      barrett_shift_u8=barrett_shift_u8,
      chunk_num_u16=chunk_num_u16,
      chunk_shift_bits=chunk_shift_bits,
  )
  sub_2u16_const = functools.partial(
      sub_2u16, mask=mask, chunk_num_u16=chunk_num_u16 * 2
  )
  cond_sub_mod_u16_local = functools.partial(
      cond_sub_mod_u16,
      modulus_array=modulus_array,
      chunk_num_u16=chunk_num_u16,
  )
  value_d = mul_shift_2u16_local(value_a, jnp.array(mu_array, dtype=jnp.uint16))
  value_e = mul_2u16_const_local(
      value_d, jnp.array(modulus_array, dtype=jnp.uint16)
  )
  value_t = jax.vmap(sub_2u16_const, in_axes=0, out_axes=0)(value_a, value_e)
  value_c = cond_sub_mod_u16_local(value_t[:, :chunk_num_u16])
  return value_c
