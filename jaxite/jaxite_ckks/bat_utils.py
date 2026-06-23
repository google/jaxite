"""Basis Aligned Transformation (BAT) utilities for CKKS on TPU."""

import jax
import jax.numpy as jnp

# Enable 64-bit precision for large integer arithmetic
jax.config.update("jax_enable_x64", True)


def matmul_bat_einsum(
    lhs: jax.Array, rhs: jax.Array, subscripts: str
) -> jax.Array:
  """Basis Aligned Transformation (BAT) based matrix multiplication.

  Args:
    lhs: input
    rhs: twiddle factor matrix
    subscripts: einsum subscripts

  Returns:
    The matrix multiplication result.
  """
  lhs_u8 = jax.lax.bitcast_convert_type(lhs, jnp.uint8)
  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
  i8_products = jnp.einsum(
      subscripts, lhs_u8, rhs, preferred_element_type=jnp.uint32
  )
  return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(-1,))


def basis_aligned_transformation(
    matrix: jnp.ndarray, moduli: list[int]
) -> jnp.ndarray:
  """Prepares a matrix for Basis Aligned Transformation (BAT)."""
  matrix_u64 = matrix.astype(jnp.uint64)
  num_bytes = 4
  matrix_u64_byteshifted = jnp.array(
      [matrix_u64 << (8 * byte_idx) for byte_idx in range(num_bytes)],
      dtype=jnp.uint64,
  )
  moduli_arr = jnp.array(moduli, dtype=jnp.uint64)
  matrix_u64_byteshifted_mod_modulus = (
      matrix_u64_byteshifted % moduli_arr
  ).astype(jnp.uint32)
  # Output shape: (4, ..., moduli, 4)
  matrix_u8 = jax.lax.bitcast_convert_type(
      matrix_u64_byteshifted_mod_modulus, jnp.uint8
  )
  return matrix_u8


def basis_aligned_transform_key(
    key_matrix: jax.Array, moduli: jax.Array | list[int]
) -> jax.Array:
  """Prepares the 4D key matrix of shape (degree, num_moduli, 2, dnum) for BAT.

  Args:
    key_matrix: The key matrix of shape (degree, num_moduli, 2, dnum).
    moduli: The moduli of the key matrix.

  Returns:
    The transformed key matrix of shape (num_blocks, block_size, num_moduli,
      2, dnum, 4, 4) in uint8.
  """
  matrix_u64 = key_matrix.astype(jnp.uint64)
  num_bytes = 4
  matrix_u64_byteshifted = jnp.array(
      [matrix_u64 << (8 * byte_idx) for byte_idx in range(num_bytes)],
      dtype=jnp.uint64,
  )  # Shape: (4, degree, num_moduli, 2, dnum)

  moduli_expanded = jnp.array(moduli, dtype=jnp.uint64).reshape(1, 1, -1, 1, 1)

  matrix_u64_byteshifted_mod_modulus = (
      matrix_u64_byteshifted % moduli_expanded
  ).astype(jnp.uint32)

  # Bitcast to uint8: shape becomes (4, degree, num_moduli, 2, dnum, 4)
  matrix_u8 = jax.lax.bitcast_convert_type(
      matrix_u64_byteshifted_mod_modulus, jnp.uint8
  )

  # Transpose to (degree, num_moduli, u, v, q, p)
  # Axes mapping:
  # 0: byte_idx (q, size 4)
  # 1: degree
  # 2: num_moduli
  # 3: u (size 2)
  # 4: v (size dnum)
  # 5: b (p, size 4)
  matrix_u8_transposed = jnp.transpose(matrix_u8, (1, 2, 3, 4, 0, 5))

  # Reshape degree dimension to (num_blocks, block_size) to optimize TPU vectorization
  degree = key_matrix.shape[0]
  if degree >= 128:
    block_size = 128
    num_blocks = degree // 128
  else:
    block_size = degree
    num_blocks = 1
  return matrix_u8_transposed.reshape(
      num_blocks, block_size, *matrix_u8_transposed.shape[1:]
  )


def matmul_bat_key_vector(
    vector_v: jax.Array, key_matrix_bat: jax.Array
) -> jax.Array:
  """Computes BAT-based matrix-vector product for 2x2 or 2xdnum key multiplication.

  Args:
    vector_v: The input vector of shape (..., degree, num_moduli, dnum).
    key_matrix_bat: The pre-transformed key matrix of shape (num_blocks,
      block_size, num_moduli, 2, dnum, 4, 4).

  Returns:
    The matrix-vector product of shape (2, ..., degree, num_moduli) in uint64.
  """
  degree = vector_v.shape[-3]
  num_moduli = vector_v.shape[-2]
  dnum = vector_v.shape[-1]

  if degree >= 128:
    block_size = 128
    num_blocks = degree // 128
  else:
    block_size = degree
    num_blocks = 1

  # Reshape degree dimension to (num_blocks, block_size) to optimize TPU vectorization
  v_reshaped = vector_v.reshape(
      *vector_v.shape[:-3], num_blocks, block_size, num_moduli, dnum
  )

  # View-cast vector_v to uint8 -> (..., num_blocks, block_size, num_moduli, dnum, 4)
  v_u8 = jax.lax.bitcast_convert_type(v_reshaped, jnp.uint8)

  # einsum subscripts to compute the 2x2 (or 2xdnum) matrix-vector multiplication
  # v_u8: ...ikjvq (where i is num_blocks, k is block_size, j is num_moduli, v is dnum, q is 4)
  # key_matrix_bat: ikjuvqp (where i is num_blocks, k is block_size, j is num_moduli, u is 2, v is dnum, q is 4, p is 4)
  # output: ...ikjup (where u is 2, p is 4)
  i8_products = jnp.einsum(
      "...ikjvq,ikjuvqp->...ikjup",
      v_u8,
      key_matrix_bat,
      preferred_element_type=jnp.uint32,
  )

  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
  # Shift and sum over the last dimension (p, size 4)
  # to reconstruct uint64 values
  # Shape after sum: (..., num_blocks, block_size, num_moduli, 2)
  summed = jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=-1)

  # Reshape to flatten num_blocks and block_size back to degree
  # Shape becomes: (..., degree, num_moduli, 2)
  summed_flat = summed.reshape(*summed.shape[:-4], degree, num_moduli, 2)

  # Transpose to (2, ..., degree, num_moduli)
  # where the components (size 2) is the first dimension
  return jnp.moveaxis(summed_flat, -1, 0)
