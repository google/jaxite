"""Basis Aligned Transformation (BAT) utilities for CKKS on TPU."""

from typing import overload
import jax
import jax.numpy as jnp
import numpy as np

# Enable 64-bit precision for large integer arithmetic
jax.config.update("jax_enable_x64", True)


def matmul_bat_einsum(
    lhs: jax.Array,
    rhs: jax.Array,
    subscripts: str,
    merge_byte_dimension: bool = False,
) -> jax.Array:
  """Basis Aligned Transformation (BAT) based matrix multiplication.

  Args:
    lhs: input
    rhs: twiddle factor matrix
    subscripts: einsum subscripts
    merge_byte_dimension: control flattening byte dimension of LHS into the
      element dimension (required for basis conversion).

  Returns:
    The matrix multiplication result.
  """
  lhs_u8 = jax.lax.bitcast_convert_type(lhs, jnp.uint8)
  if merge_byte_dimension:
    lhs_u8 = lhs_u8.reshape(*lhs_u8.shape[:-2], -1)
  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
  i8_products = jnp.einsum(
      subscripts, lhs_u8, rhs, preferred_element_type=jnp.uint32
  )
  return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(-1,))


@overload
def basis_aligned_transformation(
    matrix: np.ndarray, moduli: list[int]
) -> np.ndarray:
  ...


@overload
def basis_aligned_transformation(
    matrix: jnp.ndarray, moduli: list[int]
) -> jnp.ndarray:
  ...


def basis_aligned_transformation(
    matrix: jnp.ndarray | np.ndarray, moduli: list[int]
) -> jnp.ndarray | np.ndarray:
  """Prepares a matrix for Basis Aligned Transformation (BAT)."""
  if isinstance(matrix, np.ndarray):
    matrix_u64 = matrix.astype(np.uint64)
    num_bytes = 4
    matrix_u64_byteshifted = np.array(
        [matrix_u64 << (8 * byte_idx) for byte_idx in range(num_bytes)],
        dtype=np.uint64,
    )
    moduli_arr = np.array(moduli, dtype=np.uint64)
    matrix_u32 = (matrix_u64_byteshifted % moduli_arr).astype(np.uint32)
    view_u8 = matrix_u32.view(np.uint8)
    out_shape = matrix_u32.shape + (4,)
    return view_u8.reshape(out_shape)

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
