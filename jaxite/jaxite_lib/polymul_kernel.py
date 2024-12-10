"""Kernel for negacyclic vector-matrix polymul."""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxite.jaxite_lib import jax_helpers
from jaxite.jaxite_lib import matrix_utils


# This fallback serves as a reference implementation, but does not lower well on
# TPU due to the semantics of the vmap.
#
# in_axes = (None, 1) means that the first argument is repeated across all
# calls, while the second argument is mapped across its second index
# (column-wise)
fallback_vector_matrix_polymul = jax.jit(
    jax.vmap(matrix_utils.poly_dot_product, in_axes=(None, 1), out_axes=0)
)

# i32_as_u8_matmul is a (m,) x (m, k) -> (k,) matmul, but _i32_matmul_unreduced
# is an (m, k) x (k, n) -> (m, n) matmul. To compare, we can vmap
# i32_as_u8_matmul over the first axis.
#
# in_axes = (0, None) means that the second argument is repeated across all
# calls, while the first argument is mapped across its first axis.
fallback_i32_matmul = jax.vmap(
    matrix_utils.i32_as_u8_matmul, in_axes=(0, None), out_axes=0
)


def _i32_matmul_unreduced(lhs, rhs):
  lax = jax.lax
  m, k, n = lhs.shape[0], lhs.shape[1], rhs.shape[1]
  lhs_i8 = jnp.broadcast_to(lhs, (4, *lhs.shape))
  lhs_shift = lax.broadcasted_iota(jnp.int32, lhs_i8.shape, dimension=0) * 8
  lhs_i8 = lax.shift_right_logical(lhs_i8, lhs_shift)
  lhs_i8 = lax.bitwise_and(lhs_i8, jnp.broadcast_to(0xFF, lhs_i8.shape))
  lhs_i8 = lhs_i8.reshape((4 * m, k))

  acc = jnp.zeros((4 * m, n), dtype=jnp.int32)
  out_shift_base = lax.mul(
      lax.div(lax.broadcasted_iota(jnp.int32, (4 * m, n), dimension=0), m), 8
  )
  for rhs_shift in range(0, 32, 8):
    # TODO(b/201562458): Don't multiply lhs rows with large shift.
    rhs_i8 = lax.shift_right_logical(
        rhs, jnp.broadcast_to(rhs_shift, rhs.shape)
    )
    rhs_i8 = lax.bitwise_and(rhs_i8, jnp.broadcast_to(0xFF, rhs_i8.shape))
    # TODO(b/201562458): Use int8 matmuls once properly supported
    raw_out = lax.dot(
        lhs_i8.astype(jnp.bfloat16),
        rhs_i8.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    ).astype(jnp.int32)
    acc += jnp.left_shift(raw_out, out_shift_base + rhs_shift)
  return acc


def _vector_matrix_polymul(poly_vec1: jnp.ndarray, poly_mat2: jnp.ndarray):
  # b is the product of the RLWE dimension (e.g., 3) and the number of
  # decomposition levels in the decomposition parameters (e.g., 6).
  # n is the degree of the RLWE polynomials.
  b, n = poly_vec1.shape
  # m is the number of polynomials in the RLWE dimension (e.g., 3)
  b2, m, n2 = poly_mat2.shape
  assert b == b2 and n == n2

  # We must pad m to 8 because the TPU register sublane has size 8, and more
  # importantly, many of the pallas instructions like pltpu.roll will fail
  # if the sublane size is not a multiple of 8. This further adds the assumption
  # that the value of m is < 8. We are unlikely to need m > 8 for the
  # foreseeable future, but if we did, we would need to round up to the next
  # multiple of 8.
  real_m = m
  m = 8
  poly_mat2 = jnp.pad(
      poly_mat2,
      ((0, 0), (0, m - real_m), (0, 0)),
      mode="constant",
      constant_values=(0,),
  )

  if n % 128 != 0:
    raise ValueError(f"Input size {n} is not a multiple of 128")
  dtype = poly_vec1.dtype

  def vec_mat_polymul_kernel_single_batch(vec_ref, mat_ref, out_ref):
    chunk = jnp.broadcast_to(vec_ref[...], (128, n))
    chunk = pltpu.roll(chunk, 0, 1, stride=1, stride_axis=0)
    chunk_row_indices = jax.lax.broadcasted_iota(
        dtype=jnp.int32, shape=(128, n), dimension=0
    )
    chunk_col_indices = jax.lax.broadcasted_iota(
        dtype=jnp.int32, shape=(128, n), dimension=1
    )
    toeplitz_chunks = []
    for _ in range(0, n, 128):
      toeplitz_chunks.append(
          jnp.where(chunk_row_indices > chunk_col_indices, -chunk, chunk)
      )
      # Because the vector registers are aligned to size 128, this roll
      # operation lowers to telling the TPU to refer to a different register,
      # rather than actually applying any rolling operation. Hence, the op
      # produces no hardware instructions.
      chunk = pltpu.roll(chunk, 128, 1)
      chunk_row_indices = chunk_row_indices + 128
    vec_toeplitz = jax.lax.concatenate(toeplitz_chunks, dimension=0)

    assert vec_toeplitz.shape == (n, n)
    result = _i32_matmul_unreduced(mat_ref[...], vec_toeplitz)
    assert result.shape == (4 * m, n), result.shape
    out_ref[...] = result

  def vec_mat_polymul_kernel(vec_ref, mat_ref, out_ref):
    for b in range(vec_ref.shape[0]):
      vec_mat_polymul_kernel_single_batch(
          vec_ref.at[b], mat_ref.at[b], out_ref.at[b]
      )

  block_b = 2
  steps_b, rem_b = divmod(b, block_b)
  if rem_b:
    raise ValueError(f"b={b} is not a multiple of block_b={block_b}")

  return jnp.sum(
      pl.pallas_call(
          vec_mat_polymul_kernel,
          in_specs=(
              pl.BlockSpec((block_b, 1, n), lambda b: (b, 0, 0)),
              pl.BlockSpec((block_b, m, n), lambda b: (b, 0, 0)),
          ),
          out_specs=pl.BlockSpec((block_b, 4 * m, n), lambda b: (b, 0, 0)),
          out_shape=jax.ShapeDtypeStruct((b, 4 * m, n), jnp.int32),
          grid=(steps_b,),
      )(
          poly_vec1[:, None].astype(jnp.int32), poly_mat2.astype(jnp.int32)
      ).reshape(
          b, 4, m, n
      ),
      axis=(0, 1),
  ).astype(jnp.uint32)[:real_m]


@jax.named_call
@jax.jit
def negacyclic_vector_matrix_polymul(
    vec: jnp.ndarray, matrix: jnp.ndarray
) -> jnp.ndarray:
  """Computes a vector-matrix poly multiplication mod (X^N + 1).

  Args:
    vec: a vector of polynomials
    matrix: a matrix of polynomials

  Returns:
    the vector-matrix product of the polynomials
  """
  n_matrix = matrix.shape[-1]
  n_vec = vec.shape[-1]
  if n_matrix != n_vec:
    raise ValueError(
        "Expected polynomial degree of the inputs to match, "
        f"but found {n_vec} != {n_matrix}"
    )

  tpu_version = jax_helpers.get_tpu_version()
  if n_vec % 128 == 0 and tpu_version >= 5:
    return _vector_matrix_polymul(
        vec.astype(jnp.int32), matrix.astype(jnp.int32)
    )
  else:
    return fallback_vector_matrix_polymul(vec, matrix)


def i32_matmul_unreduced(lhs, rhs, out):
  """A helper to isolate the matmul part of the kernel to test in isolation."""
  out[...] = _i32_matmul_unreduced(lhs[...], rhs[...])


def i32_matmul(lhs, rhs):
  """A helper to isolate the matmul part of the kernel to test in isolation."""
  m, k, k2, n = lhs.shape[0], lhs.shape[1], rhs.shape[0], rhs.shape[1]
  assert k == k2
  return jnp.sum(
      pl.pallas_call(
          i32_matmul_unreduced,
          out_shape=jax.ShapeDtypeStruct((4 * m, n), jnp.int32),
      )(lhs, rhs).reshape(4, m, n),
      axis=(0),
  ).astype(jnp.uint32)
