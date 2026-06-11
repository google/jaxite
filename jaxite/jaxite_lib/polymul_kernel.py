"""Kernel for negacyclic vector-matrix polymul."""

import functools
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


@jax.jit
def bat_matmul(lhs: jax.Array, y: jax.Array):
  """Perform matrix multiplication between u8(c, m, n, 4, 4)@u32(c, n, k).

  Args:
    lhs: Input u8(c, m, n, 4, 4) Left Matrix
    y: Input u32(c, n, k) Right Matrix, will be converted to u8(c, n, k, 4)

  Returns:
    Output u32(m, k)
  """
  # u8(18, 3, 512, 4, 4) x u8(18, 512, 512, 4) -> u32(3, 512)
  rhs: jax.Array = jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint8)
  i8_products = jnp.einsum(
      "cmnpq,cnkq->cmkp",
      lhs,
      rhs,
      preferred_element_type=jnp.uint32,
  )
  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
  return jnp.sum(i8_products << shift_factors, axis=(0, 3,)).astype(jnp.uint32)


def _i32_matmul_unreduced_CGGI(lhs, rhs):
  """Performs a 32-bit integer matrix multiplication with CGGI optimization.

  This function is an optimized version of i32_matmul_unreduced for decomposed
  vectors, where the RHS matrix is Toeplitzw with decomposition base < 8.

  Args:
    lhs: The left-hand side matrix of shape (m, k).
    rhs: The right-hand side matrix of shape (k, n).

  Returns:
    The result of the matrix multiplication as a jnp.ndarray.
  """
  lax = jax.lax
  m, k, n = lhs.shape[0], lhs.shape[1], rhs.shape[1]

  # Optimized byte extraction for the CGGI trick. This splits the 32-bit
  # integers in the `lhs` matrix into their 8-bit components.
  byte0 = lax.bitwise_and(lhs, 0xFF)
  byte1 = lax.bitwise_and(lax.shift_right_logical(lhs, 8), 0xFF)
  byte2 = lax.bitwise_and(lax.shift_right_logical(lhs, 16), 0xFF)
  byte3 = lax.bitwise_and(lax.shift_right_logical(lhs, 24), 0xFF)

  # Concatenate the bytes to form a new matrix.
  lhs_i8 = jnp.concatenate(
      [
          byte0[: m // 2, :],
          byte1[m // 2 :, :],
          byte2[: m // 2, :],
          byte3[m // 2 :, :],
      ],
      axis=0,
  )

  # rhs is the Toeplitz matrix of the decomposed vector. The values are small
  # (e.g. < 64) because the decomposition base < 8. We can cast to bfloat16
  # directly without splitting into bytes, reducing the number of matmuls by 4x.
  # Perform the matrix multiplication with float32 accumulation for precision.
  raw_out = lax.dot(
      lhs_i8.astype(jnp.bfloat16),
      rhs.astype(jnp.bfloat16),
      preferred_element_type=jnp.float32,  # Ensures accumulation in float32
  ).astype(jnp.int32)
  raw_out = raw_out.reshape((4, m // 2, n))

  # Reconstruct the 32-bit integers from the 8-bit components. This is done by
  # shifting the bytes back to their original positions and summing them up.
  out_shift_base = lax.mul(
      lax.broadcasted_iota(jnp.int32, (4, m // 2, n), dimension=0), 8
  )

  shifted_out = jnp.left_shift(raw_out, out_shift_base)
  return shifted_out[0] + shifted_out[1] + shifted_out[2] + shifted_out[3]


def _decomposed_vector_matrix_polymul(
    poly_vec1: jnp.ndarray, poly_mat2: jnp.ndarray
):
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
      ((0, 0), (0, (m // 2) - real_m), (0, 0)),
      mode="constant",
      constant_values=(0,),
  )
  poly_mat2 = jnp.concatenate((poly_mat2, poly_mat2), axis=(1))
  if n % 128 != 0:
    raise ValueError(f'Input size {n} is not a multiple of 128')

  def vec_mat_polymul_kernel(vec_ref, mat_ref, out_ref):
    """Pallas kernel for polynomial multiplication."""
    for b_i in range(vec_ref.shape[0]):
      chunk = jnp.broadcast_to(vec_ref[b_i, ...], (128, n))
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

      result = _i32_matmul_unreduced_CGGI(mat_ref[b_i, ...], vec_toeplitz)
      out_ref[b_i, ...] = result

  return jnp.sum(
      pl.pallas_call(
          vec_mat_polymul_kernel,
          in_specs=(
              pl.BlockSpec((b, 1, n), lambda i: (i, 0, 0)),
              pl.BlockSpec((b, m, n), lambda i: (i, 0, 0)),
          ),
          out_specs=pl.BlockSpec((b, m // 2, n), lambda i: (i, 0, 0)),
          out_shape=jax.ShapeDtypeStruct((b, m // 2, n), jnp.int32),
          grid=(1,),
          compiler_params=pltpu.CompilerParams(
              # Set the vem limit to 32 MiB, it could be up to 128 MiB.
              vmem_limit_bytes=int(2**10 * 2**15)
          ),
      )(poly_vec1[:, None].astype(jnp.int32), poly_mat2.astype(jnp.int32)),
      axis=0,
  ).astype(jnp.uint32)[:real_m]


@jax.named_call
@functools.partial(jax.jit, static_argnums=(2,))
def negacyclic_vector_matrix_polymul(
    vec: jnp.ndarray, matrix: jnp.ndarray, decomposition_log_base: int
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
        f"matrix.shape={matrix.shape} -- vec.shape={vec.shape}"
    )

  tpu_version = jax_helpers.get_tpu_version()
  if n_vec % 128 == 0 and tpu_version >= 5 and decomposition_log_base < 8:
    return _decomposed_vector_matrix_polymul(
        vec.astype(jnp.int32), matrix.astype(jnp.int32)
    )
  else:
    return fallback_vector_matrix_polymul(vec, matrix)


@jax.named_call
@jax.jit
def negacyclic_vector_matrix_polymul_bat(
    poly_vec1: jnp.ndarray, poly_mat2: jnp.ndarray,
) -> jnp.ndarray:
  """Perform matrix multiplication between u8(c, m, n, 4, 4)@u32(c, n, k).

  Args:
    poly_vec1: Input u8(c, m, n, 4, 4) Left Matrix
    poly_mat2: Input u32(c, n, k) Right Matrix -> converted to u8(c, n, k, 4)

  Returns:
    Output u32(m, k)
  """
  n_vec = poly_vec1.shape[-1]
  n_matrix = poly_mat2.shape[-3]
  if n_matrix != n_vec:
    raise ValueError(
        "Expected polynomial degree of the inputs to match, but found"
        f" {n_vec} != {n_matrix} \n"
        f" poly_vec1.shape={poly_vec1.shape} \n"
        f" poly_mat2.shape={poly_mat2.shape} in"
        " negacyclic_vector_matrix_polymul_bat"
    )

  tpu_version = jax_helpers.get_tpu_version()
  if n_vec % 128 == 0 and tpu_version >= 5:
    # b is the product of the RLWE dimension (e.g., 3) and the number of
    # decomposition levels in the decomposition parameters (e.g., 6).
    # n is the degree of the RLWE polynomials.
    assert poly_vec1.dtype == jnp.uint32
    assert poly_mat2.dtype == jnp.uint8
    b, n = poly_vec1.shape
    # m is the number of polynomials in the RLWE dimension (e.g., 3)
    b2, m, n2, p, q = poly_mat2.shape
    assert b == b2 and n == n2
    # (18, 3, 512)
    # (18,512) -> u32(18, 512, 512)
    # for i in 18:
    #    (3, 512) x (512, 512) -> (3, 512)
    def _toeplitz_chunk(poly_vec1, vec_toeplitz):
      n = poly_vec1.shape[2]
      chunk = jnp.broadcast_to(poly_vec1[...][0], (128, n))
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
      vec_toeplitz[...] = jax.lax.concatenate(
          toeplitz_chunks, dimension=0
      ).reshape(poly_vec1.shape[0], n, n)

    vec_toeplitz_list = pl.pallas_call(
        _toeplitz_chunk,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(pl.BlockSpec((1, 1, n), lambda b: (b, 0, 0)),),
            out_specs=pl.BlockSpec((1, n, n), lambda b: (b, 0, 0)),
            grid=(18,),
        ),
        out_shape=jax.ShapeDtypeStruct((b, n, n), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(poly_vec1[:, None].astype(jnp.int32))

    return bat_matmul(poly_mat2, vec_toeplitz_list)
  else:
    original_mat = jax.lax.bitcast_convert_type(
        poly_mat2[..., :, 0], new_dtype=jnp.uint32
    )
    return fallback_vector_matrix_polymul(poly_vec1, original_mat)


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
