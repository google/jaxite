"""Util file for operations over matrices."""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxite.jaxite_lib import jax_helpers


@jax.jit
def integer_div(values: jnp.ndarray, divisor: jnp.uint32) -> jnp.ndarray:
  """Performs integer division with rounding for positive integers X, d.

  Args:
    values: A vector of values to be divided by `divisor`.
    divisor: The divisor.

  Returns:
    An `jnp.ndarray` vectorizing the division across `values`.
  """
  # If not for overflow, we could compute as: (X + d//2)//d. Adding d//2 to the
  # numerator prior to dividing rounds the result up iff the fractional part of
  # X/d is >= 0.5. This avoids floating point error due to conversion to float32
  # during division.

  # However, the above fails with uint32 for values of X, d close to 2**32. So
  # we split the division in two parts. First get the floor of the quotient,
  # X//d.
  quotient_floor = values // divisor

  # Then compute the quotient of the remainder (X%d) with rounding, which is
  # guaranteed to have a numerator small enough to avoid overflow, and always
  # results in either 0 or 1.
  rounded_fractional_part = ((values % divisor) + divisor // 2) // divisor

  # Adding them together effectively rounds up or down.
  return quotient_floor + rounded_fractional_part


@functools.partial(jax.jit, static_argnames="poly_mod_deg")
def x_power_n_minus_1(n: jnp.uint32, poly_mod_deg: jnp.uint32) -> jnp.ndarray:
  """Construct a polynomial of the form x^d - 1 for an input power d.

  The polynomial is reduced modulo (x^poly_mod_deg + 1). If n == 0, the zero
  polynomial is returned.

  Args:
    n: the nonzero exponent
    poly_mod_deg: the degree of the polynomial modulus to use for reduction. The
      output polynomial is represented by its coefficients as an array of length
      poly_mod_deg+1.

  Returns:
    An array representing the polynomial.
  """
  degree = n % (2 * poly_mod_deg)
  flip = degree // poly_mod_deg
  reduced_degree = degree % poly_mod_deg
  zeros = jnp.zeros(poly_mod_deg, dtype=jnp.uint32)
  return zeros.at[reduced_degree].set((-1) ** flip) - zeros.at[0].set(1)


@jax.jit
def int32_to_int8_arr(arr: jnp.ndarray) -> jnp.ndarray:
  """Decompose an int32 matrix into u8s."""
  return jax.lax.bitcast_convert_type(arr, new_dtype=jnp.uint8)


@jax.jit
def i32_as_u8_matmul(lhs: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
  """Multiply an (n,) by an (n, k) i32 matrix using only i8 ops."""
  if lhs.ndim != 1 or rhs.ndim != 2:
    raise ValueError(
        "lhs be 1-dim and rhs must be 2-dim, "
        f"but they were: lhs={lhs.shape}, rhs={rhs.shape}"
    )

  lhs = int32_to_int8_arr(lhs)
  rhs = int32_to_int8_arr(rhs)

  i8_products = jnp.einsum(
      "np,nkq->kpq",
      lhs,
      rhs,
      preferred_element_type=jnp.int32,
  )
  shift_factors = jnp.array(
      [
          [0, 8, 16, 24],
          [8, 16, 24, 32],
          [16, 24, 32, 40],
          [24, 32, 40, 48],
      ],
      dtype=jnp.int32,
  )
  return jnp.sum(i8_products << shift_factors, axis=(1, 2))


# https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tril.html
# For n=3, generates the following
# [[ 1 -1 -1]
#  [ 1  1 -1]
#  [ 1  1  1]]
@jax.named_call
def _generate_sign_matrix(n: int) -> jnp.ndarray:
  """Generates a sign matrix with 1s below the diagonal and -1 above."""
  up_tri = jnp.tril(jnp.ones((n, n), dtype=int), 0)
  low_tri = jnp.triu(jnp.ones((n, n), dtype=int), 1) * -1
  return up_tri + low_tri


@jax.named_call
@jax.jit
def toeplitz(x: jnp.ndarray) -> jnp.ndarray:
  """Generates a cyclic matrix with each row of the input shifted.

  For input: [1, 9, 2], generates the following matrix:
  [[1 9 2]
   [2 1 9]
   [9 2 1]]

  Args:
    x: the 1D array to shift of length n

  Returns:
    A 2D matrix of shape (n, n), with row i containing the input rolled
    rightward i times.
  """
  if len(x.shape) == 1:
    return toeplitz(x[:, None])
  n = x.shape[-2]
  m = x.shape[-1]
  if m == n:
    return x.transpose()
  if m > n:
    return x[..., :n].transpose()
  r = jnp.roll(x, m, axis=-2)
  return toeplitz(jnp.concatenate([x, r], axis=-1))


@jax.jit
def toeplitz_kernelized(x: jnp.ndarray) -> jnp.ndarray:
  """Use pltpu.roll op to implement toeplitz + sign matrix.

  Note:
    * Only works on TPU v5+.
    * Current implementation assumes
        - both input and output can fit in VMEM.
        - size of input is a multiple of 128.

  Args:
    x: the 1D array to shift of length n

  Returns:
    A 2D matrix of shape (n, n), with row i containing the input rolled
    rightward i times, with the lower-diagonal sign-flipped.
  """
  if len(x.shape) == 1:
    x = x.reshape(1, x.shape[0])
  assert len(x.shape) == 2
  n = x.shape[-1]
  if n % 128 != 0:
    raise ValueError(f"Input size {n} is not a multiple of 128")

  if x.dtype != jnp.float32 and x.dtype != jnp.int32:
    raise ValueError(f"Input {x.dtype} is not supported")

  def _toeplitz(inp_ref, out_ref):
    chunk = jnp.broadcast_to(inp_ref[...], (128, n))
    chunk = pltpu.roll(chunk, 0, 1, stride=1, stride_axis=0)
    chunk_row_indices = jax.lax.broadcasted_iota(
        dtype=jnp.int32, shape=(128, n), dimension=0
    )
    chunk_col_indices = jax.lax.broadcasted_iota(
        dtype=jnp.int32, shape=(128, n), dimension=1
    )
    for r in range(0, n, 128):
      out_ref[pl.ds(r, 128), slice(None)] = jnp.where(
          chunk_row_indices > chunk_col_indices, -chunk, chunk
      )
      # Because the vector registers are aligned to size 128, this roll
      # operation lowers to telling the TPU to refer to a different register,
      # rather than actually applying any rolling operation. Hence, the op
      # produces no hardware instructions.
      chunk = pltpu.roll(chunk, 128, 1)
      chunk_row_indices = chunk_row_indices + 128

  return pl.pallas_call(
      _toeplitz,
      out_shape=jax.ShapeDtypeStruct((n, n), x.dtype),
  )(x)


@jax.named_call
@jax.jit
def toeplitz_poly_mul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
  """Computes a poly multiplication mod (X^N + 1) where N = len(a)."""
  tpu_version = jax_helpers.get_tpu_version()
  n = a.shape[-1]
  if n % 128 == 0 and tpu_version >= 5:
    toeplitzed = toeplitz_kernelized(a.astype(jnp.int32))
    return i32_as_u8_matmul(b, toeplitzed)
  else:
    # This branch is non-optimized, does not lower well on most platforms.
    multiplier = _generate_sign_matrix(len(a))
    left_matrix = multiplier.transpose() * toeplitz(a)
    return i32_as_u8_matmul(b, left_matrix)


@jax.named_call
@jax.jit
def poly_mul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
  """Computes a poly multiplication mod (X^N + 1) where N = len(a)."""
  return toeplitz_poly_mul(a, b)


@jax.named_call
@functools.partial(jax.jit, static_argnames="log_modulus")
def monomial_mul(
    poly: jnp.ndarray, degree: int, log_modulus: int
) -> jnp.ndarray:
  """Computes `poly * X^degree mod (X^N + 1)` where N = len(poly).

  Args:
    poly: a polynomial to be multiplied by X^degree
    degree: the degree of the monomial, X^degree, which may be positive or
      negative. A negative degree is interpreted as a monomial division.
    log_modulus: the log of the coefficient modulus of the polynomial

  Returns:
    The result of the polynomial `poly` multiplied by monomial of degree
    `degree`, mod `2**log_modulus` and `X^N + 1`.

  Example:
     Args: poly = [1, 2, 3], degree = 2, log_modulus = 8
     Returns: [254, 253, 1]

  This takes advantage of the negacyclic property of polynomials mod (X^N + 1),
  as described in Example 9 (p.15) of the Joye TFHE Explainer
  (https://eprint.iacr.org/2021/1402.pdf). At each mulitplication by X, the
  polynomial coefficients are circularly shifted one position to the right and
  the entering coefficient is negated.

  Operations:
    1. After a multiplication by X^{2N}, the polynomial is unchanged. Therefore,
    we can take degree % 2N.
    2. After a multiplication by X^N, all the coefficients of the polynomial
    have been negated but the positions are unchanged.
    Therefore, if degree % 2N > N, negate the coefficients and take degree % N.
    3. Divide the coefficients into "wrapped" (negated) and "unwrapped"
    (unmodified) coefficients, and rearrange the indices accordingly:
      - unwrapped coefficients are moved from the start to the end of the
        polynomial
      - wrapped coefficients are negated and moved to the start of the
        polynomial
  """
  n = poly.shape[0]
  degree = degree % (2 * n)
  shift = degree % n
  flip = (degree // n) % 2 == 1
  indices = jax.lax.broadcasted_iota(
      dtype=jnp.int32, shape=poly.shape, dimension=0
  )
  rolled = jnp.roll(poly, degree)
  rolled = jnp.where(flip, -rolled, rolled)
  output = jnp.where(indices < shift, -rolled, rolled)

  if 0 < log_modulus < 32:
    output = jnp.mod(output, jnp.uint32(2) ** log_modulus)

  return output.astype(poly.dtype)


monomial_mul_list = jax.vmap(monomial_mul, in_axes=(0, None, None), out_axes=0)

# in_axes = (0, 0) means that we apply poly_mul to each row of two input
# matrices. out_axes = (0, ) means the output arrays are stacked one per row
poly_mul_list = jax.vmap(poly_mul, in_axes=(0, 0), out_axes=0)

# poly_mul_const_matrix(poly, list) applies poly_mul(poly, x) to each value x
# of `list`.
poly_mul_const_list = jax.vmap(poly_mul, in_axes=(None, 0), out_axes=0)

# poly_mul_const_matrix(poly, matrix) applies poly_mul(poly, x) to each value x
# of `matrix`.
poly_mul_const_matrix = jax.vmap(
    poly_mul_const_list, in_axes=(None, 0), out_axes=0
)

# Scale the elements of a matrix by a monomial.
monomial_mul_matrix = jax.vmap(
    monomial_mul_list, in_axes=(0, None, None), out_axes=0
)


def poly_dot_product(
    poly_vec1: jnp.ndarray, poly_vec2: jnp.ndarray
) -> jnp.ndarray:
  """Compute a dot product of two vectors of polynomials."""
  return jnp.sum(poly_mul_list(poly_vec1, poly_vec2), axis=0).astype(jnp.uint32)


@functools.partial(jax.jit, static_argnames="log_modulus")
def scale_by_x_power_n_minus_1(
    power: jnp.int32, matrix: jnp.ndarray, log_modulus: int
) -> jnp.ndarray:
  """An optimized poly mul for scaling a matrix of polynomials by x^n - 1.

  Args:
    power: The exponent n of x^n - 1 to scale each matrix entry by
    matrix: The matrix to be scaled.
    log_modulus: the base-2 logarithm of the polynomial coefficient modulus.

  Returns:
    An `jnp.ndarray` of the same shape as `matrix`, containing the
    entries of `matrix` each scaled by x^n - 1.
  """
  # This function can't yet be kernelized because pltpu.roll does not support
  # dynamic shifts.
  indices = jax.lax.broadcasted_iota(
      dtype=jnp.int32, shape=matrix.shape, dimension=2
  )
  n = matrix.shape[2]
  power = power % (2 * n)
  shift = power % n
  flip = (power // n) % 2 == 1
  rolled = jnp.roll(matrix, shift, axis=2)
  rolled = jnp.where(flip, -rolled, rolled)
  x_power_n_part = jnp.where(indices < shift, -rolled, rolled)
  output = x_power_n_part - matrix

  if 0 < log_modulus < 32:
    output = jnp.mod(output, jnp.uint32(2) ** log_modulus)

  return output
