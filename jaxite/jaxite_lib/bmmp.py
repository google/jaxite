"""Routines related to the BMMP17 bootstrapping trick.

Using the improved blind rotate from Bourse-Minelli-Minihold-Paillier
(BMMP17: https://eprint.iacr.org/2017/1114), a trick uses a larger
bootstrapping key to reduce the number of external products required by 1/2.
Rather than encrypt the secret key bits of the LWE key separately, we
encrypt:

BSK_{3i}   = s_{2i} * s_{2i+1},
BSK_{3i+1} = s_{2i} * (1 − s_{2i+1}),
BSK_{3i+2} = (1 − s_{2i}) * s_{2i+1}

which enables a bootstrap operation that involves 1/2 as many external
products, though this causes the bootstrapping key to be 50% larger.
"""

import functools

import jax
import jax.numpy as jnp
from jaxite.jaxite_lib import types


@jax.jit
def scale_by_x_power_n_minus_1_vanilla_jax(
    power: jnp.int32, matrix: jnp.ndarray
) -> jnp.ndarray:
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
  return x_power_n_part - matrix


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
  output = scale_by_x_power_n_minus_1_vanilla_jax(power, matrix)

  if 0 < log_modulus < 32:
    output = jnp.mod(output, jnp.uint32(2) ** log_modulus)

  return output


@jax.named_call
@functools.partial(jax.jit, static_argnums=(2))
def compute_bmmp_factors(
    coefficient_index: types.LweCiphertext,
    bsk: jnp.ndarray,
    log_coefficient_modulus: int,
):
  """Pre-process the bootstrapping key in preparation for blind rotate."""
  num_loop_terms = (coefficient_index.shape[0] - 1) // 2

  def one_bmmp_factor(j):
    power1 = coefficient_index[2 * j] + coefficient_index[2 * j + 1]
    power2 = coefficient_index[2 * j]
    power3 = coefficient_index[2 * j + 1]
    return (
        scale_by_x_power_n_minus_1(
            power1, bsk[3 * j], log_modulus=log_coefficient_modulus
        )
        + scale_by_x_power_n_minus_1(
            power2, bsk[3 * j + 1], log_modulus=log_coefficient_modulus
        )
        + scale_by_x_power_n_minus_1(
            power3, bsk[3 * j + 2], log_modulus=log_coefficient_modulus
        )
    ).astype(jnp.uint32)

  return jax.vmap(one_bmmp_factor, in_axes=(0,), out_axes=0)(
      jnp.arange(num_loop_terms)
  )
