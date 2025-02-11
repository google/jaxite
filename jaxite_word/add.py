"""TPU kernels for Evaluation of the CKKS algorithm."""

import jax
import jax.numpy as jnp


def jax_add(value_a: jax.Array, value_b: jax.Array, modulus_list: jax.Array):
  """This function processes all degree of the two input polynomials in parallel using multi-trheading.

  Assuming the input data type is jax array.

  Args:
    value_a: the first operand of the addition.
    value_b: the second operand of the addition.
    modulus_list: the list of moduli for each degree.

  Returns:
    The result of the addition.
  """
  num_elements, _, degree = value_a.shape
  modulus_broadcast = jnp.tile(
      modulus_list[None, :, None], (num_elements, 1, degree)
  )
  result = value_a + value_b
  return jnp.where(
      result > modulus_broadcast, result - modulus_broadcast, result
  )  # jnp.mod(value_a + value_b, modulus_broadcast)


def vmap_add(
    value_a: jax.Array, value_b: jax.Array, modulus_list: jax.Array
):
  """This function processes all degree of the two input polynomials in SIMD using jax.vmap.

  Assuming the input data type is jax array.

  Args:
    value_a: the first operand of the addition.
    value_b: the second operand of the addition.
    modulus_list: the list of moduli for each degree.

  Returns:
    The result of the addition.
  """
  num_elements, num_towers, degree = value_a.shape
  modulus_broadcast = jnp.tile(
      modulus_list[None, :, None], (num_elements, 1, degree)
  )

  def chunk_wise_add(value_a, value_b):
    return value_a + value_b

  def chunk_wise_subtract(value_a, value_b):
    return jnp.where(value_a > value_b, value_a - value_b, value_a)

  result = jax.vmap(chunk_wise_add)(value_a, value_b)
  return jax.vmap(chunk_wise_subtract)(result, modulus_broadcast)
