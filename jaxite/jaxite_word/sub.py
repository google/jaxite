"""TPU kernels for Evaluation of the CKKS algorithm."""

import jax
import jax.numpy as jnp


def jax_sub(value_a: jax.Array, value_b: jax.Array, modulus_list: jax.Array):
  """This function processes all degree of the two input polynomials in parallel using multi-trheading.

  Assuming the input data type is jax array of shape (n, k, d) where
    n: Number of polynomials in the ciphertext
    k: The number of limbs
    d: The degree of the polynomials

  Args:
    value_a: the first operand of the subtraction.
    value_b: the second operand of the subtraction.
    modulus_list: the list of moduli for each degree.

  Returns:
    The result of the subtraction.
  """
  num_elements, _, degree = value_a.shape
  modulus_broadcast = jnp.tile(
      modulus_list[None, :, None], (num_elements, 1, degree)
  )
  result = value_a - value_b
  result_mod_back = modulus_broadcast + result
  return jnp.where(
      value_a > value_b, result, result_mod_back
  )  # jnp.mod(value_a + value_b, modulus_broadcast)


def vmap_sub(value_a: jax.Array, value_b: jax.Array, modulus_list: jax.Array):
  """This function processes all degree of the two input polynomials in SIMD using jax.vmap.

  Assuming the input data type is jax array.

  Args:
    value_a: the first operand of the subtraction.
    value_b: the second operand of the subtraction.
    modulus_list: the list of moduli for each degree.

  Returns:
    The result of the subtraction.
  """
  num_elements, num_towers, degree = value_a.shape
  modulus_broadcast = jnp.tile(
      modulus_list[None, :, None], (num_elements, 1, degree)
  )

  def chunk_wise_subtract(value_a, value_b, mod):
    result = value_a - value_b
    return jnp.where(value_a > value_b, result, result + mod)

  return jax.vmap(chunk_wise_subtract)(value_a, value_b, modulus_broadcast)
