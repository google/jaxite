"""Tests for bmmp."""

import hypothesis
from hypothesis import strategies
import jax.numpy as jnp
from jaxite.jaxite_lib import bmmp
from jaxite.jaxite_lib import matrix_utils
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized


@hypothesis.strategies.composite
def vectors(draw, size, min_value=-(2**31), max_value=2**31 - 1):
  # Note hypothesis.extras.numpy has no build rule in google3
  return np.array(
      draw(
          strategies.lists(
              strategies.integers(min_value=min_value, max_value=max_value),
              min_size=size,
              max_size=size,
          ),
      ),
      dtype=np.int32,
  )


class BmmpTest(parameterized.TestCase):

  def test_kernel_equivalence(self):
    # create a 8x8x64 matrix, with each polynomial the same
    poly = jnp.arange(64).astype(jnp.int32)
    matrix = jnp.tile(poly, reps=jnp.array([8, 8, 1]))
    power = 4
    transformed_poly = jnp.array([-60, -62, -64, -66] + [-4] * 60)
    expected = jnp.tile(transformed_poly, reps=jnp.array([8, 8, 1]))
    actual = bmmp.scale_by_x_power_n_minus_1(power, matrix, log_modulus=32)
    np.testing.assert_array_equal(expected, actual)

  @hypothesis.given(strategies.integers(min_value=0, max_value=10), vectors(16))
  @hypothesis.settings(deadline=None)
  def test_scale_by_x_power_n_minus_1(self, power, poly):
    matrix = jnp.tile(jnp.array(list(poly)), reps=jnp.array([8, 8, 1]))
    poly_term = matrix_utils.x_power_n_minus_1(power, poly_mod_deg=16)
    expected = matrix_utils.poly_mul_const_matrix(poly_term, matrix)
    actual = matrix_utils.scale_by_x_power_n_minus_1(
        power, matrix, log_modulus=32
    )
    np.testing.assert_array_equal(expected, actual)


if __name__ == '__main__':
  absltest.main()
