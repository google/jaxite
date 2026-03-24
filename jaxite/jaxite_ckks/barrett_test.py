"""Tests for Barrett reduction."""

import hypothesis
from hypothesis import strategies as st
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import barrett
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

# Enable 64-bit precision for large integer arithmetic
jax.config.update("jax_enable_x64", True)

BATCH_SIZE = 20


@st.composite
def moduli_and_z(draw):
  """Strategy to generate a batch of moduli and corresponding inputs."""
  # Generate exactly BATCH_SIZE moduli in the range [2, 2^31 - 1]. This is
  # important because is the shape of an input array changes, jax will recompile
  # the function and that takes a few seconds. Combine this with hypothesis's
  # test loop and it causes a timeout even for small parameters.
  moduli = draw(
      st.lists(
          st.integers(min_value=2, max_value=(1 << 31) - 1),
          min_size=BATCH_SIZE,
          max_size=BATCH_SIZE,
      )
  )

  # Input must be in the range [0, m_i^2 - 1]
  z_values = [
      draw(st.integers(min_value=0, max_value=m**2 - 1)) for m in moduli
  ]
  return moduli, z_values


class BarrettTest(parameterized.TestCase):

  @parameterized.parameters(
      (1073753729, [0, 1, 1073753728, 1073753729, 2000000000]),
      (65537, [0, 1, 65536, 65537, 1000000]),
  )
  def test_modular_reduction_basic(self, modulus, inputs):
    expected = [x % modulus for x in inputs]
    constants = barrett.precompute_barrett_constants(modulus)
    unreduced = jnp.array(inputs, dtype=jnp.uint64)
    actual = barrett.modular_reduction(unreduced, constants)
    np.testing.assert_array_equal(actual, expected)

  @hypothesis.settings(deadline=None, max_examples=50)
  @hypothesis.given(moduli_and_z())
  def test_modular_reduction_hypothesis(self, moduli_and_input):
    moduli, input_values = moduli_and_input
    expected = [z_val % mod for z_val, mod in zip(input_values, moduli)]

    constants = barrett.precompute_barrett_constants(moduli)
    actual = barrett.modular_reduction(
        jnp.array(input_values, dtype=jnp.uint64), constants
    )

    np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
  absltest.main()
