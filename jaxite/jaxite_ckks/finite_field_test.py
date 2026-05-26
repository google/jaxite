"""Finite Field Test Suite

Test cases:
- Montgomery Single Modulus Context
- Barrett Single Modulus Context

Terminology:
- Modulus: Single form of modulus.

Usage:
- Specify the overall modulus for the context, and corresponding parameter
required for the modular reduction.
- Then feed "modulus" and "parameters" to the context constructor.
- Then context->modular_reduction(input) to get the reduced result for certain
inputs.
"""

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import finite_field as ff_context
from jaxite.jaxite_ckks import util
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

testing_params = [{"testcase_name": "0"}]


@parameterized.named_parameters(testing_params)
class FiniteFieldTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Setup random input data and their modmul reference results.
    self.modulus = util.find_moduli_ntt(1, 31, 16)[0]
    self.random_key = jax.random.key(0)
    self.a = jax.random.randint(
        self.random_key, (1,), 0, self.modulus - 1, dtype=jnp.int32
    )
    self.b = jax.random.randint(
        self.random_key, (1,), 0, self.modulus - 1, dtype=jnp.int32
    )
    self.ab = self.a.astype(jnp.uint64) * self.b.astype(jnp.uint64)
    self.ab_modq = (self.ab % self.modulus).astype(jnp.uint32)


  def test_montgomery_single_moduli_context(self):
    context = ff_context.MontgomeryContext(self.modulus)
    a_mont = context.to_computation_format(self.a[0].astype(jnp.uint64))
    b_mont = context.to_computation_format(self.b[0].astype(jnp.uint64))
    ab_mont = a_mont.astype(jnp.uint64) * b_mont.astype(jnp.uint64)
    result_mont = context.modular_reduction(ab_mont)
    result = context.to_original_format(result_mont.astype(jnp.uint64))
    np.testing.assert_array_equal(result[0], self.ab_modq)


  def test_barrett_single_moduli_context(self):
    context = ff_context.BarrettContext(self.modulus)
    ab = self.a.astype(jnp.uint64) * self.b.astype(jnp.uint64)
    result = context.modular_reduction(ab)
    np.testing.assert_array_equal(result[0], self.ab_modq)


if __name__ == "__main__":
  absltest.main()
