"""Finite Field Test Suite

Test cases:
- Montgomery Single Modulus Context
- Barrett Single Modulus Context
- Shoup Single Modulus Context
- Montgomery Multi Modulus Context
- Barrett Multi Modulus Context
- Shoup Multi Modulus Context

Terminology:
- Modulus -- Moduli: Single form or plural form of modulus.


Usage:
- Specify the overall moduli for the context, and corresponding parameter
required for the modular reduction.
- Then feed "moduli" and "parameters" to the context constructor.
- Then context->modular_reduction(input) to get the reduced result for certain
inputs.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jaxite.jaxite_word.ciphertext as ct
import jaxite.jaxite_word.finite_field as ff_context
import jax
import jax.numpy as jnp
import numpy as np

testing_params = [{"testcase_name": "0"}]


@parameterized.named_parameters(testing_params)
class FiniteFieldTest(parameterized.TestCase):

  def setUp(self):
    # Setup random input data and their modmul reference results.
    self.random_key = jax.random.key(0)
    # batch, (w/ element), moduli (limbs/towers), degree
    shapes = {
        "batch": 3,
        "num_elements": 2,
        "num_moduli": 4,
        "degree": 16,
        "precision": 29,
    }

    ct_a = ct.Ciphertext(shapes)
    ct_b = ct.Ciphertext(shapes, {"moduli": ct_a.get_moduli()})
    ct_ab = ct.Ciphertext(shapes, {"moduli": ct_a.get_moduli()})
    ct_ab_modq = ct.Ciphertext(shapes, {"moduli": ct_a.get_moduli()})
    ct_a.random_init()
    ct_b.random_init()
    self.a = ct_a.get_batch_ciphertext().astype(jnp.uint64)
    self.b = ct_b.get_batch_ciphertext().astype(jnp.uint64)
    self.ab = self.a * self.b
    self.ab_modq = (self.ab % ct_a.get_moduli_array()).astype(jnp.uint32)
    ct_ab.set_batch_ciphertext(self.ab)
    ct_ab_modq.set_batch_ciphertext(self.ab_modq)

    self.single_a = ct_a.get_limb(0).astype(jnp.uint64)
    self.single_b = ct_b.get_limb(0).astype(jnp.uint64)
    self.single_moduli = ct_ab.get_modulus(0)
    self.single_ab = ct_ab.get_limb(0)
    self.single_ab_modq = ct_ab_modq.get_limb(0)

    self.moduli = ct_ab.get_moduli()

  # @absltest.skip("test single implementation")
  def test_montgomery_single_moduli_context(self):
    context = ff_context.MontgomeryContext(self.single_moduli)
    single_a_mont = context.to_computation_format(
        self.single_a.astype(jnp.uint64)
    )
    single_b_mont = context.to_computation_format(
        self.single_b.astype(jnp.uint64)
    )
    single_ab_mont = single_a_mont.astype(jnp.uint64) * single_b_mont.astype(
        jnp.uint64
    )
    result_mont = context.modular_reduction(single_ab_mont)
    result = context.to_original_format(result_mont.astype(jnp.uint64))
    np.testing.assert_array_equal(result, self.single_ab_modq)

  # @absltest.skip("test single implementation")
  def test_barrett_single_moduli_context(self):
    context = ff_context.BarrettContext(self.single_moduli)
    result = context.modular_reduction(self.single_ab)
    np.testing.assert_array_equal(result, self.single_ab_modq)

  # @absltest.skip("test single implementation")
  def test_shoup_single_moduli_context(self):
    context = ff_context.ShoupContext(self.single_moduli)
    single_a_precomputed = context.precompute_constant_operand(
        self.single_a.astype(jnp.uint64)
    )
    single_ab = self.single_a.astype(jnp.uint64) * self.single_b.astype(
        jnp.uint64
    )
    single_ab_shoup = single_a_precomputed * self.single_b.astype(jnp.uint64)
    result_shoup = context.modular_reduction(single_ab, single_ab_shoup)
    result = context.to_original_format(result_shoup.astype(jnp.uint64))
    np.testing.assert_array_equal(result, self.single_ab_modq)

  # @absltest.skip("test single implementation")
  def test_montgomery_multi_moduli_context(self):
    context = ff_context.MontgomeryContext(self.moduli)
    a_mont = context.to_computation_format(self.a.astype(jnp.uint64))
    b_mont = context.to_computation_format(self.b.astype(jnp.uint64))
    ab_mont = a_mont.astype(jnp.uint64) * b_mont.astype(jnp.uint64)
    result_mont = context.modular_reduction(ab_mont)
    result = context.to_original_format(result_mont.astype(jnp.uint64))
    np.testing.assert_array_equal(result, self.ab_modq)

  # @absltest.skip("test single implementation")
  def test_barrett_multi_moduli_context(self):
    context = ff_context.BarrettContext(self.moduli)
    result = context.modular_reduction(self.ab)
    np.testing.assert_array_equal(result, self.ab_modq)

  # @absltest.skip("test single implementation")
  def test_shoup_multi_moduli_context(self):
    context = ff_context.ShoupContext(self.moduli)
    a_precomputed = context.precompute_constant_operand(
        self.a.astype(jnp.uint64)
    )
    ab = self.a.astype(jnp.uint64) * self.b.astype(jnp.uint64)
    ab_shoup = a_precomputed * self.b.astype(jnp.uint64)
    result_shoup = context.modular_reduction(ab, ab_shoup)
    result = context.to_original_format(result_shoup.astype(jnp.uint64))
    np.testing.assert_array_equal(result, self.ab_modq)

  # @absltest.skip("test single implementation")
  def test_ntt_conversion(self):
    shapes = {
        "batch": 3,
        "num_elements": 2,
        "num_moduli": 4,
        "degree": 16,
        "precision": 29,
    }
    parameters = {
        "r": 4,
        "c": 4,
        "finite_field_context": (
            ff_context.BarrettContext
        ),  # ff_context.BarrettContext, ff_context.MontgomeryContext, ff_context.ShoupContext
    }
    ct_temp = ct.Ciphertext(shapes, parameters)
    ct_temp.random_init()

    original = ct_temp.get_batch_ciphertext()

    # Check NTT round trip
    ct_temp.to_compute_format()
    ct_temp.to_ntt_form()
    ct_temp.to_coeffs_form()
    ct_temp.to_original_format()

    np.testing.assert_array_equal(original, ct_temp.get_batch_ciphertext())


class CiphertextLimbDomainConversionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.limb_shapes = {
        "batch": 2,
        "num_elements": 2,
        "num_moduli": 1,
        "degree": 16,
        "precision": 29,
    }
    self.drop_shapes = {
        "batch": 2,
        "num_elements": 2,
        "num_moduli": 3,
        "degree": 8,
        "precision": 29,
    }
    self.limb_index = 0
    base_ct = ct.Ciphertext(self.limb_shapes)
    base_ct.random_init()
    self.original = base_ct.get_batch_ciphertext()
    self.moduli = base_ct.get_moduli()

  def test_drop_last_modulus_preserves_remaining_limbs_and_context(self):
    ct_temp = ct.Ciphertext(self.drop_shapes)
    ct_temp.random_init()
    before_drop = ct_temp.get_batch_ciphertext()
    expected_remaining = before_drop[:, :, :, :-1]

    ct_temp.drop_last_modulus()

    np.testing.assert_array_equal(
        ct_temp.get_batch_ciphertext(), expected_remaining
    )
    self.assertEqual(ct_temp.num_moduli, self.drop_shapes["num_moduli"] - 1)
    self.assertEqual(
        ct_temp.ntt_ctx.ff_ctx.moduli_reduction.shape[0],
        self.drop_shapes["num_moduli"] - 1,
    )


class CiphertextArithmeticTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.shapes = {
        "batch": 1,
        "num_elements": 1,
        "num_moduli": 2,
        "degree": 8,
        "precision": 29,
    }
    self.ct1 = ct.Ciphertext(self.shapes)
    self.ct1.random_init()
    self.ct2 = ct.Ciphertext(self.shapes, {"moduli": self.ct1.get_moduli()})
    self.ct2.random_init()

    self.arr1 = self.ct1.get_batch_ciphertext()
    self.arr2 = self.ct2.get_batch_ciphertext()

  # @absltest.skip("test single implementation")
  def test_mul_ciphertext(self):
    # mul modifies in-place
    expected = self.arr1.astype(jnp.uint64) * self.arr2.astype(jnp.uint64)
    self.ct1.mul(self.ct2)
    np.testing.assert_array_equal(self.ct1.get_batch_ciphertext(), expected)

  # @absltest.skip("test single implementation")
  def test_mul_array(self):
    expected = self.arr1.astype(jnp.uint64) * self.arr2.astype(jnp.uint64)
    self.ct1.mul(self.arr2)
    np.testing.assert_array_equal(self.ct1.get_batch_ciphertext(), expected)

  # @absltest.skip("test single implementation")
  def test_modmul_ciphertext(self):
    expected_temp = self.arr1.astype(jnp.uint64) * self.arr2.astype(jnp.uint64)
    expected = self.ct1.ntt_ctx.ff_ctx.modular_reduction(expected_temp).astype(
        self.ct1.modulus_dtype
    )

    self.ct1.modmul(self.ct2)
    np.testing.assert_array_equal(self.ct1.get_batch_ciphertext(), expected)

  # @absltest.skip("test single implementation")
  def test_modmul_array(self):
    expected_temp = self.arr1.astype(jnp.uint64) * self.arr2.astype(jnp.uint64)
    expected = self.ct1.ntt_ctx.ff_ctx.modular_reduction(expected_temp).astype(
        self.ct1.modulus_dtype
    )

    self.ct1.modmul(self.arr2)
    np.testing.assert_array_equal(self.ct1.get_batch_ciphertext(), expected)

  # @absltest.skip("test single implementation")
  def test_mod_reduce(self):
    # Create a value that needs reduction
    self.ct1.ciphertext = self.ct1.ciphertext.astype(jnp.uint64) * 100
    expected = self.ct1.ntt_ctx.ff_ctx.modular_reduction(
        self.ct1.ciphertext
    ).astype(self.ct1.modulus_dtype)

    self.ct1.mod_reduce()
    np.testing.assert_array_equal(self.ct1.get_batch_ciphertext(), expected)


if __name__ == "__main__":
  absltest.main()
