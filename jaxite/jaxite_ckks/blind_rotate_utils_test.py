"""Tests for blind rotation utilities."""

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import basis_conversion
from jaxite.jaxite_ckks import blind_rotate_utils
from jaxite.jaxite_ckks import ntt_cpu
from jaxite.jaxite_ckks import types
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)


class BlindRotateUtilsTest(parameterized.TestCase):

  def test_apply_automorphism_ntt(self):
    # Test with a simple identity automorphism (g = 1)
    degree = 8
    data = jnp.arange(degree, dtype=jnp.float64).reshape(degree, 1)
    res = blind_rotate_utils.apply_automorphism_ntt(data, 1)
    np.testing.assert_array_equal(res, data)

  def test_lift_ciphertext(self):
    degree = 8
    q_limbs = [1073184769]
    p_limbs = [1073479681]
    all_moduli = q_limbs + p_limbs

    # ntt of zero is zero, e is zero
    # b = - a * sk
    a_slots = jnp.zeros((degree, len(q_limbs)), dtype=jnp.uint32)
    b_slots = jnp.zeros((degree, len(q_limbs)), dtype=jnp.uint32)
    ct = types.Ciphertext(
        data=jnp.stack([b_slots, a_slots]),
        moduli=jnp.array(q_limbs, dtype=jnp.uint32),
    )

    bc_kernel = basis_conversion.BasisConversionBarrett()
    bc_kernel.precompute_constants(all_moduli, [([0], [1])])

    lifted_ct = blind_rotate_utils.lift_ciphertext(
        ct,
        bc_kernel,
        control_index=0,
        p_limbs=jnp.array(p_limbs, dtype=jnp.uint32),
    )

    # Output dimensions should match PQ towers
    self.assertEqual(lifted_ct.data.shape, (2, degree, 2))
    np.testing.assert_array_equal(
        lifted_ct.moduli, np.array(all_moduli, dtype=np.uint32)
    )

  @parameterized.parameters(3, 5, 7)
  def test_apply_automorphism_ntt_non_trivial(self, g):
    degree = 8
    q = 1073184769
    # Create a non-trivial polynomial in coefficient domain
    poly = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint64).reshape(
        degree, 1
    )

    # Compute automorphism in coefficient domain
    poly_rot = np.zeros_like(poly)
    for i in range(degree):
      target_pow = (i * g) % (2 * degree)
      val = poly[i, 0]
      if target_pow >= degree:
        target_pow -= degree
        poly_rot[target_pow, 0] = (q - val) % q
      else:
        poly_rot[target_pow, 0] = val

    # Convert to NTT domain
    ntt_poly = ntt_cpu.ntt_negacyclic_poly(poly, [q])
    expected_ntt_poly_rot = ntt_cpu.ntt_negacyclic_poly(poly_rot, [q])

    # Apply automorphism in NTT domain
    res = blind_rotate_utils.apply_automorphism_ntt(jnp.array(ntt_poly), g)

    np.testing.assert_array_equal(np.array(res), expected_ntt_poly_rot)


if __name__ == "__main__":
  absltest.main()
