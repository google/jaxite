"""Tests for multiplication and relinearization in CKKS."""

import hypothesis
from hypothesis import strategies as st
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import barrett
from jaxite.jaxite_ckks import basis_conversion
from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import encrypt
from jaxite.jaxite_ckks import key_gen
from jaxite.jaxite_ckks import mul
from jaxite.jaxite_ckks import ntt
from jaxite.jaxite_ckks import random
from jaxite.jaxite_ckks import rescale
from jaxite.jaxite_ckks import types
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

jax.config.update('jax_enable_x64', True)

# Strategy for generating complex slots
NUM_SLOTS = 4
SLOTS_STRATEGY = st.lists(
    st.complex_numbers(
        min_magnitude=0.0,
        max_magnitude=2.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    min_size=NUM_SLOTS,
    max_size=NUM_SLOTS,
)


def _get_kernel(kernel_name, moduli=None):
  match kernel_name:
    case 'simple':
      return mul.MulPlaintextCiphertextSimple()
    case 'modular_barrett':
      assert moduli is not None
      constants = barrett.precompute_barrett_constants(moduli)
      return mul.MulPlaintextCiphertextBarrett(constants)
    case _:
      raise ValueError(f'Unknown kernel: {kernel_name}')


class PlaintextCiphertextMulTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('Simple', 'simple'),
      ('ModularBarrett', 'modular_barrett'),
  )
  def test_mul_simple(self, kernel_name):
    moduli = [17, 19]
    kernel = _get_kernel(kernel_name, moduli=moduli)
    moduli_arr = jnp.array(moduli, dtype=jnp.uint32)
    # Shape (2, 3, 2) -> (num_elements, degree, num_moduli)
    ct = types.Ciphertext(
        data=jnp.array(
            [[[1, 2], [2, 3], [3, 4]], [[5, 6], [6, 7], [7, 8]]],
            dtype=jnp.uint32,
        ),
        moduli=moduli_arr,
    )
    # Shape (3, 2) -> (degree, num_moduli)
    pt = types.Plaintext(
        data=jnp.array([[2, 3], [3, 4], [4, 5]], dtype=jnp.uint32),
        moduli=moduli_arr,
    )

    res = kernel.mul(ct, pt)

    if kernel_name == 'simple':
      expected = ct.data * pt.data
    else:
      # Barrett reduction
      expected = jnp.array(
          [[[2, 6], [6, 12], [12, 1]], [[10, 18], [1, 9], [11, 2]]],
          dtype=jnp.uint32,
      )

    np.testing.assert_array_equal(res.data, expected)
    np.testing.assert_array_equal(res.moduli, moduli_arr)

  @hypothesis.settings(max_examples=25, deadline=None)
  @hypothesis.given(
      st.lists(st.integers(min_value=0, max_value=16), min_size=2, max_size=2),
      st.lists(st.integers(min_value=0, max_value=16), min_size=2, max_size=2),
  )
  def test_mul_modular_barrett_hypothesis(self, ct_list, pt_list):
    moduli = [17, 19]
    moduli_arr = jnp.array(moduli, dtype=jnp.uint32)
    for kernel_name in ['simple', 'modular_barrett']:
      kernel = _get_kernel(kernel_name, moduli)

      # Shape (1, 1, 2) for ct, (1, 2) for pt
      ct = types.Ciphertext(
          data=jnp.array([[ct_list]], dtype=jnp.uint32),
          moduli=moduli_arr,
      )
      pt = types.Plaintext(
          data=jnp.array([pt_list], dtype=jnp.uint32),
          moduli=moduli_arr,
      )

      res = kernel.mul(ct, pt)

      if kernel_name == 'simple':
        expected = ct.data * pt.data
      else:
        expected = (
            ct.data.astype(jnp.uint64) * pt.data.astype(jnp.uint64)
        ) % moduli_arr
        expected = expected.astype(jnp.uint32)
      np.testing.assert_array_equal(res.data, expected)
      np.testing.assert_array_equal(res.moduli, moduli_arr)

  @hypothesis.settings(
      max_examples=10,
      deadline=None,
  )
  @hypothesis.given(SLOTS_STRATEGY, SLOTS_STRATEGY)
  def test_full_pipeline_mul(self, slots1, slots2):
    degree = 1024
    moduli = [335552513, 335546369]
    scale = 2**20

    encoder = encode.Encode(degree, moduli, scale)
    pt1 = encoder.encode(slots1)
    pt2 = encoder.encode(slots2)

    # Use a fixed seed for determinism in Hypothesis tests to prevent flakiness
    test_random_source = random.TestRandomSource(seed=42)
    pk, sk = key_gen.keygen(degree, moduli, random_source=test_random_source)

    encryptor = encrypt.Encrypt(pk)
    ct1 = encryptor.encrypt(pt1, random_source=test_random_source)

    # Multiply ciphertext with plaintext
    constants = barrett.precompute_barrett_constants(moduli)
    mul_kernel = mul.MulPlaintextCiphertextBarrett(constants)

    # Multiply ciphertext with plaintext
    ct_mul = mul_kernel.mul(ct1, pt2)

    decryptor = encrypt.Decrypt(sk)
    pt_dec = decryptor.decrypt(ct_mul)

    # Scale is now scale^2 after multiplication
    decoder = encode.Decode(scale * scale, len(slots1))
    decoded = decoder.decode(pt_dec)

    expected_slots = [s1 * s2 for s1, s2 in zip(slots1, slots2)]

    for s, d in zip(expected_slots, decoded):
      # We use delta=0.15 to ensure stability with degree 1024 and scale 2^20
      self.assertAlmostEqual(s.real, d.real, delta=0.15)
      self.assertAlmostEqual(s.imag, d.imag, delta=0.15)


class CiphertextCiphertextMulTest(absltest.TestCase):

  def test_encrypt_multiply_decrypt(self):
    q_towers = [536872097, 536870657, 536872001, 536870849, 536871233, 524353]
    p_towers = [1073741441, 1073740609]
    r, c = 4, 4
    degree = r * c
    dnum = 3
    scale = 281510041637249
    num_slots = 8

    pk, sk = key_gen.keygen(degree, q_towers + p_towers)
    encoder = encode.Encode(degree, q_towers, scale)
    encryptor = encrypt.Encrypt(pk)

    output_scale = (scale / q_towers[-1]) ** 2
    decoder = encode.Decode(scale=output_scale, num_slots=num_slots)

    sk_q = types.SecretKey(
        data=sk.data[:, : len(q_towers) - 1],
        moduli=sk.moduli[: len(q_towers) - 1],
    )
    evk = key_gen.gen_evaluation_key(sk_q, q_towers[:-1], p_towers, dnum)

    decryptor = encrypt.Decrypt(sk_q)

    slots1 = [0.25, 0.5, 0.75, 1, 2, 3, 4, 5]
    slots2 = [5, 4, 3, 2, 1, 0.75, 0.5, 0.25]
    expected_slots = [c1 * c2 for c1, c2 in zip(slots1, slots2)]

    pt1 = encoder.encode(slots1)
    pt2 = encoder.encode(slots2)

    ct1 = encryptor.encrypt(pt1)
    ct2 = encryptor.encrypt(pt2)

    rescaler = rescale.Rescale()
    rescaler.precompute_constants(q_towers, 1, r, c)

    drop_last_moduli = q_towers[:-1]
    extend_moduli = p_towers
    drop_last_extend_moduli = drop_last_moduli + extend_moduli

    control_indices = mul.Mul.compute_control_indices(
        drop_last_moduli, extend_moduli, dnum
    )

    bconv = basis_conversion.BasisConversionBarrett()
    bconv.precompute_constants(drop_last_extend_moduli, control_indices)

    full_ntt = ntt.NTTBarrett()
    full_ntt.precompute_constants(drop_last_extend_moduli, r, c)

    mul_kernel = mul.Mul(
        bconv=bconv,
        full_ntt=full_ntt,
    )
    mul_kernel.precompute_constants(
        q_towers, p_towers, dnum, r, c, composite_degree=1
    )

    rescaler.rescale(ct1)
    rescaler.rescale(ct2)
    ct_3elem = mul_kernel.tensor_multiply(ct1, ct2)
    res_ct = mul_kernel.relinearize(ct_3elem, evk)

    # We need to use the secret key on the rescaled ciphertext.
    # res_ct has moduli q_towers[:-1]
    pt_dec = decryptor.decrypt(res_ct)
    self.assertEqual(pt_dec.data.shape, (degree, len(q_towers) - 1))

    decoded_slots = decoder.decode(pt_dec, is_slot_form=False)

    np.testing.assert_allclose(
        np.array(decoded_slots),
        np.array(expected_slots),
        rtol=1e-5,
        atol=1e-5,
    )

  def test_relinearize_batched(self):
    q_towers = [536872097, 536870657, 536872001, 536870849, 536871233, 524353]
    p_towers = [1073741441, 1073740609]
    r, c = 4, 4
    degree = r * c
    dnum = 3
    scale = 281510041637249

    pk, sk = key_gen.keygen(degree, q_towers + p_towers)
    encoder = encode.Encode(degree, q_towers, scale)
    encryptor = encrypt.Encrypt(pk)

    sk_q = types.SecretKey(
        data=sk.data[:, : len(q_towers) - 1],
        moduli=sk.moduli[: len(q_towers) - 1],
    )
    evk = key_gen.gen_evaluation_key(sk_q, q_towers[:-1], p_towers, dnum)

    slots1 = [0.25, 0.5, 0.75, 1, 2, 3, 4, 5]
    slots2 = [5, 4, 3, 2, 1, 0.75, 0.5, 0.25]

    pt1 = encoder.encode(slots1)
    pt2 = encoder.encode(slots2)

    ct1 = encryptor.encrypt(pt1)
    ct2 = encryptor.encrypt(pt2)

    rescaler = rescale.Rescale()
    rescaler.precompute_constants(q_towers, 1, r, c)

    drop_last_moduli = q_towers[:-1]
    extend_moduli = p_towers
    drop_last_extend_moduli = drop_last_moduli + extend_moduli

    control_indices = mul.Mul.compute_control_indices(
        drop_last_moduli, extend_moduli, dnum
    )

    bconv = basis_conversion.BasisConversionBarrett()
    bconv.precompute_constants(drop_last_extend_moduli, control_indices)

    full_ntt = ntt.NTTBarrett()
    full_ntt.precompute_constants(drop_last_extend_moduli, r, c)

    mul_kernel = mul.Mul(
        bconv=bconv,
        full_ntt=full_ntt,
    )
    mul_kernel.precompute_constants(
        q_towers, p_towers, dnum, r, c, composite_degree=1
    )

    rescaler.rescale(ct1)
    rescaler.rescale(ct2)
    ct_3elem = mul_kernel.tensor_multiply(ct1, ct2)

    expected_res = mul_kernel.relinearize(ct_3elem, evk)

    # Create a batched ciphertext
    batched_data = jnp.stack([ct_3elem.data, ct_3elem.data])
    ct_3elem_batched = types.Ciphertext(
        data=batched_data, moduli=ct_3elem.moduli
    )

    res_ct = mul_kernel.relinearize(ct_3elem_batched, evk)

    self.assertEqual(res_ct.data.shape, (2, 2, degree, len(q_towers) - 1))

    np.testing.assert_allclose(res_ct.data[0], expected_res.data)
    np.testing.assert_allclose(res_ct.data[1], expected_res.data)

  def test_mul_pytree(self):
    q_towers = [536872097, 536870657, 536872001, 536870849, 536871233, 524353]
    p_towers = [1073741441, 1073740609]
    r, c = 4, 4
    dnum = 3

    mul_kernel = mul.Mul(
        bconv=basis_conversion.BasisConversionBarrett(),
        ntt_factory=ntt.NTTBarrett,
    )
    mul_kernel.precompute_constants(
        q_towers, p_towers, dnum, r, c, composite_degree=1
    )

    children, aux_data = jax.tree_util.tree_flatten(mul_kernel)
    mul_unflattened = jax.tree_util.tree_unflatten(aux_data, children)

    self.assertTrue(mul_unflattened.is_initialized)
    self.assertEqual(
        mul_unflattened.original_moduli, mul_kernel.original_moduli
    )
    self.assertEqual(mul_unflattened.extend_moduli, mul_kernel.extend_moduli)
    self.assertIsNotNone(mul_unflattened.bconv)
    self.assertIsNotNone(mul_unflattened.ntt_current)
    self.assertIsNotNone(mul_unflattened.ntt_extend)
    self.assertIsNotNone(mul_unflattened.ks_ntt_kernels)


if __name__ == '__main__':
  absltest.main()
