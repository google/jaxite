"""Tests for Mul kernels."""

import hypothesis
from hypothesis import strategies as st
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import barrett
from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import encrypt
from jaxite.jaxite_ckks import key_gen
from jaxite.jaxite_ckks import mul
from jaxite.jaxite_ckks import random
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


class MulTest(parameterized.TestCase):

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


if __name__ == '__main__':
  absltest.main()
