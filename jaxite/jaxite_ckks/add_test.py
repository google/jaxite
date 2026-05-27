"""Tests for Add kernels."""

import hypothesis
from hypothesis import strategies as st
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import add
from jaxite.jaxite_ckks import barrett
from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import encrypt
from jaxite.jaxite_ckks import key_gen
from jaxite.jaxite_ckks import types
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

jax.config.update('jax_enable_x64', True)


def _get_kernel(kernel_name, moduli=None):
  match kernel_name:
    case 'simple':
      return add.AddSimple()
    case 'modular_barrett':
      assert moduli is not None
      constants = barrett.precompute_barrett_constants(moduli)
      return add.AddModularBarrett(constants)
    case 'modular_subtract':
      assert moduli is not None
      return add.AddModularSubtract(moduli)
    case _:
      raise ValueError(f'Unknown kernel: {kernel_name}')


class AddTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('Simple', 'simple'),
      ('ModularBarrett', 'modular_barrett'),
      ('ModularSubtract', 'modular_subtract'),
  )
  def test_add_simple(self, kernel_name):
    kernel = _get_kernel(kernel_name, moduli=[17, 19])
    a = jnp.array([[1, 2], [2, 3], [3, 4]], dtype=jnp.uint64)
    b = jnp.array([[4, 5], [5, 6], [6, 7]], dtype=jnp.uint64)
    res = kernel.add(a, b)
    expected = jnp.array([[5, 7], [7, 9], [9, 11]], dtype=jnp.uint64)
    np.testing.assert_array_equal(res, expected)

  @parameterized.named_parameters(
      ('Simple', 'simple'),
      ('ModularBarrett', 'modular_barrett'),
      ('ModularSubtract', 'modular_subtract'),
  )
  def test_add_modular_barrett(self, kernel_name):
    moduli = [17, 19]
    kernel = _get_kernel(kernel_name, moduli)

    # Shape (2, 2) -> (degree, num_moduli)
    a = jnp.array([[10, 15], [5, 10]], dtype=jnp.uint64)
    b = jnp.array([[8, 5], [15, 12]], dtype=jnp.uint64)

    res = kernel.add(a, b)

    if kernel_name == 'simple':
      expected = a + b
    else:
      expected = jnp.array([[1, 1], [3, 3]], dtype=jnp.uint64)
    np.testing.assert_array_equal(res, expected)

  @parameterized.named_parameters(
      ('Simple', 'simple'),
      ('ModularBarrett', 'modular_barrett'),
      ('ModularSubtract', 'modular_subtract'),
  )
  def test_add_modular_subtract(self, kernel_name):
    moduli = [17, 19]
    kernel = _get_kernel(kernel_name, moduli)

    # Inputs must be reduced
    a = jnp.array([[10, 15], [5, 10]], dtype=jnp.uint64)
    b = jnp.array([[8, 5], [10, 8]], dtype=jnp.uint64)

    res = kernel.add(a, b)

    if kernel_name == 'simple':
      expected = a + b
    else:
      expected = jnp.array([[1, 1], [15, 18]], dtype=jnp.uint64)
    np.testing.assert_array_equal(res, expected)

  @hypothesis.settings(max_examples=25, deadline=None)
  @hypothesis.given(
      st.lists(st.integers(min_value=0, max_value=16), min_size=2, max_size=2),
      st.lists(st.integers(min_value=0, max_value=16), min_size=2, max_size=2),
  )
  def test_add_modular_subtract_hypothesis(self, a_list, b_list):
    moduli = [17, 19]
    for kernel_name in ['simple', 'modular_barrett', 'modular_subtract']:
      kernel = _get_kernel(kernel_name, moduli)

      a = jnp.array(a_list, dtype=jnp.uint64)
      b = jnp.array(b_list, dtype=jnp.uint64)

      res = kernel.add(a, b)

      moduli_arr = jnp.array(moduli, dtype=jnp.uint64)
      if kernel_name == 'simple':
        expected = a + b
      else:
        expected = (a + b) % moduli_arr
      np.testing.assert_array_equal(res, expected)

  @parameterized.named_parameters(
      ('Simple', 'simple'),
      ('ModularBarrett', 'modular_barrett'),
      ('ModularSubtract', 'modular_subtract'),
  )
  def test_full_pipeline_add(self, kernel_name):
    degree = 8
    moduli = [335552513, 335546369]
    scale = 2**10

    slots1 = [
        complex(1.0, 2.0),
        complex(3.0, 4.0),
        complex(5.0, 6.0),
        complex(7.0, 8.0),
    ]
    slots2 = [
        complex(0.5, 0.5),
        complex(1.5, 1.5),
        complex(2.5, 2.5),
        complex(3.5, 3.5),
    ]

    encoder = encode.Encode()
    encoder.precompute_constants(degree, moduli, scale)
    pt1 = encoder.encode(slots1)
    pt2 = encoder.encode(slots2)

    pk, sk = key_gen.keygen(degree, moduli)

    encryptor = encrypt.Encrypt()
    encryptor.precompute_constants(pk)
    ct1 = encryptor.encrypt(pt1)
    ct2 = encryptor.encrypt(pt2)

    # Add ciphertexts
    add_kernel = _get_kernel(kernel_name, moduli)

    ct_add_data = add_kernel.add(ct1.data, ct2.data)
    ct_add = types.Ciphertext(data=ct_add_data, moduli=ct1.moduli)

    decryptor = encrypt.Decrypt()
    decryptor.precompute_constants(sk)
    pt_dec = decryptor.decrypt(ct_add)

    decoder = encode.Decode()
    decoder.precompute_constants(scale, len(slots1))
    decoded = decoder.decode(pt_dec)

    expected_slots = [s1 + s2 for s1, s2 in zip(slots1, slots2)]

    for s, d in zip(expected_slots, decoded):
      self.assertAlmostEqual(s.real, d.real, delta=0.5)
      self.assertAlmostEqual(s.imag, d.imag, delta=0.5)


if __name__ == '__main__':
  absltest.main()
