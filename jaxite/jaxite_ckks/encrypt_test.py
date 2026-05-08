"""Tests for CKKS encrypt/decrypt routines."""

import random as std_random

import hypothesis
from hypothesis import strategies as st
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import encrypt
from jaxite.jaxite_ckks import key_gen
from jaxite.jaxite_ckks import random
from jaxite.jaxite_ckks import types
import numpy as np

from absl.testing import absltest

jax.config.update('jax_enable_x64', True)


class EncryptTest(absltest.TestCase):

  @hypothesis.settings(max_examples=25, deadline=None)
  @hypothesis.given(
      st.lists(
          st.complex_numbers(min_magnitude=0, max_magnitude=10),
          min_size=4,
          max_size=4,
      ),
      st.floats(min_value=2**10, max_value=2**15),
      st.integers(min_value=0, max_value=2**32 - 1),
  )
  def test_encrypt_decrypt_loop(self, slots, scale, seed):
    degree = 8
    moduli = [335552513, 335546369]
    random_source = random.TestRandomSource(seed)

    pk, sk = key_gen.keygen(degree, moduli, random_source=random_source)
    pt = encode.encode(slots, degree, moduli, scale)

    ct = encrypt.encrypt(pt, pk, random_source=random_source)
    decrypted_pt = encrypt.decrypt(ct, sk)

    decoded = encode.decode(decrypted_pt, scale, len(slots))

    for s, d in zip(slots, decoded):
      self.assertAlmostEqual(s.real, d.real, delta=0.2)
      self.assertAlmostEqual(s.imag, d.imag, delta=0.2)

  @hypothesis.settings(max_examples=25, deadline=None)
  @hypothesis.given(st.integers(min_value=0, max_value=2**32 - 1))
  def test_exact_encrypt_decrypt(self, seed):
    # This tests encrypt/decrypt without cryptographic noise, which allows for
    # exactness checking to verify the underlying algebra.
    degree = 8
    moduli = [335552513, 335546369]
    rng = std_random.Random(seed)

    pt_data = np.zeros((degree, len(moduli)), dtype=jnp.uint64)
    for i, q in enumerate(moduli):
      pt_data[:, i] = [rng.randint(0, q - 1) for _ in range(degree)]
    pt = types.Plaintext(
        jnp.array(pt_data), jnp.array(moduli, dtype=jnp.uint64)
    )

    random_source = random.ZeroNoiseRandomSource()
    pk, sk = key_gen.keygen(degree, moduli, random_source=random_source)
    ct = encrypt.encrypt(pt, pk, random_source=random_source)
    decrypted_pt = encrypt.decrypt(ct, sk)

    np.testing.assert_array_equal(decrypted_pt.data, pt.data)

  def test_jax_compatibility(self):
    """Ensure Ciphertext and keys are valid JAX types."""
    degree = 8
    moduli = [335552513]
    scale = 2**10
    slots = [1.0, 2.0, 3.0, 4.0]

    pt = encode.encode(slots, degree, moduli, scale)
    pk, sk = key_gen.keygen(degree, moduli)
    ct = encrypt.encrypt(pt, pk)

    @jax.jit
    def get_data(c):
      return c.data

    data = get_data(ct)
    np.testing.assert_array_equal(data, ct.data)


if __name__ == '__main__':
  absltest.main()
