"""Tests for CKKS encrypt/decrypt routines."""

import random as std_random

import hypothesis
from hypothesis import strategies as st
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import encrypt
from jaxite.jaxite_ckks import key_gen
from jaxite.jaxite_ckks import ntt_cpu
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

    encoder = encode.Encode(degree, moduli, scale)
    pt = encoder.encode(slots)

    encryptor = encrypt.Encrypt(pk)
    ct = encryptor.encrypt(pt, random_source=random_source)

    decryptor = encrypt.Decrypt(sk)
    decrypted_pt = decryptor.decrypt(ct)

    decoder = encode.Decode(scale, len(slots))
    decoded = decoder.decode(decrypted_pt)

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

    pt_data = np.zeros((degree, len(moduli)), dtype=np.uint64)
    for i, q in enumerate(moduli):
      pt_data[:, i] = [rng.randint(0, q - 1) for _ in range(degree)]

    pt_data_ntt = ntt_cpu.ntt_negacyclic_poly(pt_data, moduli)
    pt = types.Plaintext(
        jnp.array(pt_data_ntt, dtype=jnp.uint32),
        jnp.array(moduli, dtype=jnp.uint32),
    )

    random_source = random.ZeroNoiseRandomSource()
    pk, sk = key_gen.keygen(degree, moduli, random_source=random_source)

    encryptor = encrypt.Encrypt(pk)
    ct = encryptor.encrypt(pt, random_source=random_source)

    decryptor = encrypt.Decrypt(sk)
    decrypted_pt = decryptor.decrypt(ct)

    np.testing.assert_array_equal(np.array(decrypted_pt.data), pt_data)

  def test_encrypt_moduli_mismatch_raises_error(self):
    degree = 8
    moduli = [335552513, 335546369]
    pk, _ = key_gen.keygen(degree, moduli)

    pt_data = np.zeros((degree, len(moduli)), dtype=np.uint32)
    bad_moduli = [335546369, 335552513]
    pt = types.Plaintext(
        jnp.array(pt_data), jnp.array(bad_moduli, dtype=jnp.uint32)
    )

    encryptor = encrypt.Encrypt(pk)
    with self.assertRaisesRegex(
        ValueError,
        "Plaintext moduli must match the prefix of public key moduli",
    ):
      encryptor.encrypt(pt)

  def test_decrypt_moduli_mismatch_raises_error(self):
    degree = 8
    moduli = [335552513, 335546369]
    pk, sk = key_gen.keygen(degree, moduli)

    encoder = encode.Encode(degree, moduli, scale=2**10)
    pt = encoder.encode([1.0, 2.0, 3.0, 4.0])

    encryptor = encrypt.Encrypt(pk)
    ct = encryptor.encrypt(pt)

    bad_moduli = [335546369, 335552513]
    bad_ct = types.Ciphertext(ct.data, jnp.array(bad_moduli, dtype=jnp.uint32))

    decryptor = encrypt.Decrypt(sk)
    with self.assertRaisesRegex(
        ValueError,
        "Ciphertext moduli must match the prefix of secret key moduli",
    ):
      decryptor.decrypt(bad_ct)

  def test_jax_compatibility(self):
    """Ensure Ciphertext and keys are valid JAX types."""
    degree = 8
    moduli = [335552513]
    scale = 2**10
    slots = [1.0, 2.0, 3.0, 4.0]

    encoder = encode.Encode(degree, moduli, scale)
    pt = encoder.encode(slots)

    pk, sk = key_gen.keygen(degree, moduli)

    encryptor = encrypt.Encrypt(pk)
    ct = encryptor.encrypt(pt)

    @jax.jit
    def get_data(c):
      return c.data

    data = get_data(ct)
    np.testing.assert_array_equal(data, ct.data)


if __name__ == '__main__':
  absltest.main()
