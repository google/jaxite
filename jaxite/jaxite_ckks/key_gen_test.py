"""Tests for key generation."""

from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import encrypt
from jaxite.jaxite_ckks import key_gen
from jaxite.jaxite_ckks import ntt_cpu
import numpy as np

from absl.testing import absltest

TEST_PRIMES = (
    1_073_692_673,
    1_073_643_521,
    1_073_479_681,
    1_073_430_529,
)


class KeyGenTest(absltest.TestCase):

  def test_keygen_with_hamming_weight(self):
    degree = 4
    moduli = [TEST_PRIMES[0]]
    hamming_weight = 2

    _, sk = key_gen.keygen(degree, moduli, hamming_weight=hamming_weight)

    # Convert secret key back to time domain to verify Hamming weight
    s_coeffs = ntt_cpu.intt_negacyclic_poly(sk.data, moduli)

    # Check first tower (all towers should be identical for binary keys)
    s_coeffs_single = s_coeffs[:, 0]

    # Verify Hamming weight (number of 1s)
    self.assertEqual(np.sum(s_coeffs_single == 1), hamming_weight)
    # Verify all other elements are 0
    self.assertEqual(np.sum(s_coeffs_single == 0), degree - hamming_weight)

  def test_column_key(self):

    scale = 2**10

    degree = 4
    moduli = [TEST_PRIMES[0]]
    hamming_weight = 2

    pk, sk = key_gen.keygen(degree, moduli, hamming_weight=hamming_weight)

    nonzero_indices = []
    for i in range(len(sk.data)):
      if sk.data[i]:
        nonzero_indices.append(i)
    cm_keys = key_gen.gen_cm_keys(nonzero_indices, pk, scale)

    self.assertEqual(len(cm_keys), len(nonzero_indices))

    decryptor = encrypt.Decrypt(sk)
    num_slots = degree // 2
    decoder = encode.Decode(scale, num_slots)

    for i in range(len(cm_keys)):
      self.assertEqual(len(cm_keys[i]), len(nonzero_indices))
      for j in range(len(cm_keys[i])):
        plaintext = decryptor.decrypt(cm_keys[i][j])
        cleartext = decoder.decode(plaintext)

        if i == j:
          np.testing.assert_allclose(cleartext, [1] * num_slots, atol=0.2)
        else:
          np.testing.assert_allclose(cleartext, [0] * num_slots, atol=0.2)


if __name__ == "__main__":
  absltest.main()
