"""Tests for key generation."""

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import key_gen
from jaxite.jaxite_ckks import ntt_cpu
from jaxite.jaxite_ckks import types
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

    pk, sk = key_gen.keygen(degree, moduli, hamming_weight=hamming_weight)

    # Convert secret key back to time domain to verify Hamming weight
    s_coeffs = ntt_cpu.intt_negacyclic_poly(sk.data, moduli)

    # Check first tower (all towers should be identical for binary keys)
    s_coeffs_single = s_coeffs[:, 0]

    # Verify Hamming weight (number of 1s)
    self.assertEqual(np.sum(s_coeffs_single == 1), hamming_weight)
    # Verify all other elements are 0
    self.assertEqual(np.sum(s_coeffs_single == 0), degree - hamming_weight)


if __name__ == "__main__":
  absltest.main()
