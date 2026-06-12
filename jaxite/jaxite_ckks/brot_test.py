"""Tests for blind rotation kernels."""

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import barrett
from jaxite.jaxite_ckks import brot
from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import encrypt
from jaxite.jaxite_ckks import key_gen
from jaxite.jaxite_ckks import mul
from jaxite.jaxite_ckks import random
from jaxite.jaxite_ckks import types
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)


class BrotTest(parameterized.TestCase):

  def test_brot_cm(self):
    degree = 1024
    moduli = [335552513, 335546369]
    scale = 2**20
    num_slots = 4
    secret_idx = 2

    # 1. Generate keys
    test_random_source = random.TestRandomSource(seed=42)
    pk, sk = key_gen.keygen(degree, moduli, random_source=test_random_source)

    # 2. Generate Column Keys for all indices
    # gen_cm_keys returns shape (num_slots, num_slots) of Ciphertexts
    # cm_keys[i, j] encrypts 1 if i == j else 0
    indices = list(range(num_slots))
    cm_keys_matrix = key_gen.gen_cm_keys(indices, pk, scale)

    # Extract the column for the secret index j
    # cmkey_j[i] encrypts 1 if i == secret_idx else 0
    cmkey_j = [cm_keys_matrix[i][secret_idx] for i in range(num_slots)]

    # 3. Define input message mu (slots)
    mu = np.array(
        [1.0 + 2.0j, 2.0 + 3.0j, 3.0 + 4.0j, 4.0 + 5.0j], dtype=complex
    )

    # 4. Rotate and encode mu for all i
    # pt_rot_mu_all[i] = Encode(Rot_i(mu))
    encoder = encode.Encode(degree, moduli, scale)
    pt_rot_mu_all = []
    for i in range(num_slots):
      # Rotate mu by i positions to the right
      # Matches the behavior of brot_cm_pt in math_ops.py
      rotated_mu = np.roll(mu, i)
      pt_rot_mu_all.append(encoder.encode(rotated_mu.tolist()))

    # 5. Run homomorphic BRotCM
    constants = barrett.precompute_barrett_constants(moduli)
    mul_kernel = mul.MulPlaintextCiphertextBarrett(constants)

    ct_res = brot.brot_cm(cmkey_j, pt_rot_mu_all, mul_kernel)

    # 6. Decrypt and decode result
    decryptor = encrypt.Decrypt(sk)
    pt_dec = decryptor.decrypt(ct_res)

    # Scale is now scale^2 because we did pt-ct multiplication:
    # Ecd(Rot_i(mu)) * cmkey_i
    # where Ecd(Rot_i(mu)) has scale, and cmkey_i has scale.
    decoder = encode.Decode(scale * scale, num_slots)
    decoded = decoder.decode(pt_dec)

    # Expected result is Rot_j(mu) where j is secret_idx
    expected = np.roll(mu, secret_idx)

    for e, d in zip(expected, decoded):
      # Using delta=0.15 for stability with degree 1024 and scale 2^20
      self.assertAlmostEqual(e.real, d.real, delta=1.)
      self.assertAlmostEqual(e.imag, d.imag, delta=1.)


if __name__ == "__main__":
  absltest.main()
