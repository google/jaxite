# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for blind rotation kernels."""

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import barrett
from jaxite.jaxite_ckks import basis_conversion
from jaxite.jaxite_ckks import blind_rotate
from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import encrypt
from jaxite.jaxite_ckks import key_gen
from jaxite.jaxite_ckks import mul
from jaxite.jaxite_ckks import random
from jaxite.jaxite_ckks import rescale
from jaxite.jaxite_ckks import types
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)


def _negacyclic_roll(arr: np.ndarray, shift: int) -> np.ndarray:
  res = np.roll(arr, shift)
  if shift > 0:
    res[:shift] = -res[:shift]
  return res


class BlindRotateTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("standard", 4, 2),
      ("secret_idx_0", 4, 0),
      ("secret_idx_N_minus_1", 4, 3),
      ("secret_idx_greater_than_N_div_2", 8, 5),
      ("dense_encoding", 1024, 512),
  )
  def test_blind_rotate_cm(self, num_slots, secret_idx):
    degree = max(1024, 2 * num_slots)
    moduli = [1073184769, 1073479681]
    scale = 2**22

    # 1. Generate keys with zero noise for exact algebraic correctness in test
    test_random_source = random.ZeroNoiseRandomSource()
    pk, sk = key_gen.keygen(degree, moduli, random_source=test_random_source)

    encoder = encode.Encode(degree, moduli, scale)
    encryptor = encrypt.Encrypt(pk)

    # 2. Generate Column Keys for secret index j
    # cmkey_j[i] encrypts 1 if i == secret_idx else 0
    all_zeroes = [complex(0)] * num_slots
    all_ones = [complex(1)] * num_slots

    plain_0 = encoder.encode(all_zeroes)
    plain_1 = encoder.encode(all_ones)

    cmkey_j = []
    for i in range(num_slots):
      if i == secret_idx:
        cmkey_j.append(
            encryptor.encrypt(plain_1, random_source=test_random_source)
        )
      else:
        cmkey_j.append(
            encryptor.encrypt(plain_0, random_source=test_random_source)
        )

    # 3. Define input message mu (slots)
    mu = np.array(
        [complex(x % 4 + 1, x % 4 + 2) for x in range(num_slots)], dtype=complex
    )

    # 4. Rotate and encode mu for all i
    # pt_rot_mu_all[i] = Encode(Rot_i(mu))
    pt_rot_mu_all = []
    for i in range(num_slots):
      # Rotate mu by i positions to the right with sign-flip (negacyclic)
      rotated_mu = _negacyclic_roll(mu, i)
      pt_rot_mu_all.append(encoder.encode(rotated_mu.tolist()))

    # 5. Run homomorphic BRotCM
    rescale_kernel = rescale.Rescale()
    rescale_kernel.precompute_constants(
        moduli, num_rescales=1, r=32, c=degree // 32
    )

    constants = barrett.precompute_barrett_constants(moduli)
    mul_kernel = mul.MulPlaintextCiphertextBarrett(constants)

    ct_res = blind_rotate.brot_cm(
        cmkey_j, pt_rot_mu_all, mul_kernel, rescale_kernel
    )

    # 6. Decrypt and decode result
    sk_q = types.SecretKey(
        data=sk.data[:, :1],
        moduli=np.array([moduli[0]], dtype=np.uint32),
    )
    decryptor = encrypt.Decrypt(sk_q)
    pt_dec = decryptor.decrypt(ct_res)

    # Scale is now scale^2 / P because we rescaled:
    scale_rescaled = (scale * scale) / moduli[1]
    decoder = encode.Decode(scale_rescaled, num_slots)
    decoded = decoder.decode(pt_dec)

    # Expected result is Rot_j(mu) where j is secret_idx
    expected = _negacyclic_roll(mu, secret_idx)
    for e, d in zip(expected, decoded):
      # Using delta=1.0 for stability with degree 1024 and scale 2^22 after
      # rescaling
      self.assertAlmostEqual(e.real, d.real, delta=1.0)
      self.assertAlmostEqual(e.imag, d.imag, delta=1.0)

  @parameterized.named_parameters(
      ("secret_idx_0", 4, 0),
      ("secret_idx_1", 4, 1),
      ("secret_idx_2", 4, 2),
      ("secret_idx_3", 4, 3),
      ("dense_8_slots", 8, 5),
      ("dense_16_slots", 16, 9),
  )
  def test_brot_mux(self, num_slots, secret_idx):
    degree = max(1024, 2 * num_slots)
    r = 32
    q_limbs = [1073184769]
    p_limbs = [1073479681]
    all_moduli = q_limbs + p_limbs
    scale = 2**22

    # 1. Generate keys
    test_random_source = random.ZeroNoiseRandomSource()
    pk_q, sk_q = key_gen.keygen(
        degree, q_limbs, random_source=test_random_source
    )
    # 2. Generate Mux Rotation Key for secret index
    # We choose the secret index bits corresponding to secret_idx
    num_bits = int(np.log2(num_slots))
    secret_bits = [int((secret_idx >> k) & 1) for k in range(num_bits)]
    mux_key = key_gen.gen_mux_rotation_key(
        sk=sk_q,
        secret_bits=secret_bits,
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        random_source=test_random_source,
    )

    # 3. Setup input ciphertext ct_in
    mu = np.array(
        [complex(x % 4 + 1, x % 4 + 2) for x in range(num_slots)], dtype=complex
    )

    bc_kernel = basis_conversion.BasisConversionBarrett()
    bc_kernel.precompute_constants(all_moduli, [([0], [1])])
    mul_constants = barrett.precompute_barrett_constants(all_moduli)
    mul_kernel = mul.MulPlaintextCiphertextBarrett(mul_constants)

    rescale_kernel = rescale.Rescale()
    rescale_kernel.precompute_constants(
        all_moduli, num_rescales=1, r=r, c=degree // r
    )

    encoder_q = encode.Encode(degree, q_limbs, scale)
    encryptor_q = encrypt.Encrypt(pk_q)

    plain_mu = encoder_q.encode(mu.tolist())
    ct_in = encryptor_q.encrypt(plain_mu, random_source=test_random_source)

    # 4. Run homomorphic BRotMux
    ct_res = blind_rotate.brot_mux(
        ct_in=ct_in,
        mux_key=mux_key,
        p_limbs=jnp.array(p_limbs, dtype=jnp.uint32),
        bc_kernel=bc_kernel,
        control_index=0,
        mul_kernel=mul_kernel,
        rescale_kernel=rescale_kernel,
    )

    # 5. Decrypt and verify result
    decryptor_q = encrypt.Decrypt(sk_q)
    pt_dec = decryptor_q.decrypt(ct_res)

    decoder = encode.Decode(scale, num_slots)
    decoded = decoder.decode(pt_dec)

    full_slots = degree // 2
    mu_full = np.zeros(full_slots, dtype=complex)
    mu_full[:num_slots] = mu
    expected_full = _negacyclic_roll(mu_full, secret_idx)
    expected = expected_full[:num_slots]

    for e, d in zip(expected, decoded):
      self.assertAlmostEqual(e.real, d.real, delta=1.5)
      self.assertAlmostEqual(e.imag, d.imag, delta=1.5)


if __name__ == "__main__":
  absltest.main()
