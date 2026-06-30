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

import hypothesis
from hypothesis import strategies as st
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import blind_rotate
from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import encrypt
from jaxite.jaxite_ckks import key_gen
from jaxite.jaxite_ckks import ntt
from jaxite.jaxite_ckks import random
from jaxite.jaxite_ckks import types
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)


def _cyclic_roll(arr: np.ndarray, shift: int) -> np.ndarray:
  return np.roll(arr, shift)


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
      rotated_mu = _cyclic_roll(mu, i)
      pt_rot_mu_all.append(encoder.encode(rotated_mu.tolist()))

    # 5. Run homomorphic BRotCM
    brot_kernel = blind_rotate.BlindRotation()
    brot_kernel.precompute_constants(
        q_limbs=moduli[:1],
        p_limbs=moduli[1:],
        dnum=1,
        r=32,
        c=degree // 32,
        num_rescales=1,
    )

    ct_res = brot_kernel.brot_cm(cmkey_j, pt_rot_mu_all)

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
    expected = _cyclic_roll(mu, secret_idx)
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
    encoder_q = encode.Encode(degree, q_limbs, scale)
    encryptor_q = encrypt.Encrypt(pk_q)
    plain_mu = encoder_q.encode(mu.tolist())
    ct_in = encryptor_q.encrypt(plain_mu, random_source=test_random_source)

    # 4. Run homomorphic BRotMux
    brot_kernel = blind_rotate.BlindRotation()
    brot_kernel.precompute_constants(
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        dnum=1,
        r=r,
        c=degree // r,
        num_rescales=1,
    )

    ct_res = brot_kernel.brot_mux(
        ct_in=ct_in,
        mux_key=mux_key,
        p_limbs=jnp.array(p_limbs, dtype=jnp.uint32),
        control_index=0,
    )

    # 5. Decrypt and verify result
    decryptor_q = encrypt.Decrypt(sk_q)
    pt_dec = decryptor_q.decrypt(ct_res)
    decoder = encode.Decode(scale, num_slots)
    decoded = decoder.decode(pt_dec)

    full_slots = degree // 2
    mu_full = np.zeros(full_slots, dtype=complex)
    mu_full[:num_slots] = mu
    expected_full = _cyclic_roll(mu_full, secret_idx)
    expected = expected_full[:num_slots]
    for e, d in zip(expected, decoded):
      self.assertAlmostEqual(e.real, d.real, delta=1.5)
      self.assertAlmostEqual(e.imag, d.imag, delta=1.5)

  @parameterized.named_parameters(
      ("slots_4_r_1", 4, 1),
      ("slots_4_r_2", 4, 2),
      ("slots_8_r_3", 8, 3),
  )
  def test_brot_mux_cyclic_identity(self, num_slots, r):
    degree = 2 * num_slots
    q_limbs = [1073184769]
    p_limbs = [1073479681]
    scale = 2**22

    # 1. Generate keys
    test_random_source = random.ZeroNoiseRandomSource()
    pk_q, sk_q = key_gen.keygen(
        degree, q_limbs, random_source=test_random_source
    )

    # 2. Generate Mux keys for r and len - r
    num_bits = int(np.log2(num_slots))

    secret_bits_r = [int((r >> k) & 1) for k in range(num_bits)]
    mux_key_r = key_gen.gen_mux_rotation_key(
        sk=sk_q,
        secret_bits=secret_bits_r,
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        random_source=test_random_source,
    )

    len_minus_r = num_slots - r
    secret_bits_len_minus_r = [
        int((len_minus_r >> k) & 1) for k in range(num_bits)
    ]
    mux_key_len_minus_r = key_gen.gen_mux_rotation_key(
        sk=sk_q,
        secret_bits=secret_bits_len_minus_r,
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        random_source=test_random_source,
    )

    # 3. Setup input ciphertext ct_in
    mu = np.array(
        [complex(x % 4 + 1, x % 4 + 2) for x in range(num_slots)], dtype=complex
    )
    r_factor = 4
    c_factor = degree // r_factor
    encoder_q = encode.Encode(degree, q_limbs, scale)
    encryptor_q = encrypt.Encrypt(pk_q)
    plain_mu = encoder_q.encode(mu.tolist())
    ct_in = encryptor_q.encrypt(plain_mu, random_source=test_random_source)

    # 4. Perform first rotation by r
    brot_kernel = blind_rotate.BlindRotation()
    brot_kernel.precompute_constants(
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        dnum=1,
        r=r_factor,
        c=c_factor,
        num_rescales=1,
    )

    ct_r = brot_kernel.brot_mux(
        ct_in=ct_in,
        mux_key=mux_key_r,
        p_limbs=jnp.array(p_limbs, dtype=jnp.uint32),
        control_index=0,
    )

    # 5. Perform second rotation by len - r
    ct_final = brot_kernel.brot_mux(
        ct_in=ct_r,
        mux_key=mux_key_len_minus_r,
        p_limbs=jnp.array(p_limbs, dtype=jnp.uint32),
        control_index=0,
    )

    # 6. Decrypt and verify
    decryptor_q = encrypt.Decrypt(sk_q)
    pt_dec = decryptor_q.decrypt(ct_final)
    decoder = encode.Decode(scale, num_slots)
    decoded = decoder.decode(pt_dec)

    expected = mu
    for e, d in zip(expected, decoded):
      self.assertAlmostEqual(e.real, d.real, delta=1.5)
      self.assertAlmostEqual(e.imag, d.imag, delta=1.5)


class BlindRotationHypothesisTest(absltest.TestCase):

  Q_LIMBS = [1073184769]
  P_LIMBS = [1073479681]
  ALL_MODULI = Q_LIMBS + P_LIMBS
  DEGREE = 16
  NUM_SLOTS = 8
  SCALE = 2**22
  THETA = 2
  SECRET_IDX = 3

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.test_random_source = random.ZeroNoiseRandomSource()
    cls.pk_pq, cls.sk_pq = key_gen.keygen(
        cls.DEGREE, cls.ALL_MODULI, random_source=cls.test_random_source
    )
    cls.pk_q = types.PublicKey(
        cls.pk_pq.data[:, :, :1], np.array(cls.Q_LIMBS, dtype=np.uint64)
    )
    cls.sk_q = types.SecretKey(
        cls.sk_pq.data[:, :1], np.array(cls.Q_LIMBS, dtype=np.uint64)
    )

    # 1. CM key
    encoder_pq = encode.Encode(cls.DEGREE, cls.ALL_MODULI, cls.SCALE)
    encryptor_pq = encrypt.Encrypt(cls.pk_pq)
    plain_0 = encoder_pq.encode([complex(0)] * cls.NUM_SLOTS)
    plain_1 = encoder_pq.encode([complex(1)] * cls.NUM_SLOTS)
    cls.cmkey_j = []
    for i in range(cls.NUM_SLOTS):
      if i == cls.SECRET_IDX:
        cls.cmkey_j.append(
            encryptor_pq.encrypt(plain_1, random_source=cls.test_random_source)
        )
      else:
        cls.cmkey_j.append(
            encryptor_pq.encrypt(plain_0, random_source=cls.test_random_source)
        )

    # 2. Mux key
    num_bits = int(np.log2(cls.NUM_SLOTS))
    secret_bits = [int((cls.SECRET_IDX >> k) & 1) for k in range(num_bits)]
    cls.mux_key = key_gen.gen_mux_rotation_key(
        sk=cls.sk_q,
        secret_bits=secret_bits,
        q_limbs=cls.Q_LIMBS,
        p_limbs=cls.P_LIMBS,
        random_source=cls.test_random_source,
    )

    # Setup BlindRotation kernel
    cls.brot_kernel = blind_rotate.BlindRotation()
    cls.brot_kernel.precompute_constants(
        q_limbs=cls.Q_LIMBS,
        p_limbs=cls.P_LIMBS,
        dnum=1,
        r=4,
        c=4,
        num_rescales=1,
    )

    # Setup joint NTT for encoding
    cls.ntt_pq = ntt.NTTBarrett()
    cls.ntt_pq.precompute_constants(cls.ALL_MODULI, r=4, c=4)

    cls.encoder_pq = encode.Encode(cls.DEGREE, cls.ALL_MODULI, cls.SCALE)
    cls.encoder_q = encode.Encode(cls.DEGREE, cls.Q_LIMBS, cls.SCALE)
    cls.encryptor_q = encrypt.Encrypt(cls.pk_q)
    cls.decryptor_q = encrypt.Decrypt(cls.sk_q)
    cls.decoder = encode.Decode(cls.SCALE, cls.NUM_SLOTS)
    scale_rescaled = (cls.SCALE * cls.SCALE) / cls.P_LIMBS[0]
    cls.decoder_cm = encode.Decode(scale_rescaled, cls.NUM_SLOTS)

  @hypothesis.settings(max_examples=10, deadline=None)
  @hypothesis.given(
      slots=st.lists(
          st.complex_numbers(min_magnitude=0, max_magnitude=5),
          min_size=8,
          max_size=8,
      )
  )
  def test_brot_cm_hypothesis(self, slots):
    mu = np.array(slots, dtype=complex)
    pt_rot_mu_all = []
    for i in range(self.NUM_SLOTS):
      rotated_mu = _cyclic_roll(mu, i)
      pt_rot_mu_all.append(self.encoder_pq.encode(rotated_mu.tolist()))
    ct_res = self.brot_kernel.brot_cm(self.cmkey_j, pt_rot_mu_all)

    pt_dec = self.decryptor_q.decrypt(ct_res)
    decoded = self.decoder_cm.decode(pt_dec)

    mu_full = np.zeros(self.DEGREE // 2, dtype=complex)
    mu_full[: self.NUM_SLOTS] = mu
    expected_full = _cyclic_roll(mu_full, self.SECRET_IDX)
    expected = expected_full[: self.NUM_SLOTS]

    for e, d in zip(expected, decoded):
      self.assertAlmostEqual(e.real, d.real, delta=1.5)
      self.assertAlmostEqual(e.imag, d.imag, delta=1.5)

  @hypothesis.settings(max_examples=10, deadline=None)
  @hypothesis.given(
      slots=st.lists(
          st.complex_numbers(min_magnitude=0, max_magnitude=5),
          min_size=8,
          max_size=8,
      )
  )
  def test_brot_mux_hypothesis(self, slots):
    mu = np.array(slots, dtype=complex)
    pt_mu = self.encoder_q.encode(mu.tolist())
    ct_in = self.encryptor_q.encrypt(
        pt_mu, random_source=self.test_random_source
    )

    ct_res = self.brot_kernel.brot_mux(
        ct_in=ct_in,
        mux_key=self.mux_key,
        p_limbs=jnp.array(self.P_LIMBS, dtype=jnp.uint32),
        control_index=0,
    )

    pt_dec = self.decryptor_q.decrypt(ct_res)
    decoded = self.decoder.decode(pt_dec)

    mu_full = np.zeros(self.DEGREE // 2, dtype=complex)
    mu_full[: self.NUM_SLOTS] = mu
    expected_full = _cyclic_roll(mu_full, self.SECRET_IDX)
    expected = expected_full[: self.NUM_SLOTS]

    for e, d in zip(expected, decoded):
      self.assertAlmostEqual(e.real, d.real, delta=1.5)
      self.assertAlmostEqual(e.imag, d.imag, delta=1.5)


if __name__ == "__main__":
  absltest.main()
