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

"""Tests for homomorphic ciphertext rotation."""

import jax.numpy as jnp
from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import encrypt
from jaxite.jaxite_ckks import key_gen
from jaxite.jaxite_ckks import random
from jaxite.jaxite_ckks import rotate

from absl.testing import absltest
from absl.testing import parameterized


class RotateTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("rotate_1_slots", 1),
      ("rotate_2_slots", 2),
      ("rotate_5_slots", 5),
      ("rotate_neg_1_slots", -1),
      ("rotate_neg_3_slots", -3),
  )
  def test_rotate(self, shift):
    degree = 1024
    num_slots = degree // 4
    q_limbs = [1073184769]
    p_limbs = [1073479681]
    scale = 2**22

    # 1. Key Generation
    test_random_source = random.ZeroNoiseRandomSource()
    pk, sk = key_gen.keygen(degree, q_limbs, random_source=test_random_source)

    # Generate rotation key (automorphism key)
    rot_key = key_gen.gen_rotation_key(
        sk=sk,
        j=shift,
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        dnum=1,
        random_source=test_random_source,
    )

    # 2. Encode and encrypt input slots
    slots = [float(i) for i in range(num_slots)]
    encoder = encode.Encode(degree, q_limbs, scale)
    pt = encoder.encode(slots)
    encryptor = encrypt.Encrypt(pk)
    ct = encryptor.encrypt(pt, random_source=test_random_source)

    # 3. Perform homomorphic rotation
    rotate_kernel = rotate.Rotate()
    rotate_kernel.precompute_constants(
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        dnum=1,
        r=32,
        c=degree // 32,
        num_rescales=1,
    )

    ct_rot = rotate_kernel.rotate(
        ct=ct,
        rot_key=rot_key,
        j=shift,
        p_limbs=jnp.array(p_limbs, dtype=jnp.uint32),
    )

    # 4. Decrypt and decode
    decryptor = encrypt.Decrypt(sk)
    pt_dec = decryptor.decrypt(ct_rot)
    decoder = encode.Decode(scale, num_slots)
    decoded = decoder.decode(pt_dec)

    # Verify shift with 0-fill
    if shift > 0:
      expected = slots[shift:] + [0.0] * shift
    elif shift < 0:
      expected = [0.0] * (-shift) + slots[:shift]
    else:
      expected = slots

    for e, d in zip(expected, decoded):
      self.assertAlmostEqual(e, d.real, delta=1.5)
      self.assertAlmostEqual(0.0, d.imag, delta=1.5)

  @parameterized.named_parameters(
      ("shift_1", 1),
      ("shift_5", 5),
      ("shift_neg_3", -3),
  )
  def test_rotate_identity(self, shift):
    degree = 1024
    num_slots = degree // 4
    n = degree // 2
    q_limbs = [1073184769]
    p_limbs = [1073479681]
    scale = 2**22

    # 1. Key Generation
    test_random_source = random.ZeroNoiseRandomSource()
    pk, sk = key_gen.keygen(degree, q_limbs, random_source=test_random_source)

    # Generate rotation key for shift j
    rot_key_j = key_gen.gen_rotation_key(
        sk=sk,
        j=shift,
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        dnum=1,
        random_source=test_random_source,
    )

    # Generate rotation key for shift n - j
    rot_key_n_minus_j = key_gen.gen_rotation_key(
        sk=sk,
        j=n - shift,
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        dnum=1,
        random_source=test_random_source,
    )

    # 2. Encode and encrypt input slots
    slots = [float(i) for i in range(num_slots)]
    encoder = encode.Encode(degree, q_limbs, scale)
    pt = encoder.encode(slots)
    encryptor = encrypt.Encrypt(pk)
    ct = encryptor.encrypt(pt, random_source=test_random_source)

    # 3. Perform homomorphic rotations
    rotate_kernel = rotate.Rotate()
    rotate_kernel.precompute_constants(
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        dnum=1,
        r=32,
        c=degree // 32,
        num_rescales=1,
    )

    # Rotate by j
    ct_rot1 = rotate_kernel.rotate(
        ct=ct,
        rot_key=rot_key_j,
        j=shift,
        p_limbs=jnp.array(p_limbs, dtype=jnp.uint32),
    )

    # Rotate by n - j
    ct_rot2 = rotate_kernel.rotate(
        ct=ct_rot1,
        rot_key=rot_key_n_minus_j,
        j=n - shift,
        p_limbs=jnp.array(p_limbs, dtype=jnp.uint32),
    )

    # 4. Decrypt and decode
    decryptor = encrypt.Decrypt(sk)
    pt_dec = decryptor.decrypt(ct_rot2)
    decoder = encode.Decode(scale, num_slots)
    decoded = decoder.decode(pt_dec)

    # Verify that we got the original slots back
    for e, d in zip(slots, decoded):
      self.assertAlmostEqual(e, d.real, delta=1.5)
      self.assertAlmostEqual(0.0, d.imag, delta=1.5)


if __name__ == "__main__":
  absltest.main()
