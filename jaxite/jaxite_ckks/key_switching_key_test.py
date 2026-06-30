"""Tests for key switching."""

import math

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import barrett
from jaxite.jaxite_ckks import basis_conversion
from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import encrypt
from jaxite.jaxite_ckks import key_gen
from jaxite.jaxite_ckks import key_switching
from jaxite.jaxite_ckks import mul
from jaxite.jaxite_ckks import ntt_cpu
from jaxite.jaxite_ckks import random
from jaxite.jaxite_ckks import types
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)

TEST_PRIMES = (
    1_073_692_673,
    1_073_643_521,
    1_073_479_681,
    1_073_430_529,
    1_071_513_601,
    1_070_727_169,
)


class KeySwitchingTest(parameterized.TestCase):

  def test_gen_key_switching_key(self):
    degree = 4
    q_limbs = [TEST_PRIMES[0]]
    p_limbs = [TEST_PRIMES[1]]
    dnum = 1

    _, source_sk = key_gen.keygen(degree, q_limbs)
    _, dest_sk = key_gen.keygen(degree, q_limbs)

    ksk = key_gen.gen_key_switching_key(
        source_key=source_sk,
        dest_key=dest_sk,
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        dnum=dnum,
    )

    all_moduli = q_limbs + p_limbs
    all_moduli_u64 = np.array(all_moduli, dtype=np.uint64).reshape(1, -1)

    temp_sk = key_gen.extend_secret_key(dest_sk, all_moduli)

    p_val = math.prod(p_limbs)

    expected_scaled_key = key_gen.compute_scaled_source_key_partition(
        source_key=source_sk,
        q_limbs=q_limbs,
        p_val=p_val,
        all_moduli_len=len(all_moduli),
        start_idx=0,
        end_idx=len(q_limbs),
    )

    a_part = ksk.a[0]
    b_part = ksk.b[0]

    ct = types.Ciphertext(
        data=jnp.stack([b_part, a_part]),
        moduli=jnp.array(all_moduli, dtype=jnp.uint32),
    )

    decryptor = encrypt.Decrypt(temp_sk)
    decrypted_pt = decryptor.decrypt(ct)

    sum_val = decrypted_pt.data.astype(np.uint64)

    # expected_scaled_key is in NTT domain, convert it to coefficient domain
    expected_scaled_key_coeffs = ntt_cpu.intt_negacyclic_poly(
        expected_scaled_key, all_moduli
    )

    diff = (
        sum_val + all_moduli_u64 - expected_scaled_key_coeffs
    ) % all_moduli_u64

    for j, q in enumerate(all_moduli):
      coeffs = diff[:, j]
      centered_coeffs = np.where(
          coeffs > q // 2, coeffs.astype(np.int64) - q, coeffs.astype(np.int64)
      )
      np.testing.assert_array_less(np.abs(centered_coeffs), 20)

  @parameterized.named_parameters(
      (
          "dnum_1",
          [TEST_PRIMES[0], TEST_PRIMES[1]],
          [TEST_PRIMES[2], TEST_PRIMES[3]],
          1,
      ),
      (
          "dnum_2",
          [TEST_PRIMES[0], TEST_PRIMES[1]],
          [TEST_PRIMES[2], TEST_PRIMES[3]],
          2,
      ),
      (
          "dnum_3",
          [TEST_PRIMES[0], TEST_PRIMES[1], TEST_PRIMES[2]],
          [TEST_PRIMES[3], TEST_PRIMES[4], TEST_PRIMES[5]],
          3,
      ),
  )
  def test_key_switch(self, q_limbs, p_limbs, dnum):
    degree = 16
    num_slots = 8
    scale = 2**20

    test_random_source = random.ZeroNoiseRandomSource()

    pk_source, sk_source = key_gen.keygen(
        degree, q_limbs, random_source=test_random_source
    )
    _, sk_dest = key_gen.keygen(
        degree, q_limbs, random_source=test_random_source
    )

    ksk = key_gen.gen_key_switching_key(
        source_key=sk_source,
        dest_key=sk_dest,
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        dnum=dnum,
        random_source=test_random_source,
    )

    all_moduli = q_limbs + p_limbs

    # 1. Setup Basis Conversion
    bc_kernel = basis_conversion.BasisConversionBarrett()
    limbs_per_part = math.ceil(len(q_limbs) / dnum)
    bc_pairs = []
    for i in range(dnum):
      start_idx = i * limbs_per_part
      end_idx = min(start_idx + limbs_per_part, len(q_limbs))
      in_indices = list(range(start_idx, end_idx))
      out_indices = [j for j in range(len(all_moduli)) if j not in in_indices]
      bc_pairs.append((in_indices, out_indices))
    bc_kernel.precompute_constants(all_moduli, bc_pairs)

    # 2. Setup Barrett Mul
    barrett_constants_pq = barrett.precompute_barrett_constants(all_moduli)
    mul_kernel = mul.MulPlaintextCiphertextBarrett(barrett_constants_pq)

    # 3. Setup KeySwitcher
    key_switcher = key_switching.KeySwitcher()
    key_switcher.precompute_constants(
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        dnum=dnum,
        r=4,
        c=4,
    )

    # 4. Encrypt some message under source_sk
    mu = np.array(
        [complex(x % 4 + 1.5, 0.0) for x in range(num_slots)], dtype=complex
    )
    encoder_q = encode.Encode(degree, q_limbs, scale)
    encryptor_q = encrypt.Encrypt(pk_source)
    plain_mu = encoder_q.encode(mu.tolist())
    ct_in = encryptor_q.encrypt(plain_mu, random_source=test_random_source)

    # 5. Run Key Switch
    ct_switched = key_switcher.key_switch(
        ct=ct_in,
        ksk=ksk,
        p_limbs=jnp.array(p_limbs, dtype=jnp.uint32),
        bc_kernel=bc_kernel,
        mul_kernel=mul_kernel,
        start_control_index=0,
    )

    # Decrypt modulo Q of ct_in for checking
    decryptor_q = encrypt.Decrypt(sk_source)
    pt_dec_in = decryptor_q.decrypt(ct_in)

    # 6. Decrypt under sk_dest modulo QP (No rescaling!)
    dest_sk_ext = key_gen.extend_secret_key(sk_dest, all_moduli)
    decryptor_qp = encrypt.Decrypt(dest_sk_ext)
    pt_dec_qp = decryptor_qp.decrypt(ct_switched)

    # 7. Verify modulo Q: should be equal to (pt_dec_in * P) % Q
    p_val = math.prod(p_limbs)
    q_slice = np.array(q_limbs, dtype=np.uint64)
    p_mod_q = np.array([p_val % int(q) for q in q_slice], dtype=np.uint64)
    expected_q = (
        pt_dec_in.data.astype(np.uint64) * p_mod_q.reshape(1, -1)
    ) % q_slice.reshape(1, -1)

    diff_q = (
        pt_dec_qp.data[:, : len(q_limbs)].astype(np.int64)
        - expected_q.astype(np.int64)
    ) % q_slice
    diff_q_centered = np.where(diff_q > q_slice // 2, diff_q - q_slice, diff_q)
    np.testing.assert_array_less(np.abs(diff_q_centered), 20)

    # 8. Verify modulo P: should be exactly 0 (modulo noise)
    p_slice = np.array(p_limbs, dtype=np.uint64)
    diff_p = pt_dec_qp.data[:, len(q_limbs) :].astype(np.int64)
    diff_p_centered = np.where(diff_p > p_slice // 2, diff_p - p_slice, diff_p)
    np.testing.assert_array_less(np.abs(diff_p_centered), 20)


if __name__ == "__main__":
  absltest.main()
