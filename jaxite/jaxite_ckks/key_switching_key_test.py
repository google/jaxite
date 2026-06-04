"""Tests for key switching."""

import math

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import encrypt
from jaxite.jaxite_ckks import key_gen
from jaxite.jaxite_ckks import ntt_cpu
from jaxite.jaxite_ckks import types
import numpy as np

from absl.testing import absltest

jax.config.update("jax_enable_x64", True)

TEST_PRIMES = (
    1_073_692_673,
    1_073_643_521,
    1_073_479_681,
    1_073_430_529,
)


class KeySwitchingTest(absltest.TestCase):

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


if __name__ == "__main__":
  absltest.main()
