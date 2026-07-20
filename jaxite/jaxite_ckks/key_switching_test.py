"""Tests for key switching kernels."""

import math
import hypothesis
from hypothesis import strategies as st
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
from jaxite.jaxite_ckks import rescale
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
)


def _gen_ksk_and_key1(
    source_key: types.SecretKey,
    dest_key: types.SecretKey,
    q_limbs: list[int],
    p_limbs: list[int],
    random_source: random.RandomSource | None = None,
) -> tuple[types.EvaluationKeys, types.Ciphertext]:
  """Generates standard KSK and key1 for testing."""
  random_source = random_source or random.SecureRandomSource()
  degree = source_key.data.shape[0]
  all_moduli = q_limbs + p_limbs
  dest_key_ext = key_gen.extend_secret_key(dest_key, all_moduli)

  # 1. Generate standard key switching key (source to dest)
  ksk = key_gen.gen_key_switching_key(
      source_key=source_key,
      dest_key=dest_key,
      q_limbs=q_limbs,
      p_limbs=p_limbs,
      dnum=1,
      random_source=random_source,
      dest_key_ext=dest_key_ext,
  )

  # 2. Generate key1: encrypt P under dest_key (extended to all_moduli)
  p_val = math.prod(p_limbs)
  scaled_val = p_val % np.array(all_moduli, dtype=np.uint64).reshape(1, -1)
  target1 = np.ones((degree, len(all_moduli)), dtype=np.uint64) * scaled_val

  a_coeffs = random_source.gen_uniform_poly(degree, all_moduli)
  e_coeffs = random_source.gen_gaussian_poly(degree, all_moduli)
  a_slots = ntt_cpu.ntt_negacyclic_poly(a_coeffs, all_moduli)
  e_slots = ntt_cpu.ntt_negacyclic_poly(e_coeffs, all_moduli)

  prod = (a_slots * dest_key_ext.data) % np.array(
      all_moduli, dtype=np.uint64
  ).reshape(1, -1)
  b_slots = (
      e_slots
      + target1
      + np.array(all_moduli, dtype=np.uint64).reshape(1, -1)
      - prod
  ) % np.array(all_moduli, dtype=np.uint64).reshape(1, -1)

  key1_data = np.stack([b_slots, a_slots])
  key1 = types.Ciphertext(
      data=jnp.array(key1_data), moduli=jnp.array(all_moduli, dtype=jnp.uint64)
  )

  return ksk, key1


# Strategy to generate (degree, mu_list, r, c) for hypothesis tests.
@st.composite
def degree_mu_r_c_strategy(draw):
  k = draw(st.sampled_from([4, 5, 6]))
  degree = 2**k
  mu_list = draw(
      st.lists(
          st.complex_numbers(
              min_magnitude=0.0,
              max_magnitude=2.0,
              allow_nan=False,
              allow_infinity=False,
          ),
          min_size=degree // 2,
          max_size=degree // 2,
      )
  )
  a = draw(st.integers(min_value=1, max_value=k - 1))
  r = 2**a
  c = 2 ** (k - a)
  return degree, mu_list, r, c


class KeySwitchingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("dnum_1", 1),
      ("dnum_2", 2),
  )
  def test_key_switcher_rns(self, dnum):
    degree = 16
    q_limbs = [TEST_PRIMES[0], TEST_PRIMES[1]]
    p_limbs = [TEST_PRIMES[2]]
    all_moduli = q_limbs + p_limbs
    scale = 2**20

    test_random_source = random.ZeroNoiseRandomSource()
    pk_src, sk_src = key_gen.keygen(
        degree, q_limbs, random_source=test_random_source
    )
    _, sk_dst = key_gen.keygen(
        degree, q_limbs, random_source=test_random_source
    )

    ksk = key_gen.gen_key_switching_key(
        source_key=sk_src,
        dest_key=sk_dst,
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        dnum=dnum,
        random_source=test_random_source,
    )

    bc_kernel = basis_conversion.BasisConversionBarrett()
    ks_control_indices = mul.Mul.compute_control_indices(q_limbs, p_limbs, dnum)
    bc_kernel.precompute_constants(all_moduli, ks_control_indices)

    barrett_constants_pq = barrett.precompute_barrett_constants(all_moduli)
    mul_kernel = mul.MulPlaintextCiphertextBarrett(barrett_constants_pq)

    # 1. Encrypt message under sk_src
    mu = np.array(
        [complex(x % 4 + 1, x % 4 + 2) for x in range(degree // 2)],
        dtype=complex,
    )
    encoder = encode.Encode(degree, q_limbs, scale)
    encryptor_src = encrypt.Encrypt(pk_src)
    ct_in = encryptor_src.encrypt(
        encoder.encode(mu.tolist()), random_source=test_random_source
    )

    # 2. Key switch to sk_dst
    switcher = key_switching.KeySwitcher()
    switcher.precompute_constants(
        q_limbs,
        p_limbs,
        dnum,
        r=4,
        c=4,
        bc_kernel=bc_kernel,
        mul_kernel=mul_kernel,
    )

    ct_switched_qp = switcher.key_switch(
        ct=ct_in,
        ksk=ksk,
        start_control_index=1,
    )

    # 3. Rescale ct_switched_qp to ct_switched_q
    rescale_kernel = rescale.Rescale()
    rescale_kernel.precompute_constants(
        all_moduli, num_rescales=len(p_limbs), r=4, c=4
    )
    rescale_kernel.rescale(ct_switched_qp)

    # 4. Decrypt under sk_dst and decode
    decryptor_dst = encrypt.Decrypt(sk_dst)
    pt_dec = decryptor_dst.decrypt(ct_switched_qp)

    decoder = encode.Decode(scale, degree // 2)
    decoded = decoder.decode(pt_dec)

    for e, d in zip(mu, decoded):
      self.assertAlmostEqual(e.real, d.real, delta=1e-1)
      self.assertAlmostEqual(e.imag, d.imag, delta=1e-1)

  def test_key_switcher_bat(self):
    degree = 16
    q_limbs = [TEST_PRIMES[0], TEST_PRIMES[1]]
    p_limbs = [TEST_PRIMES[2]]
    all_moduli = q_limbs + p_limbs
    scale = 2**20

    test_random_source = random.ZeroNoiseRandomSource()
    pk_src, sk_src = key_gen.keygen(
        degree, q_limbs, random_source=test_random_source
    )
    _, sk_dst = key_gen.keygen(
        degree, q_limbs, random_source=test_random_source
    )

    # Generate BAT key matrix
    ksk, key1 = _gen_ksk_and_key1(
        source_key=sk_src,
        dest_key=sk_dst,
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        random_source=test_random_source,
    )

    bc_kernel = basis_conversion.BasisConversionBarrett()
    # BC from Q (2 limbs) to P (1 limb)
    bc_kernel.precompute_constants(all_moduli, [([0, 1], [2])])

    barrett_constants_pq = barrett.precompute_barrett_constants(all_moduli)
    mul_kernel = mul.MulPlaintextCiphertextBarrett(barrett_constants_pq)

    rescale_kernel = rescale.Rescale()
    rescale_kernel.precompute_constants(all_moduli, num_rescales=1, r=4, c=4)

    # 1. Encrypt message under sk_src
    mu = np.array(
        [complex(x % 4 + 1, x % 4 + 2) for x in range(degree // 2)],
        dtype=complex,
    )
    encoder = encode.Encode(degree, q_limbs, scale)
    encryptor_src = encrypt.Encrypt(pk_src)
    ct_in = encryptor_src.encrypt(
        encoder.encode(mu.tolist()), random_source=test_random_source
    )

    # 2. Key switch using BATKeySwitcher
    r, c = 4, 4
    switcher = key_switching.BATKeySwitcher()
    switcher.precompute_constants(
        q_limbs,
        p_limbs,
        r=r,
        c=c,
        bc_kernel=bc_kernel,
        mul_kernel=mul_kernel,
        rescale_kernel=rescale_kernel,
    )
    key_matrix_bat = switcher.transform_key_to_bat(ksk, key1)
    ct_switched_q = switcher.key_switch(
        ct=ct_in,
        key_matrix_bat=key_matrix_bat,
        control_index=0,
    )

    # 3. Decrypt under sk_dst and decode
    decryptor_dst = encrypt.Decrypt(sk_dst)
    pt_dec = decryptor_dst.decrypt(ct_switched_q)

    decoder = encode.Decode(scale, degree // 2)
    decoded = decoder.decode(pt_dec)

    for e, d in zip(mu, decoded):
      self.assertAlmostEqual(e.real, d.real, delta=1e-1)
      self.assertAlmostEqual(e.imag, d.imag, delta=1e-1)


class KeySwitcherHypothesisTest(parameterized.TestCase):

  @hypothesis.settings(max_examples=50, deadline=None)
  @hypothesis.given(
      degree_mu_r_c_strategy(),
      st.sampled_from([1, 2]),
      st.sampled_from([2**15, 2**20, 2**25, 2**30]),
  )
  def test_key_switcher_rns_hypothesis(self, deg_mu_r_c, dnum, scale):
    degree, mu_list, r, c = deg_mu_r_c
    q_limbs = [TEST_PRIMES[0], TEST_PRIMES[1]]
    p_limbs = [TEST_PRIMES[2]]
    all_moduli = q_limbs + p_limbs

    test_random_source = random.ZeroNoiseRandomSource()
    pk_src, sk_src = key_gen.keygen(
        degree, q_limbs, random_source=test_random_source
    )
    _, sk_dst = key_gen.keygen(
        degree, q_limbs, random_source=test_random_source
    )

    ksk = key_gen.gen_key_switching_key(
        source_key=sk_src,
        dest_key=sk_dst,
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        dnum=dnum,
        random_source=test_random_source,
    )

    bc_kernel = basis_conversion.BasisConversionBarrett()
    ks_control_indices = mul.Mul.compute_control_indices(q_limbs, p_limbs, dnum)
    bc_kernel.precompute_constants(all_moduli, ks_control_indices)

    barrett_constants_pq = barrett.precompute_barrett_constants(all_moduli)
    mul_kernel = mul.MulPlaintextCiphertextBarrett(barrett_constants_pq)

    # 1. Encrypt message under sk_src
    mu = np.array(mu_list, dtype=complex)
    encoder = encode.Encode(degree, q_limbs, scale)
    encryptor_src = encrypt.Encrypt(pk_src)
    ct_in = encryptor_src.encrypt(
        encoder.encode(mu.tolist()), random_source=test_random_source
    )

    # 2. Key switch to sk_dst
    switcher = key_switching.KeySwitcher()
    switcher.precompute_constants(
        q_limbs,
        p_limbs,
        dnum,
        r=r,
        c=c,
        bc_kernel=bc_kernel,
        mul_kernel=mul_kernel,
    )

    ct_switched_qp = switcher.key_switch(
        ct=ct_in,
        ksk=ksk,
        start_control_index=1,
    )

    # 3. Rescale ct_switched_qp to ct_switched_q
    rescale_kernel = rescale.Rescale()
    rescale_kernel.precompute_constants(
        all_moduli, num_rescales=len(p_limbs), r=r, c=c
    )
    rescale_kernel.rescale(ct_switched_qp)

    # 4. Decrypt under sk_dst and decode
    decryptor_dst = encrypt.Decrypt(sk_dst)
    pt_dec = decryptor_dst.decrypt(ct_switched_qp)

    decoder = encode.Decode(scale, degree // 2)
    decoded = decoder.decode(pt_dec)

    for e, d in zip(mu, decoded):
      self.assertAlmostEqual(e.real, d.real, delta=1e-1)
      self.assertAlmostEqual(e.imag, d.imag, delta=1e-1)

  @hypothesis.settings(max_examples=50, deadline=None)
  @hypothesis.given(
      degree_mu_r_c_strategy(),
      st.sampled_from([2**15, 2**20, 2**25, 2**30]),
  )
  def test_key_switcher_bat_hypothesis(self, deg_mu_r_c, scale):
    degree, mu_list, r, c = deg_mu_r_c
    q_limbs = [TEST_PRIMES[0], TEST_PRIMES[1]]
    p_limbs = [TEST_PRIMES[2]]
    all_moduli = q_limbs + p_limbs

    test_random_source = random.ZeroNoiseRandomSource()
    pk_src, sk_src = key_gen.keygen(
        degree, q_limbs, random_source=test_random_source
    )
    _, sk_dst = key_gen.keygen(
        degree, q_limbs, random_source=test_random_source
    )

    # Generate BAT key matrix
    ksk, key1 = _gen_ksk_and_key1(
        source_key=sk_src,
        dest_key=sk_dst,
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        random_source=test_random_source,
    )

    bc_kernel = basis_conversion.BasisConversionBarrett()
    # BC from Q (2 limbs) to P (1 limb)
    bc_kernel.precompute_constants(all_moduli, [([0, 1], [2])])

    barrett_constants_pq = barrett.precompute_barrett_constants(all_moduli)
    mul_kernel = mul.MulPlaintextCiphertextBarrett(barrett_constants_pq)

    rescale_kernel = rescale.Rescale()
    rescale_kernel.precompute_constants(all_moduli, num_rescales=1, r=r, c=c)

    # 1. Encrypt message under sk_src
    mu = np.array(mu_list, dtype=complex)
    encoder = encode.Encode(degree, q_limbs, scale)
    encryptor_src = encrypt.Encrypt(pk_src)
    ct_in = encryptor_src.encrypt(
        encoder.encode(mu.tolist()), random_source=test_random_source
    )

    # 2. Key switch using BATKeySwitcher
    switcher = key_switching.BATKeySwitcher()
    switcher.precompute_constants(
        q_limbs,
        p_limbs,
        r=r,
        c=c,
        bc_kernel=bc_kernel,
        mul_kernel=mul_kernel,
        rescale_kernel=rescale_kernel,
    )
    key_matrix_bat = switcher.transform_key_to_bat(ksk, key1)
    ct_switched_q = switcher.key_switch(
        ct=ct_in,
        key_matrix_bat=key_matrix_bat,
        control_index=0,
    )

    # 3. Decrypt under sk_dst and decode
    decryptor_dst = encrypt.Decrypt(sk_dst)
    pt_dec = decryptor_dst.decrypt(ct_switched_q)

    decoder = encode.Decode(scale, degree // 2)
    decoded = decoder.decode(pt_dec)

    for e, d in zip(mu, decoded):
      self.assertAlmostEqual(e.real, d.real, delta=1e-1)
      self.assertAlmostEqual(e.imag, d.imag, delta=1e-1)


if __name__ == "__main__":
  absltest.main()
