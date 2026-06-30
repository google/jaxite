"""Tests for conjugate automorphism."""

import math

import hypothesis
from hypothesis import strategies as st
import jax.numpy as jnp
from jaxite.jaxite_ckks import barrett
from jaxite.jaxite_ckks import basis_conversion
from jaxite.jaxite_ckks import conjugate
from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import encrypt
from jaxite.jaxite_ckks import key_gen
from jaxite.jaxite_ckks import mul
from jaxite.jaxite_ckks import ntt_cpu
from jaxite.jaxite_ckks import random
from jaxite.jaxite_ckks import rescale
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized


class ConjugateTest(parameterized.TestCase):

  def _run_conjugate_test(
      self, degree, num_slots, q_limbs, p_limbs, dnum, use_noise, mu=None
  ):
    all_moduli = q_limbs + p_limbs
    scale = 2**20

    if use_noise:
      test_random_source = random.TestRandomSource(123)
    else:
      test_random_source = random.ZeroNoiseRandomSource()

    pk_q, sk_q = key_gen.keygen(
        degree, q_limbs, random_source=test_random_source
    )

    conj_key = key_gen.gen_conjugate_key(
        sk=sk_q,
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        dnum=dnum,
        random_source=test_random_source,
    )

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

    barrett_constants_pq = barrett.precompute_barrett_constants(all_moduli)
    mul_kernel = mul.MulPlaintextCiphertextBarrett(barrett_constants_pq)
    rescale_kernel = rescale.Rescale()
    rescale_kernel.precompute_constants(
        moduli=all_moduli,
        num_rescales=len(p_limbs),
        r=4,
        c=4,
    )

    conjugate_kernel = conjugate.Conjugation()
    conjugate_kernel.precompute_constants(
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        dnum=dnum,
        r=4,
        c=4,
    )

    if mu is None:
      mu = np.array(
          [complex(x % 4 + 1.5, x % 4 - 2.5) for x in range(num_slots)],
          dtype=complex,
      )
    encoder_q = encode.Encode(degree, q_limbs, scale)
    encryptor_q = encrypt.Encrypt(pk_q)

    plain_mu = encoder_q.encode(mu.tolist())
    ct_in = encryptor_q.encrypt(plain_mu, random_source=test_random_source)

    ct_res = conjugate_kernel.conjugate(
        ct=ct_in,
        conj_key=conj_key,
        p_limbs=jnp.array(p_limbs, dtype=jnp.uint32),
        bc_kernel=bc_kernel,
        mul_kernel=mul_kernel,
        rescale_kernel=rescale_kernel,
        start_control_index=0,
    )

    decryptor_q = encrypt.Decrypt(sk_q)
    pt_dec = decryptor_q.decrypt(ct_res)

    decoder = encode.Decode(scale, num_slots)
    decoded = decoder.decode(pt_dec)

    expected = np.conj(mu)

    delta = 1.0 if use_noise else 0.5
    try:
      for e, d in zip(expected, decoded):
        self.assertAlmostEqual(e.real, d.real, delta=delta)
        self.assertAlmostEqual(e.imag, d.imag, delta=delta)
    except AssertionError as err:
      print("EXPECTED:", expected.tolist())
      print("DECODED :", decoded)
      raise err

  @parameterized.named_parameters(
      ("one_q_limb_one_p_limb_dnum_1", [1073184769], [1073479681], 1, False),
      (
          "two_q_limbs_one_p_limb_dnum_1",
          [1073742113, 1073740609],
          [1073741953],
          1,
          False,
      ),
      (
          "two_q_limbs_one_p_limb_dnum_2",
          [1073742113, 1073740609],
          [1073741953],
          2,
          False,
      ),
      (
          "one_q_limb_two_p_limbs_dnum_1_with_noise",
          [1073184769],
          [1073479681, 1073741953],
          1,
          True,
      ),
      (
          "two_q_limbs_three_p_limbs_dnum_1_with_noise",
          [1073742113, 1073740609],
          [1073741953, 1073741441, 1073741857],
          1,
          True,
      ),
      (
          "two_q_limbs_three_p_limbs_dnum_2_with_noise",
          [1073742113, 1073740609],
          [1073741953, 1073741441, 1073741857],
          2,
          True,
      ),
  )
  def test_conjugate_pipeline(self, q_limbs, p_limbs, dnum, use_noise):
    self._run_conjugate_test(
        degree=16,
        num_slots=8,
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        dnum=dnum,
        use_noise=use_noise,
    )

  @hypothesis.settings(max_examples=10, deadline=None)
  @hypothesis.given(
      slots=st.lists(
          st.complex_numbers(min_magnitude=0, max_magnitude=5),
          min_size=8,
          max_size=8,
      )
  )
  def test_conjugate_hypothesis(self, slots):
    # Use P >> Q to prevent noise flooding in hypothesis tests with noise
    self._run_conjugate_test(
        degree=16,
        num_slots=8,
        q_limbs=[1073184769],
        p_limbs=[1073479681, 1073741953],
        dnum=1,
        use_noise=True,
        mu=np.array(slots, dtype=complex),
    )

  def test_conjugate_of_conjugate(self):
    degree = 16
    num_slots = 8
    q_limbs = [1073184769]
    p_limbs = [1073479681, 1073741953]

    all_moduli = q_limbs + p_limbs
    scale = 2**20

    test_random_source = random.ZeroNoiseRandomSource()

    pk_q, sk_q = key_gen.keygen(
        degree, q_limbs, random_source=test_random_source
    )

    conj_key = key_gen.gen_conjugate_key(
        sk=sk_q,
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        dnum=1,
        random_source=test_random_source,
    )

    bc_kernel = basis_conversion.BasisConversionBarrett()
    bc_kernel.precompute_constants(all_moduli, [([0], [1, 2])])

    barrett_constants_pq = barrett.precompute_barrett_constants(all_moduli)
    mul_kernel = mul.MulPlaintextCiphertextBarrett(barrett_constants_pq)
    rescale_kernel = rescale.Rescale()
    rescale_kernel.precompute_constants(
        all_moduli, num_rescales=len(p_limbs), r=4, c=4
    )

    conjugate_kernel = conjugate.Conjugation()
    conjugate_kernel.precompute_constants(
        q_limbs=q_limbs, p_limbs=p_limbs, dnum=1, r=4, c=4
    )

    mu = np.array(
        [complex(x % 4 + 1.5, x % 4 - 2.5) for x in range(num_slots)],
        dtype=complex,
    )
    encoder_q = encode.Encode(degree, q_limbs, scale)
    encryptor_q = encrypt.Encrypt(pk_q)

    plain_mu = encoder_q.encode(mu.tolist())
    ct_in = encryptor_q.encrypt(plain_mu, random_source=test_random_source)

    ct_conj = conjugate_kernel.conjugate(
        ct=ct_in,
        conj_key=conj_key,
        p_limbs=jnp.array(p_limbs, dtype=jnp.uint32),
        bc_kernel=bc_kernel,
        mul_kernel=mul_kernel,
        rescale_kernel=rescale_kernel,
        start_control_index=0,
    )

    ct_conj_conj = conjugate_kernel.conjugate(
        ct=ct_conj,
        conj_key=conj_key,
        p_limbs=jnp.array(p_limbs, dtype=jnp.uint32),
        bc_kernel=bc_kernel,
        mul_kernel=mul_kernel,
        rescale_kernel=rescale_kernel,
        start_control_index=0,
    )

    decryptor_q = encrypt.Decrypt(sk_q)
    pt_dec = decryptor_q.decrypt(ct_conj_conj)

    decoder = encode.Decode(scale, num_slots)
    decoded = decoder.decode(pt_dec)

    for e, d in zip(mu, decoded):
      self.assertAlmostEqual(e.real, d.real, delta=0.5)
      self.assertAlmostEqual(e.imag, d.imag, delta=0.5)

  def test_conjugation_ntt_preservation(self):
    degree = 8
    q = 1073184769

    np.random.seed(42)
    poly = np.random.randint(0, q, size=degree).astype(np.uint64)

    poly_rot = np.zeros_like(poly)
    poly_rot[0] = poly[0]
    for i in range(1, degree):
      val = poly[i]
      target_idx = degree - i
      poly_rot[target_idx] = (q - val) % q

    ntt_poly = ntt_cpu.ntt_negacyclic_poly(poly.reshape(degree, 1), [q])
    ntt_poly_rot = ntt_cpu.ntt_negacyclic_poly(poly_rot.reshape(degree, 1), [q])

    ntt_poly_flipped = np.flip(ntt_poly, axis=0)

    np.testing.assert_array_equal(ntt_poly_flipped, ntt_poly_rot)


if __name__ == "__main__":

  absltest.main()
