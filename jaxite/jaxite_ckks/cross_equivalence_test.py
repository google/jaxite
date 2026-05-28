"""Tests for equivalence between jaxite_word and jaxite_ckks.

Reference commit from CROSS repo: 69c46d2bf25f017e7f4a24e864ad8abb9506a5c4
"""

from importlib import resources
import json
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import encrypt
from jaxite.jaxite_ckks import mul
from jaxite.jaxite_ckks import ntt
from jaxite.jaxite_ckks import ntt_cpu
from jaxite.jaxite_ckks import random
from jaxite.jaxite_ckks import types
import numpy as np
from absl.testing import absltest

jax.config.update("jax_enable_x64", True)


class CrossEquivalenceTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    with (
        resources.files("jaxite.jaxite_ckks")
        / "cross_equivalence_test_data.json"
    ).open("r") as f:
      cls._test_data = json.load(f)

  def test_encode_equivalence(self):
    degree = 16
    scale = 563019763943521
    moduli = [1073742881, 1073742721, 1073741441, 1073741857, 524353]

    test_data = self._test_data["test_encode_equivalence"]

    slots = [complex(r, i) for r, i in test_data["slots"]]
    expected_data = np.array(test_data["expected_data"], dtype=np.uint32)

    encoder = encode.Encode(degree, moduli, scale)
    pt = encoder.encode(slots)

    np.testing.assert_array_equal(np.array(pt.data), expected_data)

  def test_encrypt_equivalence(self):
    degree = 16
    moduli = [1073742881, 1073742721, 1073741441, 1073741857, 524353]

    test_data = self._test_data["test_encrypt_equivalence"]

    plaintext_data = np.array(test_data["plaintext_data"], dtype=np.uint64)
    pt = types.Plaintext(
        data=jnp.array(plaintext_data, dtype=jnp.uint32),
        moduli=jnp.array(moduli, dtype=jnp.uint32),
    )

    ones = np.ones((degree, len(moduli)), dtype=np.uint32)
    v_coeffs = ntt_cpu.intt_negacyclic_poly(ones, moduli)
    e_coeffs = ntt_cpu.intt_negacyclic_poly(ones, moduli)

    class MockRandomSource(random.RandomSource):

      def gen_ternary_poly(self, d, m):
        return v_coeffs

      def gen_gaussian_poly(self, d, m, sigma=3.19):
        return e_coeffs

      def gen_uniform_poly(self, d, m):
        return np.zeros((d, len(m)))

    random_source = MockRandomSource()

    pk_data = np.ones((2, degree, len(moduli)), dtype=np.uint32)
    pk = types.PublicKey(data=pk_data, moduli=np.array(moduli, dtype=np.uint32))

    encryptor = encrypt.Encrypt(pk)
    ct = encryptor.encrypt(pt, random_source=random_source)

    expected_c0 = np.array(test_data["expected_c0"], dtype=np.uint32)
    expected_c1 = np.full((degree, len(moduli)), 2, dtype=np.uint32)
    expected_data = np.stack([expected_c0, expected_c1], axis=0)
    np.testing.assert_array_equal(np.array(ct.data), expected_data)

  def test_decrypt_equivalence(self):
    moduli = [1073742881, 1073742721, 1073741441, 1073741857, 524353]

    test_data = self._test_data["test_decrypt_equivalence"]

    sk_data = np.array(test_data["sk_data"], dtype=np.uint32)
    ct_c0 = np.array(test_data["ct_c0"], dtype=np.uint32)
    ct_c1 = np.array(test_data["ct_c1"], dtype=np.uint32)
    expected_result = np.array(test_data["expected_result"], dtype=np.uint32)

    sk = types.SecretKey(data=sk_data, moduli=np.array(moduli, dtype=np.uint32))
    ct_data = np.stack([ct_c0, ct_c1], axis=0)
    ct = types.Ciphertext(
        data=jnp.array(ct_data, dtype=jnp.uint32),
        moduli=jnp.array(moduli, dtype=jnp.uint32),
    )

    decryptor = encrypt.Decrypt(sk)
    pt = decryptor.decrypt(ct)

    np.testing.assert_array_equal(np.array(pt.data), expected_result)

  def test_composition_equivalence(self):
    degree = 16
    scale = 563019763943521
    moduli = [1073742881, 1073742721, 1073741441, 1073741857, 524353]

    test_data = self._test_data["test_composition_equivalence"]

    slots = [complex(r, i) for r, i in test_data["slots"]]

    encoder = encode.Encode(degree, moduli, scale)
    pt = encoder.encode(slots)

    class MockRandomSource(random.RandomSource):

      def gen_ternary_poly(self, d, m):
        return np.ones((d, len(m)), dtype=np.uint64)

      def gen_gaussian_poly(self, d, m, sigma=3.19):
        return np.ones((d, len(m)), dtype=np.uint64)

      def gen_uniform_poly(self, d, m):
        return np.zeros((d, len(m)))

    random_source = MockRandomSource()

    pk_data = np.ones((2, degree, len(moduli)), dtype=np.uint64)
    pk = types.PublicKey(data=pk_data, moduli=np.array(moduli, dtype=np.uint64))

    encryptor = encrypt.Encrypt(pk)
    ct = encryptor.encrypt(pt, random_source=random_source)

    expected_c0 = np.array(test_data["expected_c0"], dtype=np.uint64)
    expected_c1 = np.array(test_data["expected_c1"], dtype=np.uint64)

    expected_data = np.stack([expected_c0, expected_c1], axis=0)
    np.testing.assert_array_equal(np.array(ct.data), expected_data)

  def test_ntt_equivalence(self):
    moduli = [2147483489, 2147483137, 2147482817]
    r, c = 4, 4
    degree = r * c
    b = 2

    test_data = self._test_data["test_ntt_equivalence"]

    coef_in_raw = test_data["coef_in_raw"]
    eval_in_raw = test_data["eval_in_raw"]

    coef_in = jnp.concatenate(
        [
            jnp.array(coef_in_raw, dtype=jnp.uint32)
            .transpose(1, 0)
            .reshape(1, degree, -1)
            for _ in range(b)
        ],
        axis=0,
    )
    eval_in = jnp.concatenate(
        [
            jnp.array(eval_in_raw, dtype=jnp.uint32)
            .transpose(1, 0)
            .reshape(1, degree, -1)
            for _ in range(b)
        ],
        axis=0,
    )

    ntt_kernel = ntt.NTTBarrett()
    ntt_kernel.precompute_constants(moduli, r, c)

    ntt_input = coef_in.reshape(b, r, c, -1)
    ntt_output = ntt_kernel.ntt(ntt_input)

    np.testing.assert_array_equal(eval_in, ntt_output.reshape(b, degree, -1))

    intt_output = ntt_kernel.intt(ntt_output)
    np.testing.assert_array_equal(coef_in, intt_output.reshape(b, degree, -1))

  def test_mul_equivalence(self):
    q_towers = [536872097, 536870657, 536872001, 536870849, 536871233, 524353]
    p_towers = [1073741441, 1073740609]
    r, c = 4, 4
    dnum = 3

    test_data = self._test_data["test_mul_equivalence"]

    evalkey_a_vector = test_data["evalkey_a_vector"]
    evalkey_b_vector = test_data["evalkey_b_vector"]
    input_ciphertext = test_data["input_ciphertext"]
    encrypted_mult_result_ref = test_data["encrypted_mult_result_ref"]

    in_array = jnp.array(input_ciphertext, dtype=jnp.uint32)
    in_array = in_array.reshape(4, 16, 6)

    ct1_data = in_array[:2]
    ct2_data = in_array[2:]

    moduli = jnp.array(q_towers, dtype=jnp.uint64)

    ct1 = types.Ciphertext(ct1_data, moduli)
    ct2 = types.Ciphertext(ct2_data, moduli)

    mul_kernel = mul.Mul()
    mul_kernel.precompute_constants(
        q_towers, p_towers, dnum, r, c, composite_degree=1
    )

    ct_3elem = mul_kernel.tensor_multiply(ct1, ct2)

    evk_a = jnp.array(evalkey_a_vector, dtype=jnp.uint32)
    evk_b = jnp.array(evalkey_b_vector, dtype=jnp.uint32)

    evk_a_precomp = jnp.concatenate([evk_a[..., :5], evk_a[..., -2:]], axis=-1)
    evk_b_precomp = jnp.concatenate([evk_b[..., :5], evk_b[..., -2:]], axis=-1)

    evk = mul.EvaluationKeys(
        evk_a_precomp,
        evk_b_precomp,
        jnp.array(q_towers[:-1] + p_towers, dtype=jnp.uint64),
    )

    res_ct = mul_kernel.relinearize(ct_3elem, evk)

    expected_result = jnp.array(encrypted_mult_result_ref, dtype=jnp.uint32)

    np.testing.assert_array_equal(
        res_ct.data, expected_result.reshape(2, 16, 5)
    )


if __name__ == "__main__":
  absltest.main()
