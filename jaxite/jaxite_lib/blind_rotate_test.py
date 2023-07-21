"""Tests for bootstrap."""

import hypothesis
from hypothesis import strategies
import jax.numpy as jnp
from jaxite.jaxite_lib import bootstrap
from jaxite.jaxite_lib import decomposition
from jaxite.jaxite_lib import encoding
from jaxite.jaxite_lib import lwe
from jaxite.jaxite_lib import matrix_utils
from jaxite.jaxite_lib import parameters
from jaxite.jaxite_lib import random_source
from jaxite.jaxite_lib import rgsw
from jaxite.jaxite_lib import rlwe
from jaxite.jaxite_lib import test_polynomial
from jaxite.jaxite_lib import types
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized


class BlindRotateTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.log_plaintext_modulus = 32
    # These decomposition parameters result in a decomposition with B=16 and
    # L=8, resulting in a "quality" parameter (largest coefficient in the
    # signed decomposition) of 8. The error growth caused by evaluating a CMUX
    # gate in TFHE includes the product of the quality and L (see
    # b/202561578#comment49), so this choice minimizes the error growth among
    # those parameter settings that are "complete" in that there is no
    # approxmiation to the decomposition.
    self.decomposition_params = decomposition.DecompositionParameters(
        log_base=4, level_count=8
    )
    self.polynomial_modulus_degree = 4
    self.rlwe_dimension = 2
    self.noise_free_rng = random_source.CycleRng(const_normal_noise=0)

    # This encoding is appropriate for a test in which there is no noise,
    # i.e., one in which the entire bitstring is used for the message.
    self.encoding = encoding.EncodingParameters(
        total_bit_length=32,
        message_bit_length=32,
        padding_bit_length=0,
    )

    # This encoding is appropriate for a test where there is noise added,
    # and/or for when multiple ciphertexts are being added / multiplied and
    # there is a chance of overflowing the message space.
    self.noisy_encoding = encoding.EncodingParameters(
        # leaves 8 bits for noise
        total_bit_length=32,
        message_bit_length=22,
        padding_bit_length=1,
    )

    self.scheme_params = parameters.SchemeParameters(
        plaintext_modulus=2**self.log_plaintext_modulus,
        lwe_dimension=self.rlwe_dimension,
        polynomial_modulus_degree=self.polynomial_modulus_degree,
        rlwe_dimension=self.rlwe_dimension,
    )

    self.rlwe_key = rlwe.gen_key(
        params=self.scheme_params, prg=self.noise_free_rng
    )
    self.rgsw_key = rgsw.key_from_rlwe(self.rlwe_key)

  @parameterized.named_parameters(
      dict(testcase_name='by_zero', bit=0),
      dict(testcase_name='by_one', bit=1),
  )
  def test_external_product_noise_free_multiply(self, bit):
    rgsw_plaintext = rgsw.RgswPlaintext(
        modulus_degree=self.polynomial_modulus_degree, message=jnp.uint32(bit)
    )
    rgsw_ct = rgsw.encrypt(
        rgsw_plaintext,
        self.rgsw_key,
        decomposition_params=self.decomposition_params,
        prg=self.noise_free_rng,
    )

    # The secret key used for RGSW and RLWE must be the same
    np.testing.assert_array_equal(self.rgsw_key.key.data, self.rlwe_key.data)

    # the polynomial 1 + x + x^2 (mod x^4 + 1)
    rlwe_cleartext = [1, 1, 1, 0]
    rlwe_plaintext = rlwe.RlwePlaintext(
        log_coefficient_modulus=self.log_plaintext_modulus,
        modulus_degree=self.polynomial_modulus_degree,
        message=jnp.array(rlwe_cleartext, dtype=jnp.uint32),
    )
    rlwe_ciphertext = rlwe.encrypt(
        rlwe_plaintext, self.rlwe_key, prg=self.noise_free_rng
    )

    product = bootstrap.external_product(
        rgsw_ct, rlwe_ciphertext, self.decomposition_params
    )
    decrypted = rlwe.decrypt(
        product, self.rlwe_key, encoding_params=self.encoding
    )

    expected_cleartext = jnp.array(
        [bit * x for x in rlwe_cleartext], dtype=jnp.uint32
    )
    np.testing.assert_array_equal(expected_cleartext, decrypted.message)

  @parameterized.named_parameters(
      dict(testcase_name='by_zero', bit=0),
      dict(testcase_name='by_one', bit=1),
  )
  def test_external_product_noisy_multiply(self, bit):
    # It will also work with a constant normal noise, but the threshold for how
    # large the noise may be before corrupting the message is quite small.  For
    # example, setting constant noise = 5 fails, but setting constant noise = 4
    # succeeds.
    #
    # rng = random_source.CycleRng(const_normal_noise=4)
    #
    # Likewise, a normal distribution with mean zero and standard deviation 4
    # succeeds, but setting the std deviation to 5 fails.

    rng = random_source.PseudorandomSource(
        uniform_bounds=(0, 1), normal_std=4, seed=1
    )
    noisy_encoding = encoding.EncodingParameters(
        # leaves 8 bits for noise
        total_bit_length=32,
        message_bit_length=22,
        padding_bit_length=2,
    )

    rgsw_plaintext = rgsw.RgswPlaintext(
        modulus_degree=self.polynomial_modulus_degree, message=jnp.uint32(bit)
    )
    rgsw_ct = rgsw.encrypt(
        rgsw_plaintext,
        self.rgsw_key,
        decomposition_params=self.decomposition_params,
        prg=rng,
    )

    rlwe_cleartext = jnp.array([1, 2, 3, 0], dtype=jnp.uint32)
    rlwe_plaintext = encoding.encode(rlwe_cleartext, noisy_encoding)
    rlwe_plaintext = rlwe.RlwePlaintext(
        log_coefficient_modulus=self.log_plaintext_modulus,
        modulus_degree=self.polynomial_modulus_degree,
        message=jnp.array(rlwe_plaintext, dtype=jnp.uint32),
    )
    rlwe_ciphertext = rlwe.encrypt(rlwe_plaintext, self.rlwe_key, prg=rng)

    product = bootstrap.external_product(
        rgsw_ct, rlwe_ciphertext, self.decomposition_params
    )
    decrypted = rlwe.decrypt(
        product, self.rlwe_key, encoding_params=noisy_encoding
    )

    actual_cleartext = [
        encoding.decode(x, noisy_encoding) for x in decrypted.message
    ]
    expected_cleartext = jnp.array(
        [bit * x for x in rlwe_cleartext], dtype=jnp.uint32
    )
    np.testing.assert_array_equal(expected_cleartext, actual_cleartext)

  @parameterized.named_parameters(
      dict(testcase_name='control_zero', control=0),
      dict(testcase_name='control_one', control=1),
  )
  def test_cmux_noise_free(self, control):
    # test that the cmux operation works, without any encoding or noise added.
    control_pt = rgsw.RgswPlaintext(
        modulus_degree=self.polynomial_modulus_degree, message=control
    )
    control_ct = rgsw.encrypt(
        control_pt,
        self.rgsw_key,
        decomposition_params=self.decomposition_params,
        prg=self.noise_free_rng,
    )

    # The secret key used for RGSW and RLWE must be the same
    np.testing.assert_array_equal(self.rgsw_key.key.data, self.rlwe_key.data)

    # msg_1 = the polynomial 1 + x + x^2 (mod x^4 + 1)
    msg_1 = [1, 1, 1, 0]
    rlwe_pt_1 = rlwe.RlwePlaintext(
        log_coefficient_modulus=self.log_plaintext_modulus,
        modulus_degree=self.polynomial_modulus_degree,
        message=jnp.array(msg_1, dtype=jnp.uint32),
    )
    rlwe_ct_1 = rlwe.encrypt(rlwe_pt_1, self.rlwe_key, prg=self.noise_free_rng)

    # msg_2 = the polynomial x + 2*x^2 + 3*x^3 (mod x^4 + 1)
    msg_2 = [0, 1, 2, 3]
    rlwe_pt_2 = rlwe.RlwePlaintext(
        log_coefficient_modulus=self.log_plaintext_modulus,
        modulus_degree=self.polynomial_modulus_degree,
        message=jnp.array(msg_2, dtype=jnp.uint32),
    )
    rlwe_ct_2 = rlwe.encrypt(rlwe_pt_2, self.rlwe_key, prg=self.noise_free_rng)

    cmux_output = bootstrap.cmux(
        control_ct, rlwe_ct_1, rlwe_ct_2, self.decomposition_params
    )
    decrypted = rlwe.decrypt(
        cmux_output, self.rlwe_key, encoding_params=self.encoding
    )

    expected_cleartext = msg_1 if control == 0 else msg_2
    np.testing.assert_array_equal(expected_cleartext, decrypted.message)

  @parameterized.named_parameters(
      dict(testcase_name='control_zero', control=0),
      dict(testcase_name='control_one', control=1),
  )
  def test_cmux_noise_free_encoded(self, control):
    # test that the cmux operation works, when the message is encoded.
    # note: the message must be encoded with padding bits, or the message can
    # overflow the message space and result in the wrong output.
    control_pt = rgsw.RgswPlaintext(
        modulus_degree=self.polynomial_modulus_degree, message=control
    )
    control_ct = rgsw.encrypt(
        control_pt,
        self.rgsw_key,
        decomposition_params=self.decomposition_params,
        prg=self.noise_free_rng,
    )

    # The secret key used for RGSW and RLWE must be the same
    np.testing.assert_array_equal(self.rgsw_key.key.data, self.rlwe_key.data)

    # msg_1 = the polynomial 1 + x + x^2 (mod x^4 + 1)
    msg_1 = [1, 1, 1, 0]
    rlwe_encoded_1 = encoding.encode(
        jnp.array(msg_1, dtype=jnp.uint32), self.noisy_encoding
    )
    rlwe_pt_1 = rlwe.RlwePlaintext(
        log_coefficient_modulus=self.log_plaintext_modulus,
        modulus_degree=self.polynomial_modulus_degree,
        message=jnp.array(rlwe_encoded_1, dtype=jnp.uint32),
    )
    rlwe_ct_1 = rlwe.encrypt(rlwe_pt_1, self.rlwe_key, prg=self.noise_free_rng)

    # msg_ 2 = the polynomial x + 2*x^2 + 3*x^3 (mod x^4 + 1)
    msg_2 = [0, 1, 2, 3]
    rlwe_encoded_2 = encoding.encode(
        jnp.array(msg_2, dtype=jnp.uint32), self.noisy_encoding
    )
    rlwe_pt_2 = rlwe.RlwePlaintext(
        log_coefficient_modulus=self.log_plaintext_modulus,
        modulus_degree=self.polynomial_modulus_degree,
        message=jnp.array(rlwe_encoded_2, dtype=jnp.uint32),
    )
    rlwe_ct_2 = rlwe.encrypt(rlwe_pt_2, self.rlwe_key, prg=self.noise_free_rng)

    cmux_output = bootstrap.cmux(
        control_ct, rlwe_ct_1, rlwe_ct_2, self.decomposition_params
    )
    decrypted = rlwe.decrypt(
        cmux_output, self.rlwe_key, encoding_params=self.noisy_encoding
    )
    decoded = [
        encoding.decode(x, self.noisy_encoding) for x in decrypted.message
    ]

    expected_cleartext = msg_1 if control == 0 else msg_2
    np.testing.assert_array_equal(expected_cleartext, decoded)

  @parameterized.named_parameters(
      dict(testcase_name='control_zero', control=0),
      dict(testcase_name='control_one', control=1),
  )
  def test_cmux_noisy(self, control):
    noisy_rng = random_source.PseudorandomSource(normal_std=4, seed=1)

    # Because CMUX does RLWE ciphertext addition and subtraction (in addition to
    # an external mult), it accumulates more noise than just an external mult.
    # Therefore, the encoding needs to have more bits allocated for the noise.
    # Allocating 13 bits of noise allows for correct decoding with a PRG with
    # normal_std=4. However, any fewer bits of noise will fail for any PRG with
    # normal_std>1.
    extra_noisy_encoding = encoding.EncodingParameters(
        # leaves 13 bits for noise
        total_bit_length=32,
        message_bit_length=17,
        padding_bit_length=2,
    )
    control_pt = rgsw.RgswPlaintext(
        modulus_degree=self.polynomial_modulus_degree, message=control
    )
    control_ct = rgsw.encrypt(
        control_pt,
        self.rgsw_key,
        decomposition_params=self.decomposition_params,
        prg=noisy_rng,
    )

    # The secret key used for RGSW and RLWE must be the same
    np.testing.assert_array_equal(self.rgsw_key.key.data, self.rlwe_key.data)

    # msg_1 = the polynomial 1 + x + x^2 (mod x^4 + 1)
    msg_1 = [1, 1, 1, 0]
    rlwe_encoded_1 = encoding.encode(
        jnp.array(msg_1, dtype=jnp.uint32), extra_noisy_encoding
    )
    rlwe_pt_1 = rlwe.RlwePlaintext(
        log_coefficient_modulus=self.log_plaintext_modulus,
        modulus_degree=self.polynomial_modulus_degree,
        message=jnp.array(rlwe_encoded_1, dtype=jnp.uint32),
    )
    rlwe_ct_1 = rlwe.encrypt(rlwe_pt_1, self.rlwe_key, prg=noisy_rng)

    # msg_ 2 = the polynomial x + 2*x^2 + 3*x^3 (mod x^4 + 1)
    msg_2 = [0, 1, 2, 3]
    rlwe_encoded_2 = encoding.encode(
        jnp.array(msg_2, dtype=jnp.uint32), extra_noisy_encoding
    )
    rlwe_pt_2 = rlwe.RlwePlaintext(
        log_coefficient_modulus=self.log_plaintext_modulus,
        modulus_degree=self.polynomial_modulus_degree,
        message=jnp.array(rlwe_encoded_2, dtype=jnp.uint32),
    )
    rlwe_ct_2 = rlwe.encrypt(rlwe_pt_2, self.rlwe_key, prg=noisy_rng)

    cmux_output = bootstrap.cmux(
        control_ct, rlwe_ct_1, rlwe_ct_2, self.decomposition_params
    )
    decrypted = rlwe.decrypt(
        cmux_output, self.rlwe_key, encoding_params=extra_noisy_encoding
    )
    decoded = [
        encoding.decode(x, extra_noisy_encoding) for x in decrypted.message
    ]

    expected_cleartext = msg_1 if control == 0 else msg_2
    np.testing.assert_array_equal(expected_cleartext, decoded)

  @hypothesis.given(
      strategies.lists(
          strategies.integers(min_value=0, max_value=2**16 - 1),
          min_size=4,
          max_size=4,
      )
  )
  @hypothesis.settings(deadline=None)
  def test_sample_extract_noisefree(self, message):
    rlwe_encoded = encoding.encode(
        jnp.array(message, dtype=jnp.uint32), self.encoding
    )
    rlwe_pt = rlwe.RlwePlaintext(
        log_coefficient_modulus=self.log_plaintext_modulus,
        modulus_degree=self.polynomial_modulus_degree,
        message=jnp.array(rlwe_encoded, dtype=jnp.uint32),
    )
    rlwe_ct = rlwe.encrypt(rlwe_pt, self.rlwe_key, prg=self.noise_free_rng)

    extracted = bootstrap.sample_extract(rlwe_ct)

    reshaped_key = rlwe.flatten_key(self.rlwe_key)
    lwe_decrypt = lwe.decrypt(extracted, reshaped_key, self.encoding)
    np.testing.assert_array_equal(lwe_decrypt, message[0])

  @hypothesis.given(
      strategies.lists(
          strategies.integers(min_value=0, max_value=2**3 - 1),
          min_size=4,
          max_size=4,
      )
  )
  @hypothesis.settings(deadline=None)
  def test_sample_extract_noisy(self, message):
    rng = random_source.PseudorandomSource(normal_std=4, seed=1)
    noisy_encoding = encoding.EncodingParameters(
        # leaves 8 bits for noise
        total_bit_length=32,
        message_bit_length=22,
        padding_bit_length=2,
    )
    rlwe_encoded = [encoding.encode(x, noisy_encoding) for x in message]
    rlwe_pt = rlwe.RlwePlaintext(
        log_coefficient_modulus=self.log_plaintext_modulus,
        modulus_degree=self.polynomial_modulus_degree,
        message=jnp.array(rlwe_encoded, dtype=jnp.uint32),
    )
    rlwe_ct = rlwe.encrypt(rlwe_pt, self.rlwe_key, prg=rng)

    extracted = bootstrap.sample_extract(rlwe_ct)

    poly_deg, k = self.polynomial_modulus_degree, self.rlwe_dimension
    reshaped_key = lwe.LweSecretKey(
        log_modulus=self.rlwe_key.log_coefficient_modulus,
        lwe_dimension=poly_deg * k,
        key_data=self.rlwe_key.data.reshape(poly_deg * k),
    )
    lwe_decrypt = lwe.decrypt(extracted, reshaped_key, noisy_encoding)
    np.testing.assert_array_equal(
        encoding.decode(lwe_decrypt, noisy_encoding), message[0]
    )

  @parameterized.named_parameters(
      dict(testcase_name='rot_0', j=0),
      dict(testcase_name='rot_1', j=1),
      dict(testcase_name='rot_2', j=2),
      dict(testcase_name='rot_3', j=3),
      dict(testcase_name='rot_4', j=4),
      dict(testcase_name='rot_8', j=8),
      dict(testcase_name='rot_21', j=21),
      dict(testcase_name='rot_307', j=307),
      dict(testcase_name='rot_1000', j=1000),
  )
  def test_blind_rotate_encoded(self, j):
    # Test that blind rotate works, with encoding (noisy encoding)
    # but with no noise added to the encryption steps.

    # RLWE encrypt rot_poly under secret key s'.
    # Note that (0, 0, ... v) is a valid RLWE encryption of v, so we will use:
    # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 2, 3]] as our RLWE encryption.
    rot_poly = [0, 1, 2, 3]
    padding = jnp.zeros(
        shape=(self.rlwe_dimension + 1, self.polynomial_modulus_degree),
        dtype=jnp.uint32,
    )
    rot_poly_padded = padding.at[self.rlwe_dimension].set(rot_poly)
    rpp_encoded = []
    for polynomial in rot_poly_padded:
      rpp_encoded.append(
          [encoding.encode(x, self.noisy_encoding) for x in polynomial]
      )

    rot_poly_encrypted = rlwe.RlweCiphertext(
        log_coefficient_modulus=self.log_plaintext_modulus,
        modulus_degree=self.polynomial_modulus_degree,
        message=jnp.array(rpp_encoded, dtype=jnp.uint32),
    )

    lwe_key = lwe.gen_key(params=self.scheme_params, prg=self.noise_free_rng)
    bsk = bootstrap.gen_bootstrapping_key(
        lwe_sk=lwe_key,
        rgsw_sk=self.rgsw_key,
        decomposition_params=self.decomposition_params,
        prg=self.noise_free_rng,
    )
    coefficient_index = lwe.encrypt(
        plaintext=types.LwePlaintext(j), sk=lwe_key, prg=self.noise_free_rng
    )

    rotated_rlwe_ciphertext = bootstrap.blind_rotate(
        rot_polynomial=rot_poly_encrypted,
        coefficient_index=coefficient_index,
        bsk=bsk,
        decomposition_params=self.decomposition_params,
    )

    rotated_plaintext = rlwe.decrypt(
        ciphertext=rotated_rlwe_ciphertext,
        sk=self.rlwe_key,
        encoding_params=self.noisy_encoding,
    )

    # Note that because the message space has size 2**22 (set in
    # noisy_encoding), an output coefficient that gets flipped when rotating
    # beyond polynomial_modulus_degree should be inverted modulo 2**22. In
    # particular, -1 should map to 4194303, not 2**32-1 = 4294967295.
    decoded = [
        encoding.decode(x, self.noisy_encoding)
        for x in rotated_plaintext.message
    ]

    expected_cleartext = matrix_utils.monomial_mul(
        poly=jnp.array(rot_poly, dtype=jnp.uint32),
        degree=-j,
        log_modulus=self.noisy_encoding.message_bit_length,
    )
    np.testing.assert_array_equal(expected_cleartext, decoded)

  @parameterized.product(
      j=[0, 1, 2, 3, 4, 8, 21, 307],
      seed=[0, 1],
      lwe_dim=[10, 630],
  )
  def test_blind_rotate_with_noise(self, j, seed, lwe_dim):
    scheme_params = parameters.SchemeParameters(
        plaintext_modulus=2**self.log_plaintext_modulus,
        lwe_dimension=lwe_dim,
        polynomial_modulus_degree=64,
        rlwe_dimension=1,
    )

    # because the input to blind rotate is supposed to be scaled to have
    # coefficients in range 0, ..., 2N, we set the upper bound of LWE a_i
    # samples appropriately.
    lwe_rng = random_source.PseudorandomSource(
        seed=seed,
        normal_std=0,
        uniform_bounds=(0, scheme_params.polynomial_modulus_degree),
    )
    rlwe_rng = random_source.PseudorandomSource(seed=seed, normal_std=0)

    noisy_encoding = encoding.EncodingParameters(
        total_bit_length=32,
        message_bit_length=3,
        padding_bit_length=1,
    )
    rot_poly = test_polynomial.identity_test_polynomial(
        noisy_encoding, scheme_params
    )
    rot_poly_ct = test_polynomial.trivial_encryption(rot_poly, scheme_params)

    lwe_key = lwe.gen_key(params=scheme_params, prg=lwe_rng)
    rlwe_key = rlwe.gen_key(params=scheme_params, prg=rlwe_rng)
    rgsw_key = rgsw.key_from_rlwe(rlwe_key)
    bsk = bootstrap.gen_bootstrapping_key(
        lwe_sk=lwe_key,
        rgsw_sk=rgsw_key,
        decomposition_params=self.decomposition_params,
        prg=rlwe_rng,
    )
    coefficient_index = lwe.encrypt(
        plaintext=types.LwePlaintext(j), sk=lwe_key, prg=lwe_rng
    )

    rotated_rlwe_ciphertext = bootstrap.blind_rotate(
        rot_polynomial=rot_poly_ct,
        coefficient_index=coefficient_index,
        bsk=bsk,
        decomposition_params=self.decomposition_params,
    )

    rotated_plaintext = rlwe.decrypt(
        ciphertext=rotated_rlwe_ciphertext,
        sk=rlwe_key,
        encoding_params=noisy_encoding,
    )

    decoded = [
        encoding.decode(x, noisy_encoding) for x in rotated_plaintext.message
    ]

    expected_plaintext = matrix_utils.monomial_mul(
        poly=rot_poly.message, degree=-j, log_modulus=self.log_plaintext_modulus
    )
    expected_cleartext = [
        encoding.decode(x, noisy_encoding) for x in expected_plaintext
    ]
    np.testing.assert_array_equal(expected_cleartext, decoded)


if __name__ == '__main__':
  absltest.main()
