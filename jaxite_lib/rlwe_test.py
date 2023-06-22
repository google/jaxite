"""Tests for RLWE."""
import textwrap

import hypothesis
from hypothesis import strategies
import jax.numpy as jnp
from jaxite.jaxite_lib import parameters
from jaxite.jaxite_lib import random_source
from jaxite.jaxite_lib import rlwe
from jaxite.jaxite_lib import test_utils
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized


class RlweTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.dim = 10
    self.plaintext_modulus = 2**32
    self.polynomial_modulus_degree = 8
    self.rlwe_dimension = 2
    self.noise_free_rng = random_source.CycleRng(const_normal_noise=0)
    self.default_key = rlwe.gen_key(
        params=parameters.SchemeParameters(
            plaintext_modulus=self.plaintext_modulus,
            lwe_dimension=self.dim,
            polynomial_modulus_degree=self.polynomial_modulus_degree,
            rlwe_dimension=self.rlwe_dimension,
        ),
        prg=self.noise_free_rng,
    )

  @parameterized.named_parameters(
      dict(testcase_name='const_coeff_only', coeffs=[3], expected='3 x^0'),
      dict(
          testcase_name='many_coeffs',
          coeffs=[1, 0, 2, 3],
          expected='1 x^0 + 2 x^2 + 3 x^3',
      ),
  )
  def test_rlwe_plaintext_str(self, coeffs: list[int], expected: str):
    # non-message params are not needed for pretty-printing, pick arbitrarily.
    plaintext = rlwe.RlwePlaintext(
        log_coefficient_modulus=2, modulus_degree=5, message=jnp.array(coeffs)
    )
    self.assertEqual(expected, str(plaintext))

  def test_rlwe_ciphertext_str(self):
    ciphertext = rlwe.RlweCiphertext(
        log_coefficient_modulus=2,
        modulus_degree=5,
        message=jnp.array([[0, 1, 2], [2, 1, 0], [3, 3, 3]]),
    )
    expected = textwrap.dedent("""\
       [
        1 x^1 + 2 x^2
        2 x^0 + 1 x^1
        3 x^0 + 3 x^1 + 3 x^2
       ]""")
    self.assertEqual(expected, str(ciphertext))

  @hypothesis.given(
      dim=strategies.integers(min_value=1, max_value=3),
      deg=strategies.integers(min_value=1, max_value=8),
  )
  @hypothesis.settings(deadline=None)
  def test_gen_key(self, dim: int, deg: int):
    key = rlwe.gen_key(
        params=parameters.SchemeParameters(
            plaintext_modulus=2**32,
            lwe_dimension=10,
            polynomial_modulus_degree=deg,
            rlwe_dimension=dim,
        ),
        prg=self.noise_free_rng,
    )

    self.assertEqual(dim, key.rlwe_dimension)
    self.assertEqual(deg, key.modulus_degree)
    self.assertLen(key.data, dim)
    self.assertLen(key.data[0], deg)
    self.assertEqual(32, key.log_coefficient_modulus)

  @hypothesis.given(
      strategies.lists(
          strategies.integers(min_value=0, max_value=2**30 - 1),
          min_size=8,
          max_size=8,
      )
  )
  @hypothesis.settings(deadline=None)
  def test_encrypt_dimension(self, plaintext):
    rlwe_plaintext = rlwe.RlwePlaintext(
        log_coefficient_modulus=self.default_key.log_coefficient_modulus,
        modulus_degree=len(plaintext),
        message=jnp.array(plaintext, dtype=jnp.uint32),
    )
    ciphertext = rlwe.encrypt(
        rlwe_plaintext, self.default_key, prg=self.noise_free_rng
    )
    self.assertLen(ciphertext.message, self.default_key.rlwe_dimension + 1)

  @hypothesis.given(
      strategies.lists(
          strategies.integers(min_value=0, max_value=2**16 - 1),
          min_size=8,
          max_size=8,
      )
  )
  @hypothesis.settings(deadline=None)
  def test_error_free_encrypt_decrypt(self, message):
    # Tests error free encryption and decryption for a polynomial modulus deg 8.
    plaintext = jnp.left_shift(jnp.array(message, dtype=jnp.uint32), 10)
    rlwe_plaintext = rlwe.RlwePlaintext(
        log_coefficient_modulus=self.default_key.log_coefficient_modulus,
        modulus_degree=len(message),
        message=plaintext,
    )
    ciphertext = rlwe.encrypt(
        rlwe_plaintext, self.default_key, prg=self.noise_free_rng
    )
    decrypted = rlwe.decrypt(
        ciphertext,
        self.default_key,
        encoding_params=test_utils.DEFAULT_ENCODING_PARAMS,
    )
    np.testing.assert_array_equal(plaintext, decrypted.message)

  @hypothesis.settings(deadline=None)
  @hypothesis.given(
      strategies.lists(
          strategies.integers(min_value=0, max_value=2**14 - 1),
          min_size=8,
          max_size=8,
      ),
      strategies.lists(
          strategies.integers(min_value=0, max_value=2**14 - 1),
          min_size=8,
          max_size=8,
      ),
      strategies.sampled_from(random_source.VARYING_MAGNITUDE_TEST_RNGS),
  )
  def test_encrypt_add_decrypt(self, message1, message2, rng):
    # shifting left by 10 makes room for 10 least-significant bits of noise, but
    # still leaves 2 bits of padding at the top of the message to handle
    # overflow.
    plaintext1 = jnp.left_shift(jnp.array(message1, dtype=jnp.uint32), 10)
    plaintext2 = jnp.left_shift(jnp.array(message2, dtype=jnp.uint32), 10)
    rlwe_plaintext1 = rlwe.RlwePlaintext(
        log_coefficient_modulus=self.default_key.log_coefficient_modulus,
        modulus_degree=len(message1),
        message=plaintext1,
    )
    rlwe_plaintext2 = rlwe.RlwePlaintext(
        log_coefficient_modulus=self.default_key.log_coefficient_modulus,
        modulus_degree=len(message2),
        message=plaintext2,
    )

    ciphertext1 = rlwe.encrypt(rlwe_plaintext1, self.default_key, prg=rng)
    ciphertext2 = rlwe.encrypt(rlwe_plaintext2, self.default_key, prg=rng)
    ciphertext_sum = rlwe.RlweCiphertext(
        log_coefficient_modulus=self.default_key.log_coefficient_modulus,
        modulus_degree=len(ciphertext1.message),
        message=ciphertext1.message + ciphertext2.message,
    )

    decrypted = rlwe.decrypt(
        ciphertext_sum,
        self.default_key,
        encoding_params=test_utils.DEFAULT_ENCODING_PARAMS,
    )
    np.testing.assert_array_equal(plaintext1 + plaintext2, decrypted.message)

  def test_encrypt_add_decrypt_prod_security_params(self):
    encoding_params = test_utils.ENCODING_PARAMS_128_BIT_SECURITY
    rng = test_utils.RLWE_RNG_128_BIT_SECURITY
    rlwe_key = rlwe.gen_key(
        params=test_utils.SCHEME_PARAMS_128_BIT_SECURITY, prg=rng
    )

    # shifting left by 25 so the message is out of the error domain
    shift_left = (
        encoding_params.total_bit_length
        - encoding_params.message_bit_length
        - encoding_params.padding_bit_length
    )
    cleartext1 = 1
    cleartext2 = 2
    plaintext1 = jnp.left_shift(
        jnp.array(cleartext1, dtype=jnp.uint32), shift_left
    )
    plaintext2 = jnp.left_shift(
        jnp.array(cleartext2, dtype=jnp.uint32), shift_left
    )
    rlwe_plaintext1 = rlwe.RlwePlaintext(
        log_coefficient_modulus=test_utils.SCHEME_PARAMS_128_BIT_SECURITY.log_plaintext_modulus,
        modulus_degree=test_utils.SCHEME_PARAMS_128_BIT_SECURITY.polynomial_modulus_degree,
        message=plaintext1,
    )
    rlwe_plaintext2 = rlwe.RlwePlaintext(
        log_coefficient_modulus=test_utils.SCHEME_PARAMS_128_BIT_SECURITY.log_plaintext_modulus,
        modulus_degree=test_utils.SCHEME_PARAMS_128_BIT_SECURITY.polynomial_modulus_degree,
        message=plaintext2,
    )

    ciphertext1 = rlwe.encrypt(rlwe_plaintext1, rlwe_key, prg=rng)
    ciphertext2 = rlwe.encrypt(rlwe_plaintext2, rlwe_key, prg=rng)
    ciphertext_sum = rlwe.RlweCiphertext(
        log_coefficient_modulus=rlwe_key.log_coefficient_modulus,
        modulus_degree=test_utils.SCHEME_PARAMS_128_BIT_SECURITY.polynomial_modulus_degree,
        message=ciphertext1.message + ciphertext2.message,
    )

    decrypted = rlwe.decrypt(
        ciphertext_sum, rlwe_key, encoding_params=encoding_params
    )
    np.testing.assert_array_equal(plaintext1 + plaintext2, decrypted.message)


if __name__ == '__main__':
  absltest.main()
