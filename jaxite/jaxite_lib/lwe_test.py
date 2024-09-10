"""Tests for lwe."""

import math

import hypothesis
from hypothesis import strategies
import jax.numpy as jnp
from jaxite.jaxite_lib import encoding
from jaxite.jaxite_lib import lwe
from jaxite.jaxite_lib import parameters
from jaxite.jaxite_lib import random_source
from jaxite.jaxite_lib import test_utils
from jaxite.jaxite_lib import types

from absl.testing import absltest
from absl.testing import parameterized


class LweEncryptDecryptTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.noise_free_rng = random_source.CycleRng(const_normal_noise=0)
    self.dim = 10
    self.plaintext_modulus = 2**32
    self.rlwe_dimension = 2
    self.polynomial_modulus_degree = 8
    self.default_key = lwe.gen_key(
        params=parameters.SchemeParameters(
            plaintext_modulus=self.plaintext_modulus,
            lwe_dimension=self.dim,
            # rlwe_dimension is unused
            rlwe_dimension=self.rlwe_dimension,
            polynomial_modulus_degree=self.polynomial_modulus_degree,
        ),
        prg=self.noise_free_rng,
    )

  @hypothesis.given(strategies.integers(min_value=1, max_value=2**16))
  @hypothesis.settings(deadline=None)
  def test_gen_key(self, dim: int):
    key = lwe.gen_key(
        params=parameters.SchemeParameters(
            plaintext_modulus=2**32,
            lwe_dimension=dim,
            rlwe_dimension=2,
            polynomial_modulus_degree=self.polynomial_modulus_degree,
        ),
        prg=self.noise_free_rng,
    )

    self.assertEqual(dim, key.lwe_dimension)
    self.assertLen(key.key_data, dim)
    self.assertEqual(0, key.modulus)

  @hypothesis.given(strategies.integers(min_value=0, max_value=2**30 - 1))
  @hypothesis.settings(deadline=None)
  def test_encrypt_dimension(self, plaintext):
    ciphertext = lwe.encrypt(
        plaintext, self.default_key, prg=self.noise_free_rng
    )
    self.assertLen(ciphertext, self.default_key.lwe_dimension + 1)

  def test_deterministic_noise_encrypt(self):
    dim = 8
    rng = random_source.CycleRng()
    key = lwe.gen_key(
        params=parameters.SchemeParameters(
            plaintext_modulus=2**32,
            lwe_dimension=dim,
            # rlwe_dimension is unused
            rlwe_dimension=2,
            polynomial_modulus_degree=self.polynomial_modulus_degree,
        ),
        prg=rng,
    )

    expected_key = jnp.array([1, 1, 0, 0, 0, 1, 1, 1], dtype=jnp.uint32)
    expected_samples = jnp.array([1, 0, 0, 1, 0, 1, 1, 0], dtype=jnp.uint32)

    self.assertTrue((expected_key == key.key_data).all())

    # plaintext is all 1's filling up the message space
    # with top two bits set for overflow
    plaintext = (1 << dim - 2) - 1

    ciphertext = lwe.encrypt(types.LwePlaintext(plaintext), key, prg=rng)

    # manually computing the dot product of the key and the samples (3),
    # and adding the plaintext (63)
    expected_b = jnp.uint32(3 + 63)
    expected_ciphertext = jnp.concatenate(
        (expected_samples, jnp.array([expected_b]))
    )
    self.assertTrue((expected_ciphertext == ciphertext).all())

  @hypothesis.given(strategies.integers(min_value=0, max_value=2**16 - 1))
  @hypothesis.settings(deadline=None)
  def test_error_free_encrypt_decrypt(self, message):
    # shifting left by 10 makes room for 10 least-significant bits of noise, a
    # typical setup for an LWE encoded message, though for an error-free test it
    # is irrelevant.
    plaintext = message << 10
    ciphertext = lwe.encrypt(
        plaintext, self.default_key, prg=self.noise_free_rng
    )
    decrypted = lwe.decrypt(
        ciphertext,
        self.default_key,
        encoding_params=test_utils.DEFAULT_ENCODING_PARAMS,
    )
    self.assertEqual(plaintext, decrypted)

  @hypothesis.given(strategies.integers(min_value=0, max_value=2**16 - 1))
  @hypothesis.settings(deadline=None)
  def test_error_free_encrypt_decrypt_uint32(self, message):
    # encoded plaintext fills all 32 bits, with 16 bits of noise.  this test is
    # intended to catch boundary issues when using up all 32 bits.
    plaintext = jnp.uint32(message) << 16
    ciphertext = lwe.encrypt(
        plaintext, self.default_key, prg=self.noise_free_rng
    )
    encoding_params = encoding.EncodingParameters(
        total_bit_length=32, message_bit_length=14, padding_bit_length=2
    )

    decrypted = lwe.decrypt(
        ciphertext, self.default_key, encoding_params=encoding_params
    )
    self.assertEqual(plaintext, decrypted)

  @hypothesis.given(
      strategies.integers(min_value=0, max_value=2**14 - 1),
      strategies.integers(min_value=0, max_value=2**14 - 1),
      strategies.sampled_from(random_source.VARYING_MAGNITUDE_TEST_RNGS),
  )
  @hypothesis.settings(deadline=None)
  def test_encrypt_add_decrypt(self, message1, message2, rng):
    # shifting left by 10 makes room for 10 least-significant bits of noise, but
    # still leaves 2 bits of padding at the top of the message to handle
    # overflow.
    plaintext1 = message1 << 10
    plaintext2 = message2 << 10

    ciphertext1 = lwe.encrypt(plaintext1, self.default_key, prg=rng)
    ciphertext2 = lwe.encrypt(plaintext2, self.default_key, prg=rng)
    ciphertext_sum = ciphertext1 + ciphertext2

    decrypted = lwe.decrypt(
        ciphertext_sum,
        self.default_key,
        encoding_params=test_utils.DEFAULT_ENCODING_PARAMS,
    )
    self.assertEqual(plaintext1 + plaintext2, decrypted)

  @parameterized.named_parameters(
      dict(testcase_name='add_noise', sign=1),
      dict(testcase_name='subtract_noise', sign=-1),
  )
  def test_max_error_encrypt_decrypt(self, sign: int):
    message = 2**16 - 1  # all bits set
    # shifting left by 10 makes room for 10 least-significant bits of noise.
    plaintext = message << 10
    # we use only 9 fully-set bits of noise because any more noise would result
    # in an incorrect rounding.
    noise_upper_bound = (1 << 9) - 1
    just_below_too_noisy = random_source.CycleRng(
        const_normal_noise=sign * noise_upper_bound
    )
    ciphertext = lwe.encrypt(
        plaintext, self.default_key, prg=just_below_too_noisy
    )
    decrypted = lwe.decrypt(
        ciphertext,
        self.default_key,
        encoding_params=test_utils.DEFAULT_ENCODING_PARAMS,
    )
    self.assertEqual(plaintext, decrypted)

  @parameterized.named_parameters(
      dict(testcase_name='add_noise', sign=1),
      dict(testcase_name='subtract_noise', sign=-1),
  )
  def test_too_much_error_cannot_decrypt(self, sign: int):
    message = 2**16 - 1  # all bits set
    plaintext = message << 10
    # 1<<9 - 1 is the max allowable noise before a rounding error, so we add one
    # to that to get 1<<9.  Note we add one more if we're subtracting noise,
    # since half rounds up
    noise = (1 << 9) + (1 if sign == -1 else 0)
    too_noisy = random_source.CycleRng(const_normal_noise=sign * noise)
    ciphertext = lwe.encrypt(plaintext, self.default_key, prg=too_noisy)
    decrypted = lwe.decrypt(
        ciphertext,
        self.default_key,
        encoding_params=test_utils.DEFAULT_ENCODING_PARAMS,
    )
    self.assertNotEqual(plaintext, decrypted)

  def test_noiseless_embedding_succeeds(self):
    plaintext = jnp.uint32(15)  # arbitrary plaintext value
    lwe_dimension = 2**9  # arbitrary LWE dimension
    ciphertext = lwe.noiseless_embedding(plaintext, lwe_dimension)
    self.assertLen(ciphertext, lwe_dimension + 1)
    self.assertEqual(ciphertext[-1], plaintext)

  def test_encrypt_add_decrypt_prod_security_params(self):
    encoding_params = test_utils.ENCODING_PARAMS_128_BIT_SECURITY
    rng = test_utils.LWE_RNG_128_BIT_SECURITY

    lwe_key = lwe.gen_key(
        params=test_utils.SCHEME_PARAMS_128_BIT_SECURITY, prg=rng
    )

    cleartext1 = 1
    cleartext2 = 2
    plaintext1 = encoding.encode(cleartext1, encoding_params)
    plaintext2 = encoding.encode(cleartext2, encoding_params)
    ciphertext1 = lwe.encrypt(plaintext1, lwe_key, prg=rng)
    ciphertext2 = lwe.encrypt(plaintext2, lwe_key, prg=rng)
    ciphertext_sum = ciphertext1 + ciphertext2

    decrypted = lwe.decrypt(ciphertext_sum, lwe_key, encoding_params)
    self.assertEqual(
        cleartext1 + cleartext2, encoding.decode(decrypted, encoding_params)
    )


@parameterized.product(
    log_output_modulus=[6, 10, 20],
    lwe_dim=[10, 630],
    cleartext=[5, 7],
    injected_noise=[0, 123, -(2**22)],
    seed=list(range(1, 4)),
)
class LweModulusSwitchingTest(parameterized.TestCase):

  def test_modulus_switch_preserves_message(
      self,
      log_output_modulus: int,
      lwe_dim: int,
      cleartext: int,
      injected_noise: int,
      seed: int,
  ):
    """Ensure that modulus switch preserves the encrypted message."""

    # Though we do not need to decrypt/decode modulus-switched LWE ciphertexts
    # in the main cryptosystem (it is hidden inside bootstrap), for this test we
    # want to verify the modulus switch is working as intended. However, the
    # nuance in this test is that after modulus switching from Q to Q', the
    # message encrypted is changed from M to M*Q'/Q. This is interpreted in this
    # test as a change to the encoding scheme. In particular, the operation of
    # modulus switching (ignoring momentarily that it introduces additional
    # error) between two powers of two results in the message being shifted
    # rightward by log2(Q'/Q) many bits.

    # For the test cases in which lwe_dim=630, we have to take special care. The
    # error after modulus switching can be as large as lwe_dim, even when there
    # is no error in the input ciphertext. When that exceeds 2**error_bit_length
    # in `after_switch_encoding_params`, then the decrypted message will be
    # incorrect. In this case, the test could fail for some random seeds, and so
    # we elevate the log_output_modulus appropriately.
    log_output_modulus = max(
        log_output_modulus, int(math.ceil(math.log2(lwe_dim)))
    )

    rng = random_source.PseudorandomSource(normal_std=0, seed=seed)
    log_input_modulus = 32
    scheme_params = parameters.SchemeParameters(
        plaintext_modulus=2**log_input_modulus,
        lwe_dimension=lwe_dim,
        rlwe_dimension=1,  # unused
        polynomial_modulus_degree=1,  # unused
    )
    encoding_params = encoding.EncodingParameters(
        total_bit_length=log_input_modulus,
        message_bit_length=3,
        padding_bit_length=0,
    )
    after_switch_encoding_params = encoding.EncodingParameters(
        total_bit_length=log_output_modulus,
        message_bit_length=3,
        padding_bit_length=0,
    )

    cleartext = jnp.uint32(cleartext)
    plaintext = encoding.encode(cleartext, encoding_params)
    key = lwe.gen_key(params=scheme_params, prg=rng)
    ciphertext = lwe.encrypt(plaintext, key, prg=rng)
    ciphertext = ciphertext.at[-1].set(ciphertext[-1] + injected_noise)

    modulus_switched = lwe.switch_modulus(
        ciphertext, log_input_modulus, log_output_modulus
    )

    # The bits of the secret key are the same, but the modulus changes.
    after_switch_key = lwe.LweSecretKey(
        log_modulus=log_output_modulus,
        lwe_dimension=key.lwe_dimension,
        key_data=key.key_data,
    )
    plaintext_after_switch = lwe.decrypt_without_denoising(
        modulus_switched,
        after_switch_key,
    )
    cleartext_after_switch = encoding.decode(
        plaintext_after_switch, after_switch_encoding_params
    )

    self.assertEqual(cleartext, cleartext_after_switch)

    # Decrypt the original ciphertext without removing the noise so we can
    # compare that to the noise introduced by the modulus switch.
    noise_before = encoding.extract_noise(
        lwe.decrypt_without_denoising(ciphertext, key), encoding_params
    )
    noise_after = encoding.extract_noise(
        plaintext_after_switch, after_switch_encoding_params
    )

    # Bound the noise in a way that does not depend on the actual key data,
    # which can be used in other tests for assertions on parameter settings.
    # This bound depends on the uniform distribution on the secret key, and is
    # with high probability in lwe_dimension.
    # TODO(b/242187167): improve this bound to sqrt(n)
    n = lwe_dim
    divisor = 2 ** (log_input_modulus - log_output_modulus)
    key_independent_noise_upper_bound = math.sqrt(n * math.log(math.log(n)))

    self.assertEqual(injected_noise, noise_before)
    self.assertLess(
        abs(noise_after),
        abs(noise_before) / divisor + key_independent_noise_upper_bound,
    )


if __name__ == '__main__':
  absltest.main()
