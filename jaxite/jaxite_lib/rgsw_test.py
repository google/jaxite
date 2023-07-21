"""Tests for RGSW."""
import hypothesis
from hypothesis import strategies
import jax.numpy as jnp
from jaxite.jaxite_lib import decomposition
from jaxite.jaxite_lib import parameters
from jaxite.jaxite_lib import random_source
from jaxite.jaxite_lib import rgsw
from jaxite.jaxite_lib import test_utils
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized


class RgswTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.dim = 10
    self.plaintext_modulus = 2**32
    self.decomposition_params = decomposition.DecompositionParameters(
        log_base=4, level_count=4
    )
    self.polynomial_modulus_degree = 8
    self.rlwe_dimension = 2
    self.noise_free_rng = random_source.CycleRng(const_normal_noise=0)
    self.default_key = rgsw.gen_key(
        params=parameters.SchemeParameters(
            plaintext_modulus=self.plaintext_modulus,
            lwe_dimension=self.dim,
            polynomial_modulus_degree=self.polynomial_modulus_degree,
            rlwe_dimension=self.rlwe_dimension,
        ),
        prg=self.noise_free_rng,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='single_coeff',
          coeffs=[[[3]]],
          expected='\n[\n  3 x^0,\n]\n',
      ),
      dict(
          testcase_name='many_coeffs',
          coeffs=[[[1, 0, 2, 3], [0, 0, 3, 4]]],
          expected='\n[\n  1 x^0 + 2 x^2 + 3 x^3,\t3 x^2 + 4 x^3,\n]\n',
      ),
  )
  def test_rgsw_ciphertext_str(self, coeffs: list[int], expected: str):
    # non-message params are not needed for pretty-printing, pick arbitrarily.
    ciphertext = rgsw.RgswCiphertext(
        modulus_degree=self.polynomial_modulus_degree,
        log_coefficient_modulus=4,
        message=jnp.array(coeffs),
    )
    self.assertEqual(expected, str(ciphertext))

  @hypothesis.settings(deadline=None)
  @hypothesis.given(plaintext=strategies.integers(min_value=0, max_value=15))
  def test_encrypt_dimension(self, plaintext: rgsw.RgswPlaintext):
    rgsw_plaintext = rgsw.RgswPlaintext(
        modulus_degree=self.polynomial_modulus_degree,
        message=jnp.uint32(plaintext),
    )
    ciphertext = rgsw.encrypt(
        rgsw_plaintext,
        self.default_key,
        decomposition_params=self.decomposition_params,
        prg=self.noise_free_rng,
    )

    self.assertLen(
        ciphertext.message,
        (self.rlwe_dimension + 1)
        * self.decomposition_params.level_count,
    )
    for message in ciphertext.message:
      self.assertLen(message, (self.rlwe_dimension + 1))

  @hypothesis.settings(deadline=None)
  @hypothesis.given(plaintext=strategies.integers(min_value=1, max_value=15))
  def test_encrypt_diagonal(self, plaintext: int):
    """Checks that the "diagonals" are properly set.

    For RGSW, the ciphertext should be Z + m * G^T. This means all rows of the
    ciphertext should be RLWE(0), with one entry modified. Check that this
    "diagonal" element has indeed been modified, by making sure it is larger
    than the decomposed message multiplied by the decomposition params
    (for that level), minus the range that the original RLWE(0) entry
    could be in.

    This check only works when you have small randomness for RLWE encryption.
    This test is a placeholder until we have more robust tests, such as internal
    and external multiplication tests, and RGSW decryption.

    Args:
      plaintext: an RGSW plaintext, which we are using to check correct RGSW
        encryption. We are ignoring the plaintext=0 case because it wouldn't
        result in any modification to the RLWE(0) encryption. The upper bound is
        B-1 which is 15 when log_base = 4.
    """

    rgsw_plaintext = rgsw.RgswPlaintext(
        modulus_degree=self.polynomial_modulus_degree,
        message=jnp.uint32(plaintext),
    )
    ciphertext = rgsw.encrypt(
        rgsw_plaintext,
        self.default_key,
        decomposition_params=self.decomposition_params,
        prg=self.noise_free_rng,
    )
    levels = self.decomposition_params.level_count
    log_base = self.decomposition_params.log_base
    k = self.default_key.key.rlwe_dimension

    for level in range(1, levels + 1):
      decomp_multiplier = 1 << (32 - log_base * level)
      for j in range(k + 1):
        row_idx = (level - 1) + j * levels
        modified_coeff = ciphertext.message[row_idx][j][0]
        self.assertLessEqual(
            np.uint32(decomp_multiplier - log_base * levels), modified_coeff
        )
        self.assertLessEqual(
            np.uint32(plaintext * decomp_multiplier - log_base * levels),
            modified_coeff,
        )

  @parameterized.parameters(*range(16))
  def test_encrypt_decrypt_no_noise(self, plaintext: int):
    """Checks that encrypt/decrypt works using noise-free RLWE samples.

    Args:
      plaintext: an RGSW plaintext, which we are using to check correct RGSW
        encryption. The uniform_bounds are [0, B) where B=16 when log_base=4.
    """
    rgsw_plaintext = rgsw.RgswPlaintext(
        modulus_degree=self.polynomial_modulus_degree,
        message=jnp.uint32(plaintext),
    )
    ciphertext = rgsw.encrypt(
        rgsw_plaintext,
        self.default_key,
        decomposition_params=self.decomposition_params,
        prg=self.noise_free_rng,
    )
    pt_guess = rgsw.decrypt(
        ciphertext=ciphertext,
        decomposition_params=self.decomposition_params,
        sk=self.default_key,
    )
    self.assertEqual(plaintext, pt_guess.message)

  @parameterized.product(
      log_ai_bound=list(range(0, 33)),
      normal_std=[0, 1],
      seed=list(range(5)),
  )
  def test_encrypt_decrypt_with_varying_rngs(
      self, log_ai_bound: int, normal_std: int, seed: int
  ):
    """Checks that encrypt/decrypt works with a variety of RNG parameters."""
    rng = random_source.PseudorandomSource(
        uniform_bounds=(0, 2**log_ai_bound), normal_std=normal_std, seed=seed
    )
    plaintext = 1
    rgsw_key = rgsw.gen_key(
        params=parameters.SchemeParameters(
            plaintext_modulus=self.plaintext_modulus,
            lwe_dimension=self.dim,  # unused in this test
            polynomial_modulus_degree=self.polynomial_modulus_degree,
            rlwe_dimension=self.rlwe_dimension,
        ),
        prg=rng,
    )

    # Force the first index of the first polynomial to have a 1.
    # Other tests will check that other polynomials work.
    rgsw_key.key.data = rgsw_key.key.data.at[0, 0].set(1)

    rgsw_plaintext = rgsw.RgswPlaintext(
        modulus_degree=self.polynomial_modulus_degree,
        message=jnp.uint32(plaintext),
    )
    ciphertext = rgsw.encrypt(
        rgsw_plaintext,
        rgsw_key,
        decomposition_params=self.decomposition_params,
        prg=rng,
    )
    pt_guess = rgsw.decrypt(
        ciphertext=ciphertext,
        decomposition_params=self.decomposition_params,
        sk=rgsw_key,
    )
    self.assertEqual(plaintext, pt_guess.message)

  @parameterized.named_parameters(
      dict(testcase_name='sk_index_0', sk_nonzero_index=0),
      dict(testcase_name='sk_index_1', sk_nonzero_index=1),
      dict(testcase_name='sk_index_2', sk_nonzero_index=2),
      dict(testcase_name='sk_index_3', sk_nonzero_index=3),
      dict(testcase_name='sk_index_4', sk_nonzero_index=4),
  )
  def test_encrypt_decrypt_with_varying_nonzero_sk_entry(
      self, sk_nonzero_index
  ):
    """Checks that an encrypt/decrypt round works with different secret keys.

    Args:
      sk_nonzero_index: which secret key entry has a constant term of 1
    """
    # this dimension must be the same as the parameterized test range upper
    # bound, to ensure that the chosen index for a nonzero constant term is not
    # larger than the size of the secret key.
    rlwe_dimension = 5

    rng = random_source.PseudorandomSource(
        uniform_bounds=(0, 2**32), normal_std=1, seed=1
    )
    plaintext = 1
    rgsw_key = rgsw.gen_key(
        params=parameters.SchemeParameters(
            plaintext_modulus=self.plaintext_modulus,
            lwe_dimension=self.dim,  # unused in this test
            polynomial_modulus_degree=self.polynomial_modulus_degree,
            rlwe_dimension=rlwe_dimension,
        ),
        prg=rng,
    )

    # Force the first index of the chosen polynomial to have a 1.
    # All other constant terms are set to zero.
    for i in range(rlwe_dimension):
      value = 1 if i == sk_nonzero_index else 0
      rgsw_key.key.data = rgsw_key.key.data.at[i, 0].set(value)

    rgsw_plaintext = rgsw.RgswPlaintext(
        modulus_degree=self.polynomial_modulus_degree,
        message=jnp.uint32(plaintext),
    )
    ciphertext = rgsw.encrypt(
        rgsw_plaintext,
        rgsw_key,
        decomposition_params=self.decomposition_params,
        prg=rng,
    )
    pt_guess = rgsw.decrypt(
        ciphertext=ciphertext,
        decomposition_params=self.decomposition_params,
        sk=rgsw_key,
    )
    self.assertEqual(plaintext, pt_guess.message)

  def test_encrypt_decrypt_prod_security_params(self):
    rng = test_utils.RLWE_RNG_128_BIT_SECURITY
    plaintext = 13
    rgsw_key = rgsw.gen_key(
        params=test_utils.SCHEME_PARAMS_128_BIT_SECURITY, prg=rng
    )

    # Force the first index of the first polynomial to have a 1.
    # Other tests will check that other polynomials work.
    rgsw_key.key.data = rgsw_key.key.data.at[0, 0].set(1)

    rgsw_plaintext = rgsw.RgswPlaintext(
        modulus_degree=test_utils.SCHEME_PARAMS_128_BIT_SECURITY.polynomial_modulus_degree,
        message=jnp.uint32(plaintext),
    )
    ciphertext = rgsw.encrypt(
        rgsw_plaintext,
        rgsw_key,
        decomposition_params=self.decomposition_params,
        prg=rng,
    )
    pt_guess = rgsw.decrypt(
        ciphertext=ciphertext,
        decomposition_params=self.decomposition_params,
        sk=rgsw_key,
    )
    self.assertEqual(plaintext, pt_guess.message)


if __name__ == '__main__':
  absltest.main()
