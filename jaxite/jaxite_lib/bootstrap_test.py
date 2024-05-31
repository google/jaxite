"""Tests for bootstrap."""

from typing import Any, Callable

import jax.numpy as jnp
from jaxite.jaxite_lib import bootstrap
from jaxite.jaxite_lib import encoding
from jaxite.jaxite_lib import key_switch
from jaxite.jaxite_lib import lwe
from jaxite.jaxite_lib import parameters
from jaxite.jaxite_lib import random_source
from jaxite.jaxite_lib import rgsw
from jaxite.jaxite_lib import rlwe
from jaxite.jaxite_lib import test_polynomial
from jaxite.jaxite_lib import test_utils

from absl.testing import absltest
from absl.testing import parameterized

ZERO_RNG = random_source.ZeroRng()
_LOG_AI_BOUNDS = (0, 16, 32)
_SEEDS = (1, 2, 3)


class BootstrapBaseTest(parameterized.TestCase):
  """A base class for running bootstrap tests."""

  def __init__(self, *args, **kwargs):
    super(BootstrapBaseTest, self).__init__(*args, **kwargs)
    self.callback: Callable[[str, Any], None] = None

  def run_bootstrap_test(
      self,
      *,
      injected_noise: int,
      lwe_dimension: int,
      lwe_rng: random_source.RandomSource,
      message_bits: int,
      mod_degree: int,
      padding_bits: int,
      rlwe_rng: random_source.RandomSource,
      skip_assert: bool = False,
  ):
    cleartext = 2**message_bits - 1
    test_utils.assert_safe_modulus_switch(
        mod_degree, message_bits, lwe_dimension
    )

    plaintext_modulus = 2**32
    rlwe_dimension = 1
    scheme_parameters = parameters.SchemeParameters(
        plaintext_modulus=plaintext_modulus,
        lwe_dimension=lwe_dimension,
        rlwe_dimension=rlwe_dimension,
        polynomial_modulus_degree=mod_degree,
    )

    noisy_encoding = encoding.EncodingParameters(
        total_bit_length=32,
        message_bit_length=message_bits,
        padding_bit_length=padding_bits,
    )

    plaintext = encoding.encode(cleartext, noisy_encoding)
    lwe_key = lwe.gen_key(params=scheme_parameters, prg=lwe_rng)
    rlwe_key = rlwe.gen_key(params=scheme_parameters, prg=rlwe_rng)
    rgsw_key = rgsw.key_from_rlwe(rlwe_key)

    ciphertext = lwe.encrypt(plaintext, lwe_key, prg=lwe_rng)
    added_noise = jnp.array(
        [0] * lwe_key.lwe_dimension + [injected_noise], dtype=jnp.uint32
    )
    noisy_ciphertext = ciphertext + added_noise

    bsk = bootstrap.gen_bootstrapping_key(
        lwe_sk=lwe_key,
        rgsw_sk=rgsw_key,
        decomposition_params=test_utils.BSK_DECOMP_PARAMS_128_BIT_SECURITY,
        prg=rlwe_rng,
    )
    ksk = key_switch.gen_key(
        in_key=rlwe.flatten_key(rlwe_key),
        out_key=lwe_key,
        decomposition_params=test_utils.KSK_DECOMP_PARAMS_128_BIT_SECURITY,
        prg=lwe_rng,
    )

    if not self.callback:
      self.callback = test_utils.MidBootstrapDecrypter(
          scheme_params=scheme_parameters,
          encoding_params=noisy_encoding,
          lwe_key=lwe_key,
          rlwe_key=rlwe_key,
      ).decrypt

    # Construct the test polynomial
    test_poly = test_polynomial.identity_test_polynomial(
        noisy_encoding, scheme_parameters
    )
    test_poly_ciphertext = test_polynomial.trivial_encryption(
        test_poly, scheme_parameters
    )

    bootstrapped = bootstrap.bootstrap(
        noisy_ciphertext,
        test_poly_ciphertext,
        bsk,
        ksk,
        test_utils.KSK_DECOMP_PARAMS_128_BIT_SECURITY,
        test_utils.BSK_DECOMP_PARAMS_128_BIT_SECURITY,
        scheme_parameters,
        callback=self.callback,
    )

    jit_bootstrapped = bootstrap.jit_bootstrap(
        noisy_ciphertext,
        test_poly_ciphertext.message,
        bsk.encrypted_lwe_sk_bits,
        ksk.key_data,
        test_utils.KSK_DECOMP_PARAMS_128_BIT_SECURITY,
        test_utils.BSK_DECOMP_PARAMS_128_BIT_SECURITY,
        scheme_parameters,
    )

    self.assertEqual(len(jit_bootstrapped), len(bootstrapped))
    for i in range(len(jit_bootstrapped)):
      self.assertEqual(jit_bootstrapped[i], bootstrapped[i])

    decrypted = lwe.decrypt(
        bootstrapped, lwe_key, encoding_params=noisy_encoding
    )
    actual_cleartext = encoding.decode(decrypted, noisy_encoding)
    if skip_assert:
      return

    self.assertEqual(actual_cleartext, cleartext)

    if injected_noise:
      remaining_noise = encoding.extract_noise(
          lwe.decrypt_without_denoising(bootstrapped, lwe_key), noisy_encoding
      )

      if rlwe_rng == ZERO_RNG:
        self.assertEqual(remaining_noise, 0)
      else:
        self.assertLess(abs(remaining_noise), 3 * injected_noise)


@parameterized.product(
    log_ai_bound=_LOG_AI_BOUNDS,
    seed=_SEEDS,
)
class BootstrapTest(BootstrapBaseTest):

  def test_3_bit_bootstrap(self, log_ai_bound, seed):
    message_bits = 3
    padding_bits = 1
    lwe_dimension = 4
    mod_degree = 64

    rng = random_source.PseudorandomSource(
        uniform_bounds=(0, 2**log_ai_bound),
        normal_std=1,
        seed=seed,
    )
    injected_noise = 2 ** (32 - padding_bits - message_bits - 2) - 1

    self.run_bootstrap_test(
        injected_noise=injected_noise,
        lwe_dimension=lwe_dimension,
        lwe_rng=rng,
        message_bits=message_bits,
        mod_degree=mod_degree,
        padding_bits=padding_bits,
        rlwe_rng=rng,
    )

  def test_3_bit_bootstrap_larger_lwe_dimension(
      self, log_ai_bound: int, seed: int
  ):
    message_bits = 3
    padding_bits = 1
    lwe_dimension = 100
    mod_degree = 512
    # TODO(b/339715397): make the kernel work for degree 1024
    # mod_degree = 1024

    rng = random_source.PseudorandomSource(
        uniform_bounds=(0, 2**log_ai_bound),
        normal_std=1,
        seed=seed,
    )
    injected_noise = 2 ** (32 - padding_bits - message_bits - 3) - 1

    self.run_bootstrap_test(
        injected_noise=injected_noise,
        lwe_dimension=lwe_dimension,
        lwe_rng=rng,
        message_bits=message_bits,
        mod_degree=mod_degree,
        padding_bits=padding_bits,
        rlwe_rng=rng,
    )

  @absltest.skip("b/325287870")
  def test_6_bit_bootstrap(self, log_ai_bound: int, seed: int):
    message_bits = 6
    padding_bits = 1
    lwe_dimension = 30
    mod_degree = 2048

    rng = random_source.PseudorandomSource(
        uniform_bounds=(0, 2**log_ai_bound),
        normal_std=1,
        seed=seed,
    )
    injected_noise = 2 ** (32 - padding_bits - message_bits - 2) - 1

    self.run_bootstrap_test(
        injected_noise=injected_noise,
        lwe_dimension=lwe_dimension,
        lwe_rng=rng,
        message_bits=message_bits,
        mod_degree=mod_degree,
        padding_bits=padding_bits,
        rlwe_rng=rng,
    )

  def test_3_bit_bootstrap_prod_decomp_params(
      self, log_ai_bound: int, seed: int
  ):
    message_bits = 3
    padding_bits = 1
    lwe_dimension = 30
    mod_degree = 512

    rng = random_source.PseudorandomSource(
        uniform_bounds=(0, 2**log_ai_bound),
        normal_std=1,
        seed=seed,
    )
    injected_noise = 2 ** (32 - padding_bits - message_bits - 2) - 1

    self.run_bootstrap_test(
        injected_noise=injected_noise,
        lwe_dimension=lwe_dimension,
        lwe_rng=rng,
        message_bits=message_bits,
        mod_degree=mod_degree,
        padding_bits=padding_bits,
        rlwe_rng=rng,
    )


class ProdSecurityTest(BootstrapBaseTest):
  """Test 128-bit security parameters for bootstrap."""

  def test_1_bit_bootstrap_prod_params(self):
    message_bits = 1
    padding_bits = 1
    lwe_dimension = test_utils.SCHEME_PARAMS_128_BIT_SECURITY.lwe_dimension

    mod_degree = (
        test_utils.SCHEME_PARAMS_128_BIT_SECURITY.polynomial_modulus_degree
    )

    test_utils.assert_safe_modulus_switch(
        mod_degree, message_bits, lwe_dimension
    )

    lwe_rng = test_utils.LWE_RNG_128_BIT_SECURITY
    rlwe_rng = test_utils.RLWE_RNG_128_BIT_SECURITY
    injected_noise = 2 ** (32 - padding_bits - message_bits - 2) - 1

    self.run_bootstrap_test(
        injected_noise=injected_noise,
        lwe_dimension=lwe_dimension,
        lwe_rng=lwe_rng,
        message_bits=message_bits,
        mod_degree=mod_degree,
        padding_bits=padding_bits,
        rlwe_rng=rlwe_rng,
    )

  def test_3_bit_bootstrap_prod_params(self):
    message_bits = 3
    padding_bits = 1
    lwe_dimension = test_utils.SCHEME_PARAMS_128_BIT_SECURITY.lwe_dimension

    mod_degree = (
        test_utils.SCHEME_PARAMS_128_BIT_SECURITY.polynomial_modulus_degree
    )

    test_utils.assert_safe_modulus_switch(
        mod_degree, message_bits, lwe_dimension
    )

    lwe_rng = test_utils.LWE_RNG_128_BIT_SECURITY
    rlwe_rng = test_utils.RLWE_RNG_128_BIT_SECURITY
    injected_noise = 2 ** (32 - padding_bits - message_bits - 2) - 1

    self.run_bootstrap_test(
        injected_noise=injected_noise,
        lwe_dimension=lwe_dimension,
        lwe_rng=lwe_rng,
        message_bits=message_bits,
        mod_degree=mod_degree,
        padding_bits=padding_bits,
        rlwe_rng=rlwe_rng,
    )


class ConsistencyTest(BootstrapBaseTest):
  """A test suite to check for race conditions or other nondeterminism."""

  def test_3_bit_consistency(self):
    log_ai_bound = 32
    seed = 2
    message_bits = 3
    padding_bits = 1
    lwe_dimension = 50
    mod_degree = 1024
    injected_noise = 0

    assert mod_degree / message_bits >= lwe_dimension + 1

    consistency = test_utils.ConsistencyChecker()
    self.callback = consistency.store

    for _ in range(3):
      rng = random_source.PseudorandomSource(
          uniform_bounds=(0, 2**log_ai_bound),
          normal_std=0,
          seed=seed,
      )
      self.run_bootstrap_test(
          injected_noise=injected_noise,
          lwe_dimension=lwe_dimension,
          lwe_rng=rng,
          message_bits=message_bits,
          mod_degree=mod_degree,
          padding_bits=padding_bits,
          rlwe_rng=rng,
          skip_assert=True,
      )

    consistency.check(asserter=self.assertEqual)


if __name__ == "__main__":
  absltest.main()
