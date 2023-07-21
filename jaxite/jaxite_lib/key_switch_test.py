"""Tests for key_switch."""
import jax.numpy as jnp
from jaxite.jaxite_lib import decomposition
from jaxite.jaxite_lib import encoding
from jaxite.jaxite_lib import key_switch
from jaxite.jaxite_lib import lwe
from jaxite.jaxite_lib import parameters
from jaxite.jaxite_lib import random_source
from jaxite.jaxite_lib import rlwe
from jaxite.jaxite_lib import test_utils
from absl.testing import absltest
from absl.testing import parameterized

ZERO_RNG = random_source.ZeroRng()
A_I_BOUNDS = [2**i for i in range(33)]


class KeySwitchTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.dim = 5
    self.plaintext_modulus = 2**32
    self.decomposition_params = decomposition.DecompositionParameters(
        log_base=8, level_count=4
    )
    self.polynomial_modulus_degree = 4
    self.rlwe_dimension = 2
    self.in_params = parameters.SchemeParameters(
        plaintext_modulus=2**32,
        lwe_dimension=20,
        rlwe_dimension=2,
        polynomial_modulus_degree=self.polynomial_modulus_degree,
    )
    self.out_params = parameters.SchemeParameters(
        plaintext_modulus=2**32,
        lwe_dimension=5,
        rlwe_dimension=2,
        polynomial_modulus_degree=self.polynomial_modulus_degree,
    )

  def test_key_switch_gen(self):
    rng = random_source.PseudorandomSource(
        uniform_bounds=(0, 1), normal_std=0, seed=1
    )
    in_key = lwe.gen_key(self.in_params, rng)
    out_key = lwe.gen_key(self.out_params, rng)
    ksk = key_switch.gen_key(self.decomposition_params, rng, in_key, out_key)
    self.assertEqual(
        (
            in_key.lwe_dimension,
            self.decomposition_params.level_count,
            out_key.lwe_dimension + 1,
        ),
        ksk.key_data.shape,
    )

  @parameterized.product(
      ai_bound=A_I_BOUNDS,
      message=[0, 1, 2**10 - 1, 2**16, 2**31 - 1],
  )
  def test_switch_key_error_free(self, ai_bound: int, message: int):
    # a no-op encoding is used because the RNG is error-free.
    encoding_params = encoding.EncodingParameters(
        total_bit_length=32, message_bit_length=32, padding_bit_length=0
    )
    plaintext = encoding.encode(message, encoding_params)
    rng = random_source.PseudorandomSource(
        uniform_bounds=(0, ai_bound), normal_std=0, seed=1
    )
    in_key = lwe.gen_key(self.in_params, rng)
    out_key = lwe.gen_key(self.out_params, rng)
    ksk = key_switch.gen_key(self.decomposition_params, rng, in_key, out_key)
    in_ciphertext = lwe.encrypt(plaintext, in_key, prg=rng)
    out_ciphertext = key_switch.switch_key(ksk, in_ciphertext)
    decrypted = lwe.decrypt(
        out_ciphertext, out_key, encoding_params=encoding_params
    )
    self.assertEqual(plaintext, decrypted)

  @parameterized.product(
      ai_bound=A_I_BOUNDS,
      message=[0, 1, 2**10 - 1, 2**16],
  )
  def test_switch_key_with_error(self, ai_bound: int, message: int):
    encoding_params = encoding.EncodingParameters(
        total_bit_length=32, message_bit_length=17, padding_bit_length=2
    )
    plaintext = encoding.encode(message, encoding_params)
    rng = random_source.PseudorandomSource(
        uniform_bounds=(0, ai_bound), normal_std=4, seed=1
    )
    in_key = lwe.gen_key(self.in_params, rng)
    out_key = lwe.gen_key(self.out_params, rng)
    ksk = key_switch.gen_key(self.decomposition_params, rng, in_key, out_key)
    in_ciphertext = lwe.encrypt(plaintext, in_key, prg=rng)
    out_ciphertext = key_switch.switch_key(ksk, in_ciphertext)
    decrypted = lwe.decrypt(
        out_ciphertext, out_key, encoding_params=encoding_params
    )
    self.assertEqual(plaintext, decrypted)

  @parameterized.parameters(1, 4, 7)
  def test_key_switch_128_bit_security(self, message: int):
    encoding_params = encoding.EncodingParameters(
        total_bit_length=32,
        message_bit_length=3,
        padding_bit_length=1,
    )

    scheme_params = parameters.SchemeParameters(
        plaintext_modulus=2**32,
        lwe_dimension=630,
        rlwe_dimension=1,
        polynomial_modulus_degree=1024,
    )
    ksk_decomp_params = decomposition.DecompositionParameters(
        log_base=4,
        level_count=8,
    )

    plaintext = encoding.encode(message, encoding_params)
    lwe_rng = test_utils.LWE_RNG_128_BIT_SECURITY
    rlwe_rng = test_utils.RLWE_RNG_128_BIT_SECURITY
    lwe_key = lwe.gen_key(params=scheme_params, prg=lwe_rng)
    rlwe_key = rlwe.gen_key(params=scheme_params, prg=rlwe_rng)

    in_key = rlwe.flatten_key(rlwe_key)
    out_key = lwe_key

    ksk = key_switch.gen_key(
        in_key=in_key,
        out_key=out_key,
        decomposition_params=ksk_decomp_params,
        prg=lwe_rng,
    )

    in_ciphertext = lwe.encrypt(plaintext, in_key, prg=lwe_rng)
    injected_noise = 2 ** (21) - 1
    added_noise = jnp.array(
        [0] * in_key.lwe_dimension + [injected_noise], dtype=jnp.uint32
    )
    in_ciphertext = in_ciphertext + added_noise

    out_ciphertext = key_switch.switch_key(ksk, in_ciphertext)
    decrypted = lwe.decrypt(
        out_ciphertext, out_key, encoding_params=encoding_params
    )

    initial_noise = encoding.extract_noise(
        lwe.decrypt_without_denoising(in_ciphertext, in_key), encoding_params
    )
    plaintext_post_key_switch = lwe.decrypt_without_denoising(
        out_ciphertext, out_key
    )
    cleartext_post_key_switch = encoding.decode(
        plaintext_post_key_switch, encoding_params
    )
    noise_post_key_switch = encoding.extract_noise(
        plaintext_post_key_switch, encoding_params
    )
    print(f"Original cleartext = {message}")
    print(f"Original ciphertext noise = {initial_noise}")
    print(f"Cleartext post key switch = {cleartext_post_key_switch}")
    print(f"Ciphertext noise post key switch = {noise_post_key_switch}")
    self.assertEqual(plaintext, decrypted)


if __name__ == "__main__":
  absltest.main()
