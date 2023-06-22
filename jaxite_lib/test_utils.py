"""Test utilities."""

import itertools
import math
from typing import Optional

import jax.numpy as jnp
from jaxite.jaxite_bool import bool_params
from jaxite.jaxite_lib import encoding
from jaxite.jaxite_lib import lwe
from jaxite.jaxite_lib import parameters
from jaxite.jaxite_lib import rlwe

DEFAULT_ENCODING_PARAMS = encoding.EncodingParameters(
    total_bit_length=26, message_bit_length=14, padding_bit_length=2
)
ENCODING_PARAMS_128_BIT_SECURITY = encoding.EncodingParameters(
    total_bit_length=32, message_bit_length=6, padding_bit_length=1
)

LWE_RNG_128_BIT_SECURITY = bool_params.get_lwe_rng_for_128_bit_security(2)
RLWE_RNG_128_BIT_SECURITY = bool_params.get_rlwe_rng_for_128_bit_security(2)
SCHEME_PARAMS_128_BIT_SECURITY = bool_params.SCHEME_PARAMS_128_BIT_SECURITY

BSK_DECOMP_PARAMS_128_BIT_SECURITY = (
    bool_params.BSK_DECOMP_PARAMS_128_BIT_SECURITY
)
KSK_DECOMP_PARAMS_128_BIT_SECURITY = (
    bool_params.KSK_DECOMP_PARAMS_128_BIT_SECURITY
)


class ConsistencyChecker:
  """A helper for checking consistency of intermediate values across runs."""

  def __init__(self):
    self.insertion_order = []
    self.records = dict()

  def store(self, name: str, value: jnp.ndarray) -> None:
    value = value.tolist()
    if name in self.records:
      self.records[name].append(value)
    else:
      self.insertion_order.append(name)
      self.records[name] = [value]

  def check(self, asserter: ... = None) -> None:
    assert asserter is not None
    for name in self.insertion_order:
      print(f"Consistency check for {name}")
      values = self.records[name]
      for x, y in itertools.combinations(values, 2):
        asserter(x, y)


def assert_safe_modulus_switch(
    mod_degree: int, message_bits: int, lwe_dimension: int
):
  """Guard against corruption of the message due to modulus switch error.

  The ratio of mod_degree to 2**message_bits defines the size of a block in the
  test polynomial. Half that is the radius of the block, and that defines the
  error allowed in the post-modulus-switched ciphertext before it corrupts the
  message. Modulus switch can add up to sqrt(lwe_dim * log(log(lwe_dim))) error,
  and this implies a limit on how large lwe_dim can be before mod_degree must
  increase.

  Args:
    mod_degree: the degree of the RLWE polynomial modulus
    message_bits: the log2 of the size of the message space.
    lwe_dimension: the number of samples used in the LWE ciphertexts.

  Returns:
    None. Assertion failure if params are set improperly.
  """
  test_poly_block_radius = 0.5 * mod_degree / 2**message_bits
  # add one because the rounding of the `b` term also introduces error while
  # lwe_dimension tracks the number of samples only.
  n = 1 + lwe_dimension
  # TODO(b/242187167): show that the bounds can be tightened to sqrt(n).
  mod_switch_extra_error = math.sqrt(n)
  assert mod_switch_extra_error <= test_poly_block_radius, (
      f"Modulus switch error may be as large as {mod_switch_extra_error}, "
      "but the test polynomial repetition blocks "
      f"only have radius {test_poly_block_radius}. "
      "This risks the error from modulus switch corrupting the message. "
      f"(mod_degree={mod_degree}, lwe_dim={lwe_dimension}, "
      f"message_space={2**message_bits})"
  )


class MidBootstrapDecrypter:
  """A helper for decrypting intermediate encrypted values during bootstrap.

  Pass a reference to MidBootstrapDecrypter.decrypt to `bootstrap` as the
  `callback` argument.
  """

  def __init__(
      self,
      scheme_params: parameters.SchemeParameters,
      encoding_params: encoding.EncodingParameters,
      lwe_key: lwe.LweSecretKey,
      rlwe_key: rlwe.RlweSecretKey,
  ):
    self.scheme_params = scheme_params
    self.encoding_params = encoding_params
    self.lwe_key = lwe_key
    self.rlwe_key = rlwe_key
    self.starting_cleartext = None

  def _noise_and_bits(self, noise) -> str:
    # A helper to format the noise and its log magnitude. Noise may be zero, in
    # which case treat its log as 1 to avoid domain error.
    abs_bits = math.log2(max(1, abs(noise)))
    return f"{noise} ({abs_bits} bits)"

  def decrypt(
      self,
      name: str,
      value: jnp.ndarray,
      callback_lut: Optional[list[int]] = None,
  ) -> None:
    """Decrypt an intermediate value of bootstrap and report noise growth."""
    lwe_dim = self.scheme_params.lwe_dimension

    if name == "initial":
      ciphertext = value
      plaintext = lwe.decrypt(
          ciphertext,
          self.lwe_key,
          encoding_params=self.encoding_params,
      )
      initial_noise = encoding.extract_noise(
          lwe.decrypt_without_denoising(ciphertext, self.lwe_key),
          self.encoding_params,
      )
      cleartext = encoding.decode(plaintext, self.encoding_params)
      self.starting_cleartext = cleartext
      print(f"Cleartext at bootstrap start = {cleartext}")
      print(
          "Ciphertext noise at bootstrap start = "
          + self._noise_and_bits(initial_noise)
      )

    if name == "approx_ciphertext":
      modulus_switched = value
      log_output_modulus = self.scheme_params.log_mod_degree + 1
      post_mod_switch_key = lwe.LweSecretKey(
          log_modulus=log_output_modulus,
          lwe_dimension=lwe_dim,
          key_data=self.lwe_key.key_data,
      )
      post_mod_switch_encoding_params = encoding.EncodingParameters(
          total_bit_length=log_output_modulus,
          message_bit_length=self.encoding_params.message_bit_length,
          padding_bit_length=self.encoding_params.padding_bit_length,
      )
      plaintext_post_mod_switch = lwe.decrypt_without_denoising(
          modulus_switched, post_mod_switch_key
      )
      cleartext_post_mod_switch = encoding.decode(
          plaintext_post_mod_switch, post_mod_switch_encoding_params
      )
      noise_after = encoding.extract_noise(
          plaintext_post_mod_switch, post_mod_switch_encoding_params
      )
      print(f"Cleartext post modulus switch = {cleartext_post_mod_switch}")
      print(
          "Ciphertext noise post modulus switch = "
          + self._noise_and_bits(noise_after)
      )
      if self.starting_cleartext != cleartext_post_mod_switch:
        print(
            f"DECRYPTION_FAILURE: expected {self.starting_cleartext} "
            f"but got {cleartext_post_mod_switch}"
        )

    if name == "rotated":
      rotated_rlwe_ciphertext = rlwe.RlweCiphertext(
          log_coefficient_modulus=self.rlwe_key.log_coefficient_modulus,
          modulus_degree=self.rlwe_key.modulus_degree,
          message=value,
      )
      rotated_plaintext = rlwe.decrypt(
          ciphertext=rotated_rlwe_ciphertext,
          sk=self.rlwe_key,
          encoding_params=self.encoding_params,
      )
      rotated_decoded = [
          encoding.decode(x, self.encoding_params)
          for x in rotated_plaintext.message
      ]
      print(f"Cleartext rotated polynomial = {rotated_decoded}")

    if name == "extracted":
      extracted = value
      reshaped_key = rlwe.flatten_key(self.rlwe_key)
      plaintext_post_sample_extract = lwe.decrypt_without_denoising(
          extracted, reshaped_key
      )
      cleartext_post_sample_extract = encoding.decode(
          plaintext_post_sample_extract, self.encoding_params
      )
      noise_post_sample_extract = encoding.extract_noise(
          plaintext_post_sample_extract, self.encoding_params
      )
      print(f"Cleartext post sample extract = {cleartext_post_sample_extract}")
      print(
          "Ciphertext noise post sample extract = "
          + self._noise_and_bits(noise_post_sample_extract)
      )

      expected_cleartext = self.starting_cleartext
      if callback_lut:
        expected_cleartext = callback_lut[self.starting_cleartext]
      if expected_cleartext != cleartext_post_sample_extract:
        print(
            f"DECRYPTION_FAILURE: expected {expected_cleartext} "
            f"but got {cleartext_post_sample_extract}. "
        )
        if callback_lut:
          print(
              f"Starting cleartext was {self.starting_cleartext} "
              f"and LUT was {callback_lut}."
          )

    if name == "key_switched":
      key_switched = value
      plaintext_post_key_switch = lwe.decrypt_without_denoising(
          key_switched, self.lwe_key
      )
      cleartext_post_key_switch = encoding.decode(
          plaintext_post_key_switch, self.encoding_params
      )
      noise_post_key_switch = encoding.extract_noise(
          plaintext_post_key_switch, self.encoding_params
      )
      print(f"Cleartext post key switch = {cleartext_post_key_switch}")
      print(
          "Ciphertext noise post key switch = "
          + self._noise_and_bits(noise_post_key_switch)
      )

      expected_cleartext = self.starting_cleartext
      if callback_lut:
        expected_cleartext = callback_lut[self.starting_cleartext]
      if expected_cleartext != cleartext_post_key_switch:
        print(
            f"DECRYPTION_FAILURE: expected {expected_cleartext} "
            f"but got {cleartext_post_key_switch}."
        )
        if callback_lut:
          print(
              f"Starting cleartext was {self.starting_cleartext} "
              f"and LUT was {callback_lut}."
          )
