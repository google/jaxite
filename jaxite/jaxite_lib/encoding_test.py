"""Tests for encoding and decoding logic."""

import hypothesis
from hypothesis import strategies
from jax import numpy as jnp
from jaxite.jaxite_lib import encoding
from jaxite.jaxite_lib import types
from absl.testing import absltest
from absl.testing import parameterized


class EncodingParametersTest(absltest.TestCase):
  """Tests encoding.EncodingParameters initialization."""

  def test_invalid_total_bit_length_raises(self):
    with self.assertRaises(ValueError):
      _ = encoding.EncodingParameters(
          total_bit_length=0, message_bit_length=4, padding_bit_length=2
      )

  def test_invalid_message_bit_length_raises(self):
    with self.assertRaises(ValueError):
      _ = encoding.EncodingParameters(
          total_bit_length=8, message_bit_length=0, padding_bit_length=2
      )

  def test_invalid_padding_bit_length_raises(self):
    with self.assertRaises(ValueError):
      _ = encoding.EncodingParameters(
          total_bit_length=8, message_bit_length=4, padding_bit_length=-1
      )

  def test_unavailable_space_raises(self):
    with self.assertRaises(ValueError):
      _ = encoding.EncodingParameters(
          total_bit_length=8, message_bit_length=7, padding_bit_length=2
      )


class EncodingDecodingTest(parameterized.TestCase):
  """Exercises encoding and decoding logic."""

  @parameterized.named_parameters(
      dict(testcase_name='_with_min', cleartext=0),
      dict(testcase_name='_with_max', cleartext=2**16 - 1),
  )
  def test_encode_decode_message_bounds_succeeds(
      self, cleartext: types.LweCleartext
  ):
    """Tests encoding+decoding produces exactly the input cleartext."""
    params = encoding.EncodingParameters(
        total_bit_length=32, message_bit_length=16, padding_bit_length=0
    )

    encoded: types.LwePlaintext = encoding.encode(cleartext, params)
    decoded: types.LweCleartext = encoding.decode(encoded, params)
    self.assertEqual(decoded, cleartext)

  @hypothesis.given(strategies.integers(min_value=0, max_value=2**16 - 1))
  @hypothesis.settings(deadline=None)
  def test_encode_decode_succeeds(self, cleartext: types.LweCleartext):
    params = encoding.EncodingParameters(
        total_bit_length=32, message_bit_length=16, padding_bit_length=8
    )

    encoded: types.LwePlaintext = encoding.encode(cleartext, params)
    decoded: types.LweCleartext = encoding.decode(encoded, params)
    self.assertEqual(decoded, cleartext)

  @hypothesis.given(
      strategies.integers(min_value=0, max_value=2**24 - 1),
      # max_value is the max allowed noise before it corrupts the message
      strategies.integers(min_value=0, max_value=2**5 - 1),
  )
  @hypothesis.settings(deadline=None)
  def test_encode_add_noise_decode_succeeds(
      self, cleartext: types.LweCleartext, noise: int
  ):
    params = encoding.EncodingParameters(
        total_bit_length=32, message_bit_length=24, padding_bit_length=2
    )
    encoded = encoding.encode(cleartext, params)
    noisy_encoded = encoded | noise
    decoded = encoding.decode(noisy_encoded, params)
    self.assertEqual(cleartext, decoded)

  @parameterized.named_parameters(
      dict(testcase_name='_with_greater_than_max', cleartext=2**8),
      dict(testcase_name='_with_less_than_min', cleartext=-1),
  )
  def test_encode_invalid_message_raises(self, cleartext: types.LweCleartext):
    params = encoding.EncodingParameters(
        total_bit_length=16, message_bit_length=8, padding_bit_length=4
    )

    with self.assertRaises(ValueError):
      _ = encoding.encode(cleartext, params)

  @hypothesis.given(
      strategies.integers(min_value=0, max_value=2**24 - 1),
      # max_value is the max allowed noise before it corrupts the message
      strategies.integers(min_value=0, max_value=2**5 - 1),
  )
  @hypothesis.settings(deadline=None)
  def test_encode_add_noise_extract_noise(
      self, cleartext: types.LweCleartext, noise: int
  ):
    params = encoding.EncodingParameters(
        total_bit_length=32, message_bit_length=24, padding_bit_length=2
    )
    encoded = encoding.encode(cleartext, params)
    noisy_encoded = encoded | noise
    extracted_noise = encoding.extract_noise(noisy_encoded, params)
    self.assertEqual(noise, extracted_noise)

  def test_encode_test_polynomial_coefficients(self):
    # needs three bits
    coefficients = jnp.arange(7, dtype=jnp.uint32)
    params = encoding.EncodingParameters(
        total_bit_length=32, message_bit_length=3, padding_bit_length=1
    )

    encoded = encoding.encode(coefficients, params)
    self.assertEqual(list(encoded), list(coefficients << 28))

  def test_encode_test_polynomial_coefficients_too_large(self):
    # needs 4 bits, and not allowed to use padding
    coefficients = jnp.arange(10, dtype=jnp.uint32)
    params = encoding.EncodingParameters(
        total_bit_length=32, message_bit_length=3, padding_bit_length=1
    )

    with self.assertRaises(ValueError):
      encoding.encode(coefficients, params)


if __name__ == '__main__':
  absltest.main()
