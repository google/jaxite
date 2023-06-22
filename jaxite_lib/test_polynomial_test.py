"""Tests for test_polynomial."""
import itertools
import math

import jax.numpy as jnp
from jaxite.jaxite_lib import encoding
from jaxite.jaxite_lib import parameters
from jaxite.jaxite_lib import test_polynomial
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

THREE_BIT_ENCODING = encoding.EncodingParameters(
    total_bit_length=32,
    message_bit_length=3,
    padding_bit_length=2,
)

DEGREE_32_POLY = parameters.SchemeParameters(
    plaintext_modulus=2**32,
    lwe_dimension=0,  # unused
    polynomial_modulus_degree=2**5,
    rlwe_dimension=0,  # unused
)


class TestPolynomialTest(parameterized.TestCase):

  def test_gen_polynomial_modded(self):
    encoding_params = encoding.EncodingParameters(
        total_bit_length=4,
        message_bit_length=2,
        padding_bit_length=2,
    )

    scheme_params = parameters.SchemeParameters(
        plaintext_modulus=2**32,
        lwe_dimension=0,  # unused
        polynomial_modulus_degree=8,
        rlwe_dimension=0,  # unused
    )

    coefficients = jnp.array([1, 1, 0, 0], dtype=jnp.uint32)
    expected_coefficients = jnp.array(
        [1, 1, 1, 0, 0, 0, 0, 3], dtype=jnp.uint32
    )

    test_poly = test_polynomial.gen_test_polynomial(
        coefficients, encoding_params, scheme_params
    )

    np.testing.assert_array_equal(expected_coefficients, test_poly.message)

  def test_manually_gen_identity_test_polynomial(self):
    encoding_params = THREE_BIT_ENCODING
    scheme_params = DEGREE_32_POLY

    test_poly = test_polynomial.gen_test_polynomial(
        jnp.arange(2**encoding_params.message_bit_length, dtype=jnp.uint32),
        encoding_params,
        scheme_params,
    )

    # pyformat: disable
    # yapf: disable
    expected_coeffs = jnp.array([
        0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4,
        5, 5, 5, 5,
        6, 6, 6, 6,
        7, 7, 7, 7,
        # the backwards roll by half of one padding block
        0, 0,
    ], dtype=jnp.uint32)
    # pyformat: enable
    # yapf: enable

    np.testing.assert_array_equal(
        expected_coeffs << encoding_params.error_bit_length, test_poly.message
    )

  def test_manually_gen_nonidentity_test_polynomial(self):
    encoding_params = THREE_BIT_ENCODING
    scheme_params = DEGREE_32_POLY

    coeffs = jnp.array([2, 1, 3, 4, 0, 0, 0, 0])
    test_poly = test_polynomial.gen_test_polynomial(
        coeffs, encoding_params, scheme_params
    )

    # pyformat: disable
    # yapf: disable
    expected_coeffs = jnp.array([
        2, 2,
        1, 1, 1, 1,
        3, 3, 3, 3,
        4, 4, 4, 4,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        6, 6,  # == -2 mod 8
    ], dtype=jnp.uint32)
    # pyformat: enable
    # yapf: enable

    np.testing.assert_array_equal(
        expected_coeffs << encoding_params.error_bit_length, test_poly.message
    )

  @parameterized.named_parameters(
      dict(testcase_name='too_small', coeffs=[1, 2, 3, 4]),
      dict(testcase_name='too_big', coeffs=range(32)),
  )
  def test_gen_test_polynomial_wrong_dims(self, coeffs):
    coeffs = jnp.array(coeffs, dtype=jnp.uint32)
    with self.assertRaises(ValueError):
      test_polynomial.gen_test_polynomial(
          coeffs, THREE_BIT_ENCODING, DEGREE_32_POLY
      )

  @parameterized.named_parameters(
      dict(testcase_name='small_cleartext_space', p=2**3, q=2**7, N=2**5),
      dict(
          testcase_name='medium_cleartext_space', p=2**6, q=2**9, N=2**7
      ),
      dict(
          testcase_name='large_cleartext_space', p=2**10, q=2**17, N=2**14
      ),
  )
  def test_gen_test_polynomial_parameterized(self, p, q, N):  # pylint: disable=invalid-name
    encoding_params = encoding.EncodingParameters(
        total_bit_length=int(math.log2(q)),
        message_bit_length=int(math.log2(p)),
        padding_bit_length=1,  # Leaves log2(q/p) - 1 bits of error
    )
    scheme_params = parameters.SchemeParameters(
        lwe_dimension=2,  # Unused
        plaintext_modulus=q,
        rlwe_dimension=2,  # Unused
        polynomial_modulus_degree=N,
    )

    test_poly = test_polynomial.gen_test_polynomial(
        jnp.arange(2**encoding_params.message_bit_length, dtype=jnp.uint32),
        encoding_params,
        scheme_params,
    )
    block_size = N // p

    layered_coeffs = [
        [i % p] * (block_size // 2 if i % p == 0 else block_size)
        for i in range(p + 1)
    ]
    coeffs = list(itertools.chain.from_iterable(layered_coeffs))
    expected_coeffs = jnp.array(coeffs, dtype=jnp.uint32)
    np.testing.assert_array_equal(
        expected_coeffs << encoding_params.error_bit_length, test_poly.message
    )


if __name__ == '__main__':
  absltest.main()
