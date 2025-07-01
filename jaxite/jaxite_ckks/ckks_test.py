"""Basic tests for CKKS.

CPU-based tests:
- Encode & decode, with additive and multiplicative homomorphisms;
- Encrypt & decrypt;

TPU-based tests:
- Homomorphic add and subtraction.
"""

from concurrent import futures
import random

import jax
from jaxite.jaxite_ckks import ckks
from jaxite.jaxite_ckks import rns
from jaxite.jaxite_word import add
from jaxite.jaxite_word import sub
import parameterized

from absl.testing import absltest
from absl.testing import parameterized as parameterized_test


ProcessPoolExecutor = futures.ProcessPoolExecutor

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_traceback_filtering', 'off')


@parameterized.parameterized_class([
    # The followings are toy parameters that should only be used for testing.
    # In general, we instantiate the CKKS scheme with parameters defining
    # the ring R_Q = Z[X] / (Q, X^N+1), a scaling factor used in the canonical
    # embedding encoding, and bit precisions that should be achieved in the
    # encoding and encryption.
    {
        'degree': 8,
        'moduli': [335552513],
        'scaling_factor': 2**9,
        'encoding_precision': 5,
        'encryption_precision': 3,
    },
    {
        'degree': 16,
        'moduli': [335552513, 65537],
        'scaling_factor': 2**16,
        'encoding_precision': 10,
        'encryption_precision': 8,
    },
    {
        'degree': 1024,
        'moduli': [335552513, 65537],
        'scaling_factor': 2**16,
        'encoding_precision': 7,
        'encryption_precision': 5,
    },
])
class CkksTest(parameterized_test.TestCase):

  def setUp(self):
    super().setUp()
    self.rns_params = rns.RnsParams(self.degree, self.moduli)
    self.ntt_params = self.rns_params.ntt_params

  def _random_slots(self, degree: int) -> list[complex]:
    """Generate a list of random complex numbers having norm <= 1."""
    rand_complex = lambda: complex(random.uniform(-1, 1), random.uniform(-1, 1))
    return [rand_complex() for _ in range(degree >> 1)]

  def _compare_with_precision(
      self, a: list[complex], b: list[complex], bit_precision: int = 10
  ):
    """Check if two lists of numbers are close componentwise."""
    assert len(a) == len(b)
    percision_bound = pow(2, -bit_precision)
    for i in range(len(a)):
      assert abs(a[i] - b[i]) < percision_bound, (
          f'bad precision: a[{i}] = {a[i]}, b[{i}] = {b[i]}, expected precision'
          f' = {bit_precision} bits'
      )

  def test_encoder(self):
    """Encode and then decode should not lose too much precision."""
    encoder = ckks.CkksEncoder(
        degree=self.degree,
        moduli=self.moduli,
        scaling_factor=self.scaling_factor,
    )
    slots = self._random_slots(self.degree)
    plaintext = encoder.encode(slots)
    self.assertEqual(plaintext.degree, self.degree)
    self.assertEqual(plaintext.moduli, self.moduli)

    decoded = encoder.decode(plaintext)
    self._compare_with_precision(
        slots, decoded, bit_precision=self.encoding_precision
    )

  def test_encoding_additive_homomorphism(self):
    """The encoding scheme should be approximately additive."""
    encoder = ckks.CkksEncoder(
        degree=self.degree,
        moduli=self.moduli,
        scaling_factor=self.scaling_factor,
    )
    slots0 = self._random_slots(self.degree)
    slots1 = self._random_slots(self.degree)
    poly0 = encoder.encode(slots0)
    poly1 = encoder.encode(slots1)
    poly_sum = poly0 + poly1
    decoded = encoder.decode(poly_sum)
    expected = [slots0[i] + slots1[i] for i in range(self.degree >> 1)]
    self._compare_with_precision(
        expected, decoded, bit_precision=self.encoding_precision
    )

  def test_encoding_multiplicative_homomorphism(self):
    """The encoding scheme should be approximately multiplicative (slotwise)."""
    encoder = ckks.CkksEncoder(
        degree=self.degree,
        moduli=self.moduli,
        scaling_factor=self.scaling_factor,
    )
    slots0 = self._random_slots(self.degree)
    slots1 = self._random_slots(self.degree)
    poly0 = encoder.encode(slots0)
    poly1 = encoder.encode(slots1)
    # Make sure the polynomials are in the NTT form before multiplication.
    poly0.to_ntt_form(self.ntt_params)
    poly1.to_ntt_form(self.ntt_params)

    poly_prod = poly0 * poly1
    # Convert the product to the coefficient form before decoding.
    poly_prod.to_coeffs_form(self.ntt_params)
    decoded = encoder.decode(poly_prod)
    assert len(decoded) == self.degree >> 1

    # poly_prof encodes the component-wise product under square of the
    # scaling factor. So normalize it before decoding.
    for i in range(self.degree >> 1):
      decoded[i] /= self.scaling_factor
    expected = [slots0[i] * slots1[i] for i in range(self.degree >> 1)]
    self._compare_with_precision(
        expected, decoded, bit_precision=self.encoding_precision
    )

  def test_encrypt_decrypt(self):
    encoder = ckks.CkksEncoder(
        degree=self.degree,
        moduli=self.moduli,
        scaling_factor=self.scaling_factor,
    )
    secret_key = ckks.gen_secret_key(self.degree, self.moduli)
    slots = self._random_slots(self.degree)
    ciphertext = ckks.encrypt(secret_key, slots, encoder, self.rns_params)
    decoded = ckks.decrypt(secret_key, ciphertext, encoder, self.rns_params)
    self._compare_with_precision(
        slots, decoded, bit_precision=self.encryption_precision
    )

  @parameterized_test.named_parameters(
      {
          'testcase_name': 'jax_add',
          'test_target': add.jax_add,
      },
      {
          'testcase_name': 'vmap_add',
          'test_target': add.vmap_add,
      },
  )
  def test_homomorphic_add_with(self, test_target):
    encoder = ckks.CkksEncoder(
        degree=self.degree,
        moduli=self.moduli,
        scaling_factor=self.scaling_factor,
    )
    secret_key = ckks.gen_secret_key(self.degree, self.moduli)
    slots0 = self._random_slots(self.degree)
    slots1 = self._random_slots(self.degree)
    ciphertext0 = ckks.encrypt(secret_key, slots0, encoder, self.rns_params)
    ciphertext1 = ckks.encrypt(secret_key, slots1, encoder, self.rns_params)

    modulus_list, ciphertext_data0 = ciphertext0.to_jnp_array()
    _, ciphertext_data1 = ciphertext1.to_jnp_array()
    jax_results = test_target(ciphertext_data0, ciphertext_data1, modulus_list)
    ciphertext_sum = ckks.gen_ciphertext_from_jnp_array(
        self.degree, self.moduli, jax_results
    )
    decoded = ckks.decrypt(secret_key, ciphertext_sum, encoder, self.rns_params)
    expected = [slots0[i] + slots1[i] for i in range(self.degree >> 1)]
    self._compare_with_precision(
        expected, decoded, bit_precision=self.encryption_precision
    )

  @parameterized_test.named_parameters(
      {
          'testcase_name': 'jax_sub',
          'test_target': sub.jax_sub,
      },
      {
          'testcase_name': 'vmap_sub',
          'test_target': sub.vmap_sub,
      },
  )
  def test_homomorphic_sub_with(self, test_target):
    encoder = ckks.CkksEncoder(
        degree=self.degree,
        moduli=self.moduli,
        scaling_factor=self.scaling_factor,
    )
    secret_key = ckks.gen_secret_key(self.degree, self.moduli)
    slots0 = self._random_slots(self.degree)
    slots1 = self._random_slots(self.degree)
    ciphertext0 = ckks.encrypt(secret_key, slots0, encoder, self.rns_params)
    ciphertext1 = ckks.encrypt(secret_key, slots1, encoder, self.rns_params)

    modulus_list, ciphertext_data0 = ciphertext0.to_jnp_array()
    _, ciphertext_data1 = ciphertext1.to_jnp_array()
    jax_results = test_target(ciphertext_data0, ciphertext_data1, modulus_list)
    ciphertext_diff = ckks.gen_ciphertext_from_jnp_array(
        self.degree, self.moduli, jax_results
    )
    decoded = ckks.decrypt(
        secret_key, ciphertext_diff, encoder, self.rns_params
    )
    expected = [slots0[i] - slots1[i] for i in range(self.degree >> 1)]
    self._compare_with_precision(
        expected, decoded, bit_precision=self.encryption_precision
    )


class CkksNegativeTest(parameterized_test.TestCase):
  """Testing negative cases for CKKS implementation."""

  def setUp(self):
    super().setUp()
    self.degree = 8
    self.moduli = [12289]
    self.scaling_factor = 2**5

  @parameterized_test.named_parameters(
      {
          'testcase_name': 'zero_degree',
          'invalid_degree': 0,
      },
      {
          'testcase_name': 'odd_degree',
          'invalid_degree': 7,
      },
  )
  def test_create_encoder_with_invalid_degree(self, invalid_degree):
    with self.assertRaises(ValueError):
      ckks.CkksEncoder(
          degree=invalid_degree,
          moduli=self.moduli,
          scaling_factor=self.scaling_factor,
      )

  def test_encode_with_too_many_slots(self):
    encoder = ckks.CkksEncoder(
        degree=self.degree,
        moduli=self.moduli,
        scaling_factor=self.scaling_factor,
    )
    slots = [complex(0, 0)] * (self.degree // 2 + 1)
    with self.assertRaises(ValueError):
      encoder.encode(slots)

  def test_decode_with_invalid_degree(self):
    encoder = ckks.CkksEncoder(
        degree=self.degree,
        moduli=self.moduli,
        scaling_factor=self.scaling_factor,
    )
    # Create a polynomial with an invalid degree.
    invalid_degree = self.degree - 1
    coeffs = [[0] * invalid_degree for _ in self.moduli]
    plaintext = rns.RnsPolynomial(
        invalid_degree, self.moduli, coeffs, is_ntt=False
    )
    with self.assertRaises(ValueError):
      encoder.decode(plaintext)

  def test_decode_with_ntt_polynomial(self):
    encoder = ckks.CkksEncoder(
        degree=self.degree,
        moduli=self.moduli,
        scaling_factor=self.scaling_factor,
    )
    # Create a polynomial in the NTT form.
    coeffs = [[0] * self.degree for _ in self.moduli]
    plaintext = rns.RnsPolynomial(self.degree, self.moduli, coeffs, is_ntt=True)
    with self.assertRaises(ValueError):
      encoder.decode(plaintext)


if __name__ == '__main__':
  absltest.main()
