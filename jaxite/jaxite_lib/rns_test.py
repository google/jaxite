"""Tests for RnsPolynomial."""

import random

from jaxite.jaxite_lib import rns
import parameterized

from absl.testing import absltest
from absl.testing import parameterized as parameterized_test


@parameterized.parameterized_class([
    {"degree": 8, "moduli": [12289]},
    {"degree": 16, "moduli": [12289, 65537]},
    {"degree": 1024, "moduli": [12289, 65537]},
])
class RnsPolynomialTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.ntt_params = [rns.Ntt(self.degree, modulus) for modulus in self.moduli]

  def _random_coeffs(self, degree: int, modulus: int) -> list[int]:
    return [random.randint(0, modulus) for _ in range(degree)]

  def _random_rns_polynomial(
      self, degree: int, moduli: list[int], is_ntt: bool
  ) -> rns.RnsPolynomial:
    coeffs = [self._random_coeffs(degree, modulus) for modulus in moduli]
    return rns.RnsPolynomial(degree, moduli, coeffs, is_ntt=is_ntt)

  def test_ntt(self):
    ntt = rns.Ntt(self.degree, self.moduli[0])
    coeffs = self._random_coeffs(self.degree, self.moduli[0])
    # NTT^-1( NTT( coeffs )) should be the same as coeffs.
    evals = list(coeffs)
    ntt.forward(evals)
    coeffs_back = list(evals)
    ntt.backward(coeffs_back)
    self.assertEqual(coeffs, coeffs_back)

  def test_rns_polynomial_addition(self):
    poly0 = self._random_rns_polynomial(self.degree, self.moduli, is_ntt=False)
    poly1 = self._random_rns_polynomial(self.degree, self.moduli, is_ntt=False)

    # First compute a + b in the coefficient form.
    poly_sum0 = poly0 + poly1
    assert not poly_sum0.is_ntt

    # Then compute a + b in the NTT form. The result (once converted back to the
    # coefficient form) should be the same.
    poly0.to_ntt_form(self.ntt_params)
    poly1.to_ntt_form(self.ntt_params)
    assert poly0.is_ntt
    assert poly1.is_ntt
    poly_sum1 = poly0 + poly1
    assert poly_sum1.is_ntt
    poly_sum1.to_coeffs_form(self.ntt_params)
    self.assertEqual(poly_sum1, poly_sum0)

  def test_rns_polynomial_negation(self):
    zero_coeffs = [[0] * self.degree for _ in range(len(self.moduli))]
    zero = rns.RnsPolynomial(self.degree, self.moduli, zero_coeffs)

    poly0 = self._random_rns_polynomial(self.degree, self.moduli, is_ntt=False)
    poly0_neg = -poly0
    assert not poly0_neg.is_ntt
    poly0_sum = poly0 + poly0_neg
    assert not poly0_sum.is_ntt
    self.assertEqual(poly0_sum, zero)

    poly1 = self._random_rns_polynomial(self.degree, self.moduli, is_ntt=True)
    poly1_neg = -poly1
    assert poly1_neg.is_ntt
    poly1_sum = poly1 + poly1_neg
    assert poly1_sum.is_ntt
    poly1_sum.to_coeffs_form(self.ntt_params)
    self.assertEqual(poly1_sum, zero)

  def test_rns_polynomial_multiplication(self):
    a = self._random_rns_polynomial(self.degree, self.moduli, is_ntt=True)
    b = self._random_rns_polynomial(self.degree, self.moduli, is_ntt=True)
    c = self._random_rns_polynomial(self.degree, self.moduli, is_ntt=True)

    # First we compute (a + b) * c
    ab = a + b
    abc = ab * c
    assert abc.is_ntt
    # Then we compute a * c + b * c
    ac = a * c
    bc = b * c
    acbc = ac + bc
    assert acbc.is_ntt
    # Check (a + b) * c = a * c + b * c)
    self.assertEqual(abc, acbc)
    
    
class RnsNegativeTest(parameterized_test.TestCase):
  """Testing negative cases for RNS implementation."""
  
  def setUp(self):
    super().setUp()
    self.degree = 8
    self.moduli = [12289]
    

  @parameterized_test.named_parameters(
      {
          'testcase_name': 'n_is_zero',
          'invalid_n': 0,
      },
      {
          'testcase_name': 'n_is_odd',
          'invalid_n': 7,
      },
  )    
  def test_create_ntt_with_invalid_n(self, invalid_n):
    with self.assertRaises(ValueError):
      rns.Ntt(invalid_n, self.moduli[0])
      
  def test_create_ntt_with_invalid_q(self):
    invalid_q = 2 * self.degree # set q = 2*N which isn't NTT-friendly
    with self.assertRaises(ValueError):
      rns.Ntt(self.degree, invalid_q)
      
  def test_add_polynomials_with_different_degrees(self):
    coeffs0 = [[0] * self.degree for _ in range(len(self.moduli))]
    coeffs1 = [[0] * (self.degree * 2) for _ in range(len(self.moduli))]
    poly0 = rns.RnsPolynomial(self.degree, self.moduli, coeffs0, is_ntt=False)
    poly1 = rns.RnsPolynomial(self.degree * 2, self.moduli, coeffs1, is_ntt=False)
    with self.assertRaises(ValueError):
      poly0 + poly1
      
  def test_add_polynomials_with_incompatible_coeffs(self):
    coeffs0 = [[0] * self.degree for _ in range(len(self.moduli))]
    coeffs1 = [[0] * self.degree for _ in range(len(self.moduli) + 1)]
    poly0 = rns.RnsPolynomial(self.degree, self.moduli, coeffs0, is_ntt=False)
    poly1 = rns.RnsPolynomial(self.degree, self.moduli, coeffs1, is_ntt=False)
    with self.assertRaises(ValueError):
      poly0 + poly1
      
  def test_add_polynomials_with_different_moduli(self):
    moduli0 = self.moduli
    moduli1 = moduli0 + [65537]
    coeffs0 = [[0] * self.degree for _ in range(len(moduli0))]
    coeffs1 = [[0] * self.degree for _ in range(len(moduli1))]
    poly0 = rns.RnsPolynomial(self.degree, moduli0, coeffs0, is_ntt=False)
    poly1 = rns.RnsPolynomial(self.degree, moduli1, coeffs1, is_ntt=False)
    with self.assertRaises(ValueError):
      poly0 + poly1
      
  def test_add_polynomials_with_different_forms(self):
    coeffs = [[0] * self.degree for _ in range(len(self.moduli))]
    poly0 = rns.RnsPolynomial(self.degree, self.moduli, coeffs, is_ntt=False)
    poly1 = rns.RnsPolynomial(self.degree, self.moduli, coeffs, is_ntt=True)
    with self.assertRaises(ValueError):
      poly0 + poly1
      
  def test_multiply_polynomials_in_coefficient_form(self):
    coeffs = [[0] * self.degree for _ in range(len(self.moduli))]
    poly0 = rns.RnsPolynomial(self.degree, self.moduli, coeffs, is_ntt=False)
    poly1 = rns.RnsPolynomial(self.degree, self.moduli, coeffs, is_ntt=False)
    with self.assertRaises(ValueError):
      poly0 * poly1
        
      

if __name__ == "__main__":
  absltest.main()
