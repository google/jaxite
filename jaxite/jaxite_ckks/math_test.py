"""Tests for math utilities."""

from jaxite.jaxite_ckks import math
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized


class MathTest(parameterized.TestCase):

  @parameterized.parameters(
      (12, {2, 3}),
      (30, {2, 3, 5}),
      (13, {13}),
  )
  def test_prime_factors(self, n, expected_factors):
    self.assertEqual(math.prime_factors(n), expected_factors)

  @parameterized.parameters(
      (7, 3),
      (11, 2),
  )
  def test_find_generator(self, q, expected_generator):
    self.assertEqual(math.find_generator(q), expected_generator)

  @parameterized.parameters(
      (8, 17),  # Power of 2
      (16, 17),  # Power of 2 (m = q-1)
      (4, 13),  # Power of 2
      (6, 7),  # Non-power of 2 (m = q-1)
      (3, 7),  # Non-power of 2
  )
  def test_root_of_unity(self, m, q):
    omega = math.root_of_unity(m, q)
    # Check it is an m-th root
    self.assertEqual(pow(omega, m, q), 1)
    # Check it is primitive (omega^(m/p) != 1 for all prime factors p of m)
    factors = math.prime_factors(m)
    for p in factors:
      self.assertNotEqual(pow(omega, m // p, q), 1)
    # Check it is the minimum canonical root
    for k in range(1, m):
      if np.gcd(k, m) == 1:
        psi = pow(omega, k, q)
        self.assertGreaterEqual(psi, omega)

  def test_gen_twiddle_matrix(self):
    q = 17
    omega = 2  # 8th root of unity mod 17
    rows = 4
    cols = 4
    np_twiddle = math.gen_twiddle_matrix(rows, cols, q, omega)

    # Verify every element T[r, c] == omega^(r*c) mod q
    r_indices = np.arange(rows)
    c_indices = np.arange(cols)
    exponent_matrix = np.outer(r_indices, c_indices)
    expected = np.power(omega, exponent_matrix) % q
    np.testing.assert_array_equal(np_twiddle, expected)

  def test_gen_twiddle_matrix_inv(self):
    q = 17
    omega = 2
    rows = 4
    cols = 4
    twiddle = math.gen_twiddle_matrix(rows, cols, q, omega)
    twiddle_inv = math.gen_twiddle_matrix_inv(rows, cols, q, omega)

    # (T * T_inv) mod q should have specific properties,
    # but simplest is just checking T_inv[r, c] == omega^(-r*c) mod q
    inv_omega = pow(omega, -1, q)
    r_indices = np.arange(rows)
    c_indices = np.arange(cols)
    exponent_matrix = np.outer(r_indices, c_indices)
    expected = np.power(inv_omega, exponent_matrix) % q
    np.testing.assert_array_equal(twiddle_inv, expected)

  def test_get_bit_reverse_perm(self):
    self.assertEqual(math.get_bit_reverse_perm(4), [0, 2, 1, 3])
    self.assertEqual(math.get_bit_reverse_perm(8), [0, 4, 2, 6, 1, 5, 3, 7])


if __name__ == "__main__":
  absltest.main()
