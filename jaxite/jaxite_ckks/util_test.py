from jaxite.jaxite_ckks import util
from absl.testing import absltest


class UtilsTest(absltest.TestCase):

  def test_is_prime_deterministic(self):
    self.assertTrue(util.is_prime_deterministic(17))
    self.assertFalse(util.is_prime_deterministic(18))

  def test_find_moduli_ntt(self):
    self.assertEqual(util.find_moduli_ntt(1, 31, 16), [2147483489])
    self.assertEqual(util.find_moduli_ntt(2, 31, 16), [2147483489, 2147483249])

if __name__ == "__main__":
  absltest.main()
