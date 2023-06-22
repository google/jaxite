from jaxite.jaxite_bool import bool_encoding
from jaxite.jaxite_bool import lut
from jaxite.jaxite_lib import parameters
from absl.testing import absltest

TEST_SCHEME_PARAMS = parameters.SchemeParameters(
    lwe_dimension=4,
    plaintext_modulus=2**32,
    rlwe_dimension=2,
    polynomial_modulus_degree=8,
)


class LutTest(absltest.TestCase):

  def test_str(self):
    table = lut.LookUpTable(num_inputs=3, truth_table=0x4F)
    expected = """000 -> 1
001 -> 1
010 -> 1
011 -> 1
100 -> 0
101 -> 0
110 -> 1
111 -> 0"""
    self.assertEqual(expected, str(table))

  def test_repr(self):
    table = lut.LookUpTable(num_inputs=3, truth_table=0x4F)
    expected = 'LookUpTable(num_inputs=3, truth_table=79, )'
    self.assertEqual(expected, repr(table))

  def test_from_callable(self):
    def lut_fn(x: bool, y: bool, z: bool) -> bool:
      if not x:
        return True
      elif y and not z:
        return True
      return False

    table = lut.from_callable(num_inputs=3, fn=lut_fn)
    self.assertEqual(79, table.truth_table)

  def test_as_cleartext_list(self):
    # tested fn is XOR
    expected = [
        bool_encoding.CLEARTEXT_FALSE,
        bool_encoding.CLEARTEXT_TRUE,
        bool_encoding.CLEARTEXT_TRUE,
        bool_encoding.CLEARTEXT_FALSE,
    ]
    table = lut.from_callable(num_inputs=2, fn=lambda x, y: x != y)
    self.assertEqual(expected, table.as_cleartext_list)

  def test_lut_cache_and(self):
    cache = lut.LutCache(scheme_params=TEST_SCHEME_PARAMS)
    self.assertEqual(8, cache.lut_by_name('and').truth_table)

  def test_lut_cache_generate_lut(self):
    cache = lut.LutCache(scheme_params=TEST_SCHEME_PARAMS)
    self.assertEqual(10, cache.lut(num_inputs=3, lut_as_int=10).truth_table)

  def test_lut_cache_key_error(self):
    cache = lut.LutCache(scheme_params=TEST_SCHEME_PARAMS)
    self.assertRaises(ValueError, lambda: cache.lut_by_name('wat'))


if __name__ == '__main__':
  absltest.main()
