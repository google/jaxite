"""Boolean gate API tests that chain multiple gates together."""

from jaxite.jaxite_bool import bool_encoding
from jaxite.jaxite_bool import bool_params
from jaxite.jaxite_bool import jaxite_bool
from jaxite.jaxite_lib import test_utils
from absl.testing import absltest
from absl.testing import parameterized


class BoolMultigateTest(parameterized.TestCase):
  """Boolean gate API tests that chain multiple gates together."""

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    cls.lwe_rng = bool_params.get_lwe_rng_for_128_bit_security(1)
    cls.rlwe_rng = bool_params.get_rlwe_rng_for_128_bit_security(1)
    cls.boolean_params = bool_params.get_params_for_128_bit_security()
    cls.client_key_set = jaxite_bool.ClientKeySet(
        cls.boolean_params,
        lwe_rng=cls.lwe_rng,
        rlwe_rng=cls.rlwe_rng,
    )
    cls.callback = test_utils.MidBootstrapDecrypter(
        scheme_params=cls.boolean_params.scheme_params,
        encoding_params=bool_encoding.ENCODING_PARAMS,
        lwe_key=cls.client_key_set.lwe_sk,
        rlwe_key=cls.client_key_set.rlwe_sk,
    ).decrypt
    cls.server_key_set = jaxite_bool.ServerKeySet(
        cls.client_key_set,
        cls.boolean_params,
        lwe_rng=cls.lwe_rng,
        rlwe_rng=cls.rlwe_rng,
        bootstrap_callback=cls.callback,
    )

  def test_boolean_gate_chained(self) -> None:
    ct_true = jaxite_bool.encrypt(True, self.client_key_set, self.lwe_rng)
    ct_false = jaxite_bool.encrypt(False, self.client_key_set, self.lwe_rng)
    not_false = jaxite_bool.not_(ct_false, self.boolean_params)
    or_false = jaxite_bool.or_(
        not_false, ct_false, self.server_key_set, self.boolean_params
    )
    and_true = jaxite_bool.and_(
        or_false, ct_true, self.server_key_set, self.boolean_params
    )
    xor_true = jaxite_bool.xor_(
        and_true, ct_true, self.server_key_set, self.boolean_params
    )
    actual = jaxite_bool.decrypt(xor_true, self.client_key_set)

    expected = (((not False) or False) and True) != True  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
    self.assertEqual(actual, expected)

  @parameterized.named_parameters(
      dict(testcase_name='_with_TTT', i0=True, i1=True, i2=True),
      dict(testcase_name='_with_TTF', i0=True, i1=True, i2=False),
      dict(testcase_name='_with_TFT', i0=True, i1=False, i2=True),
      dict(testcase_name='_with_TFF', i0=True, i1=False, i2=False),
      dict(testcase_name='_with_FTT', i0=False, i1=True, i2=True),
      dict(testcase_name='_with_FTF', i0=False, i1=True, i2=False),
      dict(testcase_name='_with_FFT', i0=False, i1=False, i2=True),
      dict(testcase_name='_with_FFF', i0=False, i1=False, i2=False),
  )
  def test_chained_and_succeeds(self, i0: bool, i1: bool, i2: bool) -> None:
    i0_ct = jaxite_bool.encrypt(i0, self.client_key_set, self.lwe_rng)
    i1_ct = jaxite_bool.encrypt(i1, self.client_key_set, self.lwe_rng)
    i2_ct = jaxite_bool.encrypt(i2, self.client_key_set, self.lwe_rng)

    and_1 = jaxite_bool.and_(
        i0_ct, i1_ct, self.server_key_set, self.boolean_params
    )
    # Check intermediate result
    output_and_1 = jaxite_bool.decrypt(and_1, self.client_key_set)
    self.assertEqual(output_and_1, i0 and i1)

    and_2 = jaxite_bool.and_(
        and_1, i2_ct, self.server_key_set, self.boolean_params
    )
    output_and_2 = jaxite_bool.decrypt(and_2, self.client_key_set)
    self.assertEqual(output_and_2, i0 and i1 and i2)

    # The test should pass for all combinations of "and"-ed ciphertexts
    and_3 = jaxite_bool.and_(
        i2_ct, i1_ct, self.server_key_set, self.boolean_params
    )
    # Check intermediate result
    output_and_3 = jaxite_bool.decrypt(and_3, self.client_key_set)
    self.assertEqual(output_and_3, i2 and i1)

    and_4 = jaxite_bool.and_(
        and_3, i0_ct, self.server_key_set, self.boolean_params
    )
    output_and_4 = jaxite_bool.decrypt(and_4, self.client_key_set)
    self.assertEqual(output_and_4, i0 and i1 and i2)

  @parameterized.product(
      seed=[1, 5, 9],
  )
  def test_seeds(self, seed: int) -> None:
    # Testing different seeds using a chained operation:
    # ((((not False) or False) and True) xor True)

    # Make sure we are using 128-bit security for RNGs
    lwe_rng_128_bit_security = bool_params.get_lwe_rng_for_128_bit_security(
        seed
    )
    rlwe_rng_128_bit_security = bool_params.get_rlwe_rng_for_128_bit_security(
        seed
    )
    modified_server_key_set = jaxite_bool.ServerKeySet(
        self.client_key_set,
        self.boolean_params,
        lwe_rng=lwe_rng_128_bit_security,
        rlwe_rng=rlwe_rng_128_bit_security,
        bootstrap_callback=self.callback,
    )

    ct_true = jaxite_bool.encrypt(
        True, self.client_key_set, lwe_rng_128_bit_security
    )
    ct_false = jaxite_bool.encrypt(
        False, self.client_key_set, lwe_rng_128_bit_security
    )
    not_false = jaxite_bool.not_(ct_false, self.boolean_params)
    or_false = jaxite_bool.or_(
        not_false, ct_false, modified_server_key_set, self.boolean_params
    )
    and_true = jaxite_bool.and_(
        or_false, ct_true, modified_server_key_set, self.boolean_params
    )
    xor_true = jaxite_bool.xor_(
        and_true, ct_true, modified_server_key_set, self.boolean_params
    )
    actual = jaxite_bool.decrypt(xor_true, self.client_key_set)

    expected = (((not False) or False) and True) != True  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
    self.assertEqual(actual, expected)


if __name__ == '__main__':
  absltest.main()
