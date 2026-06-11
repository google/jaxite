"""Unit tests for the Boolean Gate API."""
from jaxite.jaxite_bool import bool_encoding
from jaxite.jaxite_bool import bool_params
from jaxite.jaxite_bool import jaxite_bool
from jaxite.jaxite_lib import decomposition
from jaxite.jaxite_lib import encoding
from jaxite.jaxite_lib import test_utils
from absl.testing import absltest
from absl.testing import parameterized

FUNC_NAME_TO_LAMBDA = {
    'and_': lambda x, y: x and y,
    'andny_': lambda x, y: not x and y,
    'andyn_': lambda x, y: x and not y,
    'nand_': lambda x, y: not (x and y),
    'nor_': lambda x, y: not (x or y),
    'or_': lambda x, y: x or y,
    'orny_': lambda x, y: not x or y,
    'oryn_': lambda x, y: x or not y,
    'xnor_': lambda x, y: x == y,
    'xor_': lambda x, y: x != y,
}


class BoolBasicOperationsTest(parameterized.TestCase):
  """A suite of unit tests using a real server_key_set and client_key_set."""

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

  @parameterized.named_parameters(
      dict(testcase_name='_with_T', value=True),
      dict(testcase_name='_with_F', value=False),
  )
  def test_boolean_gate_constant_succeeds(self, value: bool) -> None:
    expected_cleartext = (
        bool_encoding.CLEARTEXT_TRUE if value else bool_encoding.CLEARTEXT_FALSE
    )
    expected_value = encoding.encode(
        expected_cleartext, bool_encoding.ENCODING_PARAMS
    )

    # Assert that the constant "gate" is a noiseless/trivial embedding
    actual_ct = jaxite_bool.constant(value, self.boolean_params)
    self.assertLen(
        actual_ct, self.boolean_params.scheme_params.lwe_dimension + 1
    )
    self.assertEqual(actual_ct[-1], expected_value)

  @parameterized.named_parameters(
      dict(testcase_name='_with_T', value=True),
      dict(testcase_name='_with_F', value=False),
  )
  def test_boolean_gate_not_succeeds(self, value: bool) -> None:
    ct = jaxite_bool.encrypt(value, self.client_key_set, self.lwe_rng)
    actual_ct = jaxite_bool.not_(ct, self.boolean_params)
    actual = jaxite_bool.decrypt(actual_ct, self.client_key_set)
    self.assertEqual(actual, not value)

  @parameterized.product(
      func_name=[
          'and_',
          'andny_',
          'andyn_',
          'nand_',
          'nor_',
          'or_',
          'orny_',
          'oryn_',
          'xnor_',
          'xor_',
      ],
      inputs=[
          (False, False),
          (False, True),
          (True, False),
          (True, True),
      ],
  )
  def test_two_bit_gates(
      self, func_name: str, inputs: tuple[bool, bool]
  ) -> None:
    expected_func = FUNC_NAME_TO_LAMBDA[func_name]
    lhs, rhs = inputs
    lhs_ct = jaxite_bool.encrypt(lhs, self.client_key_set, self.lwe_rng)
    rhs_ct = jaxite_bool.encrypt(rhs, self.client_key_set, self.lwe_rng)
    actual_func = getattr(jaxite_bool, func_name)
    actual_ct = actual_func(
        lhs_ct, rhs_ct, self.server_key_set, self.boolean_params
    )
    actual = jaxite_bool.decrypt(actual_ct, self.client_key_set)
    self.assertEqual(actual, expected_func(lhs, rhs))

  @parameterized.named_parameters(
      dict(testcase_name='_with_TTT', v1=True, v0=True, c=True),
      dict(testcase_name='_with_TTF', v1=True, v0=True, c=False),
      dict(testcase_name='_with_TFT', v1=True, v0=False, c=True),
      dict(testcase_name='_with_TFF', v1=True, v0=False, c=False),
      dict(testcase_name='_with_FTT', v1=False, v0=True, c=True),
      dict(testcase_name='_with_FTF', v1=False, v0=True, c=False),
      dict(testcase_name='_with_FFT', v1=False, v0=False, c=True),
      dict(testcase_name='_with_FFF', v1=False, v0=False, c=False),
  )
  def test_boolean_gate_cmux_succeeds(
      self, v1: bool, v0: bool, c: bool
  ) -> None:
    c_ct = jaxite_bool.encrypt(c, self.client_key_set, self.lwe_rng)
    v0_ct = jaxite_bool.encrypt(v0, self.client_key_set, self.lwe_rng)
    v1_ct = jaxite_bool.encrypt(v1, self.client_key_set, self.lwe_rng)
    actual_ct = jaxite_bool.cmux_(
        v1_ct, v0_ct, c_ct, self.server_key_set, self.boolean_params
    )
    actual = jaxite_bool.decrypt(actual_ct, self.client_key_set)
    self.assertEqual(actual, v1 if c else v0)

  def test_lut2_asymmetric(self) -> None:
    # Intended to guard against a prior bug where the arguments to lut2/lut3
    # were passed in reverse order. In this case we can choose a truth table for
    # which, if the roles of the inputs are swapped, you get the wrong output.
    truth_table = 0b1010
    a = False
    b = True
    # wrong way: {a, b}, gives 0b1010 >> 1 == 1
    # right way: {b, a}, gives 0b1010 >> 2 == 0

    a_ct = jaxite_bool.encrypt(a, self.client_key_set, self.lwe_rng)
    b_ct = jaxite_bool.encrypt(b, self.client_key_set, self.lwe_rng)
    actual_ct = jaxite_bool.lut2(
        a_ct, b_ct, truth_table, self.server_key_set, self.boolean_params
    )
    actual = jaxite_bool.decrypt(actual_ct, self.client_key_set)
    self.assertEqual(actual, False)

  # TODO(cathieyun): Add the b=3, L=7, case if we replace signed decomposition
  # with unsigned decomposition, as those parameters should pass in that case.
  # b/243840786
  @parameterized.named_parameters(
      dict(testcase_name='_b=2_L=10', decomp_log_base=2, l=10),
      dict(testcase_name='_b=2_L=8', decomp_log_base=2, l=8),
      dict(testcase_name='_b=4_L=5', decomp_log_base=4, l=5),
  )
  def test_ksk_decomposition_params(self, decomp_log_base: int, l: int) -> None:
    # Testing ksk decomp parameters using the cmux gate
    c = True
    v0 = True
    v1 = False

    # pylint: disable=protected-access
    modified_boolean_params = self.boolean_params
    modified_boolean_params._ks_decomp_params = (
        decomposition.DecompositionParameters(
            log_base=decomp_log_base, level_count=l
        )
    )
    # pylint: enable=protected-access
    # Make sure we are using 128-bit security for RNGs
    lwe_rng_128_bit_security = bool_params.get_lwe_rng_for_128_bit_security(1)
    rlwe_rng_128_bit_security = bool_params.get_rlwe_rng_for_128_bit_security(1)
    modified_server_key_set = jaxite_bool.ServerKeySet(
        self.client_key_set,
        modified_boolean_params,
        lwe_rng=lwe_rng_128_bit_security,
        rlwe_rng=rlwe_rng_128_bit_security,
        bootstrap_callback=self.callback,
    )

    c_ct = jaxite_bool.encrypt(c, self.client_key_set, lwe_rng_128_bit_security)
    v0_ct = jaxite_bool.encrypt(
        v0, self.client_key_set, lwe_rng_128_bit_security
    )
    v1_ct = jaxite_bool.encrypt(
        v1, self.client_key_set, lwe_rng_128_bit_security
    )

    actual_ct = jaxite_bool.cmux_(
        v1_ct, v0_ct, c_ct, modified_server_key_set, modified_boolean_params
    )
    actual = jaxite_bool.decrypt(actual_ct, self.client_key_set)
    self.assertEqual(actual, v1 if c else v0)

  @parameterized.named_parameters(
      dict(testcase_name='_b=4_L=8', decomp_log_base=4, l=8),
      # TODO(b/335701655): odd L results in tensor shapes that conflict with
      # the TPU kernel's requirements in polymul_kernel.py.
      # dict(testcase_name='_b=4_L=7', decomp_log_base=4, l=7),
      dict(testcase_name='_b=4_L=6', decomp_log_base=4, l=6),
  )
  def test_bsk_decomposition_params(self, decomp_log_base: int, l: int) -> None:
    # Testing bsk using a chained add: F and F and F
    # This test should fail for b=4, L=5

    # pylint: disable=protected-access
    modified_boolean_params = self.boolean_params
    modified_boolean_params._bs_decomp_params = (
        decomposition.DecompositionParameters(
            log_base=decomp_log_base, level_count=l
        )
    )
    # pylint: enable=protected-access
    # Make sure we are using 128-bit security for RNGs
    lwe_rng_128_bit_security = bool_params.get_lwe_rng_for_128_bit_security(1)
    rlwe_rng_128_bit_security = bool_params.get_rlwe_rng_for_128_bit_security(1)
    modified_server_key_set = jaxite_bool.ServerKeySet(
        self.client_key_set,
        modified_boolean_params,
        lwe_rng=lwe_rng_128_bit_security,
        rlwe_rng=rlwe_rng_128_bit_security,
        bootstrap_callback=self.callback,
    )

    false_1 = jaxite_bool.encrypt(
        False, self.client_key_set, lwe_rng_128_bit_security
    )
    false_2 = jaxite_bool.encrypt(
        False, self.client_key_set, lwe_rng_128_bit_security
    )
    false_3 = jaxite_bool.encrypt(
        False, self.client_key_set, lwe_rng_128_bit_security
    )
    and_1 = jaxite_bool.and_(
        false_1, false_2, modified_server_key_set, modified_boolean_params
    )
    and_2 = jaxite_bool.and_(
        and_1, false_3, modified_server_key_set, modified_boolean_params
    )
    output_and_2 = jaxite_bool.decrypt(and_2, self.client_key_set)
    self.assertEqual(output_and_2, False)


if __name__ == '__main__':
  absltest.main()
