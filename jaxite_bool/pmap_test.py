"""Tests of jaxite_bool API that use pmap to parallelize across gates.

This test is separated from the other tests because it can only be run on TPUs.
"""

from jaxite.jaxite_bool import bool_params
from jaxite.jaxite_bool import jaxite_bool
from absl.testing import absltest
from absl.testing import parameterized


class PmapTest(parameterized.TestCase):
  """Tests of jaxite_bool API that use pmap to parallelize across gates."""

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
    cls.server_key_set = jaxite_bool.ServerKeySet(
        cls.client_key_set,
        cls.boolean_params,
        lwe_rng=cls.lwe_rng,
        rlwe_rng=cls.rlwe_rng,
    )

  def test_pmap_lut3(self) -> None:
    ct_true = jaxite_bool.encrypt(True, self.client_key_set, self.lwe_rng)
    ct_false = jaxite_bool.encrypt(False, self.client_key_set, self.lwe_rng)

    # For input (a, b, c, tt),
    # each output is constructed as (tt >> 0b{a, b, c}) & 1
    inputs = [
        (ct_true, ct_false, ct_true, 221),  # false
        (ct_true, ct_true, ct_false, 221),  # true
        # Forge only gives tests 2 cores, so we can't test parallelism beyond
        # two operations at once.
        # 2: (ct_false, ct_false, ct_false, 220),  # false
    ]
    outputs = jaxite_bool.pmap_lut3(
        inputs, self.server_key_set, self.boolean_params
    )

    output_cleartexts = [
        jaxite_bool.decrypt(value, self.client_key_set) for value in outputs
    ]
    expected = [False, True]
    self.assertEqual(expected, output_cleartexts)

  def test_pmap_lut2(self) -> None:
    ct_true = jaxite_bool.encrypt(True, self.client_key_set, self.lwe_rng)
    ct_false = jaxite_bool.encrypt(False, self.client_key_set, self.lwe_rng)

    # For input (a, b, tt),
    # each output is constructed as (tt >> 0b{a, b}) & 1
    inputs = [
        (ct_true, ct_false, 13),  # false
        (ct_true, ct_true, 13),  # true
        # Forge only gives tests 2 cores, so we can't test parallelism beyond
        # two operations at once.
    ]
    outputs = jaxite_bool.pmap_lut2(
        inputs, self.server_key_set, self.boolean_params
    )

    output_cleartexts = [
        jaxite_bool.decrypt(value, self.client_key_set) for value in outputs
    ]
    expected = [False, True]
    self.assertEqual(expected, output_cleartexts)


if __name__ == '__main__':
  absltest.main()
