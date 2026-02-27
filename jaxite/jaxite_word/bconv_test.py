from absl.testing import absltest
from absl.testing import parameterized
import jaxite.jaxite_word.bconv as bconv
import jax
import jax.numpy as jnp
import numpy as np
from jaxite.jaxite_word.util import random_batched_ciphertext

# Use 64-bit precision as in bconv.py
jax.config.update("jax_enable_x64", True)

TEST_PARAMS = [
    (
        "L2_to_L5",
        [
            [
                180089039,
                904401266,
                277587483,
                381410246,
                867235356,
                971323117,
                934942938,
                338146069,
                129667711,
                97559399,
                337422188,
                364870460,
                916966745,
                312366062,
                762079964,
                605485434,
            ],
            [
                540094309,
                1034680811,
                1057648335,
                677992674,
                650354195,
                558219774,
                502221165,
                503532224,
                1049911792,
                146837876,
                560962740,
                820076664,
                58915608,
                1034452760,
                724437159,
                68291682,
            ],
        ],
        [1073741441, 1073740609],
        [268437409, 268436801, 268435361, 268435649, 524353],
        [249077041, 824663761],
        [
            [268428382, 268430206, 268434526, 268433662, 390018],
            [268429214, 268431038, 268435358, 268434494, 390850],
        ],
        [
            [
                127196115,
                177098281,
                103398386,
                262465714,
                225857559,
                213539642,
                56845406,
                173328911,
                21637023,
                13036123,
                259867486,
                247888119,
                190104469,
                18415021,
                107052173,
                152967426,
            ],
            [
                168457304,
                81199027,
                93565169,
                186078678,
                45587255,
                266135885,
                57716353,
                256503901,
                42940759,
                230451532,
                167299604,
                7499360,
                30178241,
                217184571,
                253380763,
                263628678,
            ],
            [
                36832759,
                236716332,
                248966405,
                220314486,
                69166987,
                6190101,
                204761211,
                194300152,
                254116210,
                106417675,
                161103783,
                244499620,
                193481707,
                80047389,
                247286681,
                101528753,
            ],
            [
                57170724,
                202300871,
                265775502,
                100961596,
                165644129,
                105852026,
                62793519,
                256209638,
                261792224,
                19441022,
                102172131,
                193804848,
                207565270,
                260633034,
                136301593,
                126040258,
            ],
            [
                7075121,
                3849323,
                6595938,
                5951409,
                7277876,
                7164094,
                6139709,
                4952812,
                4506557,
                5030408,
                7544214,
                3494637,
                8199168,
                9397516,
                5558747,
                9127873,
            ],
        ],  # , [258532, 178852, 303702, 183526, 461287, 347505, 371826, 233635, 311733, 311231, 203272, 348519, 333873, 483515, 315217, 213872]],
    ),
]


class BConvContextTest(parameterized.TestCase):
  # @absltest.skip("Skip a single test")
  @parameterized.named_parameters(*TEST_PARAMS)
  def test_barrett_context(
      self,
      partCtCloneCoef,
      original_moduli,
      target_moduli,
      QHatInvModq,
      QHatModp,
      reference_result,
  ):
    """Verifies that basis_change works with BarrettContext"""
    key = jax.random.PRNGKey(0)
    in_tower = jax.numpy.array(partCtCloneCoef, dtype=jnp.uint64).T
    reference_result = jax.numpy.array(reference_result, dtype=jnp.uint64).T

    # New API setup
    overall_moduli = original_moduli + target_moduli
    original_index = list(range(len(original_moduli)))
    target_index = list(range(len(original_moduli), len(overall_moduli)))

    _bconv = bconv.BConvBarrett(overall_moduli)
    _bconv.control_gen([(original_index, target_index)])

    in_formatted = _bconv.ff_ctx_origin[0].to_computation_format(in_tower)
    out_formatted = _bconv.basis_change(in_formatted)
    out = _bconv.ff_ctx_target[0].to_original_format(out_formatted)

    np.testing.assert_array_equal(reference_result, out)

  # @absltest.skip("Skip a single test")
  @parameterized.named_parameters(*TEST_PARAMS)
  def test_bat_lazy_context(
      self,
      partCtCloneCoef,
      original_moduli,
      target_moduli,
      QHatInvModq,
      QHatModp,
      reference_result,
  ):
    """Verifies that basis_change works with BATLazyContext"""
    key = jax.random.PRNGKey(0)
    in_tower = jax.numpy.array(partCtCloneCoef, dtype=jnp.uint64).T
    reference_result = jax.numpy.array(reference_result, dtype=jnp.uint64).T

    overall_moduli = original_moduli + target_moduli
    original_index = list(range(len(original_moduli)))
    target_index = list(range(len(original_moduli), len(overall_moduli)))

    _bconv = bconv.BConvBATLazy(overall_moduli)
    _bconv.control_gen([(original_index, target_index)])

    in_formatted = _bconv.ff_ctx_origin[0].to_computation_format(in_tower)
    out_formatted = _bconv.basis_change(in_formatted)
    out = _bconv.ff_ctx_target[0].to_original_format(out_formatted)

    # BATLazy produces result congruent mod p, but not necessarily fully reduced.
    # Hence we check the post modular reduction results.
    target_moduli_arr = jnp.array(target_moduli, dtype=jnp.uint64)
    diff = (
        out.astype(jnp.int64) - reference_result.astype(jnp.int64)
    ) % target_moduli_arr.astype(jnp.int64)
    np.testing.assert_array_equal(diff, jnp.zeros_like(diff))

  def test_multiple_control_gen(self):
    """Verifies that BConv supports multiple control generations."""
    # Define a simple setup
    # overall_moduli = [q0, q1, p0, p1]
    # Config 0: [q0] -> [p0]
    # Config 1: [q1] -> [p1]

    # Using small primes for easy verification
    q0, q1 = 17, 19
    p0, p1 = 23, 29
    overall_moduli = [q0, q1, p0, p1]

    # Config 0
    original_index_0 = [0]
    target_index_0 = [2]

    # Config 1
    original_index_1 = [1]
    target_index_1 = [3]

    _bconv = bconv.BConvBarrett(overall_moduli)
    _bconv.control_gen(
        [(original_index_0, target_index_0), (original_index_1, target_index_1)]
    )

    # Test Config 0
    val = 15
    in_tower_0 = jnp.array(
        [[val]], dtype=jnp.uint64
    )  # shape (1, 1) to match (d, q) ? sizeQ=1
    in_tower = jnp.array([[[15]]], dtype=jnp.uint64)  # (1, 1, 1)
    out_0 = _bconv.basis_change(in_tower, control_index=0)
    self.assertEqual(out_0[0, 0, 0], 15)

    in_tower_1 = jnp.array([[[20]]], dtype=jnp.uint64)
    out_1 = _bconv.basis_change(in_tower_1, control_index=1)
    self.assertEqual(out_1[0, 0, 0], 1)

    # Quick check for non-interference
    in_tower_0_b = jnp.array([[[20]]], dtype=jnp.uint64)
    out_0_b = _bconv.basis_change(
        in_tower_0_b, control_index=0
    )  # 20 mod 17 -> 3 -> 3
    self.assertEqual(out_0_b[0, 0, 0], 3)


class BConvBATTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Define some example moduli. Both fit in 32 bits (required for BAT assumption).
    # These are from basis_change_test.py (approx 2^27)
    self.original_moduli = [
        134219681,
        134218433,
        134219009,
        1073741857,
        1073740609,
    ]
    self.target_moduli = [268435361, 268435009, 6710893, 1067031829]

    self.overall_moduli = self.original_moduli + self.target_moduli
    self.original_index = list(range(len(self.original_moduli)))
    self.target_index = list(
        range(len(self.original_moduli), len(self.overall_moduli))
    )

    self.bconv = bconv.BConvBarrett(self.overall_moduli)
    self.bconv.control_gen([(self.original_index, self.target_index)])

  # @absltest.skip("Skip a single test")
  def test_basis_change_bat_vs_standard(self):
    """Verifies that basis_change_bat produces the same result as basis_change"""
    key = jax.random.PRNGKey(0)

    # Dimensions
    batch = 1
    elements = 2
    d = 128  # small ring dim
    sizeQ = len(self.original_moduli)
    in_tower = random_batched_ciphertext(
        (batch, elements, d, sizeQ), self.original_moduli, jnp.uint32
    )

    # Expected result (Standard)
    expected = self.bconv.basis_change(in_tower)

    # Actual result (BAT)
    actual = self.bconv.basis_change_bat(in_tower)
    target_moduli_arr = jnp.array(self.target_moduli, dtype=jnp.uint64)
    diff = (
        actual.astype(jnp.int64) - expected.astype(jnp.int64)
    ) % target_moduli_arr.astype(jnp.int64)
    np.testing.assert_array_equal(diff, jnp.zeros_like(diff))

  # @absltest.skip("Skip a single test")
  def test_basis_change_bat_random_big(self):
    """Test with larger shapes to ensure robustness."""
    key = jax.random.PRNGKey(1)
    batch = 4
    elements = 4
    d = 8
    sizeQ = len(self.original_moduli)
    shape = (batch, elements, d, sizeQ)

    in_tower = random_batched_ciphertext(
        shape, self.original_moduli, jnp.uint32
    )

    expected = self.bconv.basis_change(in_tower)
    actual = self.bconv.basis_change_bat(in_tower)
    target_moduli_arr = jnp.array(self.target_moduli, dtype=jnp.uint64)
    diff = (
        actual.astype(jnp.int64) - expected.astype(jnp.int64)
    ) % target_moduli_arr.astype(jnp.int64)
    np.testing.assert_array_equal(diff, jnp.zeros_like(diff))


if __name__ == "__main__":
  absltest.main()
