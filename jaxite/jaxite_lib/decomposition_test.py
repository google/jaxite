"""Tests for decomposition logic."""

import hypothesis
from hypothesis import strategies
import jax.numpy as jnp
from jaxite.jaxite_lib import decomposition
import numpy as np
from absl.testing import absltest  # fmt: skip

NUM_BITS = 32
BASE_LOG = 8
BASE = 1 << BASE_LOG

MAX_SIGNED_REPRESENTABLE = (
    (BASE // 2 - 1) * (BASE ** (NUM_BITS // BASE_LOG) - 1) // (BASE - 1)
)


class DecomposeTest(absltest.TestCase):

  def test_decompose_specific_example(self):
    x = 1340987234
    decomposed = decomposition.decompose(
        x, base_log=4, num_levels=3, total_bit_length=NUM_BITS
    )
    np.testing.assert_array_equal(decomposed, [4, 15, 14])

  def test_decomposition_summand_specific_example(self):
    x = jnp.uint32(1)
    num_levels = 8
    res = decomposition.recomposition_summands(
        x, base_log=3, num_levels=num_levels, total_bit_length=32
    )
    np.testing.assert_array_equal(
        res, [536870912, 67108864, 8388608, 1048576, 131072, 16384, 2048, 256]
    )

  def test_recompose_specific_example(self):
    x = jnp.array([4, 15, 14])
    recomposed = decomposition.recompose(
        x, base_log=4, total_bit_length=NUM_BITS
    )
    self.assertEqual(recomposed, 1340080128)

  @hypothesis.settings(deadline=None)
  @hypothesis.given(
      strategies.integers(min_value=0, max_value=2**NUM_BITS - 1)
  )
  def test_exact_reconstruction(self, x: int):
    decomposed = decomposition.decompose(
        jnp.uint32(x),
        # 4 * 8 = 32 ensures all bits are in the decomposition
        base_log=4,
        num_levels=8,
        total_bit_length=NUM_BITS,
    )
    recomposed = decomposition.recompose(
        decomposed, base_log=4, total_bit_length=NUM_BITS
    )
    self.assertEqual(x, int(recomposed))

  @hypothesis.settings(deadline=None)
  @hypothesis.given(
      strategies.integers(min_value=0, max_value=2**NUM_BITS - 1),
      strategies.integers(min_value=2, max_value=BASE_LOG),
      strategies.integers(min_value=1, max_value=4),
  )
  def test_decompose_recompose(self, x: int, base_log: int, num_levels: int):
    # base_log=2, num_levels=3
    # x        = 0 b 1111_0101_1100_0000
    #                      ^ lowest-order bit preserved be recomp(decomp(x))
    # max_diff = 0 b 0000_0011_1111_1111
    max_diff = (1 << (NUM_BITS - base_log * num_levels)) - 1
    decomposed = decomposition.decompose(
        jnp.uint32(x),
        base_log=base_log,
        num_levels=num_levels,
        total_bit_length=NUM_BITS,
    )
    recomposed = decomposition.recompose(
        decomposed, base_log=base_log, total_bit_length=NUM_BITS
    )
    self.assertLessEqual(recomposed, np.uint32(x))
    self.assertLessEqual(
        abs(x - int(recomposed)),
        max_diff,
        (
            f'\ndecomposed={decomposed}\nrecomposed={int(recomposed)}'
            f'\nx={x}\ndiff={abs(x-int(recomposed))}'
        ),
    )

  def test_gadget_matrix(self):
    params = decomposition.DecompositionParameters(
        log_base=2, level_count=3
    )
    gadget_matrix = decomposition.gadget_matrix(
        decomp_params=params, vector_length=2, total_bit_length=NUM_BITS
    )
    expected = np.array([
        [1 / 4, 1 / 16, 1 / 64, 0, 0, 0],
        [0, 0, 0, 1 / 4, 1 / 16, 1 / 64],
    ])

    np.testing.assert_array_equal(expected, gadget_matrix)

  def test_gadget_matrix_32_bit(self):
    params = decomposition.DecompositionParameters(
        log_base=BASE_LOG, level_count=4
    )
    gadget_matrix = decomposition.gadget_matrix(
        decomp_params=params, vector_length=2, total_bit_length=NUM_BITS
    )
    expected = np.array([
        [1 / 256, 1 / 256**2, 1 / 256**3, 1 / 256**4, 0, 0, 0, 0],
        [
            0,
            0,
            0,
            0,
            1 / 256,
            1 / 256**2,
            1 / 256**3,
            1 / 256**4,
        ],
    ])
    np.testing.assert_array_almost_equal(expected, gadget_matrix, decimal=16)

  @hypothesis.settings(deadline=None)
  @hypothesis.given(
      # for some reason, values too close to zero hit approximation errors
      # and cause the equality comparison to fail
      strategies.floats(min_value=0.01, max_value=0.49),
      strategies.floats(min_value=0.01, max_value=0.49),
  )
  @hypothesis.example(0.5, 0.5)
  def test_gadget_inverse_dot_gadget(self, x: float, y: float):
    params = decomposition.DecompositionParameters(
        log_base=BASE_LOG, level_count=4
    )
    gadget_matrix = decomposition.gadget_matrix(
        decomp_params=params, vector_length=2, total_bit_length=NUM_BITS
    )
    vector = jnp.array([x, y], dtype=jnp.float32)
    decomposed = decomposition.inverse_gadget(
        vector, params, total_bit_length=NUM_BITS
    )
    recomposed = decomposed.dot(gadget_matrix.T)
    np.testing.assert_array_equal(vector, recomposed)

  @hypothesis.settings(deadline=None)
  @hypothesis.given(
      strategies.integers(min_value=0, max_value=MAX_SIGNED_REPRESENTABLE)
  )
  @hypothesis.example(2047)
  @hypothesis.example(MAX_SIGNED_REPRESENTABLE)
  def test_signed_decomposition(self, number):
    num_levels = 4
    decomp = decomposition.signed_decomposition(
        number, BASE_LOG, num_levels=num_levels, total_bit_length=NUM_BITS
    )
    reconstructed = sum(
        jnp.uint32(digit) * jnp.uint32(BASE) ** (num_levels - i - 1)
        for (i, digit) in enumerate(decomp)
    )
    self.assertEqual(reconstructed, number)

  def test_decompose_rlwe_ciphertext_vmap_compatibility(self):
    decomposition_params = decomposition.DecompositionParameters(
        log_base=4,
        level_count=8,
        total_bit_length=32,
    )
    rlwe_ct = jnp.array(
        [
            [1, 2, 3, 4],
            [2, 2, 2, 2],
            [2**24, 2**16, 2**8, 1],
        ],
        dtype=jnp.uint32,
    )

    actual = decomposition.decompose_rlwe_ciphertext(
        rlwe_ct, decomposition_params
    )
    expected = jnp.array(
        [
            # polynomial 0, (level, coefficient) sub-matrix
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            # polynomial 1
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 2, 2, 2],
            # polynomial 2
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=jnp.uint32,
    )

    np.testing.assert_array_equal(expected, actual)


if __name__ == '__main__':
  absltest.main()
