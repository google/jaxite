"""Tests for NTT OpenFHE comparison."""

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import ntt
import numpy as np
from absl.testing import absltest


class NTTOpenFHETest(absltest.TestCase):

  def test_ntt_openfhe_comparison(self):
    """Byte-for-byte comparison against OpenFHE backend.

    This test case is ported from ntt_mm_test.py and represents a known
    correct transformation from the OpenFHE library.
    """
    r, c = 4, 4
    moduli = [134219681, 134219681, 134219681]
    # Input data from ntt_mm_test.py (3 moduli, 16 coefficients each)
    coef_in_raw = [
        [
            105825732,
            68433452,
            36629220,
            126901109,
            89469849,
            106633716,
            15102657,
            108374459,
            68789927,
            23451922,
            93538050,
            20585372,
            30604976,
            37517995,
            65644325,
            102451383,
        ],
        [
            105825732,
            68433452,
            36629220,
            126901109,
            89469849,
            106633716,
            15102657,
            108374459,
            68789927,
            23451922,
            93538050,
            20585372,
            30604976,
            37517995,
            65644325,
            102451383,
        ],
        [
            105825732,
            68433452,
            36629220,
            126901109,
            89469849,
            106633716,
            15102657,
            108374459,
            68789927,
            23451922,
            93538050,
            20585372,
            30604976,
            37517995,
            65644325,
            102451383,
        ],
    ]
    eval_in_raw = [
        [
            26196696,
            45475009,
            10055359,
            23277424,
            69041040,
            71916973,
            73894069,
            3311254,
            44646798,
            49882443,
            28097016,
            70484730,
            10811958,
            11946041,
            61318182,
            19099272,
        ],
        [
            26196696,
            45475009,
            10055359,
            23277424,
            69041040,
            71916973,
            73894069,
            3311254,
            44646798,
            49882443,
            28097016,
            70484730,
            10811958,
            11946041,
            61318182,
            19099272,
        ],
        [
            26196696,
            45475009,
            10055359,
            23277424,
            69041040,
            71916973,
            73894069,
            3311254,
            44646798,
            49882443,
            28097016,
            70484730,
            10811958,
            11946041,
            61318182,
            19099272,
        ],
    ]

    # Reshape to (R, C, M) as expected by our NTT kernel.
    # coef_in_raw is (M, N). We want (R, C, M).
    coef_in = (
        jnp.array(coef_in_raw, dtype=jnp.uint32)
        .transpose(1, 0)
        .reshape(r, c, len(moduli))
    )
    eval_in = (
        jnp.array(eval_in_raw, dtype=jnp.uint32)
        .transpose(1, 0)
        .reshape(r, c, len(moduli))
    )

    ntt_kernel = ntt.NTTBarrett()
    ntt_kernel.precompute_constants(moduli, r, c)

    transformed = ntt_kernel.ntt(coef_in)
    np.testing.assert_array_equal(transformed, eval_in)

    recovered = ntt_kernel.intt(transformed)
    np.testing.assert_array_equal(recovered, coef_in)


if __name__ == "__main__":
  absltest.main()
