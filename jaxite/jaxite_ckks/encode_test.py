"""Tests for CKKS encode/decode routines."""

import hypothesis
from hypothesis import strategies as st
from jaxite.jaxite_ckks import encode
import numpy as np
from absl.testing import absltest


class EncodeTest(absltest.TestCase):

  @hypothesis.settings(max_examples=5, deadline=None)
  @hypothesis.given(
      st.lists(
          st.complex_numbers(min_magnitude=0, max_magnitude=100),
          min_size=4,
          max_size=4,
      ),
      st.floats(min_value=2**10, max_value=2**20),
  )
  def test_encode_decode_loop(self, slots, scale):
    degree = 8
    moduli = [335552513, 335546369]  # Two 25-bit primes

    pt = encode.encode(slots, degree, moduli, scale)
    decoded = encode.decode(pt, scale, len(slots))

    for s, d in zip(slots, decoded):
      self.assertAlmostEqual(s.real, d.real, delta=1.0)
      self.assertAlmostEqual(s.imag, d.imag, delta=1.0)


class CrossDiffTest(absltest.TestCase):
  """Tests ensuring the exact same behavior as the CROSS reference code."""

  def test_encode_diff(self):
    degree = 16
    num_slots = degree // 2

    scale = 563019763943521
    q_towers = [1073742881, 1073742721, 1073741441, 1073741857, 524353]
    moduli = q_towers

    slots = [
        complex(0.25, 0),
        complex(0.5, 0),
        complex(0.75, 0),
        complex(1, 0),
        complex(2, 0),
        complex(3, 0),
        complex(4, 0),
        complex(5, 0),
    ]

    pt = encode.encode(slots, degree, moduli, scale)

    # Expected values generated from CROSS
    expected_data = np.array(
        [
            [136867625, 992729062, 1062901950, 64309566, 48023],
            [448217487, 21710597, 452417311, 776229866, 220065],
            [459110530, 1072702960, 135576683, 13425321, 107178],
            [399986586, 819955793, 346580006, 305401790, 87720],
            [338761771, 467831082, 833645419, 435102310, 305645],
            [253416924, 808362513, 770044521, 509826858, 509546],
            [529757713, 1036319305, 429707580, 665276930, 22149],
            [922456207, 726996332, 105156242, 988523858, 10556],
            [56346712, 330072408, 553702770, 645102273, 136241],
            [322124327, 359500508, 1054902493, 1008698515, 420817],
            [822180619, 673061236, 208387012, 877663497, 188715],
            [843740957, 603132359, 321561487, 67265671, 102123],
            [410369751, 388650230, 1058737500, 1009707013, 347437],
            [448727365, 430265802, 497160630, 382861955, 371814],
            [276075424, 22743178, 297873459, 418401481, 384996],
            [309009688, 991696481, 143704361, 422137951, 407445],
        ],
        dtype=np.uint32,
    )

    np.testing.assert_array_equal(np.array(pt.data), expected_data)
    decoded = encode.decode(pt, scale, num_slots=num_slots)
    np.testing.assert_allclose(decoded, slots, atol=1e-03)


if __name__ == '__main__':
  absltest.main()
