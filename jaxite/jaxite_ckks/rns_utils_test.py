"""Tests for RNS utilities."""

import functools
import math
import hypothesis
from hypothesis import strategies as st
from jaxite.jaxite_ckks import rns_utils
from absl.testing import absltest
from absl.testing import parameterized


@st.composite
def moduli_and_values(draw):
  """Strategy to generate a batch of moduli and original integers."""
  # Generate between 1 and 10 moduli in the range [2, 2^31 - 1].
  # They must be pairwise coprime for CRT to work.
  moduli = []
  while len(moduli) < 10:
    m = draw(st.integers(min_value=2, max_value=(1 << 31) - 1))
    if all(math.gcd(m, existing_m) == 1 for existing_m in moduli):
      moduli.append(m)
    if draw(st.booleans()):
      break

  if not moduli:
    moduli = [2]

  q_prod = functools.reduce(lambda a, b: a * b, moduli)
  # Generate between 1 and 20 integers in [0, q_prod - 1].
  values = draw(
      st.lists(
          st.integers(min_value=0, max_value=q_prod - 1),
          min_size=1,
          max_size=20,
      )
  )
  return moduli, values


class RnsUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      ([3, 5, 7], [10, 20, 30]),
      ([1073753729, 1073738977], [1, 2, 3]),
  )
  def test_reconstruct_crt_basic(self, moduli, values):
    q_prod = functools.reduce(lambda a, b: a * b, moduli)
    # residues[i][j] = values[j] % moduli[i]
    residues = [[v % m for v in values] for m in moduli]
    reconstructed = rns_utils.reconstruct_crt(residues, moduli)
    expected = [v % q_prod for v in values]
    self.assertEqual(reconstructed, expected)

  # CRT reconstruction involves very large integers (product of up to 10
  # 31-bit moduli), which triggers a Hypothesis health check.
  @hypothesis.settings(
      deadline=None,
      max_examples=100,
      suppress_health_check=[hypothesis.HealthCheck.large_base_example],
  )
  @hypothesis.given(moduli_and_values())
  def test_reconstruct_crt_hypothesis(self, data):
    moduli, values = data
    q_prod = functools.reduce(lambda a, b: a * b, moduli)

    # Compute residues: shape (num_moduli, num_values)
    residues = [[v % m for v in values] for m in moduli]

    reconstructed = rns_utils.reconstruct_crt(residues, moduli)
    expected = [v % q_prod for v in values]

    self.assertEqual(reconstructed, expected)


if __name__ == "__main__":
  absltest.main()
