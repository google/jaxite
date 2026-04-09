"""Tests for NTT."""

import functools
import hypothesis
from hypothesis import strategies as st
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import math as ckks_math
from jaxite.jaxite_ckks import ntt
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

# A list of NTT-friendly primes (around 30 bits).
# P = k * 16384 + 1, so they support N up to 8192 (2N=16384).
TEST_PRIMES = (
    1_073_692_673,
    1_073_643_521,
    1_073_479_681,
    1_073_430_529,
)


def ntt_reference(v, q, omega, r, c):
  """Naive 4-step NTT for reference."""
  # Step 1: NTT on columns
  omega_col = pow(omega, c, q)
  v_mat = v.reshape((r, c))
  y = np.zeros((r, c), dtype=np.uint64)
  for j in range(c):
    col = v_mat[:, j]
    y[:, j] = ckks_math.gen_twiddle_matrix(r, r, q, omega_col) @ col % q

  # Step 2: Multiply by twiddle matrix
  twiddle = ckks_math.gen_twiddle_matrix(r, c, q, omega)
  y = (y * twiddle) % q

  # Step 3: NTT on rows
  omega_row = pow(omega, r, q)
  z = np.zeros((r, c), dtype=np.uint64)
  for i in range(r):
    row = y[i, :]
    z[i, :] = ckks_math.gen_twiddle_matrix(c, c, q, omega_row) @ row % q

  # Step 4: Transpose and flatten
  return z.T.flatten()


class NTTTest(parameterized.TestCase):

  @parameterized.parameters(
      (4, 4, [TEST_PRIMES[0]]),
      (8, 8, [TEST_PRIMES[0], TEST_PRIMES[1]]),
  )
  def test_ntt_intt_identity(self, r, c, moduli):
    ntt_kernel = ntt.NTTBarrett()
    ntt_kernel.precompute_constants(moduli, r, c)

    key = jax.random.PRNGKey(42)
    v = jax.random.randint(
        key, (1, r, c, len(moduli)), 0, min(moduli), dtype=jnp.uint32
    )

    transformed = ntt_kernel.ntt(v)
    recovered = ntt_kernel.intt(transformed)

    np.testing.assert_allclose(v, recovered)

  @hypothesis.given(
      r_log2=st.integers(min_value=1, max_value=6),
      c_log2=st.integers(min_value=1, max_value=6),
  )
  @hypothesis.settings(deadline=None, max_examples=30)
  def test_ntt_intt_identity_property(self, r_log2, c_log2):
    r = 2**r_log2
    c = 2**c_log2
    moduli = [TEST_PRIMES[0]]

    ntt_kernel = ntt.NTTBarrett()
    ntt_kernel.precompute_constants(moduli, r, c)

    key = jax.random.PRNGKey(0)
    v = jax.random.randint(
        key, (1, r, c, len(moduli)), 0, moduli[0], dtype=jnp.uint32
    )

    transformed = ntt_kernel.ntt(v)
    recovered = ntt_kernel.intt(transformed)

    np.testing.assert_allclose(v, recovered)

  def test_ntt_linearity(self):
    r, c = 4, 4
    moduli = [TEST_PRIMES[0]]
    q = moduli[0]
    scale = 123

    ntt_kernel = ntt.NTTBarrett()
    ntt_kernel.precompute_constants(moduli, r, c)

    key = jax.random.PRNGKey(123)
    v1 = jax.random.randint(key, (r, c, 1), 0, q, dtype=jnp.uint32)
    v2 = jax.random.randint(
        jax.random.PRNGKey(124), (r, c, 1), 0, q, dtype=jnp.uint32
    )

    # ntt(v1 + v2) == ntt(v1) + ntt(v2)
    sum_v = (v1.astype(jnp.uint64) + v2.astype(jnp.uint64)) % q
    ntt_sum = ntt_kernel.ntt(sum_v.astype(jnp.uint32))

    ntt_v1 = ntt_kernel.ntt(v1)
    ntt_v2 = ntt_kernel.ntt(v2)
    sum_ntt = (ntt_v1.astype(jnp.uint64) + ntt_v2.astype(jnp.uint64)) % q

    np.testing.assert_allclose(ntt_sum, sum_ntt.astype(jnp.uint32))


if __name__ == "__main__":
  absltest.main()
