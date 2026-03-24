"""Tests for basis conversion."""

import functools
import hypothesis
from hypothesis import strategies as st
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import basis_conversion
from jaxite.jaxite_ckks import rns_utils
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

# Enable 64-bit precision for large integer arithmetic
jax.config.update("jax_enable_x64", True)


def verify_approximate_basis_conversion(
    in_tower: jax.Array,
    out_tower: jax.Array,
    orig_moduli: list[int],
    target_moduli: list[int],
):
  """Verifies the mathematical correctness of approximate basis conversion.

  Approximate basis conversion from a source basis Q = {q1, ..., qk} to a
  target basis P = {p1, ..., pl} takes an RNS representation of an integer x
  and produces an RNS representation of an integer y such that:

      y = x + v * Q

  where v is a small integer in the range [0, k), where k is the number of
  moduli in the source basis.

  This function performs the following checks for each value in the input:

  1. Reconstructs the original integer x from its residues mod q_i.
  2. For each target modulus p_j, verifies that the output residue y_j
     satisfies:

     y_j = (x + v * Q) mod p_j  =>  (y_j - x) mod p_j = (v * Q) mod p_j
     for some v in [0, k).

  Args:
    in_tower: Input coefficients in source basis, shape (..., sizeQ).
    out_tower: Output coefficients in target basis, shape (..., sizeP).
    orig_moduli: The list of moduli in the source basis.
    target_moduli: The list of moduli in the target basis.
  """
  # Q is the product of source moduli
  q_prod = functools.reduce(lambda a, b: a * b, orig_moduli)
  num_source_moduli = len(orig_moduli)

  # Flatten the leading dimensions to process all elements
  flat_in = np.array(in_tower).reshape(-1, num_source_moduli)
  flat_out = np.array(out_tower).reshape(-1, len(target_moduli))

  # Reconstruct the integers x from residues (source basis)
  # residues shape: (num_source_moduli, num_elements)
  reconstructed_x = rns_utils.reconstruct_crt(flat_in.T.tolist(), orig_moduli)

  for i, x in enumerate(reconstructed_x):
    for j, p in enumerate(target_moduli):
      actual_y_j = int(flat_out[i, j])

      # Check the property: (actual_y_j - x) % p == (v * q_prod) % p
      diff = (actual_y_j - (x % p)) % p
      q_mod_p = q_prod % p

      # For approximate conversion, v is in [0, k)
      found_v = any(diff == (v * q_mod_p) % p for v in range(num_source_moduli))
      if not found_v:
        raise AssertionError(
            f"Result {actual_y_j} at index {i}, modulus {p} "
            f"not of form (x + vQ) mod P. x%p={x%p}, "
            f"Q%p={q_mod_p}, diff={diff}"
        )


def generate_rns_value(ring_dim, moduli, key):
  """Generate an RNS value for the given ring_dim and moduli."""
  return jnp.concatenate(
      [
          jax.random.randint(
              key, shape=(ring_dim, 1), minval=0, maxval=m, dtype=jnp.uint64
          )
          for m in moduli
      ],
      axis=1,
  )


# A list of 40 large distinct primes (around 30 bits) for testing.
# For the purpose of basis conversion, these need not be NTT-friendly primes.
TEST_PRIMES = [
    536870951,
    536871001,
    536871017,
    536871019,
    536871029,
    536871061,
    536871089,
    536871091,
    536871119,
    536871131,
    536871157,
    536871173,
    536871191,
    536871199,
    536871233,
    536871259,
    536871263,
    536871301,
    536871311,
    536871319,
    536871331,
    536871337,
    536871367,
    536871389,
    536871421,
    536871427,
    536871449,
    536871463,
    536871481,
    536871499,
    536871523,
    536871527,
    536871551,
    536871563,
    536871583,
    536871613,
    536871637,
    536871649,
    536871703,
]


class BasisConversionBarrettTest(parameterized.TestCase):

  @parameterized.parameters(
      ([1073753729, 1073738977, 1073753281, 1073739041], 16),
      ([1073753729, 1073738977, 1073753281, 1073739041], 64),
      ([1073753729, 1073738977, 1073753281, 1073739041], 32768),
  )
  def test_basis_conversion(self, moduli, ring_dim):
    # Define two scenarios:
    # 0: [q0, q1] -> [q2, q3]
    # 1: [q0, q1, q2] -> [q3]
    control_indices = [
        ([0, 1], [2, 3]),
        ([0, 1, 2], [3]),
    ]

    kernel = basis_conversion.BasisConversionBarrett()
    kernel.precompute_constants(moduli, control_indices)

    key = jax.random.key(0)

    for control_index, (orig_idx, target_idx) in enumerate(control_indices):
      orig_moduli = [moduli[i] for i in orig_idx]
      target_moduli = [moduli[i] for i in target_idx]

      in_tower = generate_rns_value(ring_dim, orig_moduli, key)
      out_tower = kernel.basis_change(in_tower, control_index=control_index)
      verify_approximate_basis_conversion(
          in_tower, out_tower, orig_moduli, target_moduli
      )

  @hypothesis.settings(deadline=None, max_examples=50)
  @hypothesis.given(
      # We need at least 1 prime for each basis. Total between 2 and 30.
      num_total_moduli=st.integers(min_value=2, max_value=30),
      # log2 of ring_dim, power of 2 between 4 (2^2) and 128 (2^7)
      log2_ring_dim=st.integers(min_value=2, max_value=7),
      # random seed for jax
      seed=st.integers(min_value=0, max_value=2**32 - 1),
  )
  def test_hypothesis(self, num_total_moduli, log2_ring_dim, seed):
    all_moduli = TEST_PRIMES[:num_total_moduli]

    # Random split halfway through the list of moduli.
    # split point between 1 and num_total_moduli - 1
    split = max(1, num_total_moduli // 2)
    orig_idx = list(range(split))
    target_idx = list(range(split, num_total_moduli))

    control_indices = [(orig_idx, target_idx)]

    kernel = basis_conversion.BasisConversionBarrett()
    kernel.precompute_constants(all_moduli, control_indices)

    key = jax.random.key(seed)
    orig_moduli = [all_moduli[i] for i in orig_idx]
    target_moduli = [all_moduli[i] for i in target_idx]

    ring_dim = 2**log2_ring_dim
    in_tower = generate_rns_value(ring_dim, orig_moduli, key)
    out_tower = kernel.basis_change(in_tower, control_index=0)

    verify_approximate_basis_conversion(
        in_tower, out_tower, orig_moduli, target_moduli
    )


if __name__ == "__main__":
  absltest.main()
