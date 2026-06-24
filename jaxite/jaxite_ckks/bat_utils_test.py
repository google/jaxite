"""Tests for basis-aligned transformation (BAT) utility functions."""

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import bat_utils
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

# Enable 64-bit precision for large integer arithmetic
jax.config.update("jax_enable_x64", True)


class BatUtilsTest(parameterized.TestCase):

  def test_bat_key_vector_matmul(self):
    degree = 8
    num_moduli = 2
    moduli_list = [1073184769, 1073479681]
    moduli = jnp.array(moduli_list, dtype=jnp.uint32)

    key = jax.random.key(0)
    k0, k1, k2, k3 = jax.random.split(key, 4)

    # 1. Generate random key0 and key1 of shape (2, degree, num_moduli)
    key0 = jax.random.randint(
        k0,
        shape=(2, degree, num_moduli),
        minval=0,
        maxval=2**30,
        dtype=jnp.uint32,
    )
    key1 = jax.random.randint(
        k1,
        shape=(2, degree, num_moduli),
        minval=0,
        maxval=2**30,
        dtype=jnp.uint32,
    )

    # Pack into key_matrix (degree, num_moduli, 2, 2)
    # key0 corresponds to column 0, key1 to column 1
    # row 0 has key0[0] and key1[0]; row 1 has key0[1] and key1[1]
    stacked = jnp.stack(
        [key0, key1], axis=1
    )  # Shape: (2, 2, degree, num_moduli)
    key_matrix = jnp.transpose(
        stacked, (2, 3, 0, 1)
    )  # Shape: (degree, num_moduli, 2, 2)

    # 2. Generate random plaintexts a and b of shape (degree, num_moduli)
    a = jax.random.randint(
        k2, shape=(degree, num_moduli), minval=0, maxval=2**30, dtype=jnp.uint32
    )
    b = jax.random.randint(
        k3, shape=(degree, num_moduli), minval=0, maxval=2**30, dtype=jnp.uint32
    )
    vector_v = jnp.stack([a, b], axis=-1)  # Shape: (degree, num_moduli, 2)

    # 3. Compute expected product using exact modular arithmetic
    # prod0 = key0 * a (element-wise over degree and moduli)
    # prod1 = key1 * b
    # expected = (prod0 + prod1) % moduli
    moduli_expanded = moduli.reshape(1, 1, -1)
    prod0 = (key0.astype(jnp.uint64) * a.astype(jnp.uint64)) % moduli_expanded
    prod1 = (key1.astype(jnp.uint64) * b.astype(jnp.uint64)) % moduli_expanded
    expected = (prod0 + prod1) % moduli_expanded

    # 4. Perform BAT pre-transformation on key_matrix
    key_matrix_transposed = jnp.transpose(key_matrix, (0, 2, 3, 1))
    key_matrix_bat_u8 = bat_utils.basis_aligned_transformation(
        key_matrix_transposed, moduli_list
    )
    key_matrix_bat = jnp.transpose(key_matrix_bat_u8, (1, 4, 2, 3, 0, 5))

    # 5. Run BAT matrix-vector multiplication
    actual_uint64 = bat_utils.matmul_bat_einsum(
        vector_v, key_matrix_bat, "...ijvq,ijuvqp->...ijup"
    )
    actual_uint64_transposed = jnp.moveaxis(actual_uint64, -1, 0)
    actual = (actual_uint64_transposed % moduli_expanded).astype(jnp.uint32)

    np.testing.assert_array_equal(np.array(actual), np.array(expected))

  def test_basis_aligned_transformation_shape(self):
    degree = 8
    moduli = [1073184769, 1073479681]
    matrix = jnp.ones((degree, len(moduli)), dtype=jnp.uint32)
    actual = bat_utils.basis_aligned_transformation(matrix, moduli)
    self.assertEqual(actual.shape, (4, degree, len(moduli), 4))
    self.assertEqual(actual.dtype, jnp.uint8)

  @parameterized.parameters(True, False)
  def test_large_values_no_overflow(self, merge_byte_dimension):
    degree = 8
    moduli = [1073184769, 1073479681]
    num_moduli = len(moduli)
    max_val = 2**32 - 5
    lhs = jax.random.randint(
        jax.random.key(0), (degree, num_moduli), 0, max_val, dtype=jnp.uint32
    )
    rhs = jax.random.randint(
        jax.random.key(1), (degree, num_moduli), 0, max_val, dtype=jnp.uint32
    )

    if merge_byte_dimension:
      # If merging, we need to do matrix multiplication setup.
      # Let's say rhs is (num_moduli, num_moduli)
      rhs_square = jax.random.randint(
          jax.random.key(1),
          (num_moduli, num_moduli),
          0,
          max_val,
          dtype=jnp.uint32,
      )
      rhs_bat_raw = bat_utils.basis_aligned_transformation(rhs_square, moduli)
      rhs_bat = rhs_bat_raw.transpose(1, 0, 2, 3).reshape(-1, num_moduli, 4)
      actual = bat_utils.matmul_bat_einsum(
          lhs, rhs_bat, "...q,qpb->...pb", merge_byte_dimension=True
      )
      expected_list = []
      for p in range(num_moduli):
        p_mod = moduli[p]
        lhs_u64 = lhs.astype(jnp.uint64)
        rhs_col_u64 = rhs_square[:, p].astype(jnp.uint64)
        # Compute dot product using element-wise multiplication and sum.
        prod = (
            jnp.sum((lhs_u64 * rhs_col_u64.reshape(1, -1)) % p_mod, axis=1)
            % p_mod
        )
        expected_list.append(prod)
      expected = jnp.stack(expected_list, axis=-1).astype(jnp.uint32)
    else:
      rhs_bat = bat_utils.basis_aligned_transformation(rhs, moduli)
      actual = bat_utils.matmul_bat_einsum(
          lhs, rhs_bat, "...mq,q...mb->...mb", merge_byte_dimension=False
      )
      expected = (lhs.astype(jnp.uint64) * rhs.astype(jnp.uint64)) % jnp.array(
          moduli, dtype=jnp.uint64
      )

    np.testing.assert_array_equal(
        actual % jnp.array(moduli, dtype=jnp.uint64), expected
    )

  def test_bat_merge_byte_dimension_rank1_lhs(self):
    original_moduli = [1073184769, 1073479681]
    target_moduli = [1073741825, 1073872897, 1074003969]
    num_q = len(original_moduli)
    num_p = len(target_moduli)

    lhs = jax.random.randint(
        jax.random.key(0), (num_q,), 0, 2**30, dtype=jnp.uint32
    )
    rhs = jax.random.randint(
        jax.random.key(1), (num_q, num_p), 0, 2**30, dtype=jnp.uint32
    )

    rhs_bat_raw = bat_utils.basis_aligned_transformation(rhs, target_moduli)
    rhs_bat = rhs_bat_raw.transpose(1, 0, 2, 3).reshape(-1, num_p, 4)

    actual = bat_utils.matmul_bat_einsum(
        lhs, rhs_bat, "q,qpb->pb", merge_byte_dimension=True
    )

    expected_list = []
    for p in range(num_p):
      p_mod = target_moduli[p]
      lhs_u64 = lhs.astype(jnp.uint64)
      rhs_col_u64 = rhs[:, p].astype(jnp.uint64)
      # Compute dot product using element-wise multiplication and sum.
      prod = jnp.sum((lhs_u64 * rhs_col_u64) % p_mod) % p_mod
      expected_list.append(prod)
    expected = jnp.stack(expected_list, axis=-1).astype(jnp.uint32)

    np.testing.assert_array_equal(
        actual % jnp.array(target_moduli, dtype=jnp.uint64), expected
    )


if __name__ == "__main__":
  absltest.main()
