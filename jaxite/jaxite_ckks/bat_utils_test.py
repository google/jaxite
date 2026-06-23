"""Tests for bat_utils."""

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
    moduli = jnp.array([1073184769, 1073479681], dtype=jnp.uint32)

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
    key_matrix_bat = bat_utils.basis_aligned_transform_key(key_matrix, moduli)

    # 5. Run BAT matrix-vector multiplication
    actual_uint64 = bat_utils.matmul_bat_key_vector(vector_v, key_matrix_bat)
    actual = (actual_uint64 % moduli_expanded).astype(jnp.uint32)

    np.testing.assert_array_equal(np.array(actual), np.array(expected))


if __name__ == "__main__":
  absltest.main()
