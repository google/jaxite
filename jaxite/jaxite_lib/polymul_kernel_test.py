import jax.numpy as jnp
from jaxite.jaxite_lib import polymul_kernel
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized


_SEEDS = list(range(3))


def random(shape, dtype=np.int32):
  return jnp.array(
      np.random.randint(low=0, high=2**31 - 1, size=shape, dtype=dtype)
  )


class PolymulKernelTest(parameterized.TestCase):

  @parameterized.product(seed=_SEEDS)
  def test_i32_matmul_vs_reference(self, seed: int):
    np.random.seed(seed)
    lhs = random(shape=(24, 512))  # leading dimension must be a multiple of 8
    rhs = random(shape=(512, 512))
    expected = polymul_kernel.fallback_i32_matmul(lhs, rhs).astype(jnp.uint32)
    actual = polymul_kernel.i32_matmul(lhs, rhs)
    np.testing.assert_array_equal(expected, actual)

  def test_vector_matrix_vs_reference(self):
    vector = random(shape=(18, 512))
    matrix = random(shape=(18, 3, 512))
    expected = polymul_kernel.fallback_vector_matrix_polymul(vector, matrix)
    actual = polymul_kernel.negacyclic_vector_matrix_polymul(vector, matrix)
    np.testing.assert_array_equal(expected, actual)

  @parameterized.product(
      seed=_SEEDS,
  )
  def test_many_seeds(self, seed: int):
    np.random.seed(seed)
    vector = random(shape=(18, 512), dtype=jnp.uint32)
    matrix = random(shape=(18, 3, 512), dtype=jnp.uint32)
    expected = polymul_kernel.fallback_vector_matrix_polymul(vector, matrix)
    actual = polymul_kernel.negacyclic_vector_matrix_polymul(vector, matrix)
    np.testing.assert_array_equal(expected, actual)


if __name__ == "__main__":
  absltest.main()
