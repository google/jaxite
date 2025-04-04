"""A module for operations on test CKKS evaluation kernels including.

- Modsub
- HESub
"""

from concurrent import futures
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxite.jaxite_word import sub

from absl.testing import absltest
from absl.testing import parameterized


ProcessPoolExecutor = futures.ProcessPoolExecutor

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


class CKKSEvalSubTest(parameterized.TestCase):
  """A base class for running bootstrap tests."""

  def __init__(self, *args, **kwargs):
    super(CKKSEvalSubTest, self).__init__(*args, **kwargs)
    self.debug = False  # dsiable it from printing the test input values
    self.modulus_element_0_tower_0 = 1152921504606748673  # 60 (k=60->2k=120)
    self.modulus_element_0_tower_1 = 268664833  # 28 (k=28->2k=56)
    self.modulus_element_0_tower_2 = 557057  # 19 (k=19->2k=38)
    self.random_key = jax.random.key(0)
    self.in_c1 = [
        [761974115069642497, 186812814, 396780],
        [1119697542422587247, 195711320, 415240],
    ]
    self.in_c2 = [
        [723287396072165360, 91967352, 112274],
        [251652059326221653, 111494737, 534294],
    ]
    self.refer_sub_result = [
        [38686718997477137, 94845462, 284506],
        [868045483096365594, 84216583, 438003],
    ]

    self.random_key = jax.random.key(0)

  def random(self, shape, modulus_list, dtype=jnp.int32):
    assert len(modulus_list) == shape[1]

    return jnp.concatenate(
        [
            jax.random.randint(
                self.random_key,
                shape=(shape[0], 1, shape[2]),
                minval=0,
                maxval=bound,
                dtype=dtype,
            )
            for bound in modulus_list
        ],
        axis=1,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="jax_sub",
          test_target=sub.jax_sub,
          modulus_list=[1152921504606748673, 268664833, 557057],
          shape=(2, 3, 16384),  # number of elements, number of towers, degree
      ),
      dict(
          testcase_name="vmap_sub",
          test_target=sub.vmap_sub,
          modulus_list=[1152921504606748673, 268664833, 557057],
          shape=(2, 3, 16384),  # number of elements, number of towers, degree
      ),
  )
  def test_sub(
      self,
      test_target: Callable[[Any, Any, Any], Any],
      modulus_list=jax.Array,
      shape=tuple[int, int, int],
  ):
    """This function tests the sub function using Python native integer data type with arbitrary precision.

    This test finishes in 1.05 second.

    Args:
      test_target: The function to test.
      modulus_list: A jax.Array of integers.
      shape: A tuple of integers representing the shape of the input arrays.
    """
    # Only test a single element to save comparison time,
    # Correctness-wise, it's sufficient for sub.
    value_a = self.random(shape, modulus_list, dtype=jnp.uint64)
    value_b = self.random(shape, modulus_list, dtype=jnp.uint64)
    for i in range(shape[0]):
      for j in range(shape[1]):
        value_a = value_a.at[i, j, 0].set(self.in_c1[i][j])
        value_b = value_b.at[i, j, 0].set(self.in_c2[i][j])
    assert value_a.shape == shape
    assert value_b.shape == shape
    modulus_list = jnp.array(modulus_list, dtype=jnp.uint64)
    refer_sub_result = jnp.array(self.refer_sub_result, dtype=jnp.uint64)
    result = test_target(value_a, value_b, modulus_list)
    self.assertEqual(result[:, :, 0].all(), refer_sub_result.all())


if __name__ == "__main__":
  absltest.main()
