"""A module for operations on test CKKS evaluation kernels including.

- ModAdd
- HEAdd
- HESub
- HEMul
- HERotate
"""

from concurrent import futures
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxite.jaxite_word import add

from absl.testing import absltest
from absl.testing import parameterized


ProcessPoolExecutor = futures.ProcessPoolExecutor

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


class CKKSEvalKernelsTest(parameterized.TestCase):
  """A base class for running bootstrap tests."""

  def __init__(self, *args, **kwargs):
    super(CKKSEvalKernelsTest, self).__init__(*args, **kwargs)
    self.debug = False  # dsiable it from printing the test input values
    self.modulus_element_0_tower_0 = 1152921504606748673
    self.modulus_element_0_tower_1 = 268664833
    self.modulus_element_0_tower_2 = 557057
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
          testcase_name="jax_add",
          test_target=add.jax_add,
          modulus_list=[1152921504606748673, 268664833, 557057],
          shape=(2, 3, 16384),  # number of elements, number of towers, degree
      ),
      dict(
          testcase_name="vmap_add",
          test_target=add.vmap_add,
          modulus_list=[1152921504606748673, 268664833, 557057],
          shape=(2, 3, 16384),  # number of elements, number of towers, degree
      ),
  )
  def test_add(
      self,
      test_target: Callable[[Any, Any, Any], Any],
      modulus_list=jax.Array,
      shape=tuple[int, int, int],
  ):
    """This function tests the add function using Python native integer data type with arbitrary precision.

    This test finishes in 1.05 second.

    Args:
      test_target: The function to test.
      modulus_list: A jax.Array of integers.
      shape: A tuple of integers representing the shape of the input arrays.
    """
    # Only test a single element to save comparison time,
    # Correctness-wise, it's sufficient for add.
    value_a = self.random(shape, modulus_list, dtype=jnp.uint64)
    value_b = self.random(shape, modulus_list, dtype=jnp.uint64)
    assert value_a.shape == shape
    assert value_b.shape == shape
    result_a_plus_b = []
    for element_id in range(value_a.shape[0]):
      result_a_plus_b_one_element = []
      for tower_id in range(value_a.shape[1]):
        add_res = int(value_b[element_id, tower_id, 0]) + int(
            value_a[element_id, tower_id, 0]
        )
        if add_res > modulus_list[tower_id]:
          add_res = add_res - modulus_list[tower_id]
        result_a_plus_b_one_element.append(add_res)
      result_a_plus_b.append(result_a_plus_b_one_element)
    result_a_plus_b = jnp.array(result_a_plus_b, dtype=jnp.uint64)
    modulus_list = jnp.array(modulus_list, dtype=jnp.uint64)
    result = test_target(value_a, value_b, modulus_list)
    self.assertEqual(result[:, :, 0].all(), result_a_plus_b.all())


if __name__ == "__main__":
  absltest.main()
