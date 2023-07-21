"""Tests for random_source."""

import jax.numpy as jnp
from jaxite.jaxite_lib import random_source
from absl.testing import absltest
from absl.testing import parameterized


class ShapeGeneratorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.constant_function = lambda: 1

  def test_valid_shape(self):
    test_shape = (10, 10)
    result = random_source._shape_generator(self.constant_function, test_shape)
    self.assertEqual(result.shape, test_shape)

  def test_1d_shape_is_valid(self):
    test_shape = (10,)
    result = random_source._shape_generator(self.constant_function, test_shape)
    self.assertEqual(result.shape, test_shape)

  def test_nd_shape_is_valid(self):
    test_shape = (2, 2, 2, 2)
    result = random_source._shape_generator(self.constant_function, test_shape)
    self.assertEqual(result.shape, test_shape)

  def test_invalid_shape(self):
    test_shape = (-1, 1)
    with self.assertRaises(ValueError):
      _ = random_source._shape_generator(self.constant_function, test_shape)


class AllRngsTest(absltest.TestCase):

  def test_sk_uniform_is_binary(self):
    for rng_class in random_source.ALL_RNGS:
      data = [int(x) for x in rng_class().sk_uniform(shape=(100,))]
      non_binary_values = set(data) - set([0, 1])
      self.assertEmpty(non_binary_values)


class CycleRngTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_shape = (10,)
    self.const_normal_noise = 10
    self.rng = random_source.CycleRng(self.const_normal_noise)

  def test_uniform_matches_random_data(self):
    expected = jnp.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 0])
    self.assertTrue(jnp.all(self.rng.uniform(self.test_shape) == expected))

  def test_rounded_normal_matches_const_normal_noise(self):
    self.assertTrue(
        jnp.all(
            self.rng.rounded_normal(self.test_shape)
            == self.const_normal_noise * jnp.ones(self.test_shape)
        )
    )


@parameterized.parameters(
    random_source.SystemRandomSource(),
    random_source.PseudorandomSource(),
)
class CryptographicallySecureRandomSourceTest(parameterized.TestCase):
  """Test cryptographically-secure random sources.

  Seeding will have no effect, since we cannot test these generators
  deterministically.
  """

  def test_uniform_valid_and_correct_shape(
      self, rng: random_source.RandomSource
  ):
    test_shape = (10, 10)
    result = rng.uniform(test_shape)
    self.assertEqual(result.shape, test_shape)

  def test_rounded_normal_valid_and_correct_shape(
      self, rng: random_source.RandomSource
  ):
    test_shape = (10, 10)
    result = rng.rounded_normal(test_shape)
    self.assertEqual(result.shape, test_shape)

  def test_rounded_normal_correct_type(self, rng: random_source.RandomSource):
    test_shape = (10, 10)
    result = rng.rounded_normal(test_shape, dtype=jnp.int32)
    self.assertEqual(result.dtype, jnp.int32)


@parameterized.parameters(
    random_source.SystemRandomSource(uniform_bounds=(0, 100)),
    random_source.PseudorandomSource(uniform_bounds=(0, 100)),
)
def test_uniform_elements_within_bounds(self, rng: random_source.RandomSource):
  test_shape = (100, 100)
  result = rng.uniform(test_shape)
  self.assertTrue(jnp.all((result >= 0) & (result <= 100)))


class NormalOnlyRandomSourceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # The generator we're passing is expected to *always* produce a value of 0
    # uniform distribution.
    self.rng = random_source.NormalOnlyRng()

  def test_uniform_valid_and_correct_shape(self):
    test_shape = (10, 10)
    result = self.rng.uniform(test_shape)
    self.assertEqual(result.shape, test_shape)

  def test_rounded_normal_valid_and_correct_shape(self):
    test_shape = (10, 10)
    result = self.rng.rounded_normal(test_shape)
    self.assertEqual(result.shape, test_shape)

  def test_rounded_normal_correct_type(self):
    test_shape = (10, 10)
    result = self.rng.rounded_normal(test_shape, dtype=jnp.int32)
    self.assertEqual(result.dtype, jnp.int32)


class ConstantUniformRandomSourceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # The generator we're passing is expected to produce constant values from
    # its uniform distribution.
    self.const_uniform = 7
    self.rng = random_source.ConstantUniformRng(
        const_uniform=self.const_uniform
    )

  def test_uniform_valid_and_correct_shape(self):
    test_shape = (10, 10)
    result = self.rng.uniform(test_shape)
    self.assertEqual(result.shape, test_shape)

  def test_rounded_normal_valid_and_correct_shape(self):
    test_shape = (10, 10)
    result = self.rng.rounded_normal(test_shape)
    self.assertEqual(result.shape, test_shape)

  def test_rounded_normal_correct_type(self):
    test_shape = (10, 10)
    result = self.rng.rounded_normal(test_shape, dtype=jnp.int32)
    self.assertEqual(result.dtype, jnp.int32)


class ZeroRandomSourceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Both the Uniform distribution and the rounded Normal distribution are
    # expected to have all elements set to zero.
    self.rng = random_source.ZeroRng()

  def test_uniform_valid_and_correct_shape(self):
    test_shape = (10, 10)
    result = self.rng.uniform(test_shape)
    self.assertEqual(result.shape, test_shape)

  def test_rounded_normal_valid_and_correct_shape(self):
    test_shape = (10, 10)
    result = self.rng.rounded_normal(test_shape)
    self.assertEqual(result.shape, test_shape)

  def test_rounded_normal_elements_equal_zero(self):
    test_shape = (10, 10)
    result = self.rng.rounded_normal(test_shape)
    self.assertTrue(jnp.all(result == 0))

  def test_rounded_normal_correct_type(self):
    test_shape = (10, 10)
    result = self.rng.rounded_normal(test_shape, dtype=jnp.int32)
    self.assertEqual(result.dtype, jnp.int32)


if __name__ == '__main__':
  absltest.main()
