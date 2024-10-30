"""Tests for matrix_utils."""
import decimal

import hypothesis
from hypothesis import strategies
import jax
import jax.numpy as jnp
from jaxite.jaxite_lib import jax_helpers
from jaxite.jaxite_lib import matrix_utils
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

POLY_MUL_IMPLS = [
    matrix_utils.toeplitz_poly_mul,
]


@hypothesis.strategies.composite
def matrices(draw, shape, min_value=-(2**31), max_value=2**31 - 1):
  # Note hypothesis.extras.numpy has no build rule in google3
  nrows, ncols = shape
  return np.array(
      draw(
          strategies.lists(
              strategies.lists(
                  strategies.integers(min_value=min_value, max_value=max_value),
                  min_size=ncols,
                  max_size=ncols,
              ),
              min_size=nrows,
              max_size=nrows,
          )
      ),
      dtype=np.int32,
  )


@hypothesis.strategies.composite
def vectors(draw, size, min_value=-(2**31), max_value=2**31 - 1):
  # Note hypothesis.extras.numpy has no build rule in google3
  return np.array(
      draw(
          strategies.lists(
              strategies.integers(min_value=min_value, max_value=max_value),
              min_size=size,
              max_size=size,
          ),
      ),
      dtype=np.int32,
  )


class MatrixUtilsTest(parameterized.TestCase):

  def test_generate_sign_matrix(self):
    sign_matrix = matrix_utils._generate_sign_matrix(3)
    self.assertEqual(sign_matrix.tolist(), [[1, -1, -1], [1, 1, -1], [1, 1, 1]])

  def test_get_cyclic_matrix(self):
    inp = jnp.array([1, 9, 2], dtype=int)
    cyclic_matrix = matrix_utils.toeplitz(inp)
    self.assertEqual(cyclic_matrix.tolist(), [[1, 9, 2], [2, 1, 9], [9, 2, 1]])

  def cast_float64_to_int32(self, x):
    i32max = np.iinfo(np.int32).max
    i32min = np.iinfo(np.int32).min
    if x > i32max:
      return jnp.int32(i32min + x % (i32max + 1))
    elif x < i32min:
      return jnp.int32((x % i32min) + i32max + 1)
    else:
      return jnp.int32(x)

  def _np_polymul(self, poly1, poly2, q_mod=None):
    # poly_mod represents the polynomial to divide by: x^N + 1, N = len(a)
    poly_mod = jnp.zeros(len(poly1) + 1, jnp.uint32)
    poly_mod = poly_mod.at[0].set(1)
    poly_mod = poly_mod.at[len(poly1)].set(1)

    # Reversing the list order because numpy polymul interprets the polynomial
    # with higher-order coefficients first, whereas our code does the opposite
    np_mul = np.polymul(list(reversed(poly1)), list(reversed(poly2)))
    (_, np_poly_mod) = np.polydiv(np_mul, poly_mod)
    np_q_mod = np.mod(np_poly_mod, q_mod) if q_mod else np_poly_mod
    np_pad = np.pad(
        np_q_mod,
        (len(poly1) - len(np_poly_mod), 0),
        'constant',
        constant_values=(0, 0),
    )
    result = jnp.array([
        self.cast_float64_to_int32(x) for x in list(reversed(np_pad))
    ])
    return result

  @hypothesis.given(
      vectors(10),
      vectors(10),
      strategies.sampled_from(POLY_MUL_IMPLS),
  )
  @hypothesis.settings(deadline=None)
  def test_poly_mul(self, poly1, poly2, impl):
    expected = self._np_polymul(poly1, poly2, q_mod=None)
    actual = impl(
        jnp.array(poly1, dtype=int),
        jnp.array(poly2, dtype=int),
    )
    np.testing.assert_array_equal(actual, expected)

  @parameterized.named_parameters(
      dict(testcase_name='no_mul', degree=0, expected=[0, 1, 2, 3]),
      dict(testcase_name='<n', degree=2, expected=[254, 253, 0, 1]),
      dict(testcase_name='>n', degree=7, expected=[1, 2, 3, 0]),
      dict(testcase_name='2n', degree=8, expected=[0, 1, 2, 3]),
      dict(testcase_name='>2n', degree=9, expected=[253, 0, 1, 2]),
  )
  def test_monomial_mul(self, degree, expected):
    poly = jnp.array([0, 1, 2, 3], dtype=int)
    log_modulus = 8
    mono_mul_output = matrix_utils.monomial_mul(poly, degree, log_modulus)
    self.assertEqual(expected, mono_mul_output.tolist())

  @parameterized.named_parameters(
      dict(testcase_name='no_mul', degree=0, expected=[0, 255, 254, 253]),
      dict(testcase_name='<n', degree=2, expected=[2, 3, 0, 255]),
      dict(testcase_name='>n', degree=7, expected=[255, 254, 253, 0]),
      dict(testcase_name='2n', degree=8, expected=[0, 255, 254, 253]),
      dict(testcase_name='>2n', degree=9, expected=[3, 0, 255, 254]),
  )
  def test_monomial_mul_neg(self, degree, expected):
    poly = jnp.array([0, -1, -2, -3], dtype=int)
    log_modulus = 8
    mono_mul_output = matrix_utils.monomial_mul(poly, degree, log_modulus)
    self.assertEqual(expected, mono_mul_output.tolist())

  @parameterized.named_parameters(
      dict(testcase_name='no_div', degree=0, expected=[0, 1, 2, 3]),
      dict(testcase_name='<n', degree=2, expected=[2, 3, 0, 255]),
      dict(testcase_name='>n', degree=7, expected=[253, 0, 1, 2]),
      dict(testcase_name='2n', degree=8, expected=[0, 1, 2, 3]),
      dict(testcase_name='>2n', degree=9, expected=[1, 2, 3, 0]),
  )
  def test_monomial_div(self, degree, expected):
    poly = jnp.array([0, 1, 2, 3], dtype=int)
    log_modulus = 8
    mono_div_output = matrix_utils.monomial_mul(poly, -degree, log_modulus)
    self.assertEqual(expected, mono_div_output.tolist())

  @parameterized.named_parameters(
      dict(testcase_name='no_div', degree=0, expected=[0, 255, 254, 253]),
      dict(testcase_name='<n', degree=2, expected=[254, 253, 0, 1]),
      dict(testcase_name='>n', degree=7, expected=[3, 0, 255, 254]),
      dict(testcase_name='2n', degree=8, expected=[0, 255, 254, 253]),
      dict(testcase_name='>2n', degree=9, expected=[255, 254, 253, 0]),
  )
  def test_monomial_div_neg(self, degree, expected):
    poly = jnp.array([0, -1, -2, -3], dtype=int)
    log_modulus = 8
    mono_div_output = matrix_utils.monomial_mul(poly, -degree, log_modulus)
    self.assertEqual(expected, mono_div_output.tolist())

  @parameterized.named_parameters(
      dict(
          testcase_name='no_rounding',
          values=[8, 16],
          divisor=4,
          expected=[2, 4],
      ),
      dict(
          testcase_name='round_up', values=[8, 23], divisor=3, expected=[3, 8]
      ),
      dict(
          testcase_name='round_down', values=[7, 6], divisor=3, expected=[2, 2]
      ),
      dict(
          testcase_name='exact_half_round_up',
          values=[1, 3, 5, 7],
          divisor=2,
          expected=[1, 2, 3, 4],
      ),
  )
  def test_integer_div(self, values, divisor, expected):
    actual = matrix_utils.integer_div(
        jnp.array(values, dtype=jnp.uint32), jnp.uint32(divisor)
    )
    expected = jnp.array(expected, dtype=jnp.uint32)
    np.testing.assert_array_equal(expected, actual)

  @hypothesis.given(
      strategies.integers(min_value=0, max_value=2**31 - 1),
      strategies.integers(min_value=1, max_value=2**31 - 1),
  )
  @hypothesis.settings(deadline=None)
  def test_integer_div_hypothesis(self, value, divisor):
    # Decimal is needed because Python round() uses "banker's rounding" and
    # rounds to even. We need to force rounding up in all cases.
    expected = (decimal.Decimal(value) / divisor).to_integral(
        rounding=decimal.ROUND_HALF_UP
    )
    expected = jnp.array([expected], dtype=jnp.uint32)
    actual = matrix_utils.integer_div(
        jnp.array([value], dtype=jnp.uint32), jnp.uint32(divisor)
    )
    np.testing.assert_array_equal(expected, actual)

  @parameterized.named_parameters(
      dict(
          testcase_name='<n',
          degree=2,
          expected=[2**32 - 2, 2**32 - 3, 0, 1],
      ),
      dict(testcase_name='>2n', degree=9, expected=[2**32 - 3, 0, 1, 2]),
  )
  def test_monomial_mul_32_bit_modulus(self, degree, expected):
    expected = jnp.array(expected, dtype=jnp.uint32)
    poly = jnp.array([0, 1, 2, 3], dtype=jnp.uint32)
    log_modulus = 32
    mono_mul_output = matrix_utils.monomial_mul(poly, degree, log_modulus)
    np.testing.assert_array_equal(expected, mono_mul_output)

  def test_x_power_n_minus_1(self):
    expected = jnp.array([-1, 0, 1, 0], dtype=jnp.int32)
    actual = matrix_utils.x_power_n_minus_1(n=2, poly_mod_deg=4)
    np.testing.assert_array_equal(expected, actual)

  def test_x_power_n_minus_1_zero(self):
    expected = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
    actual = matrix_utils.x_power_n_minus_1(n=0, poly_mod_deg=4)
    np.testing.assert_array_equal(expected, actual)

  def test_x_power_n_minus_1_reduced_degree_with_sign_flip(self):
    expected = jnp.array([-1, 0, -1, 0], dtype=jnp.int32)
    actual = matrix_utils.x_power_n_minus_1(n=6, poly_mod_deg=4)
    np.testing.assert_array_equal(expected, actual)

  def test_x_power_n_minus_1_reduced_degree_without_sign_flip(self):
    expected = jnp.array([-1, 0, 1, 0], dtype=jnp.int32)
    actual = matrix_utils.x_power_n_minus_1(n=10, poly_mod_deg=4)
    np.testing.assert_array_equal(expected, actual)

  @hypothesis.given(vectors(15), matrices((15, 13)))
  @hypothesis.settings(deadline=None)
  def test_i32_as_u8_matmul(self, lhs, rhs):
    expected = np.dot(lhs, rhs)
    actual = matrix_utils.i32_as_u8_matmul(
        jnp.array(lhs, dtype=jnp.int32),
        jnp.array(rhs, dtype=jnp.int32),
    )
    np.testing.assert_array_equal(expected, actual)

  @hypothesis.given(vectors(512))
  @hypothesis.settings(deadline=None)
  def test_toeplitz_kernelized(self, poly):
    if jax_helpers.get_tpu_version() >= 5:
      multiplier = matrix_utils._generate_sign_matrix(len(poly))
      exp = multiplier.transpose() * matrix_utils.toeplitz(poly)
      actual = matrix_utils.toeplitz_kernelized(poly)
      np.testing.assert_array_equal(exp, actual)

  @hypothesis.given(strategies.integers(min_value=0, max_value=10), vectors(16))
  @hypothesis.settings(deadline=None)
  def test_scale_by_x_power_n_minus_1(self, power, poly):
    matrix = jnp.tile(jnp.array(list(poly)), reps=jnp.array([8, 8, 1]))
    poly_term = matrix_utils.x_power_n_minus_1(power, poly_mod_deg=16)
    expected = matrix_utils.poly_mul_const_matrix(poly_term, matrix)
    actual = matrix_utils.scale_by_x_power_n_minus_1(
        power, matrix, log_modulus=32
    )
    np.testing.assert_array_equal(expected, actual)


def test_hpmatmul_outerproduct():
  """Test the correctness of the Conv-Adapt-Conv algorithm."""
  key = jax.random.key(0)
  mat_a_shape = (4, 256)
  mat_b_shape = (mat_a_shape[1], 4)
  upper_value = (1 << 28) - 1
  modulus_32 = 4294967291
  modulus_64 = jnp.array(modulus_32, dtype=jnp.uint64)
  mat_a = jax.random.randint(key, mat_a_shape, 0, upper_value, dtype=jnp.uint32)
  mat_b = jax.random.randint(key, mat_b_shape, 0, upper_value, dtype=jnp.uint32)

  mat_reference_result = matrix_utils.hpmatmul_golden(mat_a, mat_b, modulus_32)
  mat_result_outerproduct = matrix_utils.hpmatmul_conv_adapt_outer_product(
      mat_a, mat_b
  )
  mat_result_outerproduct = mat_result_outerproduct % modulus_64

  np.testing.assert_array_equal(mat_result_outerproduct, mat_reference_result)
  print('pass testing mat_result_outerproduct == mat_reference_result')


def test_hpmatmul_bat():
  """Test the correctness of the Basis Align Transformation (BAT) algorithm."""
  key = jax.random.key(0)
  mat_a_shape = (4, 256)
  mat_b_shape = (mat_a_shape[1], 4)
  upper_value = (1 << 28) - 1
  modulus_32 = 4294967291
  modulus_64 = jnp.array(modulus_32, dtype=jnp.uint64)
  mat_a = jax.random.randint(key, mat_a_shape, 0, upper_value, dtype=jnp.uint32)
  mat_b = jax.random.randint(key, mat_b_shape, 0, upper_value, dtype=jnp.uint32)

  mat_reference_result = matrix_utils.hpmatmul_golden(mat_a, mat_b, modulus_32)
  compiled_mat_a = matrix_utils.hpmatmul_offline_compile_bat(mat_a, modulus_32)

  mat_result_bat = matrix_utils.hpmatmul_bat_adapt(compiled_mat_a, mat_b)
  mat_result_bat = mat_result_bat % modulus_64

  # Sanity Checking
  for i in range(mat_a.shape[0]):
    for j in range(mat_b.shape[1]):
      if mat_result_bat[i, j] != mat_reference_result[i][j]:
        print(
            f'mat_result_bat[{i}, {j}]={mat_result_bat[i, j]} not match'
            f' mat_reference_result[{i}, {j}]={ mat_reference_result[i][j]}'
        )

  np.testing.assert_array_equal(mat_result_bat, mat_reference_result)
  print('pass testing mat_result_bat == mat_reference_result')


def test_hpmatmul_bat_full_precision():
  """Test the correctness of the Conv-Adapt-Conv algorithm."""
  key = jax.random.key(0)
  mat_a_shape = (4, 256)
  mat_b_shape = (mat_a_shape[1], 4)
  upper_value = (1 << 32) - 1
  modulus_32 = 4294967291
  modulus_64 = jnp.array(modulus_32, dtype=jnp.uint64)
  mat_a = jax.random.randint(
      key, mat_a_shape, 0, upper_value, dtype=jnp.uint32
  )
  mat_b = jax.random.randint(
      key, mat_b_shape, 0, upper_value, dtype=jnp.uint32
  )

  mat_reference_result = matrix_utils.hpmatmul_golden(mat_a, mat_b, modulus_32)
  compiled_mat_a = matrix_utils.hpmatmul_offline_compile_bat(
      mat_a, modulus_32
  )

  mat_result_bat = matrix_utils.hpmatmul_bat_adapt(compiled_mat_a, mat_b)
  mat_result_bat = mat_result_bat % modulus_64

  # Sanity Checking
  for i in range(mat_a.shape[0]):
    for j in range(mat_b.shape[1]):
      if mat_result_bat[i, j] != mat_reference_result[i][j]:
        print(
            f'mat_result_bat[{i}, {j}]={mat_result_bat[i, j]} not match'
            f' mat_reference_result[{i}, {j}]={ mat_reference_result[i][j]}'
        )

  np.testing.assert_array_equal(mat_result_bat, mat_reference_result)
  print('pass testing mat_result_bat == mat_reference_result')

if __name__ == '__main__':
  absltest.main()
