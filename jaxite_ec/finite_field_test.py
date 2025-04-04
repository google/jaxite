import random

import jax
import jax.numpy as jnp
from jaxite.jaxite_ec import util
import jaxite.jaxite_ec.algorithm.finite_field as pyff
import jaxite.jaxite_ec.finite_field as ff
import numpy as np

from absl.testing import absltest


jax.config.update("jax_enable_x64", True)

randint = random.randint


def list_operation(a, b, func):
  return [func(ai, bi) for ai, bi in zip(a, b)]


def list_operation_three(a, b, c, func):
  return [func(ai, bi, ci) for ai, bi, ci in zip(a, b, c)]


class FiniteFieldTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.value_a = [
        0xBE4FBE5D03CE926E40E058BBDC3269C78CFAFED39796CD13EC8E9B0072DB2538DFFBCA05804574D9E2FF7EEB1DE219,
        0x008848DEFE740A67C8FC6225BF87FF5485951E2CAA9D41BB188282C8BD37CB5CD5481512FFCD394EEAB9B16EB21BE9EF,
    ]
    self.value_b = [
        0x82A0ED372BFAB8198D0667A1DC5E299C1F6C8FEB0ACD4D05A228325117BE63EAE5BABE6807F41C6C8016BDAC251CFE,
        0x01914A69C5102EFF1F674F5D30AFEEC4BD7FB348CA3E52D96D182AD44FB82305C2FE3D3634A9591AFD82DE55559C8EA6,
    ]
    self.value_c = [
        0x125E69CE765D167C0B19F8D6D6708D39C7782F33B6D320802E2FFA92BBB12DBB3897EAF9CC4CF67E487478F3C3FAD16,
        0x01AC3A384FC584EFD3E7F2C5A2927E7D454875C874A051027B9E7363D08942533EDE85DAE295D8CAB2751085206BCA76,
    ]
    self.value_a_jax = util.int_list_to_array(
        self.value_a, base=util.BASE, array_size=util.U16_CHUNK_NUM
    )
    self.value_b_jax = util.int_list_to_array(
        self.value_b, base=util.BASE, array_size=util.U16_CHUNK_NUM
    )
    self.value_c_jax = util.int_list_to_array(
        self.value_c, base=util.BASE, array_size=util.U16_CHUNK_NUM
    )

  @absltest.skip("This test is only needed if u need to generate RNS Matrix.")
  def test_generate_rns_precompute_matrix(self):
    rns_mat = util.construct_rns_matrix(util.MODULUS_377_INT)
    rns_stack_mat = rns_mat[0]
    cor_mat = rns_mat[1]
    print("Printing out RNS matrix")
    print("(")
    for i in range(len(rns_stack_mat)):
      print("(", end="")
      for j in range(len(rns_stack_mat[0])):
        if j == len(rns_stack_mat[0]) - 1:
          print(rns_stack_mat[i][j], end="), ")
        else:
          print(rns_stack_mat[i][j], end=", ")
      print("")
    print(")")

    print("Printing out Correction matrix")
    print("(")
    for i in range(len(cor_mat)):
      print("( ", end="")
      for j in range(len(cor_mat[0])):
        if j == len(cor_mat[0]) - 1:
          print(cor_mat[i][j], end="), ")
        else:
          print(cor_mat[i][j], end=", ")
      print("")
    print(")")
    print()

  def test_add_two(self):
    result_jax = ff.add_2u16(self.value_a_jax, self.value_b_jax)
    result = util.array_to_int_list(result_jax, util.BASE)

    result_ref = list_operation(self.value_a, self.value_b, lambda a, b: a + b)

    self.assertEqual(result, result_ref)

  def test_add_three(self):
    result_jax = ff.add_3u16(
        self.value_a_jax, self.value_b_jax, self.value_c_jax
    )
    result = util.array_to_int_list(result_jax, util.BASE)
    result_ref = list_operation(
        list_operation(self.value_a, self.value_b, lambda a, b: a + b),
        self.value_c,
        lambda a, b: a + b,
    )

    self.assertEqual(result, result_ref)

  def test_cond_sub_1(self):
    result_jax = ff.cond_sub_2u16(self.value_a_jax, self.value_b_jax)
    result = util.array_to_int_list(result_jax, util.BASE)

    def cond_sub(a, b):
      if a < b:
        return a + util.MODULUS_377_INT - b
      else:
        return a - b

    result_ref = list_operation(self.value_a, self.value_b, cond_sub)
    self.assertEqual(result, result_ref)

  def test_cond_sub_2(self):
    result_jax = ff.cond_sub_2u16(self.value_b_jax, self.value_a_jax)
    result = util.array_to_int_list(result_jax, util.BASE)

    def cond_sub(a, b):
      if a < b:
        return a + util.MODULUS_377_INT - b
      else:
        return a - b

    result_ref = list_operation(self.value_b, self.value_a, cond_sub)

    self.assertEqual(result, result_ref)

  def test_cond_sub_mod_1(self):
    value_list = [util.MODULUS_377_INT + 123, util.MODULUS_377_INT - 5432]
    value_jax = util.int_list_to_array(
        value_list, base=util.BASE, array_size=util.U16_CHUNK_NUM
    )
    result_jax = ff.cond_sub_mod_u16(value_jax)
    result = util.array_to_int_list(result_jax, util.BASE)

    def cond_sub_mod(a):
      if a < util.MODULUS_377_INT:
        return a
      else:
        return a - util.MODULUS_377_INT

    result_ref = [cond_sub_mod(a) for a in value_list]

    self.assertEqual(result, result_ref)

  def test_mul_1(self):
    result_jax = ff.mul_2u16(self.value_a_jax, self.value_b_jax)
    result = util.array_to_int_list(result_jax, util.BASE)
    result_ref = list_operation(self.value_a, self.value_b, lambda a, b: a * b)
    self.assertEqual(result, result_ref)

  def test_mod_mul_barrett_1(self):
    result_jax = ff.mod_mul_barrett_2u16(self.value_a_jax, self.value_b_jax)
    result = util.array_to_int_list(result_jax, util.BASE)

    def mod_mul_barrett(a, b):
      value_a_barrett = pyff.FiniteFieldElementBarrett(a, util.MODULUS_377_INT)
      value_b_barrett = pyff.FiniteFieldElementBarrett(b, util.MODULUS_377_INT)
      return (value_a_barrett * value_b_barrett).get_value()

    result_ref = list_operation(self.value_a, self.value_b, mod_mul_barrett)

    self.assertEqual(result, result_ref)

  def test_jax_mod_mul_lazy_reduction(self):
    """This test case check the jax version (TPU deployment) of the lazy reduction based modular multiplication algorithm."""
    batch_size = 16
    a_list = [randint(0, util.MODULUS_377_INT) for _ in range(batch_size)]
    b_list = [randint(0, util.MODULUS_377_INT) for _ in range(batch_size)]

    a_batch = util.int_list_to_array(
        a_list, base=util.BASE, array_size=util.U16_EXT_CHUNK_NUM
    )
    b_batch = util.int_list_to_array(
        b_list, base=util.BASE, array_size=util.U16_EXT_CHUNK_NUM
    )
    c_batch = ff.mod_mul_lazy_2u16(a_batch, b_batch)
    c_list = util.array_to_int_list(c_batch, util.BASE)
    for i in range(len(a_list)):
      np.testing.assert_equal(
          c_list[i] % util.MODULUS_377_INT,
          (a_list[i] * b_list[i]) % util.MODULUS_377_INT,
      )

  def test_jax_mod_mul_rns_reduction(self):
    """This test case check the jax version (TPU deployment) of the rns reduction based modular multiplication algorithm."""
    batch_size = 16
    a_list = [randint(0, util.MODULUS_377_INT) for _ in range(batch_size)]
    b_list = [randint(0, util.MODULUS_377_INT) for _ in range(batch_size)]

    modulus_rns_mat = util.construct_rns_matrix(util.MODULUS_377_INT)
    a_batch = util.int_list_to_array_rns(a_list)
    b_batch = util.int_list_to_array_rns(b_list)
    c_batch = ff.mod_mul_rns_2u16(a_batch, b_batch, modulus_rns_mat)
    c_list = util.array_rns_to_int_list(c_batch)
    for i in range(len(a_list)):
      np.testing.assert_equal(
          c_list[i] % util.MODULUS_377_INT,
          (a_list[i] * b_list[i]) % util.MODULUS_377_INT,
      )

  def test_jax_add_rns(self):
    max_val = [2**16 - 1 for _ in range(util.NUM_MODULI)]
    max_normal_val = [m - 1 for m in util.MODULI]
    zero = [0 for _ in range(util.NUM_MODULI)]
    values = [zero, max_val, max_normal_val]
    for a in values:
      for b in values:
        jax_a = jnp.array(a, dtype=jnp.uint16).reshape((1, util.NUM_MODULI))
        jax_b = jnp.array(b, dtype=jnp.uint16).reshape((1, util.NUM_MODULI))
        jax_sum = ff.add_rns_2u16(jax_a, jax_b, tuple(util.RNS_MODULI_T))
        jax_3sum = ff.add_rns_3u16(
            jax_a, jax_b, jax_a, tuple(util.RNS_MODULI_T)
        )
        for i in range(util.NUM_MODULI):
          np.testing.assert_equal(
              int(jax_sum[0, i]) % util.MODULI[i],
              (a[i] + b[i]) % util.MODULI[i],
          )
          np.testing.assert_equal(
              int(jax_3sum[0, i]) % util.MODULI[i],
              (a[i] + b[i] + a[i]) % util.MODULI[i],
          )

  def test_jax_sub_rns(self):
    batch_size = 16
    bound = 256 * util.NUM_MODULI * util.MODULUS_377_INT
    a_list = [randint(0, bound) for _ in range(batch_size)]
    b_list = [randint(0, bound) for _ in range(batch_size)]
    b_list[0] = bound - 1
    a_batch = util.int_list_to_array_rns(a_list)
    b_batch = util.int_list_to_array_rns(b_list)
    diff = ff.add_sub_rns_var(a_batch, ff.negate_rns_for_var_add(b_batch))
    diff_int = util.array_rns_to_int_list(diff)
    for i in range(batch_size):
      np.testing.assert_equal(
          diff_int[i] % util.MODULUS_377_INT,
          (a_list[i] - b_list[i]) % util.MODULUS_377_INT,
      )

  def test_jax_add_rns_specific_case(self):
    e = 149025596882241982990837486539530757729373308235078472950338530041138824139683871423246406325832428746678481926350
    h = 64504914146370186321601383206234327860947337385636434408981741341000348311615332297641759363669577422255115417749
    q = util.MODULUS_377_INT
    jax_a = util.int_list_to_array_rns([e])
    jax_b = util.int_list_to_array_rns([h])
    jax_sum = ff.mod_mul_rns_2u16(jax_a, jax_b)
    val_sum = util.array_rns_to_int(jax_sum[0])
    np.testing.assert_equal(val_sum % util.MODULUS_377_INT, (e * h % q))


if __name__ == "__main__":
  absltest.main()
