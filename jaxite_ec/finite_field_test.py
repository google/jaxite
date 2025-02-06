import random

import jax
import jaxite.jaxite_ec.algorithm.finite_field as pyff
import jaxite.jaxite_ec.finite_field as ff
import jaxite.jaxite_ec.util as utils
import numpy as np

from google3.perftools.accelerators.xprof.api.python import xprof_session
from absl.testing import absltest


jax.config.update("jax_enable_x64", True)

randint = random.randint


def list_operation(a, b, func):
  return [func(ai, bi) for ai, bi in zip(a, b)]


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
    self.value_a_jax = utils.int_list_to_jax_array(
        self.value_a, base=utils.BASE, array_size=utils.U16_CHUNK_NUM
    )
    self.value_b_jax = utils.int_list_to_jax_array(
        self.value_b, base=utils.BASE, array_size=utils.U16_CHUNK_NUM
    )
    self.value_c_jax = utils.int_list_to_jax_array(
        self.value_c, base=utils.BASE, array_size=utils.U16_CHUNK_NUM
    )

  def test_add_1(self):
    print("Test test_add_1", end=" ")
    session = xprof_session.XprofSession()
    session.start_session()
    result_jax = jax.vmap(ff.add_2u16)(self.value_a_jax, self.value_b_jax)
    result = utils.jax_array_to_int_list(result_jax, utils.BASE)
    result_ref = list_operation(self.value_a, self.value_b, lambda a, b: a + b)
    try:
      session_id = session.end_session_and_get_session_id()
      print(f"session_id: http://xprof/?session_id={session_id}")
    except RuntimeError as e:
      print(f"Failed to get session_id: {e}")
    self.assertEqual(result, result_ref)

  def test_add_2(self):
    print("Test test_add_2", end=" ")
    session = xprof_session.XprofSession()
    session.start_session()
    result_jax = jax.vmap(ff.add_3u16)(
        self.value_a_jax, self.value_b_jax, self.value_c_jax
    )
    result = utils.jax_array_to_int_list(result_jax, utils.BASE)
    result_ref = list_operation(
        list_operation(self.value_a, self.value_b, lambda a, b: a + b),
        self.value_c,
        lambda a, b: a + b,
    )
    try:
      session_id = session.end_session_and_get_session_id()
      print(f"session_id: http://xprof/?session_id={session_id}")
    except RuntimeError as e:
      print(f"Failed to get session_id: {e}")
    self.assertEqual(result, result_ref)

  def test_cond_sub_1(self):
    print("Test test_cond_sub_1", end=" ")
    session = xprof_session.XprofSession()
    session.start_session()
    result_jax = jax.vmap(ff.cond_sub_2u16)(self.value_a_jax, self.value_b_jax)
    result = utils.jax_array_to_int_list(result_jax, utils.BASE)

    def cond_sub(a, b):
      if a < b:
        return a + utils.MODULUS_377_INT - b
      else:
        return a - b

    result_ref = list_operation(self.value_a, self.value_b, cond_sub)
    try:
      session_id = session.end_session_and_get_session_id()
      print(f"session_id: http://xprof/?session_id={session_id}")
    except RuntimeError as e:
      print(f"Failed to get session_id: {e}")
    self.assertEqual(result, result_ref)

  def test_cond_sub_2(self):
    print("Test test_cond_sub_2", end=" ")
    session = xprof_session.XprofSession()
    session.start_session()
    result_jax = jax.vmap(ff.cond_sub_2u16)(self.value_b_jax, self.value_a_jax)
    result = utils.jax_array_to_int_list(result_jax, utils.BASE)

    def cond_sub(a, b):
      if a < b:
        return a + utils.MODULUS_377_INT - b
      else:
        return a - b

    result_ref = list_operation(self.value_b, self.value_a, cond_sub)
    try:
      session_id = session.end_session_and_get_session_id()
      print(f"session_id: http://xprof/?session_id={session_id}")
    except RuntimeError as e:
      print(f"Failed to get session_id: {e}")
    self.assertEqual(result, result_ref)

  def test_cond_sub_mod_1(self):
    print("Test test_cond_sub_mod_1", end=" ")
    session = xprof_session.XprofSession()
    session.start_session()
    value_list = [utils.MODULUS_377_INT + 123, utils.MODULUS_377_INT - 5432]
    value_jax = utils.int_list_to_jax_array(
        value_list, base=utils.BASE, array_size=utils.U16_CHUNK_NUM
    )
    result_jax = jax.vmap(ff.cond_sub_mod_u16)(value_jax)
    result = utils.jax_array_to_int_list(result_jax, utils.BASE)

    def cond_sub_mod(a, b):
      if a < utils.MODULUS_377_INT:
        return a
      else:
        return a - utils.MODULUS_377_INT

    result_ref = list_operation(value_list, value_list, cond_sub_mod)
    try:
      session_id = session.end_session_and_get_session_id()
      print(f"session_id: http://xprof/?session_id={session_id}")
    except RuntimeError as e:
      print(f"Failed to get session_id: {e}")
    self.assertEqual(result, result_ref)

  def test_mul_1(self):
    print("Test test_mul_1", end=" ")
    session = xprof_session.XprofSession()
    session.start_session()
    result_jax = jax.vmap(ff.mul_2u16)(self.value_a_jax, self.value_b_jax)
    result = utils.jax_array_to_int_list(result_jax, utils.BASE)
    result_ref = list_operation(self.value_a, self.value_b, lambda a, b: a * b)
    try:
      session_id = session.end_session_and_get_session_id()
      print(f"session_id: http://xprof/?session_id={session_id}")
    except RuntimeError as e:
      print(f"Failed to get session_id: {e}")
    self.assertEqual(result, result_ref)

  def test_mod_mul_barrett_1(self):
    print("Test test_mod_mul_barrett_1", end=" ")
    session = xprof_session.XprofSession()
    session.start_session()
    result_jax = jax.vmap(ff.mod_mul_barrett_2u16)(
        self.value_a_jax, self.value_b_jax
    )
    result = utils.jax_array_to_int_list(result_jax, utils.BASE)

    def mod_mul_barrett(a, b):
      value_a_barrett = pyff.FiniteFieldElementBarrett(a, utils.MODULUS_377_INT)
      value_b_barrett = pyff.FiniteFieldElementBarrett(b, utils.MODULUS_377_INT)
      return (value_a_barrett * value_b_barrett).get_value()

    result_ref = list_operation(self.value_a, self.value_b, mod_mul_barrett)

    try:
      session_id = session.end_session_and_get_session_id()
      print(f"session_id: http://xprof/?session_id={session_id}")
    except RuntimeError as e:
      print(f"Failed to get session_id: {e}")

    self.assertEqual(result, result_ref)

  def test_jax_mod_mul_lazy_reduction(self):
    """This test case check the jax version (TPU deployment) of the lazy reduction based modular multiplication algorithm.
    """
    batch_size = 16
    a_list = [randint(0, utils.MODULUS_377_INT) for _ in range(batch_size)]
    b_list = [randint(0, utils.MODULUS_377_INT) for _ in range(batch_size)]

    modulus_lazy_mat = ff.construct_lazy_matrix(
        utils.MODULUS_377_INT,
        chunk_precision=8,
        chunk_num_u8=utils.U8_CHUNK_NUM,
    )
    a_batch = utils.int_list_to_jax_array(
        a_list, base=utils.BASE, array_size=utils.U16_CHUNK_NUM
    )
    b_batch = utils.int_list_to_jax_array(
        b_list, base=utils.BASE, array_size=24
    )
    c_batch = ff.mod_mul_lazy_2u16(a_batch, b_batch, modulus_lazy_mat)
    c_list = utils.jax_array_to_int_list(c_batch, utils.BASE)
    for i in range(len(a_list)):
      np.testing.assert_equal(
          c_list[i] % utils.MODULUS_377_INT,
          (a_list[i] * b_list[i]) % utils.MODULUS_377_INT,
      )
    print("testing pass")


if __name__ == "__main__":
  absltest.main()
