import functools

import jax
import jax.numpy as jnp
import jaxite.jaxite_ec.algorithm.finite_field as pyff
import jaxite.jaxite_word.finite_field as ff
import jaxite.jaxite_word.util as utils

# copybara: from google3.perftools.accelerators.xprof.api.python import xprof_session
from absl.testing import absltest
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)


def list_operation(a, b, func):
  return [func(ai, bi) for ai, bi in zip(a, b)]


class FiniteFieldTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.value_a = [
        1007059853392461412,
        1039665390909095980,
    ]
    self.value_b = [
        1095667005962340998,
        1027493227371993960,
    ]
    self.value_c = [
        1007059853392567897,
        1086823018630279065,
    ]
    self.test_modulus = utils.MODULUS_DEFAULT
    self.value_a_jax = utils.int_list_to_jax_array(
        self.value_a, base=utils.BASE, array_size=utils.U16_CHUNK_NUM_DEFAULT
    )
    self.value_b_jax = utils.int_list_to_jax_array(
        self.value_b, base=utils.BASE, array_size=utils.U16_CHUNK_NUM_DEFAULT
    )
    self.value_c_jax = utils.int_list_to_jax_array(
        self.value_c, base=utils.BASE, array_size=utils.U16_CHUNK_NUM_DEFAULT
    )

  def test_add_two(self):
    """The addition results should be within the range of 16 * chunk_num_u16."""
    print("Test test_add_two")
    # input_data_shape = (num_elements, num_towers, num_degree)
    value_a_jax = utils.int_list_to_jax_array(
        self.value_a, base=utils.BASE, array_size=utils.U16_CHUNK_NUM_DEFAULT
    )
    value_b_jax = utils.int_list_to_jax_array(
        self.value_b, base=utils.BASE, array_size=utils.U16_CHUNK_NUM_DEFAULT
    )
    # copybara: session = xprof_session.XprofSession()
    # copybara: session.start_session()
    add_2u16_local = functools.partial(
        ff.add_2u16,
        mask=utils.U16_MASK,
        chunk_num_u16=utils.U16_CHUNK_NUM_DEFAULT,
        chunk_shift_bits=utils.U16_CHUNK_SHIFT_BITS,
    )
    result_jax = add_2u16_local(value_a_jax, value_b_jax)
    result = utils.jax_array_to_int_list(result_jax, utils.BASE)

    result_ref = [a + b for a, b in zip(self.value_a, self.value_b)]
    # copybara: session_id = session.end_session_and_get_session_id()
    # copybara: print(f'session_id: http://xprof/?session_id={session_id}')

    self.assertEqual(result, result_ref)

  def test_add_three(self):
    print("Test test_add_three")
    # copybara: session = xprof_session.XprofSession()
    # copybara: session.start_session()
    result_jax = ff.add_3u16(
        self.value_a_jax, self.value_b_jax, self.value_c_jax
    )
    result = utils.jax_array_to_int_list(result_jax, utils.BASE)
    result_ref = list_operation(
        list_operation(self.value_a, self.value_b, lambda a, b: a + b),
        self.value_c,
        lambda a, b: a + b,
    )
    # copybara: session_id = session.end_session_and_get_session_id()
    # copybara: print(f'session_id: http://xprof/?session_id={session_id}')

    self.assertEqual(result, result_ref)

  def test_cond_sub(self):
    print("Test test_cond_sub")
    # copybara: session = xprof_session.XprofSession()
    # copybara: session.start_session()
    result_jax = ff.cond_sub_2u16(self.value_a_jax, self.value_b_jax)
    result = utils.jax_array_to_int_list(result_jax, utils.BASE)

    def cond_sub(a, b):
      if a < b:
        return a + self.test_modulus - b
      else:
        return a - b

    result_ref = list_operation(self.value_a, self.value_b, cond_sub)

    # copybara: session_id = session.end_session_and_get_session_id()
    # copybara: print(f'session_id: http://xprof/?session_id={session_id}')

    self.assertEqual(result, result_ref)

  def test_cond_sub_batch(self):
    print("Test test_cond_sub_batch")
    # copybara: session = xprof_session.XprofSession()
    # copybara: session.start_session()
    result_jax = ff.cond_sub_2u16(self.value_b_jax, self.value_a_jax)
    result = utils.jax_array_to_int_list(result_jax, utils.BASE)

    def cond_sub(a, b):
      if a < b:
        return a + self.test_modulus - b
      else:
        return a - b

    result_ref = list_operation(self.value_b, self.value_a, cond_sub)

    # copybara: session_id = session.end_session_and_get_session_id()
    # copybara: print(f'session_id: http://xprof/?session_id={session_id}')

    self.assertEqual(result, result_ref)

  def test_cond_sub_mod(self):
    print("Test test_cond_sub_mod")
    # copybara: session = xprof_session.XprofSession()
    # copybara: session.start_session()
    value_list = [self.test_modulus + 123, self.test_modulus - 5432]
    value_jax = utils.int_list_to_jax_array(
        value_list, base=utils.BASE, array_size=utils.U16_CHUNK_NUM_DEFAULT
    )
    result_jax = ff.cond_sub_mod_u16(value_jax)
    result = utils.jax_array_to_int_list(result_jax, utils.BASE)

    def cond_sub_mod(a):
      if a < self.test_modulus:
        return a
      else:
        return a - self.test_modulus

    result_ref = [cond_sub_mod(a) for a in value_list]

    # copybara: session_id = session.end_session_and_get_session_id()
    # copybara: print(f'session_id: http://xprof/?session_id={session_id}')

    self.assertEqual(result, result_ref)

  def test_mul(self):
    print("Test test_mul")
    # copybara: session = xprof_session.XprofSession()
    # copybara: session.start_session()
    ff.mul_2u16(self.value_a_jax, self.value_b_jax)
    compiled_func = (
        jax.jit(ff.mul_2u16)
        .lower(
            jax.ShapeDtypeStruct(self.value_a_jax.shape, dtype=jnp.uint16),
            jax.ShapeDtypeStruct(self.value_b_jax.shape, dtype=jnp.uint16),
        )
        .compile()
    )
    result_jax = compiled_func(self.value_a_jax, self.value_b_jax)
    result = utils.jax_array_to_int_list(result_jax, utils.BASE)
    result_ref = list_operation(self.value_a, self.value_b, lambda a, b: a * b)
    # copybara: session_id = session.end_session_and_get_session_id()
    # copybara: print(f'session_id: http://xprof/?session_id={session_id}')
    self.assertEqual(result, result_ref)

  def test_mod_mul_barrett(self):
    print("Test test_mod_mul_barrett")
    ff_barrett = pyff.FiniteFieldElementBarrett(
        self.value_a[0], self.test_modulus
    )
    print(f"ff_barrett.mu: {ff_barrett.mu.value}")
    print(f"ff_barrett.2k: {ff_barrett.two_k.value}")

    # copybara: session = xprof_session.XprofSession()
    # copybara: session.start_session()
    result_jax = ff.mod_mul_barrett_2u16(self.value_a_jax, self.value_b_jax)
    result = utils.jax_array_to_int_list(result_jax, utils.BASE)

    def mod_mul_reference(a, b):
      return (a * b) % self.test_modulus

    result_ref = list_operation(self.value_a, self.value_b, mod_mul_reference)

    # copybara: session_id = session.end_session_and_get_session_id()
    # copybara: print(f'session_id: http://xprof/?session_id={session_id}')

    self.assertEqual(result, result_ref)

  @parameterized.named_parameters(
      (
          "test_64_bit_modulus",
          utils.MODULUS_LIST[0],
          utils.U16_CHUNK_NUM_ALL[0],
          utils.BASE,
          utils.U16_MASK,
          utils.U16_CHUNK_SHIFT_BITS,
          utils.NUM_DEGREE,
          jnp.uint64,
      ),
  )
  def test_mul_general(
      self,
      modulus,
      chunk_num_u16,
      base,
      mask,
      chunk_shift_bits,
      num_degree,
      dtype,
  ):
    print("Test test_mul_general")
    value_a = utils.random_list((num_degree), (modulus >> 1), dtype=dtype)
    value_b = utils.random_list((num_degree), (modulus >> 1), dtype=dtype)
    value_a_jax = utils.int_list_to_jax_array(
        value_a, base=base, array_size=chunk_num_u16
    )
    value_b_jax = utils.int_list_to_jax_array(
        value_b, base=base, array_size=chunk_num_u16
    )
    mul_2u16_local = functools.partial(
        ff.mul_2u16,
        mask=mask,
        chunk_num_u16=chunk_num_u16,
        chunk_shift_bits=chunk_shift_bits,
    )
    compiled_func = (
        jax.jit(mul_2u16_local)
        .lower(
            jax.ShapeDtypeStruct(value_a_jax.shape, dtype=jnp.uint16),
            jax.ShapeDtypeStruct(value_b_jax.shape, dtype=jnp.uint16),
        )
        .compile()
    )
    # copybara: session = xprof_session.XprofSession()
    # copybara: session.start_session()
    result_jax = compiled_func(value_a_jax, value_b_jax)
    # copybara: session_id = session.end_session_and_get_session_id()
    # copybara: print(f'session_id: http://xprof/?session_id={session_id}')
    result = utils.jax_array_to_int_list(result_jax, base)
    result_ref = list_operation(value_a, value_b, lambda a, b: a * b)
    self.assertEqual(result, result_ref)

  @parameterized.named_parameters(
      (
          "test_64_bit_modulus",
          utils.MODULUS_LIST[0],
          utils.U16_CHUNK_NUM_ALL[0],
          utils.BASE,
          utils.U16_MASK,
          utils.U16_CHUNK_SHIFT_BITS,
          utils.NUM_DEGREE,
          jnp.uint64,
      ),
  )
  def test_add_two_general(
      self,
      modulus,
      chunk_num_u16,
      base,
      mask,
      chunk_shift_bits,
      num_degree,
      dtype,
  ):
    """The addition results should be within the range of 16 * chunk_num_u16."""
    print("Test test_add_two_general")
    value_a = utils.random_list((num_degree), (modulus >> 1), dtype=dtype)
    value_b = utils.random_list((num_degree), (modulus >> 1), dtype=dtype)
    value_a_jax = utils.int_list_to_jax_array(
        value_a, base=base, array_size=chunk_num_u16
    )
    value_b_jax = utils.int_list_to_jax_array(
        value_b, base=base, array_size=chunk_num_u16
    )
    # copybara: session = xprof_session.XprofSession()
    # copybara: session.start_session()
    add_2u16_local = functools.partial(
        ff.add_2u16,
        mask=mask,
        chunk_num_u16=chunk_num_u16,
        chunk_shift_bits=chunk_shift_bits,
    )
    result_jax = add_2u16_local(value_a_jax, value_b_jax)
    result = utils.jax_array_to_int_list(result_jax, base)

    result_ref = [a + b for a, b in zip(value_a, value_b)]
    # copybara: session_id = session.end_session_and_get_session_id()
    # copybara: print(f'session_id: http://xprof/?session_id={session_id}')

    self.assertEqual(result, result_ref)

  @parameterized.named_parameters(
      (
          "test_64_bit_modulus",
          utils.MODULUS_LIST[0],
          utils.U16_CHUNK_NUM_ALL[0],
          utils.BARRETT_SHIFT_U8_ALL[0],
          utils.MODULUS_ARRAY_ALL[0],
          utils.MU_ARRAY_ALL[0],
          utils.BASE,
          utils.U16_MASK,
          utils.NUM_DEGREE,
          jnp.uint64,
      ),
      (
          "test_32_bit_modulus",
          utils.MODULUS_LIST[1],
          utils.U16_CHUNK_NUM_ALL[1],
          utils.BARRETT_SHIFT_U8_ALL[1],
          utils.MODULUS_ARRAY_ALL[1],
          utils.MU_ARRAY_ALL[1],
          utils.BASE,
          utils.U16_MASK,
          utils.NUM_DEGREE,
          jnp.uint32,
      ),
  )
  def test_mod_mul_general(
      self,
      modulus,
      chunk_num_u16,
      barrett_shift_u8,
      modulus_array,
      mu_array,
      base,
      mask,
      num_degree,
      dtype,
  ):
    print("Test test_mod_mul_general")
    value_a = utils.random_list((num_degree), modulus, dtype=dtype)
    value_b = utils.random_list((num_degree), modulus, dtype=dtype)
    value_a_jax = utils.int_list_to_jax_array(
        value_a, base=base, array_size=chunk_num_u16
    )
    value_b_jax = utils.int_list_to_jax_array(
        value_b, base=base, array_size=chunk_num_u16
    )
    mod_mul_2u16_local = functools.partial(
        ff.mod_mul_barrett_2u16,
        mask=mask,
        modulus_array=modulus_array,
        mu_array=mu_array,
        barrett_shift_u8=barrett_shift_u8,
        chunk_num_u16=chunk_num_u16,
        vmap_axes=(0, None),
    )
    compiled_func = (
        jax.jit(mod_mul_2u16_local)
        .lower(
            jax.ShapeDtypeStruct(value_a_jax.shape, dtype=jnp.uint16),
            jax.ShapeDtypeStruct(value_b_jax.shape, dtype=jnp.uint16),
        )
        .compile()
    )
    # copybara: session = xprof_session.XprofSession()
    # copybara: session.start_session()
    result_jax = compiled_func(value_a_jax, value_b_jax)
    # copybara: session_id = session.end_session_and_get_session_id()
    # copybara: print(f'session_id: http://xprof/?session_id={session_id}')
    result = utils.jax_array_to_int_list(result_jax, utils.BASE)

    def mod_mul_reference(a, b):
      return (a * b) % modulus

    result_ref = list_operation(value_a, value_b, mod_mul_reference)
    self.assertEqual(result, result_ref)

  @parameterized.named_parameters(
      # (
      #     "test_64_bit_modulus",
      #     utils.MODULUS_LIST[0],
      #     utils.U16_CHUNK_NUM_ALL[0],
      #     utils.BARRETT_SHIFT_U8_ALL[0],
      #     utils.MODULUS_ARRAY_ALL[0],
      #     utils.MU_ARRAY_ALL[0],
      #     utils.BASE,
      #     utils.U16_MASK,
      #     utils.NUM_DEGREE,
      #     jnp.uint64,
      # ),
      (
          "test_32_bit_modulus",
          utils.MODULUS_LIST[1],
          utils.U16_CHUNK_NUM_ALL[1],
          utils.BARRETT_SHIFT_U8_ALL[1],
          utils.MODULUS_ARRAY_ALL[1],
          utils.MU_ARRAY_ALL[1],
          utils.BASE,
          utils.U16_MASK,
          utils.NUM_DEGREE,
          jnp.uint32,
      ),
  )
  def test_mod_reduction_general(
      self,
      modulus,
      chunk_num_u16,
      barrett_shift_u8,
      modulus_array,
      mu_array,
      base,
      mask,
      num_degree,
      dtype,
  ):
    print("Test test_mod_reduction_general")
    value_a = utils.random_list((num_degree), modulus, dtype=dtype)
    value_b = utils.random_list((num_degree), modulus, dtype=dtype)
    multi_result = list_operation(value_a, value_b, lambda a, b: a * b)
    multi_result_jax = utils.int_list_to_jax_array(
        multi_result, base=base, array_size=2 * chunk_num_u16
    )
    mod_reduction_local = functools.partial(
        ff.mod_reduction_barrett_2u16,
        mask=mask,
        modulus_array=modulus_array,
        mu_array=mu_array,
        barrett_shift_u8=barrett_shift_u8,
        chunk_num_u16=chunk_num_u16,
        vmap_axes=(0, None),
    )
    compiled_func = (
        jax.jit(mod_reduction_local)
        .lower(
            jax.ShapeDtypeStruct(multi_result_jax.shape, dtype=jnp.uint16),
        )
        .compile()
    )
    # copybara: session = xprof_session.XprofSession()
    # copybara: session.start_session()
    result_jax = compiled_func(multi_result_jax)
    # copybara: session_id = session.end_session_and_get_session_id()
    # copybara: print(f'session_id: http://xprof/?session_id={session_id}')
    result = utils.jax_array_to_int_list(result_jax, utils.BASE)

    result_ref = [val % modulus for val in multi_result]
    self.assertEqual(result, result_ref)

  def test_transpose(self):
    value_a = jax.random.randint(
        jax.random.key(0),
        shape=(1048675, 1),
        minval=0,
        maxval=(2**15 - 1),
        dtype=jnp.uint32,
    )
    print(value_a.shape)

    @jax.jit
    def transpose_a(value_a: jax.Array):
      value_b = value_a.transpose(1, 0)
      value_c = value_b + value_b
      value_c = value_c.transpose(1, 0)
      value_d = jnp.multiply(value_a, value_c)
      return value_d.transpose(1, 0) + value_c

    compiled_func = (
        jax.jit(transpose_a)
        .lower(jax.ShapeDtypeStruct(value_a.shape, dtype=jnp.uint32))
        .compile()
    )

    # copybara: session = xprof_session.XprofSession()
    # copybara: session.start_session()
    compiled_func(value_a)
    # copybara: session_id = session.end_session_and_get_session_id()
    # copybara: print(f'session_id: http://xprof/?session_id={session_id}')

  def test_sub_u16(self):
    print("Test test_sub_u16")
    # ToDo: This should be efficient in Pallas.
    value_a = utils.random_array((128 * 8192,), 2**31 - 1, dtype=jnp.uint32)
    value_b = value_a - 1
    result_ref = jnp.ones(value_a.shape, dtype=jnp.uint32).tolist()
    value_a = value_a.tolist()
    value_b = value_b.tolist()
    value_a_jax = utils.int_list_to_jax_array(
        value_a, base=utils.BASE, array_size=utils.U16_CHUNK_NUM_U32
    )
    value_b_jax = utils.int_list_to_jax_array(
        value_b, base=utils.BASE, array_size=utils.U16_CHUNK_NUM_U32
    )
    compiled_func = jax.jit(ff.sub_2u16)
    print(f"value_a_jax.shape: {value_a_jax.shape}, dtype:{value_a_jax.dtype}")
    print(f"value_b_jax.shape: {value_b_jax.shape}, dtype:{value_b_jax.dtype}")
    compiled_func(value_a_jax, value_b_jax)
    # copybara: session = xprof_session.XprofSession()
    # copybara: session.start_session()
    result_jax = compiled_func(value_a_jax, value_b_jax)

    # copybara: session_id = session.end_session_and_get_session_id()
    # copybara: print(f'session_id: http://xprof/?session_id={session_id}')
    result = utils.jax_array_to_int_list(result_jax, utils.BASE)
    self.assertEqual(result, result_ref)


if __name__ == "__main__":
  absltest.main()
