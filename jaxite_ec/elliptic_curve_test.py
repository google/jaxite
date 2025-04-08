import functools

import jax
import jax.numpy as jnp
from jaxite.jaxite_ec import util
from jaxite.jaxite_ec.algorithm import config_file
import jaxite.jaxite_ec.algorithm.elliptic_curve as ec
import jaxite.jaxite_ec.elliptic_curve as jec

from absl.testing import absltest


class TestEllipticCurve(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.coordinate_num = 4
    self.batch_size = 1
    self.x1_int_ = 0x01AC3A384FC584EFD3E7F2C5A2927E7D454875C874A051027B9E7363D08942533EDE85DAE295D8CAB2751085206BCA76
    self.y1_int_ = 0x011DB83AEC88460820F4868A73B12309EE2E910526E62DB4ACCB303ABF50F86C3985A072ED07A4B81FFB82D8DD247283
    self.x2_int_ = 0x01546AF2ABB4E189E9BBC412FDBF2A8E5EC6E4A3B0AF132E21EE9CEC3EF5E226490FB98D662670FA3CFB3948B7E2A48C
    self.y2_int_ = 0x002961A558A885DF227FDB09F8BDF57AF179CB9437FF8828F13E9DF01AE55502F409AAF5058B88F2F7CCC7BC0676A5D4
    self.point_a = [self.x1_int_, self.y1_int_]
    self.point_b = [self.x2_int_, self.y2_int_]
    self.zero_twisted = [0, 1, 1, 0]
    self.ec_sys = ec.ECCSWeierstrassXYZZ(config_file.config_BLS12_377)
    self.point_a_sys = self.ec_sys.generate_point(self.point_a)
    self.point_b_sys = self.ec_sys.generate_point(self.point_b)
    assert int(self.point_a_sys.coordinates[0].value) == self.point_a[0]
    assert int(self.point_a_sys.coordinates[1].value) == self.point_a[1]
    assert int(self.point_b_sys.coordinates[0].value) == self.point_b[0]
    assert int(self.point_b_sys.coordinates[1].value) == self.point_b[1]
    self.true_result_padd = self.point_a_sys + self.point_b_sys
    self.true_result_padd_affine = self.true_result_padd.convert_to_affine()
    self.true_result_pdub_a = self.point_a_sys + self.point_a_sys
    self.true_result_pdub_a_affine = self.true_result_pdub_a.convert_to_affine()
    self.true_result_pdub_b = self.point_b_sys + self.point_b_sys
    self.true_result_pdub_b_affine = self.true_result_pdub_b.convert_to_affine()

  def test_padd_barrett_xyzz_pack(self):
    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_a + [1] * (self.coordinate_num - len(self.point_a))]
    )
    point_b_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_b + [1] * (self.coordinate_num - len(self.point_b))]
    )
    jit_padd_barrett_xyzz_pack = jax.jit(jec.padd_barrett_xyzz_pack)

    result_jax = jit_padd_barrett_xyzz_pack(point_a_jax, point_b_jax)
    result_jax = util.jax_point_pack_to_int_point_batch(result_jax)

    self.assertEqual(result_jax[0][0], self.true_result_padd[0].get_value())
    self.assertEqual(result_jax[0][1], self.true_result_padd[1].get_value())
    self.assertEqual(result_jax[0][2], self.true_result_padd[2].get_value())
    self.assertEqual(result_jax[0][3], self.true_result_padd[3].get_value())

    # performance measurement
    tasks = [
        (jit_padd_barrett_xyzz_pack, (point_a_jax, point_b_jax)),
    ]
    profile_name = "jit_padd_barrett_xyzz_pack"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_pdul_barrett_xyzz(self):
    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_a + [1] * (self.coordinate_num - len(self.point_a))]
    )
    jit_pdul_barrett_xyzz_pack = jax.jit(jec.pdul_barrett_xyzz_pack)
    result_jax = jit_pdul_barrett_xyzz_pack(point_a_jax)
    result_jax = util.jax_point_pack_to_int_point_batch(result_jax)

    self.assertEqual(result_jax[0][0], self.true_result_pdub_a[0].get_value())
    self.assertEqual(result_jax[0][1], self.true_result_pdub_a[1].get_value())
    self.assertEqual(result_jax[0][2], self.true_result_pdub_a[2].get_value())
    self.assertEqual(result_jax[0][3], self.true_result_pdub_a[3].get_value())

    # performance measurement
    tasks = [
        (jit_pdul_barrett_xyzz_pack, (point_a_jax,)),
    ]
    profile_name = "jit_pdul_barrett_xyzz_pack"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_pdul_barrett_xyzz_pack_two_no_batch(self):
    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_a + [1] * (self.coordinate_num - len(self.point_a))]
    )
    point_b_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_b + [1] * (self.coordinate_num - len(self.point_b))]
    )
    jit_pdul_barrett_xyzz_pack = jax.jit(jec.pdul_barrett_xyzz_pack)
    result_a_jax = jit_pdul_barrett_xyzz_pack(point_a_jax)
    result_a_int = util.jax_point_pack_to_int_point_batch(result_a_jax)

    result_b_jax = jit_pdul_barrett_xyzz_pack(point_b_jax)
    result_b_int = util.jax_point_pack_to_int_point_batch(result_b_jax)

    self.assertEqual(result_a_int[0][0], self.true_result_pdub_a[0].get_value())
    self.assertEqual(result_a_int[0][1], self.true_result_pdub_a[1].get_value())
    self.assertEqual(result_a_int[0][2], self.true_result_pdub_a[2].get_value())
    self.assertEqual(result_a_int[0][3], self.true_result_pdub_a[3].get_value())
    self.assertEqual(result_b_int[0][0], self.true_result_pdub_b[0].get_value())
    self.assertEqual(result_b_int[0][1], self.true_result_pdub_b[1].get_value())
    self.assertEqual(result_b_int[0][2], self.true_result_pdub_b[2].get_value())
    self.assertEqual(result_b_int[0][3], self.true_result_pdub_b[3].get_value())

    # performance measurement
    tasks = [
        (jit_pdul_barrett_xyzz_pack, (point_a_jax,)),
        (jit_pdul_barrett_xyzz_pack, (point_b_jax,)),
    ]
    profile_name = "jit_pdul_barrett_xyzz_pack"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_pdul_barrett_xyzz_pack_two_batch(self):
    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_a + [1] * (self.coordinate_num - len(self.point_a))]
    )
    point_b_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_b + [1] * (self.coordinate_num - len(self.point_b))]
    )
    batch_point = jnp.concatenate([point_a_jax, point_b_jax], axis=1)
    jit_pdul_barrett_xyzz_pack = jax.jit(jec.pdul_barrett_xyzz_pack)
    result_jax = jit_pdul_barrett_xyzz_pack(batch_point)
    result_int = util.jax_point_pack_to_int_point_batch(result_jax)

    self.assertEqual(result_int[0][0], self.true_result_pdub_a[0].get_value())
    self.assertEqual(result_int[0][1], self.true_result_pdub_a[1].get_value())
    self.assertEqual(result_int[0][2], self.true_result_pdub_a[2].get_value())
    self.assertEqual(result_int[0][3], self.true_result_pdub_a[3].get_value())
    self.assertEqual(result_int[1][0], self.true_result_pdub_b[0].get_value())
    self.assertEqual(result_int[1][1], self.true_result_pdub_b[1].get_value())
    self.assertEqual(result_int[1][2], self.true_result_pdub_b[2].get_value())
    self.assertEqual(result_int[1][3], self.true_result_pdub_b[3].get_value())

    # performance measurement
    tasks = [
        (jit_pdul_barrett_xyzz_pack, (batch_point,)),
    ]
    profile_name = "jit_pdul_barrett_xyzz_pack"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_padd_lazy_xyzz_pack(self):
    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_a + [1] * (self.coordinate_num - len(self.point_a))],
        array_size=util.U16_EXT_CHUNK_NUM,
    )
    point_b_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_b + [1] * (self.coordinate_num - len(self.point_b))],
        array_size=util.U16_EXT_CHUNK_NUM,
    )
    # lazy_mat = util.construct_lazy_matrix(util.MODULUS_377_INT)
    jit_padd_lazy_xyzz_pack = jax.jit(jec.padd_lazy_xyzz_pack)
    result_jax = jit_padd_lazy_xyzz_pack(point_a_jax, point_b_jax)
    result_jax = util.jax_point_pack_to_int_point_batch(result_jax)

    self.assertEqual(
        result_jax[0][0] % util.MODULUS_377_INT,
        self.true_result_padd[0].get_value(),
    )
    self.assertEqual(
        result_jax[0][1] % util.MODULUS_377_INT,
        self.true_result_padd[1].get_value(),
    )
    self.assertEqual(
        result_jax[0][2] % util.MODULUS_377_INT,
        self.true_result_padd[2].get_value(),
    )
    self.assertEqual(
        result_jax[0][3] % util.MODULUS_377_INT,
        self.true_result_padd[3].get_value(),
    )

    # performance measurement
    tasks = [
        (jit_padd_lazy_xyzz_pack, (point_a_jax, point_b_jax)),
    ]
    profile_name = "jit_padd_lazy_xyzz_pack"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_pdul_lazy_xyzz_pack(self):
    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [self.point_a + [1] * (self.coordinate_num - len(self.point_a))],
        array_size=util.U16_EXT_CHUNK_NUM,
    )

    # lazy_mat = util.construct_lazy_matrix(util.MODULUS_377_INT)
    jit_pdul_lazy_xyzz_pack = jax.jit(jec.pdul_lazy_xyzz_pack)
    result_jax = jit_pdul_lazy_xyzz_pack(point_a_jax)
    result_jax = util.jax_point_pack_to_int_point_batch(result_jax)

    self.assertEqual(
        result_jax[0][0] % util.MODULUS_377_INT,
        self.true_result_pdub_a[0].get_value(),
    )
    self.assertEqual(
        result_jax[0][1] % util.MODULUS_377_INT,
        self.true_result_pdub_a[1].get_value(),
    )
    self.assertEqual(
        result_jax[0][2] % util.MODULUS_377_INT,
        self.true_result_pdub_a[2].get_value(),
    )
    self.assertEqual(
        result_jax[0][3] % util.MODULUS_377_INT,
        self.true_result_pdub_a[3].get_value(),
    )

    # performance measurement
    tasks = [
        (jit_pdul_lazy_xyzz_pack, (point_a_jax,)),
    ]
    profile_name = "jit_pdul_lazy_xyzz_pack"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_padd_barrett_twisted_pack(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    twist_a = twisted_ec_sys.twist_int_coordinates(self.point_a)
    twist_b = twisted_ec_sys.twist_int_coordinates(self.point_b)

    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [twist_a], array_size=util.U16_CHUNK_NUM
    )
    point_b_jax = util.int_point_batch_to_jax_point_pack(
        [twist_b], array_size=util.U16_CHUNK_NUM
    )

    jit_padd_barrett_twisted_pack = jax.jit(jec.padd_barrett_twisted_pack)
    result_jax = jit_padd_barrett_twisted_pack(point_a_jax, point_b_jax)
    result_int = util.jax_point_pack_to_int_point_batch(result_jax)

    result_affine_point = twisted_ec_sys.generate_point(
        result_int[0], twist=False
    ).convert_to_affine()

    self.assertEqual(
        result_affine_point[0].get_value(),
        self.true_result_padd_affine[0].get_value(),
    )
    self.assertEqual(
        result_affine_point[1].get_value(),
        self.true_result_padd_affine[1].get_value(),
    )

  def test_pdul_barrett_twisted_pack(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    twist_a = twisted_ec_sys.twist_int_coordinates(self.point_a)
    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [twist_a], array_size=util.U16_CHUNK_NUM
    )

    jit_pdul_barrett_twisted_pack = jax.jit(jec.pdul_barrett_twisted_pack)
    result_jax = jit_pdul_barrett_twisted_pack(point_a_jax)
    result_int = util.jax_point_pack_to_int_point_batch(result_jax)

    result_affine_point = twisted_ec_sys.generate_point(
        result_int[0], twist=False
    ).convert_to_affine()
    self.assertEqual(
        result_affine_point[0].get_value(),
        self.true_result_pdub_a_affine[0].get_value(),
    )
    self.assertEqual(
        result_affine_point[1].get_value(),
        self.true_result_pdub_a_affine[1].get_value(),
    )

    # performance measurement
    tasks = [
        (jit_pdul_barrett_twisted_pack, (point_a_jax,)),
    ]
    profile_name = "jit_pdul_barrett_twisted_pack"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_padd_lazy_twisted_pack(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    twist_a = twisted_ec_sys.twist_int_coordinates(self.point_a)
    twist_b = twisted_ec_sys.twist_int_coordinates(self.point_b)

    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [twist_a], array_size=util.U16_EXT_CHUNK_NUM
    )
    point_b_jax = util.int_point_batch_to_jax_point_pack(
        [twist_b], array_size=util.U16_EXT_CHUNK_NUM
    )

    jit_padd_lazy_twisted_pack = jax.jit(jec.padd_lazy_twisted_pack)
    result_jax = jit_padd_lazy_twisted_pack(point_a_jax, point_b_jax)
    result_int = util.jax_point_pack_to_int_point_batch(result_jax)

    result_affine_point = twisted_ec_sys.generate_point(
        result_int[0], twist=False
    ).convert_to_affine()

    self.assertEqual(
        result_affine_point[0].get_value(),
        self.true_result_padd_affine[0].get_value(),
    )
    self.assertEqual(
        result_affine_point[1].get_value(),
        self.true_result_padd_affine[1].get_value(),
    )

  @absltest.skip("skip current test")
  def test_padd_lazy_twisted_pack_batch(self):
    for batch_size in [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
      twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
          config_file.config_BLS12_377_t
      )
      twist_a = twisted_ec_sys.twist_int_coordinates(self.point_a)
      twist_b = twisted_ec_sys.twist_int_coordinates(self.point_b)

      point_a_jax = util.int_point_batch_to_jax_point_pack(
          [twist_a], array_size=util.U16_EXT_CHUNK_NUM
      )
      point_b_jax = util.int_point_batch_to_jax_point_pack(
          [twist_b], array_size=util.U16_EXT_CHUNK_NUM
      )
      point_a_jax = jnp.broadcast_to(
          point_a_jax, (point_a_jax.shape[0], batch_size, point_a_jax.shape[-1])
      )
      point_b_jax = jnp.broadcast_to(
          point_b_jax, (point_b_jax.shape[0], batch_size, point_b_jax.shape[-1])
      )

      jit_padd_lazy_twisted_pack_batch = jax.jit(
          jax.named_call(
              functools.partial(jec.padd_lazy_twisted_pack),
              name=f"jit_padd_lazy_twisted_pack_batch_{batch_size}",
          ),
      )
      result_jax = jit_padd_lazy_twisted_pack_batch(point_a_jax, point_b_jax)
      result_int = util.jax_point_pack_to_int_point_batch(result_jax)
      result_affine_point = twisted_ec_sys.generate_point(
          result_int[0], twist=False
      ).convert_to_affine()

      self.assertEqual(
          result_affine_point.coordinates[0].value % util.MODULUS_377_INT,
          self.true_result_padd_affine[0].get_value(),
      )
      self.assertEqual(
          result_affine_point.coordinates[1].value % util.MODULUS_377_INT,
          self.true_result_padd_affine[1].get_value(),
      )
      self.assertEqual(
          result_affine_point.coordinates[2].value % util.MODULUS_377_INT,
          self.true_result_padd_affine[2].get_value(),
      )
      self.assertEqual(
          result_affine_point.coordinates[3].value % util.MODULUS_377_INT,
          self.true_result_padd_affine[3].get_value(),
      )

      # performance measurement
      tasks = [
          (jit_padd_lazy_twisted_pack_batch, (point_a_jax, point_b_jax)),
      ]
      profile_name = f"jit_padd_lazy_twisted_pack_batch_{batch_size}"
      # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_padd_same_lazy_twisted_pack(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    twist_a1 = twisted_ec_sys.twist_int_coordinates(self.point_a)
    twist_a2 = twisted_ec_sys.twist_int_coordinates(self.point_a)

    point_a1_jax = util.int_point_batch_to_jax_point_pack(
        [twist_a1], array_size=util.U16_EXT_CHUNK_NUM
    )
    point_a2_jax = util.int_point_batch_to_jax_point_pack(
        [twist_a2], array_size=util.U16_EXT_CHUNK_NUM
    )

    jit_padd_lazy_twisted_pack = jax.jit(jec.padd_lazy_twisted_pack)
    result_jax = jit_padd_lazy_twisted_pack(point_a1_jax, point_a2_jax)
    result_int = util.jax_point_pack_to_int_point_batch(result_jax)

    result_affine_point = twisted_ec_sys.generate_point(
        result_int[0], twist=False
    ).convert_to_affine()

    self.assertEqual(
        result_affine_point[0].get_value(),
        self.true_result_pdub_a_affine[0].get_value(),
    )
    self.assertEqual(
        result_affine_point[1].get_value(),
        self.true_result_pdub_a_affine[1].get_value(),
    )

    # performance measurement
    tasks = [
        (jit_padd_lazy_twisted_pack, (point_a1_jax, point_a2_jax)),
    ]
    profile_name = "jit_padd_lazy_twisted_pack"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_pdul_lazy_twisted_pack(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    twist_a = twisted_ec_sys.twist_int_coordinates(self.point_a)
    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [twist_a], array_size=util.U16_EXT_CHUNK_NUM
    )

    jit_pdul_lazy_twisted_pack = jax.jit(jec.pdul_lazy_twisted_pack)
    result_jax = jit_pdul_lazy_twisted_pack(point_a_jax)
    result_int = util.jax_point_pack_to_int_point_batch(result_jax)

    result_affine_point = twisted_ec_sys.generate_point(
        result_int[0], twist=False
    ).convert_to_affine()
    self.assertEqual(
        result_affine_point[0].get_value(),
        self.true_result_pdub_a_affine[0].get_value(),
    )
    self.assertEqual(
        result_affine_point[1].get_value(),
        self.true_result_pdub_a_affine[1].get_value(),
    )

    # performance measurement
    tasks = [
        (jit_pdul_lazy_twisted_pack, (point_a_jax,)),
    ]
    profile_name = "jit_pdul_lazy_twisted_pack"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_pneg_lazy_twisted_pack(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    twist_a = twisted_ec_sys.twist_int_coordinates(self.point_a)
    twist_b = twisted_ec_sys.twist_int_coordinates(self.point_b)

    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [twist_a], array_size=util.U16_EXT_CHUNK_NUM
    )
    point_b_jax = util.int_point_batch_to_jax_point_pack(
        [twist_b], array_size=util.U16_EXT_CHUNK_NUM
    )

    jit_padd_lazy_twisted_pack = jax.jit(jec.padd_lazy_twisted_pack)
    jit_pneg_lazy_twisted_pack = jax.jit(jec.pneg_lazy_twisted_pack)
    a_plus_b = jit_padd_lazy_twisted_pack(point_a_jax, point_b_jax)
    neg_b = jit_pneg_lazy_twisted_pack(point_b_jax)
    result_jax = jit_padd_lazy_twisted_pack(a_plus_b, neg_b)
    result_int = util.jax_point_pack_to_int_point_batch(result_jax)

    result_affine_point = twisted_ec_sys.generate_point(
        result_int[0], twist=False
    ).convert_to_affine()
    self.assertEqual(
        result_affine_point[0].get_value(), self.point_a_sys[0].get_value()
    )
    self.assertEqual(
        result_affine_point[1].get_value(), self.point_a_sys[1].get_value()
    )

    # performance measurement
    tasks = [
        (jit_pneg_lazy_twisted_pack, (point_b_jax,)),
    ]
    profile_name = "jit_pneg_lazy_twisted_pack"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_padd_rns_xyzz(self):
    point_a_jax = util.int_point_batch_to_jax_rns_point_pack(
        [self.point_a + [1] * (self.coordinate_num - len(self.point_a))]
    )
    point_b_jax = util.int_point_batch_to_jax_rns_point_pack(
        [self.point_b + [1] * (self.coordinate_num - len(self.point_b))]
    )
    rns_mat = util.construct_rns_matrix(util.MODULUS_377_INT)

    jit_padd_rns_xyzz_pack = jax.jit(
        jax.named_call(
            functools.partial(jec.padd_rns_xyzz_pack, rns_mat=rns_mat),
            name="jit_padd_rns_xyzz_pack",
        ),
    )
    result_jax = jit_padd_rns_xyzz_pack(point_a_jax, point_b_jax)
    result_jax = util.jax_rns_point_pack_to_int_point_batch(result_jax)

    self.assertEqual(
        result_jax[0][0] % util.MODULUS_377_INT,
        self.true_result_padd[0].get_value(),
    )
    self.assertEqual(
        result_jax[0][1] % util.MODULUS_377_INT,
        self.true_result_padd[1].get_value(),
    )
    self.assertEqual(
        result_jax[0][2] % util.MODULUS_377_INT,
        self.true_result_padd[2].get_value(),
    )
    self.assertEqual(
        result_jax[0][3] % util.MODULUS_377_INT,
        self.true_result_padd[3].get_value(),
    )

    # performance measurement
    tasks = [
        (jit_padd_rns_xyzz_pack, (point_a_jax, point_b_jax)),
    ]
    profile_name = "jit_padd_rns_xyzz_pack"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_pdul_rns_xyzz_pack(self):
    point_a_jax = util.int_point_batch_to_jax_rns_point_pack(
        [self.point_a + [1] * (self.coordinate_num - len(self.point_a))]
    )
    rns_mat = util.construct_rns_matrix(util.MODULUS_377_INT)

    jit_pdul_rns_xyzz_pack = jax.jit(
        jax.named_call(
            functools.partial(jec.pdul_rns_xyzz_pack, rns_mat=rns_mat),
            name="jit_pdul_rns_xyzz_pack",
        ),
    )
    result_jax = jit_pdul_rns_xyzz_pack(point_a_jax)
    result_jax = util.jax_rns_point_pack_to_int_point_batch(result_jax)

    self.assertEqual(
        result_jax[0][0] % util.MODULUS_377_INT,
        self.true_result_pdub_a[0].get_value(),
    )
    self.assertEqual(
        result_jax[0][1] % util.MODULUS_377_INT,
        self.true_result_pdub_a[1].get_value(),
    )
    self.assertEqual(
        result_jax[0][2] % util.MODULUS_377_INT,
        self.true_result_pdub_a[2].get_value(),
    )
    self.assertEqual(
        result_jax[0][3] % util.MODULUS_377_INT,
        self.true_result_pdub_a[3].get_value(),
    )

    # performance measurement
    tasks = [
        (jit_pdul_rns_xyzz_pack, (point_a_jax,)),
    ]
    profile_name = "jit_pdul_rns_xyzz_pack"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_padd_rns_twisted_pack(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    a = self.point_a_sys.coordinates[:2]
    b = self.point_b_sys.coordinates[:2]
    project_twist_a = twisted_ec_sys.generate_point(a, twist=True)
    project_twist_b = twisted_ec_sys.generate_point(b, twist=True)
    rns_mat = util.construct_rns_matrix(util.MODULUS_377_INT)
    point_a_jax = util.int_point_batch_to_jax_rns_point_pack(
        [[c.get_value() for c in project_twist_a]]
    )
    point_b_jax = util.int_point_batch_to_jax_rns_point_pack(
        [[c.get_value() for c in project_twist_b]]
    )
    jit_padd_rns_twisted_pack = jax.jit(
        jax.named_call(
            functools.partial(jec.padd_rns_twisted_pack, rns_mat=rns_mat),
            name="jit_padd_rns_twisted_pack",
        ),
    )
    point_c_jax = jit_padd_rns_twisted_pack(point_a_jax, point_b_jax)
    project_twist_sum = util.jax_rns_point_pack_to_int_point_batch(point_c_jax)[
        0
    ]
    project_twist_sum_point = twisted_ec_sys.generate_point(
        project_twist_sum, twist=False
    ).convert_to_affine()
    s = project_twist_sum_point.coordinates[:2]
    correct_s = self.true_result_padd_affine.coordinates[:2]
    self.assertEqual(
        s[0].get_value() % util.MODULUS_377_INT, correct_s[0].get_value()
    )
    self.assertEqual(
        s[1].get_value() % util.MODULUS_377_INT, correct_s[1].get_value()
    )

    # performance measurement
    tasks = [
        (jit_padd_rns_twisted_pack, (point_a_jax, point_b_jax)),
    ]
    profile_name = "jit_padd_rns_twisted_pack"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_pdul_rns_twisted_pack(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    rns_mat = util.construct_rns_matrix(util.MODULUS_377_INT)
    pg = twisted_ec_sys.generate_point(self.point_a, twist=False)
    g_correct_ff = pg << 1
    gcf_af = g_correct_ff.convert_to_affine()
    point_g_jax = util.int_point_batch_to_jax_rns_point_pack(
        [[c.get_value() for c in pg]]
    )
    jit_pdul_rns_twisted_pack = jax.jit(
        jax.named_call(
            functools.partial(jec.pdul_rns_twisted_pack, rns_mat=rns_mat),
            name="jit_pdul_rns_twisted_pack",
        ),
    )
    point_2g_jax = jit_pdul_rns_twisted_pack(point_g_jax)
    g_test = util.jax_rns_point_pack_to_int_point_batch(point_2g_jax)[0]
    gtf_af = twisted_ec_sys.generate_point(
        g_test, twist=False
    ).convert_to_affine()
    self.assertEqual(
        gtf_af.coordinates[0].get_value() % util.MODULUS_377_INT,
        gcf_af.coordinates[0].get_value(),
    )
    self.assertEqual(
        gtf_af.coordinates[1].get_value() % util.MODULUS_377_INT,
        gcf_af.coordinates[1].get_value(),
    )

    # performance measurement
    tasks = [
        (jit_pdul_rns_twisted_pack, (point_g_jax,)),
    ]
    profile_name = "jit_pdul_rns_twisted_pack"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_padd_rns_twisted_pack_new_twisted(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    twist_a = twisted_ec_sys.twist_int_coordinates(self.point_a)
    twist_b = twisted_ec_sys.twist_int_coordinates(self.point_b)

    point_a_jax = util.int_point_to_jax_rns_point_pack(twist_a)
    point_b_jax = util.int_point_to_jax_rns_point_pack(twist_b)

    rns_mat = util.construct_rns_matrix(util.MODULUS_377_INT)
    jit_padd_rns_twisted_pack = jax.jit(
        jax.named_call(
            functools.partial(jec.padd_rns_twisted_pack, rns_mat=rns_mat),
            name="jit_padd_rns_twisted_pack",
        ),
    )
    point_c_jax = jit_padd_rns_twisted_pack(point_a_jax, point_b_jax)
    project_twist_sum = util.jax_rns_point_pack_to_int_point_batch(point_c_jax)[
        0
    ]
    project_twist_sum_point = twisted_ec_sys.generate_point(
        project_twist_sum, twist=False
    ).convert_to_affine()
    s = project_twist_sum_point.coordinates[:2]
    correct_s = self.true_result_padd_affine.coordinates[:2]
    self.assertEqual(
        s[0].get_value() % util.MODULUS_377_INT, correct_s[0].get_value()
    )
    self.assertEqual(
        s[1].get_value() % util.MODULUS_377_INT, correct_s[1].get_value()
    )

    # performance measurement
    tasks = [
        (jit_padd_rns_twisted_pack, (point_a_jax, point_b_jax)),
    ]
    profile_name = "jit_padd_rns_twisted_pack_new_twist"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_pdul_rns_twisted_pack_new_twisted(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    rns_mat = util.construct_rns_matrix(util.MODULUS_377_INT)
    twist_a = twisted_ec_sys.twist_int_coordinates(self.point_a)
    point_a_jax = util.int_point_to_jax_rns_point_pack(twist_a)
    jit_pdul_rns_twisted_pack = jax.jit(
        jax.named_call(
            functools.partial(jec.pdul_rns_twisted_pack, rns_mat=rns_mat),
            name="jit_pdul_rns_twisted_pack",
        ),
    )
    point_2a_jax = jit_pdul_rns_twisted_pack(point_a_jax)
    g_test = util.jax_rns_point_pack_to_int_point_batch(point_2a_jax)[0]
    gtf_af = twisted_ec_sys.generate_point(
        g_test, twist=False
    ).convert_to_affine()
    self.assertEqual(
        gtf_af.coordinates[0].get_value() % util.MODULUS_377_INT,
        self.true_result_pdub_a_affine[0].get_value(),
    )
    self.assertEqual(
        gtf_af.coordinates[1].get_value() % util.MODULUS_377_INT,
        self.true_result_pdub_a_affine[1].get_value(),
    )

    # performance measurement
    tasks = [
        (jit_pdul_rns_twisted_pack, (point_a_jax,)),
    ]
    profile_name = "jit_pdul_rns_twisted_pack_new_twist"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_padd_rns_twisted_pack_new_twist_two_batch(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    twist_a = twisted_ec_sys.twist_int_coordinates(self.point_a)
    point_a_jax = util.int_point_to_jax_rns_point_pack(twist_a).reshape(
        util.COORDINATE_NUM, 1, util.NUM_MODULI
    )

    twist_b = twisted_ec_sys.twist_int_coordinates(self.point_b)
    point_b_jax = util.int_point_to_jax_rns_point_pack(twist_b).reshape(
        util.COORDINATE_NUM, 1, util.NUM_MODULI
    )

    batch_point = jnp.concatenate([point_a_jax, point_b_jax], axis=1)

    rns_mat = util.construct_rns_matrix(util.MODULUS_377_INT)
    jit_padd_rns_twisted_pack_two_batch = jax.jit(
        jax.named_call(
            functools.partial(jec.padd_rns_twisted_pack, rns_mat=rns_mat),
            name="jit_padd_rns_twisted_pack_two_batch",
        ),
    )
    result_batch = jit_padd_rns_twisted_pack_two_batch(batch_point, batch_point)
    project_twist_sum = util.jax_rns_point_pack_to_int_point_batch(result_batch)
    point_2a_jax = twisted_ec_sys.generate_point(
        project_twist_sum[0], twist=False
    ).convert_to_affine()
    point_2b_jax = twisted_ec_sys.generate_point(
        project_twist_sum[1], twist=False
    ).convert_to_affine()
    self.assertEqual(
        point_2a_jax[0].get_value() % util.MODULUS_377_INT,
        self.true_result_pdub_a_affine[0].get_value(),
    )
    self.assertEqual(
        point_2a_jax[1].get_value() % util.MODULUS_377_INT,
        self.true_result_pdub_a_affine[1].get_value(),
    )
    self.assertEqual(
        point_2b_jax[0].get_value() % util.MODULUS_377_INT,
        self.true_result_pdub_b_affine[0].get_value(),
    )
    self.assertEqual(
        point_2b_jax[1].get_value() % util.MODULUS_377_INT,
        self.true_result_pdub_b_affine[1].get_value(),
    )

    # performance measurement
    tasks = [
        (jit_padd_rns_twisted_pack_two_batch, (batch_point, batch_point)),
    ]
    profile_name = "jit_padd_rns_twisted_pack_two_batch"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_pdul_rns_twisted_pack_new_twist_two_batch(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    twist_a = twisted_ec_sys.twist_int_coordinates(self.point_a)
    point_a_jax = util.int_point_to_jax_rns_point_pack(twist_a).reshape(
        util.COORDINATE_NUM, 1, util.NUM_MODULI
    )

    twist_b = twisted_ec_sys.twist_int_coordinates(self.point_b)
    point_b_jax = util.int_point_to_jax_rns_point_pack(twist_b).reshape(
        util.COORDINATE_NUM, 1, util.NUM_MODULI
    )

    batch_point = jnp.concatenate([point_a_jax, point_b_jax], axis=1)

    rns_mat = util.construct_rns_matrix(util.MODULUS_377_INT)
    jit_pdul_rns_twisted_pack_two_batch = jax.jit(
        jax.named_call(
            functools.partial(jec.pdul_rns_twisted_pack, rns_mat=rns_mat),
            name="jit_pdul_rns_twisted_pack_two_batch",
        ),
    )
    result_batch = jit_pdul_rns_twisted_pack_two_batch(batch_point)
    project_twist_sum = util.jax_rns_point_pack_to_int_point_batch(result_batch)
    point_2a_jax = twisted_ec_sys.generate_point(
        project_twist_sum[0], twist=False
    ).convert_to_affine()
    point_2b_jax = twisted_ec_sys.generate_point(
        project_twist_sum[1], twist=False
    ).convert_to_affine()
    self.assertEqual(
        point_2a_jax[0].get_value() % util.MODULUS_377_INT,
        self.true_result_pdub_a_affine[0].get_value(),
    )
    self.assertEqual(
        point_2a_jax[1].get_value() % util.MODULUS_377_INT,
        self.true_result_pdub_a_affine[1].get_value(),
    )
    self.assertEqual(
        point_2b_jax[0].get_value() % util.MODULUS_377_INT,
        self.true_result_pdub_b_affine[0].get_value(),
    )
    self.assertEqual(
        point_2b_jax[1].get_value() % util.MODULUS_377_INT,
        self.true_result_pdub_b_affine[1].get_value(),
    )

    # performance measurement
    tasks = [
        (jit_pdul_rns_twisted_pack_two_batch, (batch_point,)),
    ]
    profile_name = "jit_pdul_rns_twisted_pack_two_batch"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_padd_zero_twisted_pack_new_twisted(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    twist_a = twisted_ec_sys.twist_int_coordinates(self.point_a)
    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [twist_a], array_size=util.U16_EXT_CHUNK_NUM
    )
    point_zero_jax = util.int_point_batch_to_jax_point_pack(
        [self.zero_twisted], array_size=util.U16_EXT_CHUNK_NUM
    )

    jit_padd_lazy_twisted_pack = jax.jit(
        jax.named_call(
            functools.partial(jec.padd_lazy_twisted_pack),
            name="jit_padd_lazy_twisted_pack",
        ),
    )
    point_c_jax = jit_padd_lazy_twisted_pack(point_a_jax, point_zero_jax)
    # point_c_jax = jec.padd_lazy_twisted_pack(point_a_jax, point_zero_jax)
    project_twist_sum = util.jax_point_pack_to_int_point_batch(point_c_jax)[0]
    project_twist_sum_point = twisted_ec_sys.generate_point(
        project_twist_sum, twist=False
    ).convert_to_affine()
    self.assertEqual(
        project_twist_sum_point[0].get_value() % util.MODULUS_377_INT,
        self.point_a[0],
    )
    self.assertEqual(
        project_twist_sum_point[1].get_value() % util.MODULUS_377_INT,
        self.point_a[1],
    )

  @absltest.skip("This is the known issue, which does not affect XYZZ.")
  def test_padd_zero_rns_twisted_pack_new_twisted(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    twist_a = twisted_ec_sys.twist_int_coordinates(self.point_a)

    point_a_jax = util.int_point_batch_to_jax_rns_point_pack([twist_a])
    point_zero_jax = util.int_point_batch_to_jax_rns_point_pack(
        [self.zero_twisted]
    )

    rns_mat = util.construct_rns_matrix(util.MODULUS_377_INT)
    jit_padd_rns_twisted_pack = jax.jit(
        jax.named_call(
            functools.partial(jec.padd_rns_twisted_pack, rns_mat=rns_mat),
            name="jit_padd_rns_twisted_pack",
        ),
    )
    point_c_jax = jit_padd_rns_twisted_pack(point_a_jax, point_zero_jax)
    project_twist_sum = util.jax_rns_point_pack_to_int_point_batch(point_c_jax)[
        0
    ]
    project_twist_sum_point = twisted_ec_sys.generate_point(
        project_twist_sum, twist=False
    ).convert_to_affine()
    self.assertEqual(
        project_twist_sum_point[0].get_value() % util.MODULUS_377_INT,
        self.point_a[0],
    )
    self.assertEqual(
        project_twist_sum_point[1].get_value() % util.MODULUS_377_INT,
        self.point_a[1],
    )

  @absltest.skip("This is the known issue, which does not affect XYZZ.")
  def test_padd_rns_a_point_add_zero_correctness(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    test_in_point = [
        184360877740379501345167057231241011318955851506892921023614488028185166370541128591905842464011651119609504970811,
        47698235458971847835762299820400550031713475079888046003406323907410999702258242394959839249289205517205485978635,
    ]
    twist_a = twisted_ec_sys.twist_int_coordinates(test_in_point)
    twist_b = [0, 1, 1, 0]
    point_a_jax = util.int_point_batch_to_jax_point_pack(
        [twist_a], array_size=util.U16_EXT_CHUNK_NUM
    )
    point_b_jax = util.int_point_batch_to_jax_point_pack(
        [twist_b], array_size=util.U16_EXT_CHUNK_NUM
    )
    point_a_jax_rns = util.int_point_batch_to_jax_rns_point_pack([twist_a])
    point_b_jax_rns = util.int_point_batch_to_jax_rns_point_pack([twist_b])

    point_c_jax_rns = jec.padd_rns_twisted_pack(
        point_a_jax_rns, point_b_jax_rns
    )
    project_twist_sum_rns = util.jax_rns_point_pack_to_int_point_batch(
        point_c_jax_rns
    )[0]
    affine_sum_point_rns = twisted_ec_sys.generate_point(
        project_twist_sum_rns, twist=False
    ).convert_to_affine()
    point_c_jax = jec.padd_lazy_twisted_pack(point_a_jax, point_b_jax)
    project_twist_sum = util.jax_point_pack_to_int_point_batch(point_c_jax)[0]
    affine_sum_point = twisted_ec_sys.generate_point(
        project_twist_sum, twist=False
    ).convert_to_affine()
    # In Twisted Edward Representation Verification
    self.assertEqual(
        project_twist_sum[0] % util.MODULUS_377_INT,
        project_twist_sum_rns[0] % util.MODULUS_377_INT,
    )
    self.assertEqual(
        project_twist_sum[1] % util.MODULUS_377_INT,
        project_twist_sum_rns[1] % util.MODULUS_377_INT,
    )
    self.assertEqual(
        project_twist_sum[0] % util.MODULUS_377_INT,
        project_twist_sum_rns[0] % util.MODULUS_377_INT,
    )
    self.assertEqual(
        project_twist_sum[1] % util.MODULUS_377_INT,
        project_twist_sum_rns[1] % util.MODULUS_377_INT,
    )

    # Verification in affine
    self.assertEqual(
        affine_sum_point[0].get_value() % util.MODULUS_377_INT,
        affine_sum_point_rns[0].get_value() % util.MODULUS_377_INT,
    )
    self.assertEqual(
        affine_sum_point[1].get_value() % util.MODULUS_377_INT,
        affine_sum_point_rns[1].get_value() % util.MODULUS_377_INT,
    )

  def test_padd_barrett_xyzz_pack_batch(self):
    point_a_in = util.int_point_batch_to_jax_point_pack(
        [self.point_a + [1] * (self.coordinate_num - len(self.point_a))]
    )
    point_b_in = util.int_point_batch_to_jax_point_pack(
        [self.point_b + [1] * (self.coordinate_num - len(self.point_b))]
    )
    for batch_size in [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
      point_a_jax = jnp.broadcast_to(
          point_a_in, (point_a_in.shape[0], batch_size, point_a_in.shape[-1])
      )
      point_b_jax = jnp.broadcast_to(
          point_b_in, (point_b_in.shape[0], batch_size, point_b_in.shape[-1])
      )
      jit_padd_barrett_xyzz_pack = jax.jit(
          jax.named_call(jec.padd_barrett_xyzz_pack,
                         name=f"jit_padd_barrett_xyzz_pack_{batch_size}")
      )

      result_jax = jit_padd_barrett_xyzz_pack(point_a_jax, point_b_jax)
      result_jax = util.jax_point_pack_to_int_point_batch(result_jax)

      self.assertEqual(result_jax[0][0], self.true_result_padd[0].get_value())
      self.assertEqual(result_jax[0][1], self.true_result_padd[1].get_value())
      self.assertEqual(result_jax[0][2], self.true_result_padd[2].get_value())
      self.assertEqual(result_jax[0][3], self.true_result_padd[3].get_value())

      # performance measurement
      tasks = [
          (jit_padd_barrett_xyzz_pack, (point_a_jax, point_b_jax)),
      ]
      profile_name = f"jit_padd_barrett_xyzz_pack_{batch_size}"
      # copybara: util.profile_jax_functions(tasks, profile_name)

  def test_padd_barrett_twisted_pack_batch(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    twist_a = twisted_ec_sys.twist_int_coordinates(self.point_a)
    twist_b = twisted_ec_sys.twist_int_coordinates(self.point_b)

    point_a_in = util.int_point_batch_to_jax_point_pack(
        [twist_a], array_size=util.U16_CHUNK_NUM
    )
    point_b_in = util.int_point_batch_to_jax_point_pack(
        [twist_b], array_size=util.U16_CHUNK_NUM
    )
    for batch_size in [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
      point_a_jax = jnp.broadcast_to(
          point_a_in, (point_a_in.shape[0], batch_size, point_a_in.shape[-1])
      )
      point_b_jax = jnp.broadcast_to(
          point_b_in, (point_b_in.shape[0], batch_size, point_b_in.shape[-1])
      )
      jit_padd_barrett_twisted_pack = jax.jit(
          jax.named_call(
              jec.padd_barrett_twisted_pack,
              name=f"jit_padd_barrett_twisted_pack_{batch_size}",
          )
      )
      result_jax = jit_padd_barrett_twisted_pack(point_a_jax, point_b_jax)
      result_int = util.jax_point_pack_to_int_point_batch(result_jax)

      result_affine_point = twisted_ec_sys.generate_point(
          result_int[0], twist=False
      ).convert_to_affine()

      self.assertEqual(
          result_affine_point[0].get_value(),
          self.true_result_padd_affine[0].get_value(),
      )
      self.assertEqual(
          result_affine_point[1].get_value(),
          self.true_result_padd_affine[1].get_value(),
      )
      # performance measurement
      tasks = [
          (jit_padd_barrett_twisted_pack, (point_a_jax, point_b_jax)),
      ]
      profile_name = f"jit_padd_barrett_twisted_pack_{batch_size}"
      # copybara: util.profile_jax_functions(tasks, profile_name)

  # @absltest.skip("skip current test")
  def test_padd_rns_xyzz_batch(self):
    for batch_size in [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
      point_a_jax = util.int_point_batch_to_jax_rns_point_pack(
          [self.point_a + [1] * (self.coordinate_num - len(self.point_a))]
      )
      point_b_jax = util.int_point_batch_to_jax_rns_point_pack(
          [self.point_b + [1] * (self.coordinate_num - len(self.point_b))]
      )
      point_a_jax = jnp.broadcast_to(
          point_a_jax, (point_a_jax.shape[0], batch_size, point_a_jax.shape[-1])
      )
      point_b_jax = jnp.broadcast_to(
          point_b_jax, (point_b_jax.shape[0], batch_size, point_b_jax.shape[-1])
      )
      rns_mat = util.construct_rns_matrix(util.MODULUS_377_INT)

      jit_padd_rns_xyzz_pack_batch = jax.jit(
          jax.named_call(
              functools.partial(jec.padd_rns_xyzz_pack, rns_mat=rns_mat),
              name=f"jit_padd_rns_xyzz_pack_batch_{batch_size}",
          ),
      )
      result_jax = jit_padd_rns_xyzz_pack_batch(point_a_jax, point_b_jax)
      result_jax = util.jax_rns_point_pack_to_int_point_batch(result_jax)

      self.assertEqual(
          result_jax[0][0] % util.MODULUS_377_INT,
          self.true_result_padd[0].get_value(),
      )
      self.assertEqual(
          result_jax[0][1] % util.MODULUS_377_INT,
          self.true_result_padd[1].get_value(),
      )
      self.assertEqual(
          result_jax[0][2] % util.MODULUS_377_INT,
          self.true_result_padd[2].get_value(),
      )
      self.assertEqual(
          result_jax[0][3] % util.MODULUS_377_INT,
          self.true_result_padd[3].get_value(),
      )

      # performance measurement
      tasks = [
          (jit_padd_rns_xyzz_pack_batch, (point_a_jax, point_b_jax)),
      ]
      profile_name = f"jit_padd_rns_xyzz_pack_batch_{batch_size}"
      # copybara: util.profile_jax_functions(tasks, profile_name)

  # @absltest.skip("Skip for now")
  def test_padd_rns_twisted_pack_new_twist_batch(self):
    twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(
        config_file.config_BLS12_377_t
    )
    rns_mat = util.construct_rns_matrix(util.MODULUS_377_INT)
    twist_a = twisted_ec_sys.twist_int_coordinates(self.point_a)
    twist_b = twisted_ec_sys.twist_int_coordinates(self.point_b)
    twist_a_jax = util.int_point_to_jax_rns_point_pack(twist_a).reshape(
        util.COORDINATE_NUM, 1, util.NUM_MODULI
    )
    twist_b_jax = util.int_point_to_jax_rns_point_pack(twist_b).reshape(
        util.COORDINATE_NUM, 1, util.NUM_MODULI
    )

    for batch_size in [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
      point_a_jax = jnp.broadcast_to(
          twist_a_jax, (twist_a_jax.shape[0], batch_size, twist_a_jax.shape[-1])
      )
      point_b_jax = jnp.broadcast_to(
          twist_b_jax, (twist_b_jax.shape[0], batch_size, twist_b_jax.shape[-1])
      )

      jit_padd_rns_twisted_pack_batch = jax.jit(
          jax.named_call(
              functools.partial(jec.padd_rns_twisted_pack, rns_mat=rns_mat),
              name=f"jit_padd_rns_twisted_pack_batch_{batch_size}",
          ),
      )
      result_batch = jit_padd_rns_twisted_pack_batch(point_a_jax, point_b_jax)
      project_twist_sum = util.jax_rns_point_pack_to_int_point_batch(
          result_batch
      )
      project_twist_jax = twisted_ec_sys.generate_point(
          project_twist_sum[0], twist=False
      ).convert_to_affine()

      self.assertEqual(
          project_twist_jax[0].get_value() % util.MODULUS_377_INT,
          self.true_result_padd_affine[0].get_value(),
      )
      self.assertEqual(
          project_twist_jax[1].get_value() % util.MODULUS_377_INT,
          self.true_result_padd_affine[1].get_value(),
      )

      # performance measurement
      tasks = [
          (jit_padd_rns_twisted_pack_batch, (point_a_jax, point_b_jax)),
      ]
      profile_name = f"jit_padd_rns_twisted_pack_batch_{batch_size}"
      # copybara: util.profile_jax_functions(tasks, profile_name)


if __name__ == "__main__":
  absltest.main()
