import jax
import jax.numpy as jnp
from jaxite.jaxite_ec import util
from jaxite.jaxite_ec.algorithm import config_file
import jaxite.jaxite_ec.algorithm.elliptic_curve as ec
import jaxite.jaxite_ec.elliptic_curve as jec

from google3.perftools.accelerators.xprof.api.python import xprof_session
from absl.testing import absltest


class TestEllipticCurve(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.coordinate_num = 4
    self.pdul_barrett_xyzz_jit = jax.jit(jec.pdul_barrett_xyzz_pack)
    self.padd_barrett_xyzz_jit = jax.jit(jec.padd_barrett_xyzz_pack)
    self.x1_int_ = 0x01AC3A384FC584EFD3E7F2C5A2927E7D454875C874A051027B9E7363D08942533EDE85DAE295D8CAB2751085206BCA76
    self.y1_int_ = 0x011DB83AEC88460820F4868A73B12309EE2E910526E62DB4ACCB303ABF50F86C3985A072ED07A4B81FFB82D8DD247283
    self.x2_int_ = 0x01546AF2ABB4E189E9BBC412FDBF2A8E5EC6E4A3B0AF132E21EE9CEC3EF5E226490FB98D662670FA3CFB3948B7E2A48C
    self.y2_int_ = 0x002961A558A885DF227FDB09F8BDF57AF179CB9437FF8828F13E9DF01AE55502F409AAF5058B88F2F7CCC7BC0676A5D4
    self.point_a = [self.x1_int_, self.y1_int_]
    self.point_b = [self.x2_int_, self.y2_int_]
    self.ec_sys = ec.ECCSWeierstrassXYZZ(config_file.config_BLS12_377)
    self.point_a_sys = self.ec_sys.generate_point(self.point_a)
    self.point_b_sys = self.ec_sys.generate_point(self.point_b)
    self.true_result_padd = self.point_a_sys + self.point_b_sys
    self.true_result_pdub = self.point_a_sys + self.point_a_sys

    self.point_a_jax = util.int_point_to_jax_point_pack(
        self.point_a + [1] * (self.coordinate_num - len(self.point_a))
    )
    self.point_b_jax = util.int_point_to_jax_point_pack(
        self.point_b + [1] * (self.coordinate_num - len(self.point_b))
    )

  def test_padd_barrett_xyzz(self):
    session = xprof_session.XprofSession()
    session.start_session()
    result_jax = util.jax_point_pack_to_int_point(
        self.padd_barrett_xyzz_jit(self.point_a_jax, self.point_b_jax)
    )
    try:
      session_id = session.end_session_and_get_session_id()
      print(f"session_id: http://xprof/?session_id={session_id}")
    except RuntimeError as e:
      print(f"Failed to get session_id: {e}")
    self.assertEqual(result_jax[0], self.true_result_padd[0].get_value())
    self.assertEqual(result_jax[1], self.true_result_padd[1].get_value())
    self.assertEqual(result_jax[2], self.true_result_padd[2].get_value())
    self.assertEqual(result_jax[3], self.true_result_padd[3].get_value())

  def test_pdul_barrett_xyzz(self):
    session = xprof_session.XprofSession()
    session.start_session()
    result_jax = util.jax_point_pack_to_int_point(
        self.pdul_barrett_xyzz_jit(self.point_a_jax)
    )
    try:
      session_id = session.end_session_and_get_session_id()
      print(f"session_id: http://xprof/?session_id={session_id}")
    except RuntimeError as e:
      print(f"Failed to get session_id: {e}")
    self.assertEqual(result_jax[0], self.true_result_pdub[0].get_value())
    self.assertEqual(result_jax[1], self.true_result_pdub[1].get_value())
    self.assertEqual(result_jax[2], self.true_result_pdub[2].get_value())
    self.assertEqual(result_jax[3], self.true_result_pdub[3].get_value())


if __name__ == "__main__":
  absltest.main()
