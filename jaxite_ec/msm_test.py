import os
import sys
from typing import cast

import jax
from jaxite.jaxite_ec import msm
from jaxite.jaxite_ec import pippenger
from jaxite.jaxite_ec.algorithm import config_file
from jaxite.jaxite_ec.algorithm import msm_reader as msm_reader_lib
import jaxite.jaxite_ec.algorithm.elliptic_curve as ec
import jaxite.jaxite_ec.elliptic_curve as jec
import jaxite.jaxite_ec.util as utils

from google3.pyglib import resources
from absl.testing import absltest


MSM_Pippenger = pippenger.MSMPippenger
MSM_Reader = msm_reader_lib.MSMReader
config_BLS12_377 = config_file.config_BLS12_377
MSM_DoubleAdd = msm.MSMDoubleAdd
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

BASE = 16


class MSMTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.pdul_barrett_xyzz_jit = jax.jit(jec.pdul_barrett_xyzz_pack)
    self.padd_barrett_xyzz_jit = jax.jit(jec.padd_barrett_xyzz_pack)
    self.selective_padd_barrett_xyzz_jit = jax.jit(msm.selective_padd_with_zero)

  def test_double_add_8(self):
    reader = MSM_Reader(
        scalar_file_path=resources.GetResourceFilename(
            'google3/third_party/py/jaxite/jaxite_ec/test_case/t8/zprize_msm_curve_377_scalars_dim_8_seed_0.csv'
        ),
        base_file_path=resources.GetResourceFilename(
            'google3/third_party/py/jaxite/jaxite_ec/test_case/t8/zprize_msm_curve_377_bases_dim_8_seed_0.csv'
        ),
        result_file_path=resources.GetResourceFilename(
            'google3/third_party/py/jaxite/jaxite_ec/test_case/t8/zprize_msm_curve_377_res_dim_8_seed_0.csv'
        ),
    )

    msm_algo = MSM_DoubleAdd()
    msm_algo.read_trace(reader)
    msm_algo.compute_base(self.pdul_barrett_xyzz_jit)
    msm_algo.bucket_accumulation(self.selective_padd_barrett_xyzz_jit)
    msm_algo.bucket_merge(self.padd_barrett_xyzz_jit)
    print('processing done')
    # results = utils.jax_point_coordinates_pack_to_int_point(
    #     msm_algo.result, BASE
    # )
    # ec_sys = ec.ECCSWeierstrassXYZZ(config_BLS12_377)
    # result_affine_point: ec.ECPoint = ec_sys.generate_point(
    #     results[0]
    # ).convert_to_affine()
    # coordinates = (
    #     result_affine_point[0].get_value(),
    #     result_affine_point[1].get_value(),
    # )

    # true_result = reader.get_result()
    # self.assertEqual(coordinates[0], true_result[0])
    # self.assertEqual(coordinates[1], true_result[1])
    # reader.close_files()

  # @absltest.skip("This test is temporarily disabled")
  def test_pippenger_1(self):
    slice_length = 7

    reader = MSM_Reader(
        scalar_file_path=resources.GetResourceFilename(
            'google3/third_party/py/jaxite/jaxite_ec/test_case/t1024/zprize_msm_curve_377_scalars_dim_1024_seed_0.csv'
        ),
        base_file_path=resources.GetResourceFilename(
            'google3/third_party/py/jaxite/jaxite_ec/test_case/t1024/zprize_msm_curve_377_bases_dim_1024_seed_0.csv'
        ),
        result_file_path=resources.GetResourceFilename(
            'google3/third_party/py/jaxite/jaxite_ec/test_case/t1024/zprize_msm_curve_377_res_dim_1024_seed_0.csv'
        ),
    )
    msm_algo = MSM_Pippenger(slice_length)
    msm_algo.read_trace(reader)
    msm_algo.bucket_accumulation(self.padd_barrett_xyzz_jit)
    msm_algo.bucket_reduction(self.padd_barrett_xyzz_jit)
    msm_algo.window_merge(
        self.padd_barrett_xyzz_jit, self.pdul_barrett_xyzz_jit
    )
    print('processing done')
    # if msm_algo.result is not None:
    #   results = utils.jax_point_pack_to_int_point(msm_algo.result)
    #   ec_sys = ec.ECCSWeierstrassXYZZ(config_BLS12_377)
    #   result_affine_point: ec.ECPoint = ec_sys.generate_point(
    #       results
    #   ).convert_to_affine()
    # coordinates = (
    #     result_affine_point[0].get_value(),
    #     result_affine_point[1].get_value(),
    # )

    # true_result = reader.get_result()
    # self.assertEqual(coordinates[0], true_result[0])
    # self.assertEqual(coordinates[1], true_result[1])
    #   reader.close_files()
    #   print('checking done')
    # else:
    #   print('No results producted')


if __name__ == '__main__':
  absltest.main()
