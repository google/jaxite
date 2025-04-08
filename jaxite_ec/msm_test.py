import csv
import os
import sys

import jax
from jaxite.jaxite_ec import pippenger
from jaxite.jaxite_ec.algorithm import config_file as config
import jaxite.jaxite_ec.algorithm.elliptic_curve as ec
import jaxite.jaxite_ec.util as utils

# copybara: from google3.perftools.accelerators.xprof.api.python import xprof_session
# copybara: from google3.pyglib import resources
from absl.testing import absltest
from absl.testing import parameterized

script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)

config_BLS12_377 = config.config_BLS12_377
MSM_Pippenger = pippenger.MSMPippenger
jax.config.update("jax_traceback_filtering", "off")

# Only needed when tryingt to understand the HLO dump.
os.environ["XLA_FLAGS"] = (
    "--xla_dump_to=sponge --xla_backend_optimization_level=4"
)


class MSMTest(parameterized.TestCase):
  def read_external_file(self, scalar_path, base_path, result_path):
    scalars = []
    with open(
        scalar_path, "r", newline="", encoding="utf-8"
    ) as csvfile:  # Handle potential encoding issues
      csv_reader = csv.reader(csvfile)
      for row in csv_reader:
        scalars.append(int(row[-1][13:-2], 16))

    points = []
    with open(
        base_path, "r", newline="", encoding="utf-8"
    ) as csvfile:  # Handle potential encoding issues
      csv_reader = csv.reader(csvfile)
      for row in csv_reader:
        points.append([int(row[8][13:-2], 16), int(row[-1][13:-2], 16)])

    result_ref = []
    with open(
        result_path, "r", newline="", encoding="utf-8"
    ) as csvfile:  # Handle potential encoding issues
      csv_reader = csv.reader(csvfile)
      for row in csv_reader:
        result_ref.append(int(row[7][13:-2], 16))
        result_ref.append(int(row[-1][13:-2], 16))
    return scalars, points, result_ref

  @parameterized.named_parameters(
      # (
      #     "test_4_degree",
      #     os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
      #         f"{script_dir}/jaxite_ec/test_case/t4/zprize_msm_curve_377_scalars_dim_4_seed_0.csv"
      #     ),
      #     os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
      #         f"{script_dir}/jaxite_ec/test_case/t4/zprize_msm_curve_377_bases_dim_4_seed_0.csv"
      #     ),
      #     os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
      #         f"{script_dir}/jaxite_ec/test_case/t4/zprize_msm_curve_377_res_dim_4_seed_0.csv"
      #     ),
      # ),
      (
          "test_1024_degree",
          os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
              f"{script_dir}/jaxite_ec/test_case/t1024/zprize_msm_curve_377_scalars_dim_1024_seed_0.csv"
          ),
          os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
              f"{script_dir}/jaxite_ec/test_case/t1024/zprize_msm_curve_377_bases_dim_1024_seed_0.csv"
          ),
          os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
              f"{script_dir}/jaxite_ec/test_case/t1024/zprize_msm_curve_377_res_dim_1024_seed_0.csv"
          ),
      ),
  )
  def test_pippenger(self, scalar_path, base_path, result_path):
    scalars, points, result_ref = self.read_external_file(
        scalar_path, base_path, result_path
    )
    slice_length = 4
    msm_algo = MSM_Pippenger(slice_length)
    msm_algo.initialize(scalars, points)
    jax.block_until_ready(msm_algo.bucket_accumulation())
    jax.block_until_ready(msm_algo.bucket_reduction())
    result = msm_algo.window_merge()
    result = utils.jax_point_pack_to_int_point(result)
    ec_sys = ec.ECCSWeierstrassXYZZ(config_BLS12_377)
    result_affine_point = ec_sys.generate_point(result).convert_to_affine()
    coordinates = (
        result_affine_point[0].get_value(),
        result_affine_point[1].get_value(),
    )
    self.assertEqual(coordinates[0], result_ref[0])
    self.assertEqual(coordinates[1], result_ref[1])

    # copybara: session = xprof_session.XprofSession()
    # copybara: session.start_session()

    jax.block_until_ready(msm_algo.bucket_accumulation())
    jax.block_until_ready(msm_algo.bucket_reduction())
    jax.block_until_ready(msm_algo.window_merge())
    # copybara: session_id = session.end_session_and_get_session_id()
    # copybara: print(f'session_id: http://xprof/?session_id={session_id}')


if __name__ == "__main__":
  absltest.main()
