import csv
import functools
import os
import sys

import jax
import jax.numpy as jnp
from jaxite.jaxite_ec import msm
from jaxite.jaxite_ec import pippenger
from jaxite.jaxite_ec.algorithm import config_file
import jaxite.jaxite_ec.algorithm.elliptic_curve as ec
import jaxite.jaxite_ec.elliptic_curve as jec
import jaxite.jaxite_ec.util as utils
import numpy as np

# copybara: from google3.perftools.accelerators.xprof.api.python import xprof_session
# copybara: from google3.pyglib import resources
from absl.testing import absltest
from absl.testing import parameterized

script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)

config_BLS12_377 = config_file.config_BLS12_377
MSM_DoubleAdd = msm.MSMDoubleAdd
MSM_Pippenger = pippenger.MSMPippenger
jax.config.update("jax_traceback_filtering", "off")

# Only needed when tryingt to understand the HLO dump.
# os.environ["XLA_FLAGS"] = (
#     "--xla_dump_to=sponge --xla_dump_hlo_pass_re=.*"
# )
os.environ["XLA_FLAGS"] = (
    "--xla_dump_to=sponge --xla_backend_optimization_level=4"
)
# -xla_dump_hlo_as_proto


class MSMTest(parameterized.TestCase):

  def setUp(self):
    super(MSMTest, self).setUp()

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

  # @absltest.skip("This test runs forever")  # pylint: disable=superfluous-parens
  @parameterized.named_parameters(
      (
          "test_2_degree",
          os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
              f"{script_dir}/jaxite_ec/test_case/t2/zprize_msm_curve_377_scalars_dim_2_seed_0.csv"
          ),
          os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
              f"{script_dir}/jaxite_ec/test_case/t2/zprize_msm_curve_377_bases_dim_2_seed_0.csv"
          ),
          os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
              f"{script_dir}/jaxite_ec/test_case/t2/zprize_msm_curve_377_res_dim_2_seed_0.csv"
          ),
          2,
      ),
      # (
      #     "test_8_degree",
      #     os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
      #         f"{script_dir}/jaxite_ec/test_case/t8/zprize_msm_curve_377_scalars_dim_8_seed_0.csv"
      #     ),
      #     os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
      #         f"{script_dir}/jaxite_ec/test_case/t8/zprize_msm_curve_377_bases_dim_8_seed_0.csv"
      #     ),
      #     os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
      #         f"{script_dir}/jaxite_ec/test_case/t8/zprize_msm_curve_377_res_dim_8_seed_0.csv"
      #     ),
      #     8,
      # ),
      # (
      #     "test_1024_degree",
      #     os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
      #         f"{script_dir}/jaxite_ec/test_case/t1024/zprize_msm_curve_377_scalars_dim_1024_seed_0.csv"
      #     ),
      #     os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
      #         f"{script_dir}/jaxite_ec/test_case/t1024/zprize_msm_curve_377_bases_dim_1024_seed_0.csv"
      #     ),
      #     os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
      #         f"{script_dir}/jaxite_ec/test_case/t1024/zprize_msm_curve_377_res_dim_1024_seed_0.csv"
      #     ),
      #     1024,
      # ),
  )
  def test_double_add_compile_psum_accumulation(
      self, scalar_path, base_path, result_path, msm_length
  ):
    scalars, points, result_ref = self.read_external_file(
        scalar_path, base_path, result_path
    )

    # offline procedure
    msm_algo = MSM_DoubleAdd()
    msm_algo.initialize(scalars, points)
    msm_algo.compute_base()

    # Ahead of Time Compilation for the kernel for dynamic shape
    # See https://jax.readthedocs.io/en/latest/aot.html
    # num_coordinates = 4
    # num_u16_chunks = utils.U16_CHUNK_NUM
    # point_psum_shape = (num_coordinates, msm_length, num_u16_chunks)

    # func = functools.partial(
    #     jax.named_call(msm.psum_accumulation, name="psum_accumulation"),
    #     psum_addition_length=msm_algo.psum_addition_length,
    #     length=msm_length,
    # )
    # compiled_psum_accumulation = (
    #     jax.jit(func)
    #     .lower(
    #         jax.ShapeDtypeStruct(point_psum_shape, dtype=jnp.uint16),
    #     )
    #     .compile()
    # )

    # online procedure
    # # copybara: session = xprof_session.XprofSession()
    # # copybara: session.start_session()
    point_psum = msm.point_accumulation(
        msm_algo.point_psum,
        msm_algo.base_points,
        msm_algo.overall_select,
        msm_algo.overall_non_zero_states,
        msm_algo.scalar_precision,
    )
    # result = compiled_psum_accumulation(point_psum)
    # result = func(point_psum)

    # # copybara: session_id = session.end_session_and_get_session_id()
    # # copybara: print(f'session_id: http://xprof/?session_id={session_id}')

    # result = utils.jax_point_pack_to_int_point(result)
    # ec_sys = ec.ECCSWeierstrassXYZZ(config_BLS12_377)
    # result_affine_point = ec_sys.generate_point(result).convert_to_affine()
    # coordinates = (
    #     result_affine_point[0].get_value(),
    #     result_affine_point[1].get_value(),
    # )
    # self.assertEqual(coordinates[0], result_ref[0])
    # self.assertEqual(coordinates[1], result_ref[1])

  @absltest.skip("This test takes forever to compile")  # pylint: disable=superfluous-parens
  @parameterized.named_parameters(
      (
          "test_2_degree",
          os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
              f"{script_dir}/jaxite_ec/test_case/t2/zprize_msm_curve_377_scalars_dim_2_seed_0.csv"
          ),
          os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
              f"{script_dir}/jaxite_ec/test_case/t2/zprize_msm_curve_377_bases_dim_2_seed_0.csv"
          ),
          os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
              f"{script_dir}/jaxite_ec/test_case/t2/zprize_msm_curve_377_res_dim_2_seed_0.csv"
          ),
          2,
      ),
      (
          "test_8_degree",
          os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
              f"{script_dir}/jaxite_ec/test_case/t8/zprize_msm_curve_377_scalars_dim_8_seed_0.csv"
          ),
          os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
              f"{script_dir}/jaxite_ec/test_case/t8/zprize_msm_curve_377_bases_dim_8_seed_0.csv"
          ),
          os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
              f"{script_dir}/jaxite_ec/test_case/t8/zprize_msm_curve_377_res_dim_8_seed_0.csv"
          ),
          8,
      ),
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
          1024,
      ),
  )
  def test_double_add_compile_all(
      self, scalar_path, base_path, result_path, msm_length
  ):
    scalars, points, result_ref = self.read_external_file(
        scalar_path, base_path, result_path
    )

    # offline procedure
    msm_algo = MSM_DoubleAdd()
    msm_algo.initialize(scalars, points)
    msm_algo.compute_base()
    decomposed_point = utils.int_list_to_2d_array(
        points[0] + [1, 1], utils.BASE, utils.U16_CHUNK_NUM
    )
    base_point_ref = jnp.empty(
        (msm_algo.scalar_precision,) + decomposed_point.shape,
        dtype=jnp.uint16,
    )
    base_point_ref = base_point_ref.at[0].set(decomposed_point)
    for i in range(1, msm_algo.scalar_precision):
      base_point_ref = base_point_ref.at[i].set(
          jec.pdul_barrett_xyzz_pack(base_point_ref[i - 1])
      )
    np.testing.assert_array_equal(
        msm_algo.base_points[:, 0, :, :], base_point_ref
    )

    # Ahead of Time Compilation for the kernel for dynamic shape
    # See https://jax.readthedocs.io/en/latest/aot.html
    num_coordinates = 4
    num_u16_chunks = utils.U16_CHUNK_NUM
    point_psum_shape = (msm_length, num_coordinates, num_u16_chunks)

    point_accumulation_static_configured = jax.named_call(
        functools.partial(
            msm.point_accumulation,
            scalar_precision=msm_algo.scalar_precision,
        ),
        name="point_accumulation",
    )

    # The following compilation for point accumulation is taking forever.
    base_points_shape = (
        msm_algo.scalar_precision,
        msm_length,
        num_coordinates,
        num_u16_chunks,
    )
    overall_select_shape = (msm_algo.scalar_precision, msm_length)
    overall_non_zero_states_shape = (msm_algo.scalar_precision, msm_length)
    compiled_point_accumulation = (
        jax.jit(point_accumulation_static_configured)
        .lower(
            jax.ShapeDtypeStruct(point_psum_shape, dtype=jnp.uint16),
            jax.ShapeDtypeStruct(base_points_shape, dtype=jnp.uint16),
            jax.ShapeDtypeStruct(overall_select_shape, dtype=jnp.bool),
            jax.ShapeDtypeStruct(overall_non_zero_states_shape, dtype=jnp.bool),
        )
        .compile()
    )

    # Psum accumulation has two versions -- tree based reduction or temporal
    func = jax.named_call(
        functools.partial(
            msm.psum_accumulation,
            psum_addition_length=msm_algo.psum_addition_length,
            length=msm_length,
        ),
        name="psum_accumulation",
    )

    compiled_psum_accumulation = (
        jax.jit(func)
        .lower(
            jax.ShapeDtypeStruct(point_psum_shape, dtype=jnp.uint16),
        )
        .compile()
    )

    # online procedure
    # copybara: session = xprof_session.XprofSession()
    # copybara: session.start_session()
    point_psum = compiled_point_accumulation(
        msm_algo.point_psum,
        msm_algo.base_points,
        msm_algo.overall_select,
        msm_algo.overall_non_zero_states,
    )
    result = compiled_psum_accumulation(point_psum)

    # copybara: session_id = session.end_session_and_get_session_id()
    # copybara: print(f'session_id: http://xprof/?session_id={session_id}')

    result = utils.jax_point_pack_to_int_point(result)
    ec_sys = ec.ECCSWeierstrassXYZZ(config_BLS12_377)
    result_affine_point = ec_sys.generate_point(result).convert_to_affine()
    coordinates = (
        result_affine_point[0].get_value(),
        result_affine_point[1].get_value(),
    )
    self.assertEqual(coordinates[0], result_ref[0])
    self.assertEqual(coordinates[1], result_ref[1])

  @absltest.skip("This test is tested")  # pylint: disable=superfluous-parens
  @parameterized.named_parameters(
      (
          "test_4_degree",
          os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
              f"{script_dir}/jaxite_ec/test_case/t4/zprize_msm_curve_377_scalars_dim_4_seed_0.csv"
          ),
          os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
              f"{script_dir}/jaxite_ec/test_case/t4/zprize_msm_curve_377_bases_dim_4_seed_0.csv"
          ),
          os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
              f"{script_dir}/jaxite_ec/test_case/t4/zprize_msm_curve_377_res_dim_4_seed_0.csv"
          ),
      ),
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
    # copybara: session = xprof_session.XprofSession()
    # copybara: session.start_session()

    msm_algo.bucket_accumulation()
    msm_algo.bucket_reduction()
    result = msm_algo.window_merge()
    # copybara: session_id = session.end_session_and_get_session_id()
    # copybara: print(f'session_id: http://xprof/?session_id={session_id}')

    result = utils.jax_point_pack_to_int_point(result)
    ec_sys = ec.ECCSWeierstrassXYZZ(config_BLS12_377)
    result_affine_point = ec_sys.generate_point(result).convert_to_affine()
    coordinates = (
        result_affine_point[0].get_value(),
        result_affine_point[1].get_value(),
    )
    self.assertEqual(coordinates[0], result_ref[0])
    self.assertEqual(coordinates[1], result_ref[1])


if __name__ == "__main__":
  absltest.main()
