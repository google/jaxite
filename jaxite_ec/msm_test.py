import csv
import os
import sys

import jax
import jax.numpy as jnp
from jaxite.jaxite_ec import pippenger
from jaxite.jaxite_ec import pippenger_rns
from jaxite.jaxite_ec import util
from jaxite.jaxite_ec.algorithm import config_file
import jaxite.jaxite_ec.algorithm.elliptic_curve as ec

# copybara: from google3.pyglib import resources
from absl.testing import absltest
from absl.testing import parameterized


script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)

jax.config.update("jax_traceback_filtering", "off")

# Only needed when tryingt to understand the HLO dump.
os.environ["XLA_FLAGS"] = (
    "--xla_dump_to=sponge --xla_backend_optimization_level=4"
)

TEST_PARAMS = [
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
]


def twist_coordinates_list(ec_config, coordinates_list):
  twisted_ec_sys = ec.ECCSTwistedEdwardsExtended(ec_config)
  twisted_coordinates_list = []
  untwisted_coordinates_indeices = []
  for i, coordinates in enumerate(coordinates_list):
    twisted_coordinates = twisted_ec_sys.twist_int_coordinates(coordinates)
    twisted_coordinates_list.append(twisted_coordinates)
    if twisted_coordinates == [0, 1, 1, 0]:
      untwisted_coordinates_indeices.append(i)
  return twisted_coordinates_list, untwisted_coordinates_indeices


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

  @parameterized.named_parameters(*TEST_PARAMS)
  def test_pippenger_index_selection(self, scalar_path, base_path, result_path):
    """Normal version Pippenger."""
    scalars, points, result_ref = self.read_external_file(
        scalar_path, base_path, result_path
    )
    slice_length = 4
    msm_algo = pippenger.MSMPippenger(slice_length)
    msm_algo.initialize(scalars, points)

    window_num = msm_algo.window_num
    bucket_num_per_window = msm_algo.bucket_num_per_window
    msm_length = msm_algo.msm_length
    coordinate_num = msm_algo.coordinate_num
    chunk_num = util.U16_EXT_CHUNK_NUM

    bucket_accumulation_index_scan_jit = (
        jax.jit(
            pippenger.bucket_accumulation_index_scan_algorithm,
            static_argnames="msm_length",
        )
        .lower(
            jax.ShapeDtypeStruct(
                (coordinate_num, window_num, bucket_num_per_window, chunk_num),
                dtype=jnp.uint16,
            ),
            jax.ShapeDtypeStruct(
                (msm_length, coordinate_num, chunk_num), dtype=jnp.uint16
            ),
            jax.ShapeDtypeStruct((msm_length, window_num), dtype=jnp.uint32),
            jax.ShapeDtypeStruct(
                (msm_length, window_num, bucket_num_per_window), dtype=jnp.uint8
            ),
            msm_length,
        )
        .compile()
    )

    bucket_reduction_scan_jit = (
        jax.jit(
            pippenger.bucket_reduction_scan_algorithm,
            static_argnames="bucket_num_in_window",
        )
        .lower(
            jax.ShapeDtypeStruct(
                (coordinate_num, window_num, bucket_num_per_window, chunk_num),
                dtype=jnp.uint16,
            ),
            jax.ShapeDtypeStruct(
                (coordinate_num, window_num, chunk_num), dtype=jnp.uint16
            ),
            jax.ShapeDtypeStruct(
                (coordinate_num, window_num, chunk_num), dtype=jnp.uint16
            ),
            jax.ShapeDtypeStruct(
                (bucket_num_per_window, window_num), dtype=jnp.uint8
            ),
            jax.ShapeDtypeStruct(
                (bucket_num_per_window, window_num), dtype=jnp.uint8
            ),
            jax.ShapeDtypeStruct(
                (
                    bucket_num_per_window,
                    window_num,
                ),
                dtype=jnp.uint8,
            ),
            bucket_num_per_window,
        )
        .compile()
    )

    window_merge_scan_jit = (
        jax.jit(
            pippenger.window_merge_scan_algorithm,
            static_argnames="slice_length",
        )
        .lower(
            jax.ShapeDtypeStruct(
                (coordinate_num, window_num, chunk_num), dtype=jnp.uint16
            ),
            slice_length,
        )
        .compile()
    )

    msm_algo.bucket_accumulation(bucket_accumulation_index_scan_jit)
    msm_algo.bucket_reduction(bucket_reduction_scan_jit)
    result = msm_algo.window_merge(window_merge_scan_jit)
    result = util.jax_point_pack_to_int_point(result)
    ec_sys = ec.ECCSWeierstrassXYZZ(config_file.config_BLS12_377)
    result_affine_point = ec_sys.generate_point(result).convert_to_affine()
    coordinates = (
        result_affine_point[0].get_value(),
        result_affine_point[1].get_value(),
    )
    self.assertEqual(coordinates[0], result_ref[0])
    self.assertEqual(coordinates[1], result_ref[1])

    # performance measurement
    tasks = [
        (msm_algo.bucket_accumulation, (bucket_accumulation_index_scan_jit,)),
        (msm_algo.bucket_reduction, (bucket_reduction_scan_jit,)),
        (msm_algo.window_merge, (window_merge_scan_jit,)),
    ]
    profile_name = "normal_pippenger_index_selection"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  @parameterized.named_parameters(*TEST_PARAMS)
  def test_pippenger_index_selection_twisted_edwards(
      self, scalar_path, base_path, result_path
  ):
    scalars, points, result_ref = self.read_external_file(
        scalar_path, base_path, result_path
    )
    twisted_points, untwisted_coordinates_indeices = twist_coordinates_list(
        config_file.config_BLS12_377_t, points
    )
    assert not untwisted_coordinates_indeices
    slice_length = 4
    parallel_num = 4
    msm_algo = pippenger.MSMPippengerTwisted(slice_length, parallel_num)
    msm_algo.initialize(scalars, twisted_points)

    window_num = msm_algo.window_num
    bucket_num_per_window = msm_algo.bucket_num_per_window
    msm_length = msm_algo.msm_length
    coordinate_num = msm_algo.coordinate_num
    chunk_num = util.U16_EXT_CHUNK_NUM

    batch_window_num = window_num * parallel_num
    batch_mem_length = msm_length // parallel_num

    bucket_accumulation_index_scan_jit = (
        jax.jit(
            pippenger.bucket_accumulation_index_scan_parallel_algorithm_twisted,
            static_argnames="msm_length",
        )
        .lower(
            jax.ShapeDtypeStruct(
                (
                    coordinate_num,
                    batch_window_num,
                    bucket_num_per_window,
                    chunk_num,
                ),
                dtype=jnp.uint16,
            ),
            jax.ShapeDtypeStruct(
                (batch_mem_length, coordinate_num, parallel_num, chunk_num),
                dtype=jnp.uint16,
            ),
            jax.ShapeDtypeStruct(
                (batch_mem_length, batch_window_num), dtype=jnp.uint32
            ),
            batch_mem_length,
        )
        .compile()
    )

    bucket_reduction_scan_jit = (
        jax.jit(
            pippenger.bucket_reduction_scan_algorithm_twisted,
            static_argnames="bucket_num_in_window",
        )
        .lower(
            jax.ShapeDtypeStruct(
                (
                    coordinate_num,
                    batch_window_num,
                    bucket_num_per_window,
                    chunk_num,
                ),
                dtype=jnp.uint16,
            ),
            jax.ShapeDtypeStruct(
                (coordinate_num, batch_window_num, chunk_num), dtype=jnp.uint16
            ),
            jax.ShapeDtypeStruct(
                (coordinate_num, batch_window_num, chunk_num), dtype=jnp.uint16
            ),
            bucket_num_per_window,
        )
        .compile()
    )

    batch_window_summation_jit = (
        jax.jit(
            pippenger.batch_window_summation_algorithm_twisted,
            static_argnames="point_parallel",
        )
        .lower(
            jax.ShapeDtypeStruct(
                (coordinate_num, window_num, chunk_num), dtype=jnp.uint16
            ),
            jax.ShapeDtypeStruct(
                (coordinate_num, batch_window_num, chunk_num), dtype=jnp.uint16
            ),
            parallel_num,
        )
        .compile()
    )

    window_merge_scan_jit = (
        jax.jit(
            pippenger.window_merge_scan_algorithm_twisted,
            static_argnames="slice_length",
        )
        .lower(
            jax.ShapeDtypeStruct(
                (coordinate_num, window_num, chunk_num), dtype=jnp.uint16
            ),
            slice_length,
        )
        .compile()
    )

    # HERE
    msm_algo.bucket_accumulation(bucket_accumulation_index_scan_jit)
    msm_algo.bucket_reduction(bucket_reduction_scan_jit)
    msm_algo.batch_window_summation(batch_window_summation_jit)
    result = msm_algo.window_merge(window_merge_scan_jit)
    result = util.jax_point_pack_to_int_point(result)
    # TO HERE
    ec_sys = ec.ECCSTwistedEdwardsExtended(config_file.config_BLS12_377_t)
    result_affine_point = ec_sys.generate_point(
        result, twist=False
    ).convert_to_affine()
    coordinates = (
        result_affine_point[0].get_value(),
        result_affine_point[1].get_value(),
    )
    self.assertEqual(coordinates[0], result_ref[0])
    self.assertEqual(coordinates[1], result_ref[1])

    # performance measurement
    tasks = [
        (msm_algo.bucket_accumulation, (bucket_accumulation_index_scan_jit,)),
        (msm_algo.bucket_reduction, (bucket_reduction_scan_jit,)),
        (msm_algo.batch_window_summation, (batch_window_summation_jit,)),
        (msm_algo.window_merge, (window_merge_scan_jit,)),
    ]
    profile_name = "pippenger_index_selection_twisted_edwards"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  @parameterized.named_parameters(*TEST_PARAMS)
  def test_pippenger_signed_index_selection_twisted_edwards(
      self, scalar_path, base_path, result_path
  ):
    scalars, points, result_ref = self.read_external_file(
        scalar_path, base_path, result_path
    )
    twisted_points, untwisted_coordinates_indeices = twist_coordinates_list(
        config_file.config_BLS12_377_t, points
    )
    assert not untwisted_coordinates_indeices
    slice_length = 4
    parallel_num = 4
    msm_algo = pippenger.MSMPippengerTwistedSigned(slice_length, parallel_num)
    msm_algo.initialize(scalars, twisted_points)

    window_num = msm_algo.window_num
    bucket_num_per_window = msm_algo.bucket_num_per_window
    msm_length = msm_algo.msm_length
    coordinate_num = msm_algo.coordinate_num
    chunk_num = util.U16_EXT_CHUNK_NUM

    batch_window_num = window_num * parallel_num
    batch_mem_length = msm_length // parallel_num

    bucket_accumulation_index_scan_jit = (
        jax.jit(
            pippenger.bucket_accumulation_signed_index_scan_parallel_algorithm_twisted,
            static_argnames="msm_length",
        )
        .lower(
            jax.ShapeDtypeStruct(
                (
                    coordinate_num,
                    batch_window_num,
                    bucket_num_per_window,
                    chunk_num,
                ),
                dtype=jnp.uint16,
            ),
            jax.ShapeDtypeStruct(
                (batch_mem_length, coordinate_num, parallel_num, chunk_num),
                dtype=jnp.uint16,
            ),
            jax.ShapeDtypeStruct(
                (batch_mem_length, batch_window_num), dtype=jnp.uint32
            ),
            jax.ShapeDtypeStruct(
                (batch_mem_length, batch_window_num), dtype=jnp.uint8
            ),
            batch_mem_length,
        )
        .compile()
    )

    bucket_reduction_scan_jit = (
        jax.jit(
            pippenger.bucket_reduction_scan_algorithm_twisted,
            static_argnames="bucket_num_in_window",
        )
        .lower(
            jax.ShapeDtypeStruct(
                (
                    coordinate_num,
                    batch_window_num,
                    bucket_num_per_window,
                    chunk_num,
                ),
                dtype=jnp.uint16,
            ),
            jax.ShapeDtypeStruct(
                (coordinate_num, batch_window_num, chunk_num), dtype=jnp.uint16
            ),
            jax.ShapeDtypeStruct(
                (coordinate_num, batch_window_num, chunk_num), dtype=jnp.uint16
            ),
            bucket_num_per_window,
        )
        .compile()
    )

    batch_window_summation_jit = (
        jax.jit(
            pippenger.batch_window_summation_algorithm_twisted,
            static_argnames="point_parallel",
        )
        .lower(
            jax.ShapeDtypeStruct(
                (coordinate_num, window_num, chunk_num), dtype=jnp.uint16
            ),
            jax.ShapeDtypeStruct(
                (coordinate_num, batch_window_num, chunk_num), dtype=jnp.uint16
            ),
            parallel_num,
        )
        .compile()
    )

    window_merge_scan_jit = (
        jax.jit(
            pippenger.window_merge_scan_algorithm_twisted,
            static_argnames="slice_length",
        )
        .lower(
            jax.ShapeDtypeStruct(
                (coordinate_num, window_num, chunk_num), dtype=jnp.uint16
            ),
            slice_length,
        )
        .compile()
    )

    # HERE
    msm_algo.bucket_accumulation(bucket_accumulation_index_scan_jit)
    msm_algo.bucket_reduction(bucket_reduction_scan_jit)
    msm_algo.batch_window_summation(batch_window_summation_jit)
    result = msm_algo.window_merge(window_merge_scan_jit)
    result = util.jax_point_pack_to_int_point(result)
    # TO HERE
    ec_sys = ec.ECCSTwistedEdwardsExtended(config_file.config_BLS12_377_t)
    result_affine_point = ec_sys.generate_point(
        result, twist=False
    ).convert_to_affine()
    coordinates = (
        result_affine_point[0].get_value(),
        result_affine_point[1].get_value(),
    )
    self.assertEqual(coordinates[0], result_ref[0])
    self.assertEqual(coordinates[1], result_ref[1])

    # performance measurement
    tasks = [
        (msm_algo.bucket_accumulation, (bucket_accumulation_index_scan_jit,)),
        (msm_algo.bucket_reduction, (bucket_reduction_scan_jit,)),
        (msm_algo.batch_window_summation, (batch_window_summation_jit,)),
        (msm_algo.window_merge, (window_merge_scan_jit,)),
    ]
    profile_name = "pippenger_signed_index_selection_twisted_edwards"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  # @absltest.skip("test pass")
  @parameterized.named_parameters(*TEST_PARAMS)
  def test_pippenger_index_rns_selection(
      self, scalar_path, base_path, result_path
  ):
    """RNS version Pippenger - XYZZ."""
    scalars, points, result_ref = self.read_external_file(
        scalar_path, base_path, result_path
    )
    slice_length = 4
    msm_algo = pippenger_rns.MSMPippenger(slice_length)
    msm_algo.initialize(scalars, points)

    window_num = msm_algo.window_num
    bucket_num_per_window = msm_algo.bucket_num_per_window
    msm_length = msm_algo.msm_length
    coordinate_num = msm_algo.coordinate_num
    chunk_num = util.NUM_MODULI

    bucket_accumulation_index_scan_jit = (
        jax.jit(
            pippenger_rns.bucket_accumulation_index_scan_algorithm,
            static_argnames="msm_length",
        )
        .lower(
            jax.ShapeDtypeStruct(
                (coordinate_num, window_num, bucket_num_per_window, chunk_num),
                dtype=jnp.uint16,
            ),
            jax.ShapeDtypeStruct(
                (msm_length, coordinate_num, chunk_num), dtype=jnp.uint16
            ),
            jax.ShapeDtypeStruct((msm_length, window_num), dtype=jnp.uint32),
            jax.ShapeDtypeStruct(
                (msm_length, window_num, bucket_num_per_window), dtype=jnp.uint8
            ),
            msm_length,
        )
        .compile()
    )

    bucket_reduction_scan_jit = (
        jax.jit(
            pippenger_rns.bucket_reduction_scan_algorithm,
            static_argnames="bucket_num_in_window",
        )
        .lower(
            jax.ShapeDtypeStruct(
                (coordinate_num, window_num, bucket_num_per_window, chunk_num),
                dtype=jnp.uint16,
            ),
            jax.ShapeDtypeStruct(
                (coordinate_num, window_num, chunk_num), dtype=jnp.uint16
            ),
            jax.ShapeDtypeStruct(
                (coordinate_num, window_num, chunk_num), dtype=jnp.uint16
            ),
            jax.ShapeDtypeStruct(
                (bucket_num_per_window, window_num), dtype=jnp.uint8
            ),
            jax.ShapeDtypeStruct(
                (bucket_num_per_window, window_num), dtype=jnp.uint8
            ),
            jax.ShapeDtypeStruct(
                (bucket_num_per_window, window_num), dtype=jnp.uint8
            ),
            bucket_num_per_window,
        )
        .compile()
    )

    window_merge_scan_jit = (
        jax.jit(
            pippenger_rns.window_merge_scan_algorithm,
            static_argnames="slice_length",
        )
        .lower(
            jax.ShapeDtypeStruct(
                (coordinate_num, window_num, chunk_num), dtype=jnp.uint16
            ),
            slice_length,
        )
        .compile()
    )

    # HERE
    msm_algo.bucket_accumulation(bucket_accumulation_index_scan_jit)
    msm_algo.bucket_reduction(bucket_reduction_scan_jit)
    result = msm_algo.window_merge(window_merge_scan_jit)
    result = util.jax_rns_point_pack_to_int_point(result)
    # TO HERE
    ec_sys = ec.ECCSWeierstrassXYZZ(config_file.config_BLS12_377)
    result_affine_point = ec_sys.generate_point(result).convert_to_affine()
    coordinates = (
        result_affine_point[0].get_value(),
        result_affine_point[1].get_value(),
    )
    self.assertEqual(coordinates[0], result_ref[0])
    self.assertEqual(coordinates[1], result_ref[1])

    # performance measurement
    tasks = [
        (msm_algo.bucket_accumulation, (bucket_accumulation_index_scan_jit,)),
        (msm_algo.bucket_reduction, (bucket_reduction_scan_jit,)),
        (msm_algo.window_merge, (window_merge_scan_jit,)),
    ]
    profile_name = "pippenger_index_rns_selection"
    # copybara: util.profile_jax_functions(tasks, profile_name)

  # @absltest.skip("has some bug in result")
  @parameterized.named_parameters(*TEST_PARAMS)
  def test_pippenger_index_selection_rns_twisted_edwards(
      self, scalar_path, base_path, result_path
  ):
    scalars, points, result_ref = self.read_external_file(
        scalar_path, base_path, result_path
    )
    twisted_points, untwisted_coordinates_indeices = twist_coordinates_list(
        config_file.config_BLS12_377_t, points
    )
    assert not untwisted_coordinates_indeices
    slice_length = 4
    parallel_num = 4
    msm_algo = pippenger_rns.MSMPippengerTwisted(slice_length, parallel_num)
    msm_algo.initialize(scalars, twisted_points)

    window_num = msm_algo.window_num
    bucket_num_per_window = msm_algo.bucket_num_per_window
    msm_length = msm_algo.msm_length
    coordinate_num = msm_algo.coordinate_num
    chunk_num = util.NUM_MODULI

    batch_window_num = window_num * parallel_num
    batch_mem_length = msm_length // parallel_num

    bucket_accumulation_index_scan_jit = (
        jax.jit(
            pippenger_rns.bucket_accumulation_index_scan_parallel_algorithm_twisted,
            static_argnames="msm_length",
        )
        .lower(
            jax.ShapeDtypeStruct(
                (
                    coordinate_num,
                    batch_window_num,
                    bucket_num_per_window,
                    chunk_num,
                ),
                dtype=jnp.uint16,
            ),
            jax.ShapeDtypeStruct(
                (batch_mem_length, coordinate_num, parallel_num, chunk_num),
                dtype=jnp.uint16,
            ),
            jax.ShapeDtypeStruct(
                (batch_mem_length, batch_window_num), dtype=jnp.uint32
            ),
            batch_mem_length,
        )
        .compile()
    )

    bucket_reduction_scan_jit = (
        jax.jit(
            pippenger_rns.bucket_reduction_scan_algorithm_twisted,
            static_argnames="bucket_num_in_window",
        )
        .lower(
            jax.ShapeDtypeStruct(
                (
                    coordinate_num,
                    batch_window_num,
                    bucket_num_per_window,
                    chunk_num,
                ),
                dtype=jnp.uint16,
            ),
            jax.ShapeDtypeStruct(
                (coordinate_num, batch_window_num, chunk_num), dtype=jnp.uint16
            ),
            jax.ShapeDtypeStruct(
                (coordinate_num, batch_window_num, chunk_num), dtype=jnp.uint16
            ),
            bucket_num_per_window,
        )
        .compile()
    )

    batch_window_summation_jit = (
        jax.jit(
            pippenger_rns.batch_window_summation_algorithm_twisted,
            static_argnames="point_parallel",
        )
        .lower(
            jax.ShapeDtypeStruct(
                (coordinate_num, window_num, chunk_num), dtype=jnp.uint16
            ),
            jax.ShapeDtypeStruct(
                (coordinate_num, batch_window_num, chunk_num), dtype=jnp.uint16
            ),
            parallel_num,
        )
        .compile()
    )

    window_merge_scan_jit = (
        jax.jit(
            pippenger_rns.window_merge_scan_algorithm_twisted,
            static_argnames="slice_length",
        )
        .lower(
            jax.ShapeDtypeStruct(
                (coordinate_num, window_num, chunk_num), dtype=jnp.uint16
            ),
            slice_length,
        )
        .compile()
    )

    # HERE
    msm_algo.bucket_accumulation(bucket_accumulation_index_scan_jit)
    msm_algo.bucket_reduction(bucket_reduction_scan_jit)
    msm_algo.batch_window_summation(batch_window_summation_jit)
    result = msm_algo.window_merge(window_merge_scan_jit)
    result = util.jax_rns_point_pack_to_int_point(result)
    # TO HERE
    ec_sys = ec.ECCSTwistedEdwardsExtended(config_file.config_BLS12_377_t)
    result_affine_point = ec_sys.generate_point(
        result, twist=False
    ).convert_to_affine()
    coordinates = (
        result_affine_point[0].get_value() % util.MODULUS_377_INT,
        result_affine_point[1].get_value() % util.MODULUS_377_INT,
    )
    self.assertEqual(coordinates[0], result_ref[0])
    self.assertEqual(coordinates[1], result_ref[1])

    # performance measurement
    tasks = [
        (msm_algo.bucket_accumulation, (bucket_accumulation_index_scan_jit,)),
        (msm_algo.bucket_reduction, (bucket_reduction_scan_jit,)),
        (msm_algo.batch_window_summation, (batch_window_summation_jit,)),
        (msm_algo.window_merge, (window_merge_scan_jit,)),
    ]
    profile_name = "pippenger_index_selection_rns_twisted_edwards"
    # copybara: util.profile_jax_functions(tasks, profile_name)


if __name__ == "__main__":
  absltest.main()
