import os

import jax
from jaxite.jaxite_ec import utils
import jaxite.jaxite_ec.elliptic_curve_context as ec_context
import jaxite.jaxite_ec.finite_field_context as ff_context
import jaxite.jaxite_ec.multiscalar_multiplication_context as msm_context
from jaxite.jaxite_ec.profiler import (
    PrecompiledKernelWrapper,
    Profiler,
    collect_logs,
)
from jaxite.jaxite_ec.utils import hash_args

from absl.testing import absltest
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)

MODULUS_377_INT = 0x01AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001
NUM_MODULI = 32
RNS_MODULI = utils.find_moduli_specified_number(NUM_MODULI, 28)

MSM_LENGTH_LIST = [2**10]
SLICE_BITS = 10
SCALAR_BITS = 253

TEST_PARAMS_MSM_FUSION = [
    ("msm_fusion", MSM_LENGTH_LIST),
]


def _build_msm_parameters(msm_length: int):
  ff_parameters = {
      "prime": MODULUS_377_INT,
      "rns_moduli": RNS_MODULI,
      "precision_bits": 28,
      "radix_bits": 32,
  }
  ec_parameters = {
      "finite_field_context_class": ff_context.DRNSlazyContext,
      "finite_field_parameters": ff_parameters,
      "prime": (
          258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177
      ),
      "order": (
          8444461749428370424248824938781546531375899335154063827935233455917409239041
      ),
      "a": -1,
      "twist_d": (
          122268283598675559488486339158635529096981886914877139579534153582033676785385790730042363341236035746924960903179
      ),
      "alpha": -1,
      "b": 1,
      "s": (
          10189023633222963290707194929886294091415157242906428298294512798502806398782149227503530278436336312243746741931
      ),
      "MA": (
          228097355113300204138531148905234651262148041026195375645000724271212049151994375092458297304264351187709081232384
      ),
      "MB": (
          10189023633222963290707194929886294091415157242906428298294512798502806398782149227503530278436336312243746741931
      ),
      "t": (
          23560188534917577818843641916571445935985386319233886518929971599490231428764380923487987729215299304184915158756
      ),
      "generator": [
          71222569531709137229370268896323705690285216175189308202338047559628438110820800641278662592954630774340654489393,
          6177051365529633638563236407038680211609544222665285371549726196884440490905471891908272386851767077598415378235,
      ],
  }
  return {
      "elliptic_curve_context_class": (
          ec_context.ExtendedTwistedEdwardsNDContext
      ),
      "elliptic_curve_parameters": ec_parameters,
      "coordinate_dim": 4,
      "msm_length": msm_length,
      "tile_length": msm_length,
      "slice_bits": SLICE_BITS,
      "scalar_bits": SCALAR_BITS,
      "order": ec_parameters["order"],
      "c_kernel_ret_space_ratio": 2,
  }


def _build_fusion_ctx(msm_length: int):
  """Mirrors scratch_profile_msm_fused.py: sharded fused MSM with

  pre-compiled kernels. The context's multiscalar_multiply internally
  dispatches to these already-compiled, already-sharded kernels — so it
  must NOT be re-traced under jax.jit.
  """
  params = _build_msm_parameters(msm_length)
  ctx = msm_context.FusionMSMContext(params)
  ctx.set_use_compiled_kernels(True)
  ctx.set_use_sharding(True)
  ctx.compile(parameters={"use_fused": True})
  return ctx


class FusionMSMPerformanceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if outputs_dir:
      self.output_trace_root = os.path.join(outputs_dir, "log")
    else:
      self.output_trace_root = os.path.join(os.path.dirname(__file__), "log")
    self.profiler_config = {
        "iterations": 1,
        "save_to_file": True,
        "enable_sharding": True,
    }

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    root_dir = (
        outputs_dir
        if outputs_dir
        else os.path.dirname(os.path.abspath(__file__))
    )
    print(f"Collecting logs from: {root_dir}")
    collect_logs(root_dir)

  def _create_fusion_msm_wrapper(self, kernel_name, ctx, tiled_slices):
    """Profile ``_multiscalar_multiply_fused`` — the individual jit-able kernel.

    ``ctx.multiscalar_multiply`` is a Python dispatcher that slices inputs,
    places them on shardings, and hashes into ``compiled_kernels``; it isn't
    itself a kernel. We reproduce its setup once here, then hand the
    already-compiled executable + concrete inputs to the profiler.
    """
    # Replicate the slicing ctx.multiscalar_multiply does for the fused path.
    regular_tiled_slices = tiled_slices[:, : ctx.window_num - 1]
    last_tiled_slices = tiled_slices[:, ctx.window_num - 1]
    if ctx.use_sharding:
      regular_tiled_slices = regular_tiled_slices.to_device(
          ctx.fused_reg_slices_sharding
      )
      last_tiled_slices = last_tiled_slices.to_device(
          ctx.fused_last_slices_sharding
      )

    input_arrays = [
        regular_tiled_slices,
        last_tiled_slices,
        ctx.points,
        ctx.regular_window_buckets,
        ctx.last_window_buckets,
        ctx.window_sum,
    ]

    # Pull the already-compiled kernel out of ctx — same hash scheme as
    # ctx.multiscalar_multiply uses at dispatch time.
    kernel_hash = hash_args(
        *(v for a in input_arrays for v in (a.shape, a.dtype.__str__()))
    )
    compiled_fn = ctx.compiled_kernels[kernel_hash][
        "multiscalar_multiply_fused"
    ]
    # compiled_fn = None

    return PrecompiledKernelWrapper(
        kernel_name=kernel_name,
        callable_function=compiled_fn,
        input_arrays=input_arrays,
        enable_sharding=True,
        callable_function_name="_multiscalar_multiply_fused",  # it is important for collecting the correct trace events
    )

  @parameterized.named_parameters(*TEST_PARAMS_MSM_FUSION)
  def test_fusion_msm_performance(self, msm_length_list):
    profiler_instance = Profiler(
        output_trace_path=self.output_trace_root,
        profile_naming="msm_fusion",
        configuration=self.profiler_config,
    )

    for msm_length in msm_length_list:
      ctx = _build_fusion_ctx(msm_length)
      tiled_slices = ctx.to_computational_format(None)

      kernel_name = f"msm_fusion_n{msm_length}"
      kernel_wrapper = self._create_fusion_msm_wrapper(
          kernel_name=kernel_name,
          ctx=ctx,
          tiled_slices=tiled_slices,
      )
      profiler_instance.add_profile(
          name=kernel_name,
          kernel_wrapper=kernel_wrapper,
          kernel_setting_cols={
              "msm_length": msm_length,
              "slice_bits": SLICE_BITS,
              "scalar_bits": SCALAR_BITS,
              "num_moduli": NUM_MODULI,
              "use_sharding": True,
              "use_fused": True,
          },
      )

    profiler_instance.profile_all_profilers()
    profiler_instance.post_process_all_profilers()


if __name__ == "__main__":
  absltest.main()
