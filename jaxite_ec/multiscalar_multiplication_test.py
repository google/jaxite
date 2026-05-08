import os

import jax
import jaxite.jaxite_ec.elliptic_curve_context as ec_context
import jaxite.jaxite_ec.finite_field_context as ff_context
import jaxite.jaxite_ec.multiscalar_multiplication_context as msm_context
import jaxite.jaxite_ec.utils as utils
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)

MODULUS_377_INT = 0x01AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001
NUM_MODULI = 32
RNS_MODULI = utils.find_moduli_specified_number(NUM_MODULI, 28)

MSM_DIM = 2**10
SEED = 0
MSM_TEST_DIR = os.path.join(os.path.dirname(__file__), "data", f"t{MSM_DIM}")
POINTS_PATH = os.path.join(
    MSM_TEST_DIR, f"zprize_msm_curve_377_bases_dim_{MSM_DIM}_seed_{SEED}.csv"
)
SCALARS_PATH = os.path.join(
    MSM_TEST_DIR, f"zprize_msm_curve_377_scalars_dim_{MSM_DIM}_seed_{SEED}.csv"
)
REF_RESULT_PATH = os.path.join(
    MSM_TEST_DIR, f"zprize_msm_curve_377_res_dim_{MSM_DIM}_seed_{SEED}.csv"
)


def _build_msm_parameters():
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
      "msm_length": MSM_DIM,
      "tile_length": MSM_DIM,
      "slice_bits": 6,
      "scalar_bits": 253,
      "order": ec_parameters["order"],
      "points_path": POINTS_PATH,
      "c_kernel_ret_space_ratio": 2,
  }


MSM_CONTEXT_CASES = [
    ("cpu_distribution", msm_context.CPUDistributionMSMContext, False),
    ("tpu_distribution", msm_context.TPUDistributionMSMContext, False),
    ("fusion", msm_context.FusionMSMContext, True),
]


class MsmBls12377Test(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.msm_parameters = _build_msm_parameters()
    self.scalars = utils.read_external_msm_file(SCALARS_PATH, "scalars")
    self.ref_result = utils.read_external_msm_file(
        REF_RESULT_PATH, "result_ref"
    )

  def _run_msm(self, context_class, use_fused):
    ctx = context_class(self.msm_parameters)
    ctx.set_use_compiled_kernels(True)
    compile_kwargs = {"use_fused": True} if use_fused else {}
    ctx.compile(parameters=compile_kwargs)

    tiled_slices = ctx.to_computational_format(self.scalars)
    result_m = ctx.multiscalar_multiply(tiled_slices)
    return ctx.to_original_format(result_m)

  @parameterized.named_parameters(*MSM_CONTEXT_CASES)
  def test_multiscalar_multiply_matches_reference(
      self, context_class, use_fused
  ):
    result = self._run_msm(context_class, use_fused)
    np.testing.assert_array_equal(
        np.asarray(result), np.asarray(self.ref_result)
    )


if __name__ == "__main__":
  absltest.main()
