import os

import jax
from jaxite.jaxite_ec import elliptic_curve_context as ec_context
from jaxite.jaxite_ec import finite_field_context as ff_context
from jaxite.jaxite_ec import utils
import toml

from absl.testing import absltest
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configurations.toml")

BATCH_SIZE_LIST = [128, 256, 512, 1024, 2048, 4096]

NUM_MODULI = 32

TEST_PARAMS_POINT_ADD = [
    ("point_add", BATCH_SIZE_LIST),
]

TEST_PARAMS_POINT_DOUBLE = [
    ("point_double", BATCH_SIZE_LIST),
]


def _build_ec_context():
  ec_config = toml.load(CONFIG_PATH)
  rns_moduli = utils.find_moduli_specified_number(NUM_MODULI, 28)
  finite_field_parameters = {
      "prime": ec_config["ec_parameters_bls12_377_affine"]["prime"],
      "rns_moduli": rns_moduli,
      "precision_bits": 28,
      "radix_bits": 32,
  }
  ete_cfg = ec_config["ec_parameters_bls12_377_extended_twisted_edwards"]
  ec_parameters = {
      "finite_field_context_class": ff_context.DRNSlazyContext,
      "finite_field_parameters": finite_field_parameters,
      "prime": ete_cfg["prime"],
      "order": ete_cfg["order"],
      "a": ete_cfg["a"],
      "twist_d": ete_cfg["d"],
      "alpha": ete_cfg["alpha"],
      "b": ete_cfg["b"],
      "s": ete_cfg["s"],
      "MA": ete_cfg["MA"],
      "MB": ete_cfg["MB"],
      "t": ete_cfg["t"],
      "generator": ete_cfg["generator"],
  }
  return ec_context.ExtendedTwistedEdwardsContext(ec_parameters)


def _point_add_kernel(point_a, point_b, parameters):
  return parameters["ctx"]._point_add(point_a, point_b)


def _point_double_kernel(point, parameters):
  return parameters["ctx"]._point_double(point)


class ECPointAddPerformanceTest(parameterized.TestCase):

  @parameterized.named_parameters(*TEST_PARAMS_POINT_ADD)
  def test_point_add_performance(self, batch_size_list):
    pass


if __name__ == "__main__":
  absltest.main()
