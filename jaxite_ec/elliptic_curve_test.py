import os

from jaxite.jaxite_ec import elliptic_curve_context as ec_context
from jaxite.jaxite_ec import finite_field_context as ff_context
from jaxite.jaxite_ec import utils
import numpy as np
import toml

from absl.testing import absltest
from absl.testing import parameterized

# NOTE: Ensure all tests point are on the curve
BLS12_377_TEST_CASES = [
    (
        "0",
        [
            [
                0x01AC3A384FC584EFD3E7F2C5A2927E7D454875C874A051027B9E7363D08942533EDE85DAE295D8CAB2751085206BCA76,
                0x011DB83AEC88460820F4868A73B12309EE2E910526E62DB4ACCB303ABF50F86C3985A072ED07A4B81FFB82D8DD247283,
            ],
            [
                0x0164DDDBF27670CE389E2992C0E7DAB7741F1B925EDBDC254D2BC0830BAF8E0B186F80F0DD4DE0F0EA6176E55934D45B,
                0x01908E9D77A0F8AD89AC41441F74248704E756BC59C38920617F51BFCDB738EE5B123876D489D09C9EB904A321A336EC,
            ],
        ],
        [
            [
                0x01546AF2ABB4E189E9BBC412FDBF2A8E5EC6E4A3B0AF132E21EE9CEC3EF5E226490FB98D662670FA3CFB3948B7E2A48C,
                0x002961A558A885DF227FDB09F8BDF57AF179CB9437FF8828F13E9DF01AE55502F409AAF5058B88F2F7CCC7BC0676A5D4,
            ],
            [
                0x00B0630E7F192D20443A93860275447074CE77DF559907FA1900F378D4674649BF25F85C893E2A1916B1DA57594F2E17,
                0x01ACC84F362CF60A265C011F0FE4360A15F51BECF7E2C3923FE07C66D5D113104B56E8486C64204A2A9ECD75BA0C41A7,
            ],
        ],
    ),
]


class BLS12_377_Test(parameterized.TestCase):

  def __init__(self, *args, **kwargs):
    super(BLS12_377_Test, self).__init__(*args, **kwargs)

  @parameterized.named_parameters(*BLS12_377_TEST_CASES)
  def test_ExtendedTwistedEdwards_point_add(self, point_batch_1, point_batch_2):
    ec_config = toml.load(
        os.path.join(os.path.dirname(__file__), "configurations.toml")
    )
    rns_moduli = utils.find_moduli_specified_number(32, 28)
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
    affine_cfg = ec_config["ec_parameters_bls12_377_affine"]
    ref_ec_parameters = {
        "finite_field_parameters": finite_field_parameters,
        "finite_field_context_class": ff_context.DRNSlazyContext,
        "prime": affine_cfg["prime"],
        "order": affine_cfg["order"],
        "a": affine_cfg["a"],
        "b": affine_cfg["b"],
        "generator": affine_cfg["generator"],
    }

    ec_ctx = ec_context.ExtendedTwistedEdwardsContext(ec_parameters)
    ref_ec_ctx = ec_context.CPUWeierstrassAffineContext(ref_ec_parameters)

    point_batch_1_m = ec_ctx.to_computational_format(point_batch_1)
    point_batch_2_m = ec_ctx.to_computational_format(point_batch_2)
    result_m = ec_ctx.point_add(point_batch_1_m, point_batch_2_m)
    result = ec_ctx.to_original_format(result_m)

    ref_result = ref_ec_ctx._point_add(point_batch_1, point_batch_2)

    np.testing.assert_array_equal(result, ref_result)


if __name__ == "__main__":
  absltest.main()
