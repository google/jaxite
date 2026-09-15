import jax
from jaxite.jaxite_ec import finite_field_context as ff_context
from jaxite.jaxite_ec import utils
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized


jax.config.update("jax_enable_x64", True)

FF = [
    (
        "0",
        [
            0xBE4FBE5D03CE926E40E058BBDC3269C78CFAFED39796CD13EC8E9B0072DB2538DFFBCA05804574D9E2FF7EEB1DE219,
            0x008848DEFE740A67C8FC6225BF87FF5485951E2CAA9D41BB188282C8BD37CB5CD5481512FFCD394EEAB9B16EB21BE9EF,
        ],
        [
            0x82A0ED372BFAB8198D0667A1DC5E299C1F6C8FEB0ACD4D05A228325117BE63EAE5BABE6807F41C6C8016BDAC251CFE,
            0x01914A69C5102EFF1F674F5D30AFEEC4BD7FB348CA3E52D96D182AD44FB82305C2FE3D3634A9591AFD82DE55559C8EA6,
        ],
    ),
]


class FiniteFieldTest(parameterized.TestCase):

  def __init__(self, *args, **kwargs):
    super(FiniteFieldTest, self).__init__(*args, **kwargs)

  @parameterized.named_parameters(*FF)
  def test_DRNSlazy_modular_multiply(self, value_a, value_b):
    prime = 0x01AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001
    rns_moduli = utils.find_moduli_specified_number(32, 28)
    ref_value_c = [(a * b) % prime for a, b in zip(value_a, value_b)]

    ctx = ff_context.DRNSlazyContext({
        "prime": prime,
        "rns_moduli": rns_moduli,
        "precision_bits": 28,
        "radix_bits": 32,
    })

    value_a_m = ctx.to_computational_format(value_a)
    value_b_m = ctx.to_computational_format(value_b)
    value_c_m = ctx.modular_multiply(value_a_m, value_b_m)
    value_c = ctx.to_original_format(value_c_m)

    np.testing.assert_array_equal(value_c, ref_value_c)

  @parameterized.named_parameters(*FF)
  def test_lazy_modular_multiply(self, value_a, value_b):
    prime = 0x01AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001

    ref_value_c = [(a * b) % prime for a, b in zip(value_a, value_b)]

    ctx = ff_context.CROSSLazyContext({"prime": prime, "chunk_num_u8": 48})

    value_a_m = ctx.to_computational_format(value_a)
    value_b_m = ctx.to_computational_format(value_b)
    value_c_m = ctx.modular_multiply(value_a_m, value_b_m)
    value_c = ctx.to_original_format(value_c_m)

    np.testing.assert_array_equal(value_c, ref_value_c)


if __name__ == "__main__":
  absltest.main()
