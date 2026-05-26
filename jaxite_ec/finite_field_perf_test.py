import os

import jax
import jax.numpy as jnp
from jaxite.jaxite_ec import finite_field_context as ff_context
from jaxite.jaxite_ec import profiler
from jaxite.jaxite_ec import utils

from absl.testing import absltest
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)

PRIME = 0x01AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001

BATCH_SIZE_LIST = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

NUM_MODULI_LIST = [32]  # 21 for 256-bit, 32 for 384-bit, 56 for 753-bit

TEST_PARAMS = [(f"moduli_{n}", n, BATCH_SIZE_LIST) for n in NUM_MODULI_LIST]


def _modular_multiply_kernel(a, b, parameters):
  return parameters["ctx"]._modular_multiply(a, b)


class FiniteFieldModularMultiplyPerformanceTest(parameterized.TestCase):

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
    }

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Collecting logs from: {root_dir}")
    profiler.collect_logs(root_dir)

  def _create_kernel_wrapper(self, kernel_name, ctx, batch, num_moduli):
    input_shape = (batch, num_moduli)
    return profiler.KernelWrapper(
        kernel_name=kernel_name,
        function_to_wrap=_modular_multiply_kernel,
        input_structs=[
            (input_shape, jnp.uint32),
            (input_shape, jnp.uint32),
        ],
        parameters={"ctx": ctx},
    )

  def _profile_modular_multiply(self, num_moduli, batch_size_list):
    rns_moduli = utils.find_moduli_specified_number(num_moduli, 28)

    ctx = ff_context.DRNSlazyContext({
        "prime": PRIME,
        "rns_moduli": rns_moduli,
        "precision_bits": 28,
        "radix_bits": 32,
    })

    profiler_instance = profiler.Profiler(
        output_trace_path=self.output_trace_root,
        profile_naming=f"ff_modular_multiply_moduli_{num_moduli}",
        configuration=self.profiler_config,
    )

    for batch in batch_size_list:
      kernel_name = f"ff_mod_mul_m{num_moduli}_b{batch}"
      kernel_wrapper = self._create_kernel_wrapper(
          kernel_name=kernel_name,
          ctx=ctx,
          batch=batch,
          num_moduli=num_moduli,
      )

      profiler_instance.add_profile(
          name=kernel_name,
          kernel_wrapper=kernel_wrapper,
          kernel_setting_cols={
              "num_moduli": num_moduli,
              "batch": batch,
          },
      )

    profiler_instance.profile_all_profilers()
    profiler_instance.post_process_all_profilers()

  @parameterized.named_parameters(*TEST_PARAMS)
  def test_DRNSlazy_modular_multiply_performance(
      self, num_moduli, batch_size_list
  ):
    self._profile_modular_multiply(
        num_moduli=num_moduli,
        batch_size_list=batch_size_list,
    )


if __name__ == "__main__":
  absltest.main()
