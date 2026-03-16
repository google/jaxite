from absl.testing import absltest
from absl.testing import parameterized
import jaxite.jaxite_word.finite_field as ff_context
import jax
import jaxite.jaxite_word.ntt_mm as ntt

jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import jaxite.jaxite_word.util as util
import os


NTT = [
    (
        "0",
        [134219681, 134219681, 134219681],
        None,
        3,
        4,
        4,
        [
            [
                105825732,
                68433452,
                36629220,
                126901109,
                89469849,
                106633716,
                15102657,
                108374459,
                68789927,
                23451922,
                93538050,
                20585372,
                30604976,
                37517995,
                65644325,
                102451383,
            ],
            [
                105825732,
                68433452,
                36629220,
                126901109,
                89469849,
                106633716,
                15102657,
                108374459,
                68789927,
                23451922,
                93538050,
                20585372,
                30604976,
                37517995,
                65644325,
                102451383,
            ],
            [
                105825732,
                68433452,
                36629220,
                126901109,
                89469849,
                106633716,
                15102657,
                108374459,
                68789927,
                23451922,
                93538050,
                20585372,
                30604976,
                37517995,
                65644325,
                102451383,
            ],
        ],
        [
            [
                26196696,
                45475009,
                10055359,
                23277424,
                69041040,
                71916973,
                73894069,
                3311254,
                44646798,
                49882443,
                28097016,
                70484730,
                10811958,
                11946041,
                61318182,
                19099272,
            ],
            [
                26196696,
                45475009,
                10055359,
                23277424,
                69041040,
                71916973,
                73894069,
                3311254,
                44646798,
                49882443,
                28097016,
                70484730,
                10811958,
                11946041,
                61318182,
                19099272,
            ],
            [
                26196696,
                45475009,
                10055359,
                23277424,
                69041040,
                71916973,
                73894069,
                3311254,
                44646798,
                49882443,
                28097016,
                70484730,
                10811958,
                11946041,
                61318182,
                19099272,
            ],
        ],
    ),
]


class NTTTest(parameterized.TestCase):

  def __init__(self, *args, **kwargs):
    super(NTTTest, self).__init__(*args, **kwargs)
    self.random_key = jax.random.key(0)

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_NTT_Barrett(self, q, psi, batch, r, c, coef_in, eval_in):
    b = 2  # batch size
    coef_in = jnp.concatenate(
        [
            jnp.array(coef_in, dtype=jnp.uint64)
            .transpose(1, 0)
            .reshape(1, r * c, -1)
            for _ in range(b)
        ],
        axis=0,
    ).astype(jnp.uint32)
    eval_in = jnp.concatenate(
        [
            jnp.array(eval_in, dtype=jnp.uint32)
            .transpose(1, 0)
            .reshape(1, r * c, -1)
            for _ in range(b)
        ],
        axis=0,
    ).astype(jnp.uint32)
    parameters = {
        "r": r,
        "c": c,
        "finite_field_context": ff_context.BarrettContext(moduli=q),
    }
    ntt_ctx = ntt.NTTCiphertextBarrettContext(moduli=q, parameters=parameters)
    # bit_reverse_indices = jnp.array(util.bit_reverse_indices(r*c), jnp.uint32)
    ntt_result_cf = ntt_ctx.ntt(coef_in.reshape(b, r, c, -1))
    # coef_in_br = jnp.take(ntt_result_cf.reshape(b, r*c, -1), bit_reverse_indices, axis=-2)
    np.testing.assert_array_equal(eval_in, ntt_result_cf.reshape(b, r * c, -1))
    intt_result = ntt_ctx.intt(ntt_result_cf)
    np.testing.assert_array_equal(
        coef_in, intt_result.reshape(b, r * c, -1).tolist()
    )

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_NTT_Montgomery(self, q, psi, batch, r, c, coef_in, eval_in):
    parameters = {
        "r": r,
        "c": c,
        "finite_field_context": ff_context.MontgomeryContext(moduli=q),
    }
    b = 2  # batch size
    coef_in = jnp.concatenate(
        [
            jnp.array(coef_in, dtype=jnp.uint64)
            .transpose(1, 0)
            .reshape(1, r * c, -1)
            for _ in range(b)
        ],
        axis=0,
    )
    eval_in = jnp.concatenate(
        [
            jnp.array(eval_in, dtype=jnp.uint32)
            .transpose(1, 0)
            .reshape(1, r * c, -1)
            for _ in range(b)
        ],
        axis=0,
    )

    ntt_ctx = ntt.NTTCiphertextMontgomeryContext(
        moduli=q, parameters=parameters
    )
    # bit_reverse_indices = jnp.array(util.bit_reverse_indices(r*c), jnp.uint32)
    test_in_cf = (
        ntt_ctx.to_computation_format(coef_in)
        .astype(jnp.uint32)
        .reshape(b, r, c, -1)
    )
    ntt_result_cf = ntt_ctx.ntt(test_in_cf)
    eval_recovered = ntt_ctx.to_original_format(
        ntt_result_cf.reshape(b, r * c, -1).astype(jnp.uint64)
    )
    # coef_in_br = jnp.take(eval_recovered, bit_reverse_indices, axis=-2)
    np.testing.assert_array_equal(eval_in, eval_recovered)
    intt_result = ntt_ctx.intt(ntt_result_cf)
    x_recovered = ntt_ctx.to_original_format(intt_result.reshape(b, r * c, -1))
    np.testing.assert_array_equal(
        coef_in, x_recovered.reshape(b, r * c, -1).tolist()
    )
    jit_ntt = jax.jit(ntt_ctx.ntt)
    jit_ntt(test_in_cf)
    profile_name = f"NTT_Montgomery_Performance"
    file_path = os.path.join(
        os.environ.get("TEST_TMPDIR", "/tmp"), profile_name
    )
    with jax.profiler.trace(file_path):
      jit_ntt(test_in_cf)

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_NTT_Shoup(self, q, psi, batch, r, c, coef_in, eval_in):
    b = 2
    parameters = {
        "r": r,
        "c": c,
        "finite_field_context": ff_context.ShoupContext(moduli=q),
    }
    coef_in = jnp.concatenate(
        [
            jnp.array(coef_in, dtype=jnp.uint64)
            .transpose(1, 0)
            .reshape(1, r * c, -1)
            for _ in range(b)
        ],
        axis=0,
    )
    eval_in = jnp.concatenate(
        [
            jnp.array(eval_in, dtype=jnp.uint32)
            .transpose(1, 0)
            .reshape(1, r * c, -1)
            for _ in range(b)
        ],
        axis=0,
    )

    ntt_ctx = ntt.NTTCiphertextShoupContext(moduli=q, parameters=parameters)
    ntt_result_cf = ntt_ctx.ntt(
        jnp.array(coef_in, dtype=jnp.uint32).reshape(b, r, c, -1)
    )
    eval_recovered = ntt_ctx.to_original_format(ntt_result_cf)
    np.testing.assert_array_equal(eval_in, eval_recovered.reshape(b, r * c, -1))
    intt_result = ntt_ctx.intt(ntt_result_cf)
    x_recovered = ntt_ctx.to_original_format(intt_result)
    np.testing.assert_array_equal(
        coef_in, x_recovered.reshape(b, r * c, -1).tolist()
    )

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*NTT)
  def test_NTT_BATLazy(self, q, psi, batch, r, c, coef_in, eval_in):
    b = 2  # batch size
    coef_in = jnp.concatenate(
        [
            jnp.array(coef_in, dtype=jnp.uint64)
            .transpose(1, 0)
            .reshape(1, r * c, -1)
            for _ in range(b)
        ],
        axis=0,
    ).astype(jnp.uint32)
    eval_in = jnp.concatenate(
        [
            jnp.array(eval_in, dtype=jnp.uint32)
            .transpose(1, 0)
            .reshape(1, r * c, -1)
            for _ in range(b)
        ],
        axis=0,
    ).astype(jnp.uint32)
    parameters = {
        "r": r,
        "c": c,
        "finite_field_context": ff_context.BarrettContext(moduli=q),
    }
    ntt_ctx = ntt.NTTCiphertextBATLazyContext(moduli=q, parameters=parameters)
    ntt_result_cf = ntt_ctx.ntt(coef_in.reshape(b, r, c, -1))
    np.testing.assert_array_equal(eval_in, ntt_result_cf.reshape(b, r * c, -1))
    intt_result = ntt_ctx.intt(ntt_result_cf)
    np.testing.assert_array_equal(
        coef_in, intt_result.reshape(b, r * c, -1).tolist()
    )


if __name__ == "__main__":
  absltest.main()
