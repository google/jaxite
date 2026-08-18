import jax
import jax.numpy as jnp
import jaxite.jaxite_ec.number_theory_transform_context as ntt_context
import jaxite.jaxite_ec.utils as utils
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

MODULI = 134219681

# ---------------------------------------------------------------------------
# Trailing-dimension sizing — change these to sweep field-element width.
# ---------------------------------------------------------------------------
NUM_MODULI = 21  # DRNS: number of RNS moduli (trailing dim size), 21 for 256-bit, 56 for 753-bit
PRECISION_BITS = 28  # DRNS: bit-width per modulus
RADIX_BITS = 32  # DRNS: Montgomery radix
CHUNK_NUM_U8 = 32  # CROSS: override chunk_num_u8 (None = auto from prime), 32 for 256-bit, 95 for 753-bit

TEST_VECTOR = {
    "coef_in": [
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
    "eval_in": [
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
}


def make_ntt_case(name, *layout_dims):
  return (
      name,
      MODULI,
      None,
      1,
      *layout_dims,
      TEST_VECTOR["coef_in"],
      TEST_VECTOR["eval_in"],
  )


NTT_3STEP = [make_ntt_case("0", 4, 4)]  # Layout: (r, c)
NTT_5STEP = [make_ntt_case("0", 2, 2, 4)]  # Layout: (rr, rc, c)
NTT_7STEP = [make_ntt_case("0", 2, 2, 2, 2)]  # Layout: (rr, rc, cr, cc)


# ---------------------------------------------------------------------------
# Sharded correctness configs: (case_name, spatial_params, ntt_cls, spatial_shape)
# ---------------------------------------------------------------------------
SHARDED_CORRECTNESS_CONFIGS = [
    ("3step", {"r": 4, "c": 4}, ntt_context.NTT3Step, (4, 4)),
    ("5step", {"rr": 2, "rc": 2, "c": 4}, ntt_context.NTT5Step, (2, 2, 4)),
    (
        "7step",
        {"rr": 2, "rc": 2, "cr": 2, "cc": 2},
        ntt_context.NTT7Step,
        (2, 2, 2, 2),
    ),
]


# ---------------------------------------------------------------------------
# Sharding helpers
# ---------------------------------------------------------------------------
def _create_sharding():
  """Create default batch sharding for the current device mesh."""
  available_devices = jax.devices()
  if not available_devices:
    raise RuntimeError("No devices available for sharding test.")
  if len(available_devices) == 8:
    mesh_shape = (2, 4)
  elif len(available_devices) == 4:
    mesh_shape = (2, 2)
  elif len(available_devices) == 2:
    mesh_shape = (2, 1)
  else:
    mesh_shape = (1, 1)

  mesh = jax.make_mesh(mesh_shape, ("x", "y"))
  return mesh, jax.sharding.PartitionSpec


def _batch_sharding(mesh, partition_spec, ndim):
  """NamedSharding that partitions only the leading (batch) axis."""
  axis_names = mesh.axis_names
  batch_partition = axis_names if len(axis_names) > 1 else axis_names[0]
  spec = (batch_partition,) + (None,) * (ndim - 1)
  return jax.sharding.NamedSharding(mesh, partition_spec(*spec))


def _tile_to_batch(single, shard_batch):
  """Replicate a single ``(1, *spatial, trailing)`` array across batch dim."""
  return jnp.tile(single, (shard_batch,) + (1,) * (single.ndim - 1))


def _create_drns_ff_ctx(prime):
  rns_moduli = utils.find_moduli_specified_number(NUM_MODULI, PRECISION_BITS)
  return ntt_context.DRNSLazyExtensionContext({
      "prime": prime,
      "rns_moduli": rns_moduli,
      "precision_bits": PRECISION_BITS,
      "radix_bits": RADIX_BITS,
  })


class NTTTest(parameterized.TestCase):

  # @absltest.skip("Skip DRNS NTT tests")
  @parameterized.named_parameters(*NTT_3STEP)
  def test_DRNS_NTT_3step(self, q, psi, batch, r, c, coef_in, eval_in):
    """Validate the 3-step NTT with DRNS lazy reduction."""
    rns_moduli = utils.find_moduli_specified_number(NUM_MODULI, PRECISION_BITS)
    ff_ctx = ntt_context.DRNSLazyExtensionContext({
        "prime": q,
        "rns_moduli": rns_moduli,
        "precision_bits": PRECISION_BITS,
        "radix_bits": RADIX_BITS,
    })
    ntt_ctx = ntt_context.NTT3Step(
        {"prime": q, "r": r, "c": c, "finite_field_context": ff_ctx}
    )

    # to_computational_format gives (N, num_moduli); ntt auto-reshapes
    coef_drns = ntt_ctx.to_computational_format(coef_in)

    # Forward NTT
    ntt_result = ntt_ctx.ntt(coef_drns)
    np.testing.assert_array_equal(
        eval_in, ntt_ctx.to_original_format(ntt_result)
    )

    # Inverse NTT
    intt_result = ntt_ctx.intt(ntt_result)
    np.testing.assert_array_equal(
        coef_in, ntt_ctx.to_original_format(intt_result)
    )

  # @absltest.skip("Skip DRNS NTT tests")
  @parameterized.named_parameters(*NTT_5STEP)
  def test_DRNS_NTT_5step(self, q, psi, batch, rr, rc, c, coef_in, eval_in):
    """Validate the 5-step NTT with DRNS lazy reduction."""
    rns_moduli = utils.find_moduli_specified_number(NUM_MODULI, PRECISION_BITS)
    ff_ctx = ntt_context.DRNSLazyExtensionContext({
        "prime": q,
        "rns_moduli": rns_moduli,
        "precision_bits": PRECISION_BITS,
        "radix_bits": RADIX_BITS,
    })
    ntt_ctx = ntt_context.NTT5Step(
        {"prime": q, "rr": rr, "rc": rc, "c": c, "finite_field_context": ff_ctx}
    )

    # to_computational_format gives (N, num_moduli); ntt auto-reshapes
    coef_drns = ntt_ctx.to_computational_format(coef_in)

    # Forward NTT
    ntt_result = ntt_ctx.ntt(coef_drns)
    np.testing.assert_array_equal(
        eval_in, ntt_ctx.to_original_format(ntt_result)
    )

    # Inverse NTT
    intt_result = ntt_ctx.intt(ntt_result)
    np.testing.assert_array_equal(
        coef_in, ntt_ctx.to_original_format(intt_result)
    )

  # @absltest.skip("Skip DRNS NTT tests")
  @parameterized.named_parameters(*NTT_7STEP)
  def test_DRNS_NTT_7step(
      self, q, psi, batch, rr, rc, cr, cc, coef_in, eval_in
  ):
    """Validate the 7-step NTT with DRNS lazy reduction."""
    rns_moduli = utils.find_moduli_specified_number(NUM_MODULI, PRECISION_BITS)
    ff_ctx = ntt_context.DRNSLazyExtensionContext({
        "prime": q,
        "rns_moduli": rns_moduli,
        "precision_bits": PRECISION_BITS,
        "radix_bits": RADIX_BITS,
    })
    ntt_ctx = ntt_context.NTT7Step({
        "prime": q,
        "rr": rr,
        "rc": rc,
        "cr": cr,
        "cc": cc,
        "finite_field_context": ff_ctx,
    })

    coef_drns = ntt_ctx.to_computational_format(coef_in)

    # Forward NTT
    ntt_result = ntt_ctx.ntt(coef_drns)
    np.testing.assert_array_equal(
        eval_in, ntt_ctx.to_original_format(ntt_result)
    )

    # Inverse NTT
    intt_result = ntt_ctx.intt(ntt_result)
    np.testing.assert_array_equal(
        coef_in, ntt_ctx.to_original_format(intt_result)
    )

  # @absltest.skip("Skip CROSS NTT tests")
  @parameterized.named_parameters(*NTT_3STEP)
  def test_CROSS_NTT_3step(self, q, psi, batch, r, c, coef_in, eval_in):
    """Validate the 3-step NTT with CROSS lazy matrix reduction."""
    cross_params = {"prime": q}
    if CHUNK_NUM_U8 is not None:
      cross_params["chunk_num_u8"] = CHUNK_NUM_U8
    ff_ctx = ntt_context.CROSSLazyExtensionContext(cross_params)
    ntt_ctx = ntt_context.NTT3Step(
        {"prime": q, "r": r, "c": c, "finite_field_context": ff_ctx}
    )

    coef_cross = ntt_ctx.to_computational_format(coef_in)

    ntt_result = ntt_ctx.ntt(coef_cross)
    np.testing.assert_array_equal(
        eval_in, ntt_ctx.to_original_format(ntt_result)
    )

    intt_result = ntt_ctx.intt(ntt_result)
    np.testing.assert_array_equal(
        coef_in, ntt_ctx.to_original_format(intt_result)
    )

  # @absltest.skip("Skip CROSS NTT tests")
  @parameterized.named_parameters(*NTT_5STEP)
  def test_CROSS_NTT_5step(self, q, psi, batch, rr, rc, c, coef_in, eval_in):
    """Validate the 5-step NTT with CROSS lazy matrix reduction."""
    cross_params = {"prime": q}
    if CHUNK_NUM_U8 is not None:
      cross_params["chunk_num_u8"] = CHUNK_NUM_U8
    ff_ctx = ntt_context.CROSSLazyExtensionContext(cross_params)
    ntt_ctx = ntt_context.NTT5Step(
        {"prime": q, "rr": rr, "rc": rc, "c": c, "finite_field_context": ff_ctx}
    )

    coef_cross = ntt_ctx.to_computational_format(coef_in)

    ntt_result = ntt_ctx.ntt(coef_cross)
    np.testing.assert_array_equal(
        eval_in, ntt_ctx.to_original_format(ntt_result)
    )

    intt_result = ntt_ctx.intt(ntt_result)
    np.testing.assert_array_equal(
        coef_in, ntt_ctx.to_original_format(intt_result)
    )

  # @absltest.skip("Skip CROSS NTT tests")
  @parameterized.named_parameters(*NTT_7STEP)
  def test_CROSS_NTT_7step(
      self, q, psi, batch, rr, rc, cr, cc, coef_in, eval_in
  ):
    """Validate the 7-step NTT with CROSS lazy matrix reduction."""
    cross_params = {"prime": q}
    if CHUNK_NUM_U8 is not None:
      cross_params["chunk_num_u8"] = CHUNK_NUM_U8
    ff_ctx = ntt_context.CROSSLazyExtensionContext(cross_params)
    ntt_ctx = ntt_context.NTT7Step({
        "prime": q,
        "rr": rr,
        "rc": rc,
        "cr": cr,
        "cc": cc,
        "finite_field_context": ff_ctx,
    })

    coef_cross = ntt_ctx.to_computational_format(coef_in)

    ntt_result = ntt_ctx.ntt(coef_cross)
    np.testing.assert_array_equal(
        eval_in, ntt_ctx.to_original_format(ntt_result)
    )

    intt_result = ntt_ctx.intt(ntt_result)
    np.testing.assert_array_equal(
        coef_in, ntt_ctx.to_original_format(intt_result)
    )

  # ---------------------------------------------------------------------
  # Unified NTT + NumpyCPUContext parity tests: NTT{3,5,7}Step with a
  # NumpyCPUContext backend must produce the same output as the
  # CPUCROSS{,5Step,7Step}Context legacy reference classes.
  # ---------------------------------------------------------------------
  # @absltest.skip("Skip NumpyCPU NTT tests")
  @parameterized.named_parameters(*NTT_3STEP)
  def test_NumpyCPU_NTT_3step(self, q, psi, batch, r, c, coef_in, eval_in):
    ff_ctx = ntt_context.NumpyCPUContext({"prime": q})
    ntt_ctx = ntt_context.NTT3Step(
        {"prime": q, "r": r, "c": c, "finite_field_context": ff_ctx}
    )
    coef_cpu = ntt_ctx.to_computational_format(coef_in)
    ntt_result = ntt_ctx.ntt(coef_cpu)
    np.testing.assert_array_equal(
        eval_in, ntt_ctx.to_original_format(ntt_result)
    )
    intt_result = ntt_ctx.intt(ntt_result)
    np.testing.assert_array_equal(
        coef_in, ntt_ctx.to_original_format(intt_result)
    )

  # @absltest.skip("Skip NumpyCPU NTT tests")
  @parameterized.named_parameters(*NTT_5STEP)
  def test_NumpyCPU_NTT_5step(self, q, psi, batch, rr, rc, c, coef_in, eval_in):
    ff_ctx = ntt_context.NumpyCPUContext({"prime": q})
    ntt_ctx = ntt_context.NTT5Step(
        {"prime": q, "rr": rr, "rc": rc, "c": c, "finite_field_context": ff_ctx}
    )
    coef_cpu = ntt_ctx.to_computational_format(coef_in)
    ntt_result = ntt_ctx.ntt(coef_cpu)
    np.testing.assert_array_equal(
        eval_in, ntt_ctx.to_original_format(ntt_result)
    )
    intt_result = ntt_ctx.intt(ntt_result)
    np.testing.assert_array_equal(
        coef_in, ntt_ctx.to_original_format(intt_result)
    )

  # @absltest.skip("Skip NumpyCPU NTT tests")
  @parameterized.named_parameters(*NTT_7STEP)
  def test_NumpyCPU_NTT_7step(
      self, q, psi, batch, rr, rc, cr, cc, coef_in, eval_in
  ):
    ff_ctx = ntt_context.NumpyCPUContext({"prime": q})
    ntt_ctx = ntt_context.NTT7Step({
        "prime": q,
        "rr": rr,
        "rc": rc,
        "cr": cr,
        "cc": cc,
        "finite_field_context": ff_ctx,
    })
    coef_cpu = ntt_ctx.to_computational_format(coef_in)
    ntt_result = ntt_ctx.ntt(coef_cpu)
    np.testing.assert_array_equal(
        eval_in, ntt_ctx.to_original_format(ntt_result)
    )
    intt_result = ntt_ctx.intt(ntt_result)
    np.testing.assert_array_equal(
        coef_in, ntt_ctx.to_original_format(intt_result)
    )


# ---------------------------------------------------------------------------
# Sharded correctness (DRNS only — CROSS correctness is covered by the
# per-backend tests above since CROSS's device-0 closure constants conflict
# with shard_map under the small-mesh jit harness).
# ---------------------------------------------------------------------------
class NTTShardedCorrectnessTest(parameterized.TestCase):
  """Verifies that batched, device-sharded NTT is element-wise correct."""

  # @absltest.skip("Skip DRNS NTT correctness tests")
  @parameterized.named_parameters(*SHARDED_CORRECTNESS_CONFIGS)
  def test_sharded_ntt_correctness(
      self, spatial_params, ctx_cls, spatial_shape
  ):
    coef_in = TEST_VECTOR["coef_in"]
    eval_in = TEST_VECTOR["eval_in"]
    ff_ctx = _create_drns_ff_ctx(MODULI)
    params = {"prime": MODULI, "finite_field_context": ff_ctx, **spatial_params}
    ntt_ctx = ctx_cls(params)

    coef_drns = ntt_ctx.to_computational_format(coef_in)
    coef_drns = ntt_ctx._ensure_ntt_shape(coef_drns)

    mesh, partition_spec = _create_sharding()
    shard_batch = len(jax.devices())
    batched = _tile_to_batch(coef_drns, shard_batch)
    sharding = _batch_sharding(mesh, partition_spec, batched.ndim)
    batched_sharded = jax.device_put(batched, sharding)

    jit_ntt = jax.jit(ntt_ctx.ntt, out_shardings=sharding)
    jit_intt = jax.jit(ntt_ctx.intt, out_shardings=sharding)

    ntt_result = jit_ntt(batched_sharded)
    ntt_host = np.asarray(ntt_result)
    for i in range(shard_batch):
      np.testing.assert_array_equal(
          eval_in,
          ntt_ctx.to_original_format(ntt_host[i]),
          err_msg=f"NTT mismatch at batch index {i}",
      )

    intt_result = jit_intt(ntt_result)
    intt_host = np.asarray(intt_result)
    for i in range(shard_batch):
      np.testing.assert_array_equal(
          coef_in,
          ntt_ctx.to_original_format(intt_host[i]),
          err_msg=f"INTT mismatch at batch index {i}",
      )


if __name__ == "__main__":
  absltest.main()
