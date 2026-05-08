"""Performance tests for batched NTT.

Profiles the six NTT designs produced by crossing
  * step count    (:class:`NTT3Step`, :class:`NTT5Step`, :class:`NTT7Step`)
  * backend       (DRNS via :class:`DRNSLazyExtensionContext`,
                   CROSS via :class:`CROSSLazyExtensionContext`)

The batch (leading) dimension is sharded across all available JAX
devices.  Performance is measured via ``jax.profiler`` through the
``KernelWrapper`` / ``Profiler`` helpers in ``profiler.py``, NOT
wall-clock time.  Sharded-correctness checks live in
``number_theory_transform_test.py``.
"""

import os

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import jaxite.jaxite_ec.number_theory_transform_context as ntt_context
from jaxite.jaxite_ec.profiler import KernelWrapper, Profiler, collect_logs
import jaxite.jaxite_ec.utils as utils

from absl.testing import absltest
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Trailing-dimension sizing — change these to sweep field-element width.
# NUM_MODULI drives the prime preset below (21 → 256-bit prime, 56 → 753-bit).
# ---------------------------------------------------------------------------
NUM_MODULI = 21  # DRNS: number of RNS moduli (trailing dim size)
PRECISION_BITS = 28  # DRNS: bit-width per modulus
RADIX_BITS = 32  # DRNS: Montgomery radix

# ---------------------------------------------------------------------------
# Prime & 2N-th-root presets, keyed by NUM_MODULI.  Each preset supplies a
# matching Q_PERF and PSI_PERF_BY_DEGREE (primitive 2*2^degree-th roots of
# unity mod Q_PERF).  The per-degree psi sidesteps utils.root_of_unity(),
# which trial-divides q-1 and is infeasible for the 753-bit prime.
# ---------------------------------------------------------------------------
_PERF_PRIME_PRESETS: dict[int, dict] = {
    # 256-bit prime — fits NUM_MODULI * PRECISION_BITS = 21 * 28 = 588 ≥ 256.
    21: {
        "q": 0x8000000000000000000000000000000000000000000000000000000070000001,
        "psi_by_degree": {
            14: (
                0x210D1D264152132AE3E5610B7E230BCD0058FE66FB35C5713527EA1FA40D1845
            ),
            16: (
                0x40D7C3F33672325E7B65C4A20B0BE07DD32F3EBB05C33DD8675D68EB3A8BDB6B
            ),
            18: (
                0x568FDDCD95737AC264EAADA546D74B051CA1B7FC5B8427DCE706674011E009E0
            ),
            20: (
                0x3AAE36A59E8E4F95E3118AA64270D0E122E0FC9585380815F737A67D613B5516
            ),
            22: (
                0x23E461BCC11091F4A355AD034B454991F9CFA113272B8DBDF38E895C68BE3702
            ),
        },
    },
    # 753-bit prime — fits NUM_MODULI * PRECISION_BITS = 56 * 28 = 1568 ≥ 753.
    56: {
        "q": (
            0x100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000023C00001
        ),
        "psi_by_degree": {
            14: (
                0x987B6025AB8620C5CB8376257CBC1FED3F6A9CD8003B6CB78C442378C5CB76DF824DDFAD28F53E6EFED050F1193BD6B114DDBAF944860CE6CEEC6A4ECCC92D690996CD24ED61167E18B61F33C7F45CA1231F30751C16AA586DA157BC12DA
            ),
            16: (
                0xC35116503813C2FEA4AA3458B0F4F8B28BBCC8BEE7004D34FB47B57FE855C6D764AF4F7CE54C9334C6C957CDE2D613405AD78F5BF210772600A9E8FBC797E35DDFCB55643E3F9ADE2C9B4AD8D08F00D7224EE0E4E176C98E61232E23114C
            ),
            18: (
                0x80DE16A04909C865C8C3954F81D93780D98B4826B7C8AEC95A3149697831B5CE3BE84CEA28866B38A0458484ACDFBFC0FB26215FFF7083F52721E3BBE094FA42CCAF9D9DF1730211F6A211BED8C8128041609A585C1234113AC33FD11FD2
            ),
            20: (
                0x4AE6B7005951673890AC5AEA5DFCA80F449C73138FBBA6EF4E26D43DDA812A3BE60C97E60450C3AA294B751F6FCAB9E3736C02191A19E87C08AB00F3A3D2B599CFC3D52A0886F3EDA8941813BCDBCD51930F838B04CFE32A1AEF52510324
            ),
            22: (
                0x26DFA808FE842C5A3C50EED23D433805D7CA3BC5FE9EDFA38C0A325159D75B8DBF7ED80054D92678DCCC4CC6B9A1D47976DFDA07D39C463F312EC45147860C46F8E4ACE696AB8B7F789EBE170FE615C18902C15F97ABC7300DBD9F1870A
            ),
        },
    },
}

if NUM_MODULI not in _PERF_PRIME_PRESETS:
  raise ValueError(
      f"No Q_PERF/PSI preset for NUM_MODULI={NUM_MODULI}; "
      f"available: {sorted(_PERF_PRIME_PRESETS)}"
  )
Q_PERF: int = _PERF_PRIME_PRESETS[NUM_MODULI]["q"]
PSI_PERF_BY_DEGREE: dict[int, int] = _PERF_PRIME_PRESETS[NUM_MODULI][
    "psi_by_degree"
]

# DRNS configs (BAT einsum + Montgomery/CRNS — runs fast, can push to deg=22).
PERF_CONFIGS_DRNS_3STEP = [
    {"degree": 14, "r1": 7, "c": 7},
    # {"degree": 16, "r1": 8, "c": 8},
    # {"degree": 18, "r1": 9, "c": 9},
    # {"degree": 20, "r1": 10, "c": 10},
    # {"degree": 22, "r1": 11, "c": 11},
]

PERF_CONFIGS_DRNS_5STEP = [
    {"degree": 14, "r1": 5, "r2": 5, "c": 4},
    # {"degree": 16, "r1": 5, "r2": 5, "c": 6},
    # {"degree": 18, "r1": 6, "r2": 6, "c": 6},
    # {"degree": 20, "r1": 6, "r2": 6, "c": 8},
    # {"degree": 22, "r1": 7, "r2": 7, "c": 8},
]

PERF_CONFIGS_DRNS_7STEP = [
    {"degree": 16, "r1": 4, "r2": 4, "c1": 4, "c2": 4},
    # {"degree": 20, "r1": 5, "r2": 5, "c1": 5, "c2": 5},
]

# CROSS configs — CROSS's fori_loop-based matmul is ~1000× slower than
# DRNS BAT per NTT, so keep sizes modest to finish in reasonable time.
PERF_CONFIGS_CROSS_3STEP = [
    {"degree": 14, "r1": 7, "c": 7},
    # {"degree": 16, "r1": 8, "c": 8},
    # {"degree": 18, "r1": 9, "c": 9},
    # {"degree": 20, "r1": 10, "c": 10},
    # {"degree": 22, "r1": 11, "c": 11},
]

PERF_CONFIGS_CROSS_5STEP = [
    {"degree": 14, "r1": 5, "r2": 5, "c": 4},
    # {"degree": 16, "r1": 5, "r2": 5, "c": 6},
    # {"degree": 18, "r1": 6, "r2": 6, "c": 6},
    # {"degree": 20, "r1": 6, "r2": 6, "c": 8},
    # {"degree": 22, "r1": 7, "r2": 7, "c": 8},
]

PERF_CONFIGS_CROSS_7STEP = [
    {"degree": 16, "r1": 4, "r2": 4, "c1": 4, "c2": 4},
    # {"degree": 20, "r1": 5, "r2": 5, "c1": 5, "c2": 5},
]


# ---------------------------------------------------------------------------
# Extension-context builders
# ---------------------------------------------------------------------------
def _create_drns_ff_ctx(prime):
  rns_moduli = utils.find_moduli_specified_number(NUM_MODULI, PRECISION_BITS)
  return ntt_context.DRNSLazyExtensionContext({
      "prime": prime,
      "rns_moduli": rns_moduli,
      "precision_bits": PRECISION_BITS,
      "radix_bits": RADIX_BITS,
  })


def _create_cross_ff_ctx(prime):
  params = {"prime": prime}
  return ntt_context.CROSSLazyExtensionContext(params)


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


# ---------------------------------------------------------------------------
# Kernel-wrapper helpers
# ---------------------------------------------------------------------------
def _ntt_kernel(input_array, parameters):
  return parameters["ctx"].ntt(input_array)


def _intt_kernel(input_array, parameters):
  return parameters["ctx"].intt(input_array)


def _shard_mapped_kernel(method_name):
  """Build a kernel entry that shard-maps ``ctx.<method_name>`` over batch.

  CROSS's NTT path uses ``jax.vmap`` internally over every leading axis
  of ``_modular_multiply``.  Under a plain jit with a batch-sharded
  input, the broadcast of replicated twiddles against the sharded input
  creates vmap axis-spec mismatches.  Running the kernel under
  ``shard_map`` gives every device a shard-local (unsharded) view, so
  all broadcast / reshape / vmap ops inside the kernel see regular-
  strided arrays while the heavy lifting still parallelizes across all
  devices at the outer (shard_map) level.
  """

  def kernel(input_array, parameters):
    ctx = parameters["ctx"]
    mesh = parameters["mesh"]
    batch_spec = parameters["batch_spec"]
    fn = getattr(ctx, method_name)
    mapped = shard_map(
        fn,
        mesh=mesh,
        in_specs=batch_spec,
        out_specs=batch_spec,
        check_rep=False,
    )
    return mapped(input_array)

  return kernel


# ---------------------------------------------------------------------------
# Performance profiling (jax.profiler traces via KernelWrapper + Profiler).
# One test method per design: 3 DRNS designs + 3 CROSS designs = 6 tests.
# ---------------------------------------------------------------------------
class NTTShardedPerformanceTest(parameterized.TestCase):
  """Profile all six NTT designs (step-count × backend) at sharded batch."""

  def setUp(self):
    super().setUp()
    outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    self.output_trace_root = (
        os.path.join(outputs_dir, "log")
        if outputs_dir
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
    )
    self.profiler_config = {
        "iterations": 1,
        "save_to_file": True,
        "enable_sharding": True,
    }

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    collect_logs(os.path.dirname(os.path.abspath(__file__)))

  # -------------------------------------------------------------------------
  # Profile driver: uniform entry point for DRNS and CROSS.  Selects the
  # right extension context, trailing-dim size, and per-device kernel
  # wrapping (plain jit for DRNS, shard_map-wrapped jit for CROSS).
  # -------------------------------------------------------------------------
  def _profile_design(
      self, variant_name, configs, ntt_cls, backend, make_params
  ):
    assert backend in ("drns", "cross")
    if backend == "drns":
      ff_ctx = _create_drns_ff_ctx(Q_PERF)
      trailing = len(ff_ctx.rns_moduli)
      trailing_key = "num_moduli"
      kernel_factory = lambda direction: (
          _ntt_kernel if direction == "ntt" else _intt_kernel
      )
    else:
      ff_ctx = _create_cross_ff_ctx(Q_PERF)
      trailing = ff_ctx.chunk_num_u32
      trailing_key = "chunk_num_u32"
      kernel_factory = lambda direction: _shard_mapped_kernel(direction)

    mesh, partition_spec = _create_sharding()
    num_devices = len(jax.devices())

    profiler = Profiler(
        output_trace_path=self.output_trace_root,
        profile_naming=f"sharded_{variant_name}_{ntt_cls.__name__}",
        configuration=self.profiler_config,
    )

    for cfg in configs:
      degree = cfg["degree"]
      spatial_params, spatial_shape = make_params(cfg)
      params = {
          "prime": Q_PERF,
          "finite_field_context": ff_ctx,
          **spatial_params,
      }
      psi = PSI_PERF_BY_DEGREE.get(degree)
      if psi is not None:
        params["psi"] = psi
      ntt_ctx = ntt_cls(params)

      ndim = 1 + len(spatial_shape) + 1
      batch_sharding = _batch_sharding(mesh, partition_spec, ndim)

      axis_names = mesh.axis_names
      batch_partition = axis_names if len(axis_names) > 1 else axis_names[0]
      batch_spec = partition_spec(batch_partition, *([None] * (ndim - 1)))

      setting_base = {
          "context": ntt_cls.__name__,
          "backend": backend,
          "degree": degree,
          "spatial_shape": str(spatial_shape),
          trailing_key: trailing,
          "num_devices": num_devices,
      }

      batch = num_devices
      input_shape = (batch,) + spatial_shape + (trailing,)
      for direction in ("ntt", "intt"):
        name = f"{variant_name}_{direction}_deg{degree}_batch{batch}"
        wrapper = KernelWrapper(
            kernel_name=name,
            function_to_wrap=kernel_factory(direction),
            input_structs=[(input_shape, jnp.uint32)],
            parameters={"ctx": ntt_ctx, "mesh": mesh, "batch_spec": batch_spec},
            mesh=mesh,
            input_shardings=(batch_sharding,),
            output_sharding=batch_sharding,
            enable_sharding=True,
        )
        profiler.add_profile(
            name=name,
            kernel_wrapper=wrapper,
            kernel_setting_cols={
                **setting_base,
                "direction": direction,
                "batch": batch,
            },
        )

    profiler.profile_all_profilers()
    profiler.post_process_all_profilers()

  # -------------------------------------------------------------------------
  # 3 DRNS-based designs
  # -------------------------------------------------------------------------
  def test_sharded_drns_3step(self):
    """DRNS 3-step NTT (``NTT3Step`` + ``DRNSLazyExtensionContext``)."""

    def make_params(cfg):
      r, c = 2 ** cfg["r1"], 2 ** cfg["c"]
      return {"r": r, "c": c}, (r, c)

    self._profile_design(
        "drns_3step",
        PERF_CONFIGS_DRNS_3STEP,
        ntt_context.NTT3Step,
        "drns",
        make_params,
    )

  def test_sharded_drns_5step(self):
    """DRNS 5-step NTT (``NTT5Step`` + ``DRNSLazyExtensionContext``)."""

    def make_params(cfg):
      rr, rc, c = 2 ** cfg["r1"], 2 ** cfg["r2"], 2 ** cfg["c"]
      return {"rr": rr, "rc": rc, "c": c}, (rr, rc, c)

    self._profile_design(
        "drns_5step",
        PERF_CONFIGS_DRNS_5STEP,
        ntt_context.NTT5Step,
        "drns",
        make_params,
    )

  def test_sharded_drns_7step(self):
    """DRNS 7-step NTT (``NTT7Step`` + ``DRNSLazyExtensionContext``)."""

    def make_params(cfg):
      rr, rc = 2 ** cfg["r1"], 2 ** cfg["r2"]
      cr, cc = 2 ** cfg["c1"], 2 ** cfg["c2"]
      return {"rr": rr, "rc": rc, "cr": cr, "cc": cc}, (rr, rc, cr, cc)

    self._profile_design(
        "drns_7step",
        PERF_CONFIGS_DRNS_7STEP,
        ntt_context.NTT7Step,
        "drns",
        make_params,
    )

  # -------------------------------------------------------------------------
  # 3 CROSS-backed designs
  # -------------------------------------------------------------------------
  def test_sharded_cross_3step(self):
    """CROSS 3-step NTT (``NTT3Step`` + ``CROSSLazyExtensionContext``)."""

    def make_params(cfg):
      r, c = 2 ** cfg["r1"], 2 ** cfg["c"]
      return {"r": r, "c": c}, (r, c)

    self._profile_design(
        "cross_3step",
        PERF_CONFIGS_CROSS_3STEP,
        ntt_context.NTT3Step,
        "cross",
        make_params,
    )

  def test_sharded_cross_5step(self):
    """CROSS 5-step NTT (``NTT5Step`` + ``CROSSLazyExtensionContext``)."""

    def make_params(cfg):
      rr, rc, c = 2 ** cfg["r1"], 2 ** cfg["r2"], 2 ** cfg["c"]
      return {"rr": rr, "rc": rc, "c": c}, (rr, rc, c)

    self._profile_design(
        "cross_5step",
        PERF_CONFIGS_CROSS_5STEP,
        ntt_context.NTT5Step,
        "cross",
        make_params,
    )

  def test_sharded_cross_7step(self):
    """CROSS 7-step NTT (``NTT7Step`` + ``CROSSLazyExtensionContext``)."""

    def make_params(cfg):
      rr, rc = 2 ** cfg["r1"], 2 ** cfg["r2"]
      cr, cc = 2 ** cfg["c1"], 2 ** cfg["c2"]
      return {"rr": rr, "rc": rc, "cr": cr, "cc": cc}, (rr, rc, cr, cc)

    self._profile_design(
        "cross_7step",
        PERF_CONFIGS_CROSS_7STEP,
        ntt_context.NTT7Step,
        "cross",
        make_params,
    )


if __name__ == "__main__":
  absltest.main()
