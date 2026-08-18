from abc import ABC, abstractmethod
from typing import Any, Dict
import jax
import jax.numpy as jnp
from jaxite.jaxite_ec import utils
from jaxite.jaxite_ec.finite_field_context import (
    CROSSLazyContext,
    DRNSlazyContext,
)
import numpy as np

jax.config.update("jax_enable_x64", True)


########################
# Helper Functions
########################
def gen_twiddle_matrix(rows, cols, q, omega):
  """Precompute twiddle matrix T where T[r, c] = omega^(r*c) mod q.

  Stored as ``dtype=object`` so the cells hold arbitrary-precision Python
  ints; needed because ``q`` can exceed 64 bits (e.g. the 753-bit ZKP
  prime used in the perf test).
  """
  twiddle_matrix = np.empty((rows, cols), dtype=object)
  for r in range(rows):
    for c in range(cols):
      twiddle_matrix[r, c] = pow(int(omega), int(r * c), int(q))
  return twiddle_matrix


def gen_twiddle_matrix_inv(rows, cols, q, omega):
  """Precompute inverse twiddle matrix T_inv where T_inv[r, c] = omega^{-(r*c)} mod q."""
  twiddle_matrix_inv = np.empty((rows, cols), dtype=object)
  for r in range(rows):
    for c in range(cols):
      twiddle_matrix_inv[r, c] = pow(int(omega), int(-r * c), int(q))
  return twiddle_matrix_inv


def get_bit_reverse_perm(n):
  """Generates bit-reversal permutation indices of size n."""
  if n <= 0:
    return []
  bits = n.bit_length() - 1
  perm = [0] * n
  for i in range(n):
    r = 0
    temp = i
    for _ in range(bits):
      r = (r << 1) | (temp & 1)
      temp >>= 1
    perm[i] = r
  return perm


########################
# Abstract Base Class
########################
class NumberTheoryTransformBase(ABC):
  """Abstract base class for all NTT context implementations."""

  @abstractmethod
  def ntt(self, v):
    """Forward Number Theory Transform."""
    pass

  @abstractmethod
  def intt(self, v):
    """Inverse Number Theory Transform."""
    pass

  @abstractmethod
  def to_computational_format(self, a):
    """Convert from plain integers to the internal computational representation."""
    pass

  @abstractmethod
  def to_original_format(self, a):
    """Convert from internal representation back to a flat list of integers."""
    pass


########################
# BAT (Basis Aligned Transformation) helpers
########################
def basis_aligned_transformation(
    matrix_drns: np.ndarray, rns_moduli
) -> jnp.ndarray:
  """Convert a 2-D DRNS twiddle matrix to BAT format for 8-bit matmul.

  Each uint32 value is byte-shifted by 0/8/16/24 bits, reduced mod each
  RNS modulus, then bitcast to uint8.

  Args:
      matrix_drns: DRNS twiddle, shape (rows, cols, num_moduli) uint32.
      rns_moduli: sequence of RNS moduli.

  Returns:
      BAT matrix of shape (rows, 4, cols, 4, num_moduli) uint8.
  """
  rows, cols, M = np.array(matrix_drns).shape
  matrix_u64 = np.array(matrix_drns, dtype=np.uint64)
  moduli = np.array(rns_moduli, dtype=np.uint64)

  # (4, rows, cols, M) — byte-shifted and reduced per channel
  shifted = np.empty((4, rows, cols, M), dtype=np.uint32)
  for s in range(4):
    shifted[s] = ((matrix_u64 << (8 * s)) % moduli).astype(np.uint32)

  # Bitcast uint32 → uint8: (4, rows, cols, M) → (4, rows, cols, M*4) → (4, rows, cols, M, 4)
  shifted_u8 = shifted.view(np.uint8).reshape(4, rows, cols, M, 4)

  # Rearrange to (rows, 4_shift, cols, 4_bytes, M)
  shifted_u8 = shifted_u8.transpose(1, 0, 2, 4, 3)
  return jnp.array(shifted_u8)


def matmul_bat_einsum(
    v: jax.Array, bat_twiddle: jax.Array, subscripts: str
) -> jax.Array:
  """BAT-based 8-bit matrix multiplication.

  Bitcasts the input from uint32 to uint8, performs an 8-bit einsum with
  the pre-processed BAT twiddle, and reconstructs the uint64 result via
  byte-shifting.

  Args:
      v: Input array (uint32). The trailing dimension (num_moduli) is expanded
        to (num_moduli, 4) by bitcast.
      bat_twiddle: BAT twiddle (uint8), pre-computed offline.
      subscripts: Einsum subscript string including the byte dimensions.

  Returns:
      uint64 result array (trailing moduli dimension, byte dim summed out).
  """
  v_u8 = jax.lax.bitcast_convert_type(v, jnp.uint8)
  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
  products = jnp.einsum(
      subscripts, v_u8, bat_twiddle, preferred_element_type=jnp.uint32
  )
  return jnp.sum(products.astype(jnp.uint64) << shift_factors, axis=-1)


########################
# BAT subscript generator (right-convention only)
########################
_BAT_RESERVED_LETTERS = frozenset("mqpj")


def _bat_subscripts_right(v_ndim: int, contract_axis: int) -> str:
  """Einsum subscripts for BAT-based right-convention matmul.

  Operand ``v`` has shape ``(*leading, M)`` with ``v_ndim == len(leading) + 1``;
  after bitcast-to-u8 it grows a trailing size-4 byte axis.  The twiddle
  handle has shape ``(K, 4, J, 4, M)``: axis 0 is the contracted dim, axis
  2 is the produced dim.  The axis at ``contract_axis`` of ``v`` (size K)
  is replaced on the output by an axis of size J.
  """
  if contract_axis < 0:
    contract_axis += v_ndim
  if not (0 <= contract_axis < v_ndim - 1):
    raise ValueError(
        "contract_axis must be a leading axis of v "
        f"(got {contract_axis} for v.ndim={v_ndim})"
    )
  letters = []
  c = ord("a")
  while len(letters) < v_ndim - 1:
    ch = chr(c)
    if ch not in _BAT_RESERVED_LETTERS:
      letters.append(ch)
    c += 1
  k = letters[contract_axis]
  v_sub = "".join(letters) + "mq"
  out_letters = letters.copy()
  out_letters[contract_axis] = "j"
  t_sub = f"{k}qjpm"
  out_sub = "".join(out_letters) + "mp"
  return f"{v_sub},{t_sub}->{out_sub}"


########################
# Extension contexts — add NTT-specific modular_matmul / twiddle helpers
# on top of the finite-field backends.  The NTT layer only ever consumes
# the abstract API exposed here, not the backend internals.
########################
class DRNSLazyExtensionContext(DRNSlazyContext):
  """``DRNSlazyContext`` + NTT-specific modular matrix multiplication.

  Exposes:
    * :py:meth:`preprocess_matmul` — 2-D integer matrix → BAT
      uint8 handle of shape ``(K, 4, J, 4, M)``.
    * :py:meth:`preprocess_elementwise` — N-D integer tensor →
      DRNS computational-format uint32 tensor of shape ``(..., M)``.
    * :py:meth:`modular_matmul` — right-convention modular matmul
      (BAT einsum + Montgomery → CRNS → Montgomery).  Only supports the
      matmul-shape used by the NTT stack (contract axis 0 of the handle).
    * :py:meth:`modular_multiply_broadcast` — broadcasting element-wise
      modular multiply for twiddle steps.
  """

  def __init__(self, parameters: Dict[str, Any]):
    super().__init__(parameters)
    self.for_ntt = True

  def preprocess_matmul(self, mat_2d) -> jnp.ndarray:
    """Encode a 2-D ``(K, J)`` integer matrix as a BAT uint8 handle."""
    arr = np.asarray(mat_2d)
    if arr.ndim != 2:
      raise ValueError(f"matmul twiddle must be 2-D, got shape {arr.shape}")
    # Reduce mod each RNS modulus in Python-int space so this stays correct
    # for primes wider than 64 bits.  Output fits in uint32 (moduli < 2^28).
    if arr.dtype != object:
      arr = arr.astype(object)
    moduli_obj = np.array(self.rns_moduli, dtype=object)
    drns_obj = (
        (arr[..., np.newaxis] % moduli_obj) << self.radix_bits
    ) % moduli_obj
    return basis_aligned_transformation(
        drns_obj.astype(np.uint32), self.rns_moduli
    )

  def preprocess_elementwise(self, mat_nd) -> jnp.ndarray:
    """Encode an arbitrary-rank integer tensor in DRNS computational form."""
    arr = np.asarray(mat_nd)
    if arr.dtype != object:
      arr = arr.astype(object)
    moduli_obj = np.array(self.rns_moduli, dtype=object)
    drns_obj = (
        (arr[..., np.newaxis] % moduli_obj) << self.radix_bits
    ) % moduli_obj
    return jnp.array(drns_obj.astype(np.uint32))

  def modular_matmul(
      self, operand: jax.Array, handle: jax.Array, contract_axis: int
  ) -> jax.Array:
    """Modular matmul contracting ``operand[contract_axis]`` against

    ``handle[0]``.  ``handle`` must come from
    :py:meth:`preprocess_matmul`.
    """
    operand = jnp.asarray(operand)
    subs = _bat_subscripts_right(operand.ndim, contract_axis)
    z = matmul_bat_einsum(operand, handle, subs)
    z = self._jax_montgomery_reduce(z)
    z = self._jax_crns(z)
    z = self._jax_montgomery_reduce(z)
    return z.astype(jnp.uint32)

  def modular_multiply_broadcast(self, a: jax.Array, b: jax.Array) -> jax.Array:
    """Element-wise modular multiply with numpy-style broadcasting."""
    return self._modular_multiply(a, b)


class CROSSLazyExtensionContext(CROSSLazyContext):
  """``CROSSLazyContext`` + NTT-specific modular matrix multiplication.

  Same four-method API as :py:class:`DRNSLazyExtensionContext`.  The
  matmul path here is a ``fori_loop`` that accumulates per-``k``
  broadcast products via ``_modular_multiply`` + ``_modular_add``
  (there is no dense-MXU-friendly representation for multi-limb CROSS
  elements).
  """

  def __init__(self, parameters: Dict[str, Any]):
    super().__init__(parameters)
    self.for_ntt = True

  def _encode_chunks(self, mat) -> jnp.ndarray:
    """Encode an integer tensor as little-endian uint32 chunks."""
    arr = np.asarray(mat)
    flat = [int(x) % self.prime for x in arr.reshape(-1).tolist()]
    n = self.chunk_num_u32
    chunks = np.zeros((len(flat), n), dtype=np.uint32)
    for i, v in enumerate(flat):
      x = v
      for j in range(n):
        chunks[i, j] = x & 0xFFFFFFFF
        x >>= 32
    return jnp.asarray(chunks.reshape(arr.shape + (n,)))

  def preprocess_matmul(self, mat_2d) -> jnp.ndarray:
    arr = np.asarray(mat_2d)
    if arr.ndim != 2:
      raise ValueError(f"matmul twiddle must be 2-D, got shape {arr.shape}")
    return self._encode_chunks(arr)

  def preprocess_elementwise(self, mat_nd) -> jnp.ndarray:
    return self._encode_chunks(mat_nd)

  def modular_multiply_broadcast(self, a: jax.Array, b: jax.Array) -> jax.Array:
    """Element-wise modular multiply with broadcasting.

    Uses nested ``vmap`` so sharded callers (running under ``shard_map``)
    see consistent axis sharding between the two operands.
    """
    shape = jnp.broadcast_shapes(a.shape, b.shape)
    a_b = jnp.broadcast_to(a, shape)
    b_b = jnp.broadcast_to(b, shape)
    fn = self._modular_multiply
    for _ in range(len(shape) - 2):
      fn = jax.vmap(fn)
    return fn(a_b, b_b)

  def modular_matmul(
      self, operand: jax.Array, handle: jax.Array, contract_axis: int
  ) -> jax.Array:
    """Modular matmul contracting ``operand[contract_axis]`` against

    ``handle[0]``.  ``handle`` must come from
    :py:meth:`preprocess_matmul` and is ``(K, J, chunks)``.
    """
    operand = jnp.asarray(operand)
    handle = jnp.asarray(handle)
    if contract_axis < 0:
      contract_axis += operand.ndim
    if not (0 <= contract_axis < operand.ndim - 1):
      raise ValueError(
          "contract_axis must be a leading axis of operand "
          f"(got {contract_axis} for operand.ndim={operand.ndim})"
      )
    nc = self.chunk_num_u32
    v_moved = jnp.moveaxis(operand, contract_axis, 0)  # (K, *rest, chunks)
    K = v_moved.shape[0]
    J = handle.shape[1]
    rest = v_moved.shape[1:-1]
    out_shape = (J,) + rest + (nc,)
    t_reshape = (J,) + (1,) * len(rest) + (nc,)

    def body(k, acc):
      vk = v_moved[k]
      tk = handle[k, :, :].reshape(t_reshape)
      prod = self.modular_multiply_broadcast(vk[None], tk)
      return self._modular_add(acc, prod)

    init = jnp.zeros(out_shape, dtype=jnp.uint32)
    out = jax.lax.fori_loop(0, K, body, init)
    return jnp.moveaxis(out, 0, contract_axis)


########################
# Unified NTT stack — one class per step count, works with either
# extension context above.  The extension context is what lets the NTT
# layer stay backend-agnostic.
########################
class NumpyCPUContext:
  """Pure-numpy CPU reference context compatible with the unified NTT stack.

  Exposes the same ``preprocess_matmul`` / ``preprocess_elementwise``
  / ``modular_matmul`` / ``modular_multiply_broadcast`` /
  ``to_computational_format`` / ``to_original_format`` API as
  :class:`DRNSLazyExtensionContext` and :class:`CROSSLazyExtensionContext`,
  so that :class:`NTT3Step` / :class:`NTT5Step` / :class:`NTT7Step`
  reproduce the same outputs as the CPU reference classes
  (:class:`CPUCROSSContext`, :class:`CPUCROSS5StepContext`,
  :class:`CPUCROSS7StepContext`).

  Each field element is stored as a single ``uint64`` with a trailing
  size-1 "chunks" axis (matching the layout the NTT layer expects).
  All arithmetic is a plain ``numpy`` multiply / tensordot followed by
  ``% prime`` — correctness only, no JAX / no TPU.
  """

  def __init__(self, parameters: Dict[str, Any]):
    self.prime = int(parameters["prime"])
    self.for_ntt = True

  # ---------- format conversion ----------

  def to_computational_format(self, a) -> np.ndarray:
    """Wrap ``a`` as an ``(..., 1)`` uint64 ndarray."""
    arr = np.asarray(a, dtype=np.uint64)
    return arr[..., np.newaxis]

  def to_original_format(self, a):
    """Reduce, flatten, return a flat list of Python ints."""
    arr = np.asarray(a, dtype=np.uint64)
    return (arr.reshape(-1) % self.prime).tolist()

  # ---------- twiddle preprocessing ----------

  def preprocess_matmul(self, mat_2d) -> np.ndarray:
    arr = np.asarray(mat_2d)
    if arr.ndim != 2:
      raise ValueError(f"matmul twiddle must be 2-D, got shape {arr.shape}")
    if arr.dtype != object:
      arr = arr.astype(object)
    return (arr % self.prime).astype(np.uint64)[..., np.newaxis]

  def preprocess_elementwise(self, mat_nd) -> np.ndarray:
    arr = np.asarray(mat_nd)
    if arr.dtype != object:
      arr = arr.astype(object)
    return (arr % self.prime).astype(np.uint64)[..., np.newaxis]

  # ---------- modular arithmetic ----------

  def modular_matmul(
      self, operand: np.ndarray, handle: np.ndarray, contract_axis: int
  ) -> np.ndarray:
    """Right-convention modular matmul along ``contract_axis``.

    ``operand`` has trailing size-1 axis (the "chunks" axis).
    ``handle`` has shape ``(K, J, 1)`` — axis 0 is the contracted dim.
    """
    op = np.asarray(operand, dtype=np.uint64)[..., 0]
    h = np.asarray(handle, dtype=np.uint64)[..., 0]
    if contract_axis < 0:
      contract_axis += op.ndim + 1  # +1 because we squeezed trailing chunks
    # tensordot contracts op.axes[contract_axis] with h.axis[0]; the new
    # J axis from h lands at the end of the result, so move it back to
    # ``contract_axis``.
    out = np.tensordot(op, h, axes=([contract_axis], [0]))
    out = np.moveaxis(out, -1, contract_axis)
    out = out % self.prime
    return out[..., np.newaxis]

  def modular_multiply_broadcast(
      self, a: np.ndarray, b: np.ndarray
  ) -> np.ndarray:
    """Broadcasting element-wise modular multiply."""
    a_arr = np.asarray(a, dtype=np.uint64)
    b_arr = np.asarray(b, dtype=np.uint64)
    return (a_arr * b_arr) % self.prime


class NTTBase(NumberTheoryTransformBase):
  """Shared scaffolding for the unified NTT classes.

  All NTT classes below follow the same algorithm skeleton (precompute
  twiddles via :py:func:`gen_twiddle_matrix`, apply bit-reversal
  permutations, then alternate matmul / element-wise twiddle steps)
  and dispatch all arithmetic through the extension context passed in
  as ``finite_field_context``.

  The ``finite_field_context`` parameter accepts either:

  * An already-constructed extension context instance
    (``DRNSLazyExtensionContext``, ``CROSSLazyExtensionContext``, or
    ``NumpyCPUContext``).
  * A backend string (``"drns"``, ``"cross"``, or ``"cpu"``), in which
    case the extension context is auto-constructed from the remaining
    parameters:

    ============  =======================================================
    ``"drns"``    ``num_moduli`` (default 21), ``precision_bits`` (28),
                  ``radix_bits`` (32).
    ``"cross"``   ``chunk_num_u8`` (default derived from prime bit-length).
    ``"cpu"``     No extra parameters.
    ============  =======================================================
  """

  @staticmethod
  def _build_ff_ctx(parameters: dict):
    """Auto-construct an extension context from top-level parameters.

    If ``finite_field_context`` is already an instance, return it as-is.
    If it's a backend string, build the matching extension context using
    ``prime`` and the optional sizing parameters from ``parameters``.
    """
    ff_ctx = parameters["finite_field_context"]
    if isinstance(ff_ctx, str):
      prime = parameters["prime"]
      backend = ff_ctx.lower()
      if backend == "drns":
        num_moduli = parameters.get("num_moduli", 21)
        precision_bits = parameters.get("precision_bits", 28)
        radix_bits = parameters.get("radix_bits", 32)
        rns_moduli = utils.find_moduli_specified_number(
            num_moduli, precision_bits
        )
        return DRNSLazyExtensionContext({
            "prime": prime,
            "rns_moduli": rns_moduli,
            "precision_bits": precision_bits,
            "radix_bits": radix_bits,
        })
      elif backend == "cross":
        ctx_params = {"prime": prime}
        if "chunk_num_u8" in parameters:
          ctx_params["chunk_num_u8"] = parameters["chunk_num_u8"]
        return CROSSLazyExtensionContext(ctx_params)
      elif backend == "cpu":
        return NumpyCPUContext({"prime": prime})
      else:
        raise ValueError(
            f"Unknown backend {ff_ctx!r}; use 'drns', 'cross', or 'cpu'"
        )
    return ff_ctx

  def __init__(self, ff_ctx, spatial_shape: tuple):
    if not getattr(ff_ctx, "for_ntt", False):
      raise TypeError(
          "finite_field_context does not declare NTT support; "
          f"{type(ff_ctx).__name__} must set ``self.for_ntt = True``"
      )
    self.ff_ctx = ff_ctx
    self._spatial_shape = spatial_shape

  def to_computational_format(self, a):
    return self.ff_ctx.to_computational_format(a)

  def to_original_format(self, a):
    a = jnp.asarray(a)
    trailing = a.shape[-1]
    return self.ff_ctx.to_original_format(a.reshape(-1, trailing))

  def _ensure_ntt_shape(self, v: jnp.ndarray) -> jnp.ndarray:
    v = jnp.asarray(v)
    expected_ndim = 1 + len(self._spatial_shape) + 1
    if v.ndim < expected_ndim:
      v = v.reshape(-1, *self._spatial_shape, v.shape[-1])
    return v


class NTT3Step(NTTBase):
  """Unified 3-step NTT.  Input/output shape: ``(B, R, C, trailing)``."""

  def __init__(self, parameters: Dict[str, Any]):
    self.prime = parameters["prime"]
    self.r = parameters["r"]
    self.c = parameters["c"]
    ff_ctx = self._build_ff_ctx(parameters)
    super().__init__(ff_ctx, spatial_shape=(self.r, self.c))

    self.transform_length = self.r * self.c
    psi = parameters.get("psi")
    self.psi = (
        int(psi)
        if psi is not None
        else utils.root_of_unity(2 * self.transform_length, self.prime)
    )
    self.omega = (self.psi**2) % self.prime

    # --- twiddle matrices (plain integer form) ---
    omega_col = pow(self.omega, self.c, self.prime)
    omega_row = pow(self.omega, self.r, self.prime)
    ntt_tf1 = gen_twiddle_matrix(self.r, self.r, self.prime, omega_col)
    ntt_tf2 = gen_twiddle_matrix(self.r, self.c, self.prime, self.omega)
    ntt_tf3 = gen_twiddle_matrix(self.c, self.c, self.prime, omega_row)

    inv_omega_col = pow(omega_col, -1, self.prime)
    inv_omega_row = pow(omega_row, -1, self.prime)
    intt_tf1 = gen_twiddle_matrix(self.c, self.c, self.prime, inv_omega_row)
    intt_tf2 = gen_twiddle_matrix_inv(self.r, self.c, self.prime, self.omega)
    col_inv = pow(self.c, -1, self.prime)
    row_inv = pow(self.r, -1, self.prime)
    intt_tf2 = (intt_tf2 * col_inv) % self.prime
    intt_tf3 = gen_twiddle_matrix(self.r, self.r, self.prime, inv_omega_col)
    intt_tf3 = (intt_tf3 * row_inv) % self.prime

    # --- bit-reversal permutations ---
    perm_r = get_bit_reverse_perm(self.r)
    perm_c = get_bit_reverse_perm(self.c)
    ntt_tf1 = ntt_tf1[perm_r, :]
    ntt_tf2 = ntt_tf2[perm_r, :]
    ntt_tf3 = ntt_tf3[:, perm_c]
    intt_tf1 = intt_tf1[perm_c, :]
    intt_tf2 = intt_tf2[perm_r, :]
    intt_tf3 = intt_tf3[:, perm_r]

    # Right-convention handles: the contracted dim is at axis 0 of the
    # handle.  Step 1 of NTT is logically "T1 @ v along R", which becomes
    # right-matmul of v against T1.T;  step 3 is already "v @ T3".
    self.ntt_t1 = ff_ctx.preprocess_matmul(ntt_tf1.T)
    self.ntt_t2 = ff_ctx.preprocess_elementwise(ntt_tf2)
    self.ntt_t3 = ff_ctx.preprocess_matmul(ntt_tf3)
    self.intt_t1 = ff_ctx.preprocess_matmul(intt_tf1)
    self.intt_t2 = ff_ctx.preprocess_elementwise(intt_tf2)
    self.intt_t3 = ff_ctx.preprocess_matmul(intt_tf3.T)

  def ntt(self, v: jnp.ndarray):
    v = self._ensure_ntt_shape(v)  # (B, R, C, trailing)
    v = self.ff_ctx.modular_matmul(v, self.ntt_t1, contract_axis=1)
    v = self.ff_ctx.modular_multiply_broadcast(v, self.ntt_t2)
    v = self.ff_ctx.modular_matmul(v, self.ntt_t3, contract_axis=2)
    return v

  def intt(self, v: jnp.ndarray):
    v = self._ensure_ntt_shape(v)
    v = self.ff_ctx.modular_matmul(v, self.intt_t1, contract_axis=2)
    v = self.ff_ctx.modular_multiply_broadcast(v, self.intt_t2)
    v = self.ff_ctx.modular_matmul(v, self.intt_t3, contract_axis=1)
    return v


class NTT5Step(NTTBase):
  """Unified 5-step NTT.  Shape: ``(B, RR, RC, C, trailing)``."""

  def __init__(self, parameters: Dict[str, Any]):
    self.prime = parameters["prime"]
    self.rr = parameters["rr"]
    self.rc = parameters["rc"]
    self.c = parameters["c"]
    ff_ctx = self._build_ff_ctx(parameters)
    super().__init__(ff_ctx, spatial_shape=(self.rr, self.rc, self.c))

    self.transform_length = self.rr * self.rc * self.c
    R = self.rr * self.rc
    psi = parameters.get("psi")
    self.psi = (
        int(psi)
        if psi is not None
        else utils.root_of_unity(2 * self.transform_length, self.prime)
    )
    self.omega = (self.psi**2) % self.prime

    omega_R = pow(self.omega, self.c, self.prime)
    omega_RR = pow(omega_R, self.rc, self.prime)
    omega_RC = pow(omega_R, self.rr, self.prime)
    omega_C = pow(self.omega, R, self.prime)

    ntt_T1 = gen_twiddle_matrix(self.rr, self.rr, self.prime, omega_RR)
    ntt_T2 = gen_twiddle_matrix(self.rr, self.rc, self.prime, omega_R)
    ntt_T3 = gen_twiddle_matrix(self.rc, self.rc, self.prime, omega_RC)
    ntt_T4 = gen_twiddle_matrix(R, self.c, self.prime, self.omega).reshape(
        self.rr, self.rc, self.c
    )
    ntt_T5 = gen_twiddle_matrix(self.c, self.c, self.prime, omega_C)

    inv_omega_RR = pow(omega_RR, -1, self.prime)
    inv_omega_RC = pow(omega_RC, -1, self.prime)
    inv_omega_C = pow(omega_C, -1, self.prime)
    rr_inv = pow(self.rr, -1, self.prime)
    rc_inv = pow(self.rc, -1, self.prime)
    c_inv = pow(self.c, -1, self.prime)

    intt_T5 = gen_twiddle_matrix(self.c, self.c, self.prime, inv_omega_C)
    intt_T4 = gen_twiddle_matrix_inv(R, self.c, self.prime, self.omega).reshape(
        self.rr, self.rc, self.c
    )
    intt_T4 = (intt_T4 * c_inv) % self.prime
    intt_t3 = gen_twiddle_matrix(self.rc, self.rc, self.prime, inv_omega_RC)
    intt_T2 = gen_twiddle_matrix_inv(self.rr, self.rc, self.prime, omega_R)
    intt_T2 = (intt_T2 * rc_inv) % self.prime
    intt_T1 = gen_twiddle_matrix(self.rr, self.rr, self.prime, inv_omega_RR)
    intt_T1 = (intt_T1 * rr_inv) % self.prime

    perm_rr = get_bit_reverse_perm(self.rr)
    perm_rc = get_bit_reverse_perm(self.rc)
    perm_c = get_bit_reverse_perm(self.c)

    ntt_T1 = ntt_T1[perm_rr, :]
    ntt_T2 = ntt_T2[perm_rr, :]
    ntt_T3 = ntt_T3[:, perm_rc]
    perm_R = get_bit_reverse_perm(R)
    ntt_T4 = ntt_T4.reshape(R, self.c)[perm_R, :].reshape(
        self.rr, self.rc, self.c
    )
    ntt_T5 = ntt_T5[:, perm_c]

    intt_T5 = intt_T5[perm_c, :]
    intt_T4 = intt_T4.reshape(R, self.c)[perm_R, :].reshape(
        self.rr, self.rc, self.c
    )
    intt_t3 = intt_t3[perm_rc, :]
    intt_T2 = intt_T2[perm_rr, :]
    intt_T1 = intt_T1[:, perm_rr]

    # Matmul twiddles: transpose when the original step was left-matmul.
    self.ntt_T1 = ff_ctx.preprocess_matmul(
        ntt_T1.T
    )  # left-mat → right via transpose
    self.ntt_T2 = ff_ctx.preprocess_elementwise(ntt_T2)  # (RR, RC, trailing)
    self.ntt_T3 = ff_ctx.preprocess_matmul(ntt_T3)  # right-mat
    self.ntt_T4 = ff_ctx.preprocess_elementwise(ntt_T4)  # (RR, RC, C, trailing)
    self.ntt_T5 = ff_ctx.preprocess_matmul(ntt_T5)  # right-mat

    self.intt_T5 = ff_ctx.preprocess_matmul(intt_T5)
    self.intt_T4 = ff_ctx.preprocess_elementwise(intt_T4)
    self.intt_T3 = ff_ctx.preprocess_matmul(intt_t3)
    self.intt_T2 = ff_ctx.preprocess_elementwise(intt_T2)
    self.intt_T1 = ff_ctx.preprocess_matmul(
        intt_T1.T
    )  # undo the step-1 left-mat

  def ntt(self, v: jnp.ndarray):
    v = self._ensure_ntt_shape(v)  # (B, RR, RC, C, trailing)
    v = self.ff_ctx.modular_matmul(v, self.ntt_T1, contract_axis=1)
    v = self.ff_ctx.modular_multiply_broadcast(v, self.ntt_T2[:, :, None, :])
    v = self.ff_ctx.modular_matmul(v, self.ntt_T3, contract_axis=2)
    v = self.ff_ctx.modular_multiply_broadcast(v, self.ntt_T4[None])
    v = self.ff_ctx.modular_matmul(v, self.ntt_T5, contract_axis=3)
    return v

  def intt(self, v: jnp.ndarray):
    v = self._ensure_ntt_shape(v)
    v = self.ff_ctx.modular_matmul(v, self.intt_T5, contract_axis=3)
    v = self.ff_ctx.modular_multiply_broadcast(v, self.intt_T4[None])
    v = self.ff_ctx.modular_matmul(v, self.intt_T3, contract_axis=2)
    v = self.ff_ctx.modular_multiply_broadcast(v, self.intt_T2[:, :, None, :])
    v = self.ff_ctx.modular_matmul(v, self.intt_T1, contract_axis=1)
    return v


class NTT7Step(NTTBase):
  """Unified 7-step NTT.  Shape: ``(B, RR, RC, CR, CC, trailing)``."""

  def __init__(self, parameters: Dict[str, Any]):
    self.prime = parameters["prime"]
    self.rr = parameters["rr"]
    self.rc = parameters["rc"]
    self.cr = parameters["cr"]
    self.cc = parameters["cc"]
    ff_ctx = self._build_ff_ctx(parameters)
    super().__init__(ff_ctx, spatial_shape=(self.rr, self.rc, self.cr, self.cc))

    r_total = self.rr * self.rc
    c_total = self.cr * self.cc
    self.transform_length = r_total * c_total
    psi = parameters.get("psi")
    self.psi = (
        int(psi)
        if psi is not None
        else utils.root_of_unity(2 * self.transform_length, self.prime)
    )
    self.omega = (self.psi**2) % self.prime

    omega_r = pow(self.omega, c_total, self.prime)
    omega_rr = pow(omega_r, self.rc, self.prime)
    omega_rc = pow(omega_r, self.rr, self.prime)
    omega_c = pow(self.omega, r_total, self.prime)
    omega_cr = pow(omega_c, self.cc, self.prime)
    omega_cc = pow(omega_c, self.cr, self.prime)

    ntt_T1 = gen_twiddle_matrix(self.rr, self.rr, self.prime, omega_rr)
    ntt_T2 = gen_twiddle_matrix(self.rr, self.rc, self.prime, omega_r)
    ntt_T3 = gen_twiddle_matrix(self.rc, self.rc, self.prime, omega_rc)
    ntt_T4 = gen_twiddle_matrix(
        r_total, c_total, self.prime, self.omega
    ).reshape(self.rr, self.rc, self.cr, self.cc)
    ntt_T5 = gen_twiddle_matrix(self.cr, self.cr, self.prime, omega_cr)
    ntt_T6 = gen_twiddle_matrix(self.cr, self.cc, self.prime, omega_c)
    ntt_T7 = gen_twiddle_matrix(self.cc, self.cc, self.prime, omega_cc)

    inv_omega_rr = pow(omega_rr, -1, self.prime)
    inv_omega_rc = pow(omega_rc, -1, self.prime)
    inv_omega_cr = pow(omega_cr, -1, self.prime)
    inv_omega_cc = pow(omega_cc, -1, self.prime)
    rr_inv = pow(self.rr, -1, self.prime)
    rc_inv = pow(self.rc, -1, self.prime)
    cr_inv = pow(self.cr, -1, self.prime)
    cc_inv = pow(self.cc, -1, self.prime)

    intt_T7 = gen_twiddle_matrix(self.cc, self.cc, self.prime, inv_omega_cc)
    intt_T6 = gen_twiddle_matrix_inv(self.cr, self.cc, self.prime, omega_c)
    intt_T6 = (intt_T6 * cc_inv) % self.prime
    intt_T5 = gen_twiddle_matrix(self.cr, self.cr, self.prime, inv_omega_cr)
    intt_T5 = (intt_T5 * cr_inv) % self.prime
    intt_T4 = gen_twiddle_matrix_inv(
        r_total, c_total, self.prime, self.omega
    ).reshape(self.rr, self.rc, self.cr, self.cc)
    intt_T3 = gen_twiddle_matrix(self.rc, self.rc, self.prime, inv_omega_rc)
    intt_T2 = gen_twiddle_matrix_inv(self.rr, self.rc, self.prime, omega_r)
    intt_T2 = (intt_T2 * rc_inv) % self.prime
    intt_T1 = gen_twiddle_matrix(self.rr, self.rr, self.prime, inv_omega_rr)
    intt_T1 = (intt_T1 * rr_inv) % self.prime

    perm_rr = get_bit_reverse_perm(self.rr)
    perm_rc = get_bit_reverse_perm(self.rc)
    perm_cr = get_bit_reverse_perm(self.cr)
    perm_cc = get_bit_reverse_perm(self.cc)

    ntt_T1 = ntt_T1[perm_rr, :]
    ntt_T2 = ntt_T2[perm_rr, :]
    ntt_T3 = ntt_T3[:, perm_rc]
    perm_r = get_bit_reverse_perm(r_total)
    ntt_T4 = ntt_T4.reshape(r_total, c_total)[perm_r, :].reshape(
        self.rr, self.rc, self.cr, self.cc
    )
    ntt_T5 = ntt_T5[perm_cr, :]
    ntt_T6 = ntt_T6[perm_cr, :]
    ntt_T7 = ntt_T7[:, perm_cc]

    intt_T7 = intt_T7[perm_cc, :]
    intt_T6 = intt_T6[perm_cr, :]
    intt_T5 = intt_T5[:, perm_cr]
    intt_T4 = intt_T4.reshape(r_total, c_total)[perm_r, :].reshape(
        self.rr, self.rc, self.cr, self.cc
    )
    intt_T3 = intt_T3[perm_rc, :]
    intt_T2 = intt_T2[perm_rr, :]
    intt_T1 = intt_T1[:, perm_rr]

    # Steps 1 and 5 are logically left-matmul; steps 3 and 7 are right.
    self.ntt_T1 = ff_ctx.preprocess_matmul(ntt_T1.T)
    self.ntt_T2 = ff_ctx.preprocess_elementwise(ntt_T2)
    self.ntt_T3 = ff_ctx.preprocess_matmul(ntt_T3)
    self.ntt_T4 = ff_ctx.preprocess_elementwise(ntt_T4)
    self.ntt_T5 = ff_ctx.preprocess_matmul(ntt_T5.T)
    self.ntt_T6 = ff_ctx.preprocess_elementwise(ntt_T6)
    self.ntt_T7 = ff_ctx.preprocess_matmul(ntt_T7)

    self.intt_T7 = ff_ctx.preprocess_matmul(intt_T7)
    self.intt_T6 = ff_ctx.preprocess_elementwise(intt_T6)
    self.intt_T5 = ff_ctx.preprocess_matmul(intt_T5.T)
    self.intt_T4 = ff_ctx.preprocess_elementwise(intt_T4)
    self.intt_T3 = ff_ctx.preprocess_matmul(intt_T3)
    self.intt_T2 = ff_ctx.preprocess_elementwise(intt_T2)
    self.intt_T1 = ff_ctx.preprocess_matmul(intt_T1.T)

  def ntt(self, v: jnp.ndarray):
    v = self._ensure_ntt_shape(v)  # (B, RR, RC, CR, CC, trailing)
    # Steps 1-3: inner NTT on R = RR * RC.
    v = self.ff_ctx.modular_matmul(v, self.ntt_T1, contract_axis=1)
    v = self.ff_ctx.modular_multiply_broadcast(
        v, self.ntt_T2[:, :, None, None, :]
    )
    v = self.ff_ctx.modular_matmul(v, self.ntt_T3, contract_axis=2)
    # Step 4: outer twiddle (broadcast over batch only).
    v = self.ff_ctx.modular_multiply_broadcast(v, self.ntt_T4[None])
    # Steps 5-7: inner NTT on C = CR * CC.
    v = self.ff_ctx.modular_matmul(v, self.ntt_T5, contract_axis=3)
    v = self.ff_ctx.modular_multiply_broadcast(
        v, self.ntt_T6[None, None, None, :, :, :]
    )
    v = self.ff_ctx.modular_matmul(v, self.ntt_T7, contract_axis=4)
    return v

  def intt(self, v: jnp.ndarray):
    v = self._ensure_ntt_shape(v)
    v = self.ff_ctx.modular_matmul(v, self.intt_T7, contract_axis=4)
    v = self.ff_ctx.modular_multiply_broadcast(
        v, self.intt_T6[None, None, None, :, :, :]
    )
    v = self.ff_ctx.modular_matmul(v, self.intt_T5, contract_axis=3)
    v = self.ff_ctx.modular_multiply_broadcast(v, self.intt_T4[None])
    v = self.ff_ctx.modular_matmul(v, self.intt_T3, contract_axis=2)
    v = self.ff_ctx.modular_multiply_broadcast(
        v, self.intt_T2[:, :, None, None, :]
    )
    v = self.ff_ctx.modular_matmul(v, self.intt_T1, contract_axis=1)
    return v
