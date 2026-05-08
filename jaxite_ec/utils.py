import csv
from functools import lru_cache
import hashlib
import math
import os
import pickle
from typing import Any, Callable, List, Optional
import warnings
import jax
from jax import export
from jax import sharding as shd
import jax.numpy as jnp
import numpy as np
import toml

# =============================================================================
# Load configurations
# =============================================================================
_config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configurations.toml"
)
with open(_config_path, "r", encoding="utf-8") as _f:
  _config = toml.load(_f)
_serialized_jax_kernel_dir = _config.get(
    "serialized_jax_kernel_dir", "./deployments/"
)
if "TEST_TMPDIR" in os.environ:
  _serialized_jax_kernel_dir = os.path.join(
      os.environ["TEST_TMPDIR"], "deployments"
  )

if not os.path.exists(_serialized_jax_kernel_dir):
  os.makedirs(_serialized_jax_kernel_dir)
_hash_length = _config.get("hash_length", 8)


# =============================================================================
# Number theory transform related helper functions
# =============================================================================
def gen_twiddle_matrix(rows, cols, q, omega):
  """Precompute the twiddle matrix T of shape (rows, cols), where T[r, c] = omega^(r*c) mod q.

  Args:
      rows: The number of rows in the matrix.
      cols: The number of columns in the matrix.
      q: The modulus.
      omega: The primitive root of unity.

  Returns:
      The twiddle matrix.
  """
  warnings.warn(
      "gen_twiddle_matrix is deprecated. Use"
      " jaxite_ntt.number_theory_transform.utils.gen_twiddle_matrix instead.",
      UserWarning,
      stacklevel=2,
  )
  twiddle_matrix = np.zeros((rows, cols), dtype=int)
  for r in range(rows):
    for c in range(cols):
      twiddle_matrix[r, c] = pow(int(omega), int(r * c), int(q))
  return twiddle_matrix


def gen_twiddle_matrix_inv(rows, cols, q, omega):
  """Precompute the inverse twiddle matrix T_inv of shape (rows, cols).

  T_inv[r, c] = omega^{- (r*c)} mod q.

  Args:
      rows: The number of rows in the matrix.
      cols: The number of columns in the matrix.
      q: The modulus.
      omega: The primitive root of unity.

  Returns:
      The inverse twiddle matrix.
  """
  warnings.warn(
      "gen_twiddle_matrix_inv is deprecated. Use"
      " jaxite_ntt.number_theory_transform.utils.gen_twiddle_matrix_inv"
      " instead.",
      UserWarning,
      stacklevel=2,
  )
  twiddle_matrix_inv = np.zeros((rows, cols), dtype=int)
  for r in range(rows):
    for c in range(cols):
      twiddle_matrix_inv[r, c] = pow(int(omega), int(-r * c), int(q))
  return twiddle_matrix_inv


def prime_factors(n):
  """Return the set of prime factors of n."""
  factors = set()
  # Divide out factors of 2
  while n % 2 == 0:
    factors.add(2)
    n //= 2
  # Check odd factors from 3 to sqrt(n)
  p = 3
  while p**2 <= n:
    while n % p == 0:
      factors.add(p)
      n //= p
    p += 2
  if n > 1:
    factors.add(n)
  return factors


def find_generator(q):
  """Find a primitive root modulo q.

  Args:
    q (int): The prime modulus.

  Returns:
    A generator of GF(q)^*.

  Raises:
    ValueError: If no generator is found, indicating q is not prime.
  """
  phi = q - 1
  factors = prime_factors(phi)

  # Test candidates from 2 to q-1.
  for g in range(2, q):
    is_generator = all(pow(g, phi // p, q) != 1 for p in factors)
    if is_generator:
      return g
  raise ValueError("No generator found, check that q is prime.")


def root_of_unity(m: int, q: int) -> int:
  """Canonical primitive m-th root of unity modulo q that **works with NTT**.

  Args:
    m (int): The order of the root of unity.
    q (int): The prime modulus.

  Returns:
    int: The canonical primitive m-th root of unity modulo q.

  Usage:
    root_of_unity(16, 134219681) # This works with NTT.
    computed_psi = [root_of_unity(m, q) for q in original_modulus]
  """
  assert (q - 1) % m == 0, "q-1 must be divisible by m"
  # Step 1: multiplicative generator of Z_q^*
  g = find_generator(q)
  # Step 2: raise to (q-1)/m to get an m-th root candidate
  r = pow(g, (q - 1) // m, q)
  # Step 3: among r^k with gcd(k,m)=1, pick the minimal value whose order is exactly m
  # For m=2^t, order check is psi^(m/2) == q-1 (i.e., == -1 mod q)
  candidates = []
  half = m // 2
  for k in range(1, m):
    if math.gcd(k, m) != 1:
      continue
    psi = pow(r, k, q)
    if pow(psi, half, q) == q - 1 and pow(psi, m, q) == 1:
      candidates.append(psi)
  assert candidates, "No primitive m-th root found"
  return int(min(candidates))


# =============================================================================
# Modular arithmetic related helper functions
# =============================================================================
def modular_inverse(a: int, m: int):
  t, new_t = 0, 1
  r, new_r = m, a

  while new_r != 0:
    quotient = r // new_r
    t, new_t = new_t, t - quotient * new_t
    r, new_r = new_r, r - quotient * new_r

  if r > 1:
    raise ValueError(f"{a} is not invertible modulo {m}")
  if t < 0:
    t += m

  return t


def compute_crt_factors(moduli):
  modular = math.prod(moduli)
  ms = [modular // m for m in moduli]
  ms_inv = [modular_inverse(ms[i], moduli[i]) for i in range(len(moduli))]
  return [(ms[i] * ms_inv[i]) % modular for i in range(len(moduli))]


def to_rns(x, moduli):
  return [x % m for m in moduli]


def rns_reconstruct(residues, moduli, crt_factors):
  return sum(
      [residues[i] * crt_factors[i] for i in range(len(residues))]
  ) % math.prod(moduli)


def find_moduli_specified_number(total_number, precision):
  """Finds a list of moduli close to the given precision.

  The moduli are all odd and coprime.

  Args:
    total_number: The total number of moduli requirement.
    precision: The desired precision of the moduli.

  Returns:
    A tuple containing two lists:
      - overall_moduli: A list of moduli close to the given precision.
  """
  initial_moduli = 2**precision
  overall_moduli = []
  overall_modulus = 1
  for i in range(1, 2 ** (precision >> 1) - 1):
    cur_moduli = initial_moduli - i
    if cur_moduli % 2 == 1 and math.gcd(cur_moduli, overall_modulus) == 1:
      overall_moduli.append(cur_moduli)
      overall_modulus *= cur_moduli
      if len(overall_moduli) >= total_number:
        return to_tuple(overall_moduli)

  # Find 2**31 - v
  initial_moduli = 2 ** (precision - 1)
  if len(overall_moduli) < total_number:
    for i in range(1, 2 ** (precision >> 1) - 1):
      cur_moduli = initial_moduli - i
      if cur_moduli % 2 == 1 and math.gcd(cur_moduli, overall_modulus) == 1:
        overall_moduli.append(cur_moduli)
        overall_modulus *= cur_moduli
        if len(overall_moduli) >= total_number:
          return to_tuple(overall_moduli)

  return to_tuple(overall_moduli)


def find_primes_with_bits(number, bits):
  """Returns a list of 'number' prime numbers, each with exactly 'bits' bits.

  Args:
      number (int): The number of primes to find.
      bits (int): The bit length of each prime.

  Returns:
      List[int]: List of prime numbers with the specified bit length.
  """

  def is_prime(n):
    if n < 2:
      return False
    if n == 2 or n == 3:
      return True
    if n % 2 == 0 or n % 3 == 0:
      return False
    i = 5
    w = 2
    while i * i <= n:
      if n % i == 0:
        return False
      i += w
      w = 6 - w
    return True

  primes = []
  lower = 1 << (bits - 1)
  upper = (1 << bits) - 1
  candidate = lower | 1  # ensure odd

  while candidate <= upper and len(primes) < number:
    if is_prime(candidate):
      primes.append(candidate)
    candidate += 2  # only check odd numbers

  return primes


def modular_matrix_np_u32_to_u8_bat_4d(matrix: np.ndarray, modulus: int):
  rows, cols = matrix.shape
  assert modulus <= 2**31
  matrix_u64 = matrix.astype(np.uint64)
  matrix_u64_byteshifted = np.array(
      [matrix_u64 << (8 * byte_idx) for byte_idx in range(4)], dtype=np.uint64
  )
  # shape is (4, rows, cols)
  matrix_u64_byteshifted = matrix_u64_byteshifted.transpose(1, 0, 2)
  matrix_u64_byteshifted_mod_modulus = (
      matrix_u64_byteshifted % modulus
  ).astype(np.uint32)
  matrix_u8 = matrix_u64_byteshifted_mod_modulus.view(np.uint8).reshape(
      rows, 4, cols, 4
  )
  return matrix_u8


# =============================================================================
# MSM related helper functions
# =============================================================================
def read_external_msm_file(path, type: str):
  if type == "scalars":
    scalars = []
    with open(
        path, "r", newline="", encoding="utf-8"
    ) as csvfile:  # Handle potential encoding issues
      csv_reader = csv.reader(csvfile)
      for row in csv_reader:
        scalars.append(int(row[-1][13:-2], 16))
    return scalars

  elif type == "points":
    points = []
    with open(
        path, "r", newline="", encoding="utf-8"
    ) as csvfile:  # Handle potential encoding issues
      csv_reader = csv.reader(csvfile)
      for row in csv_reader:
        points.append([int(row[8][13:-2], 16), int(row[-1][13:-2], 16)])
    return points

  elif type == "result_ref":
    result_ref = []
    with open(
        path, "r", newline="", encoding="utf-8"
    ) as csvfile:  # Handle potential encoding issues
      csv_reader = csv.reader(csvfile)
      for row in csv_reader:
        result_ref.append(int(row[7][13:-2], 16))
        result_ref.append(int(row[-1][13:-2], 16))
    return result_ref


def split_list(lst, chunk_size):
  """Splits a list into equal-sized chunks."""
  return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def slice_scalars(
    scalars: List, scalar_bits: int, slice_length: int
) -> List[List]:
  window_num = int(math.ceil(scalar_bits / slice_length))
  mask = (1 << slice_length) - 1
  slices_list = [[] for _ in range(window_num)]
  for scalar in scalars:
    for i in range(window_num):
      slices_list[i].append(scalar & mask)
      scalar >>= slice_length
  return slices_list


# =============================================================================
# shape and tuple related helper functions
# =============================================================================
def nested_list_depth(x: Any) -> int:
  """Return the nesting depth of a list.

  Examples:
      nested_list_depth([1, 2, 3])        -> 1
      nested_list_depth([[1, 2], [3, 4]]) -> 2
      nested_list_depth([[[1], [2]]])     -> 3

  Args:
      x: The value to inspect.

  Returns:
      0 if x is not a list, otherwise 1 + the maximum depth of its elements.
  """
  if not isinstance(x, list):
    return 0
  if len(x) == 0:
    return 1
  return 1 + max(nested_list_depth(item) for item in x)


def to_tuple(a):
  """Create to convert numpy array into tuple."""
  if isinstance(a, (list, tuple, np.ndarray)):
    return tuple(to_tuple(i) for i in a)
  return int(a)


def pad_jax_array(array: jnp.ndarray, target_shape: tuple) -> jax.Array:
  if array.shape == target_shape:
    return array
  assert len(array.shape) == len(
      target_shape
  ), "array and target_shape must have the same number of dimensions"
  pad_width = []
  for cur, tgt in zip(array.shape, target_shape):
    assert tgt >= cur, f"target size {tgt} is smaller than current size {cur}"
    pad_width.append((0, tgt - cur))
  return jnp.pad(array, pad_width, mode="constant", constant_values=0)


# =============================================================================
# Code structure
# =============================================================================
class JaxParameters:
  word_bits: Any = None
  rns_moduli_inv_word: Any = None
  word_mask: Any = None
  half_word_mask: Any = None
  half_word_bits: Any = None
  rns_moduli_high: Any = None
  rns_moduli_low: Any = None
  rns_moduli: Any = None
  crns_precision: Any = None
  crns_vector_g: Any = None
  crns_stacked_mat_E_with_f_T: Any = None
  rns_moduli_negate: Any = None
  rns_moduli_sub: Any = None
  twist_d: Any = None

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

  def set_parameter(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)


class JaxKernelContextBase:

  def __init__(self, use_compiled_kernels: bool = False):
    self.compiled_kernels = {}
    self.use_compiled_kernels = use_compiled_kernels
    self.use_sharding = False
    self.sharding_mesh = None
    self.mesh_axes = ()

  def set_use_compiled_kernels(self, use_compiled_kernels: bool):
    self.use_compiled_kernels = use_compiled_kernels

  def serialize(self, parameters) -> Any:
    pass

  def compile(self, parameters) -> Any:
    pass

  def context_hash(self) -> str:
    raise NotImplementedError("Subclasses must implement context_hash")

  def set_use_sharding(self, use_sharding: bool):
    if use_sharding:
      self.use_sharding = True
      mesh, partition_spec = create_sharding()
      axis_names = mesh.axis_names
      partition = axis_names if len(axis_names) > 1 else axis_names[0]
      self.sharding_mesh = mesh
      self.sharding_partition = partition
      self.sharding_partition_spec = partition_spec
      self.mesh_axes = tuple(mesh.axis_names)
    else:
      self.use_sharding = False
      self.sharding_mesh = None
      self.sharding_partition = None
      self.sharding_partition_spec = None
      self.mesh_axes = ()

  def make_named_sharding(self, spec) -> "jax.sharding.NamedSharding | None":
    """Wrap a PartitionSpec in a NamedSharding bound to the current mesh.

    Returns None if sharding is disabled.
    """
    mesh = getattr(self, "sharding_mesh", None)
    if mesh is None:
      return None
    return jax.sharding.NamedSharding(mesh, spec)

  def shard_constraint(self, x, spec):
    """Apply ``with_sharding_constraint(x, NamedSharding(mesh, spec))``.

    No-op when sharding is disabled. Intended for use inside jitted
    kernels to pin intermediate layouts.
    """
    mesh = getattr(self, "sharding_mesh", None)
    if mesh is None:
      return x
    return jax.lax.with_sharding_constraint(
        x, jax.sharding.NamedSharding(mesh, spec)
    )

  def create_named_sharding(
      self, shape: tuple, axes: list[int]
  ) -> tuple[jax.sharding.NamedSharding, tuple]:
    """Create an efficient NamedSharding for the given shape and shard axes.

    (Generated by Claude)

    For 1 axis: shards across all mesh devices on that axis. Skips sharding
    (returns replicated) if the axis is too small for the mesh size. Pads
    the axis so that shape[axis] % total_devices == 0.

    For 2 axes: tries both mesh-to-data-axis mappings and picks the one
    with less padding waste. Pads each sharded axis as needed.

    Args:
        shape: Array shape.
        axes: 1 or 2 axis indices to shard along. >=3 is not allowed.

    Returns:
        (NamedSharding, padded_shape) — the sharding and the shape after
        padding (same as input shape if no padding was needed).
    """
    assert (
        hasattr(self, "sharding_mesh") and self.sharding_mesh is not None
    ), "Sharding must be enabled first via enable_sharding()"

    if len(axes) == 0:
      spec = [None] * len(shape)
      partition_spec = jax.sharding.PartitionSpec(*spec)
      return (
          jax.sharding.NamedSharding(self.sharding_mesh, partition_spec),
          shape,
      )

    assert len(axes) >= 1, "Must specify at least 1 axis"
    assert len(axes) <= 2, "Sharding along >= 3 axes is not supported"

    mesh = self.sharding_mesh
    mesh_axis_names = mesh.axis_names  # e.g. ('x', 'y')
    mesh_axis_sizes = {name: mesh.shape[name] for name in mesh_axis_names}
    padded_shape = list(shape)

    # Minimum elements per device below which sharding is not worthwhile
    _MIN_ELEMS_PER_DEVICE = 1

    if len(axes) == 1:
      axis = axes[0]
      total_devices = math.prod(mesh_axis_sizes.values())

      # Small axis: replicate instead of sharding
      if shape[axis] < total_devices * _MIN_ELEMS_PER_DEVICE:
        spec = [None] * len(shape)
        partition_spec = jax.sharding.PartitionSpec(*spec)
        warnings.warn(
            f"Shape {shape} is too small for sharding, replicating the axis"
            f" {axis}",
            stacklevel=2,
        )
        return jax.sharding.NamedSharding(mesh, partition_spec), tuple(shape)

      # Pad so shape[axis] is divisible by total_devices
      if shape[axis] % total_devices != 0:
        warnings.warn(
            f"Shape {shape} is not divisible by total_devices {total_devices},"
            f" padding the axis {axis} to"
            f" {math.ceil(shape[axis] / total_devices) * total_devices}",
            stacklevel=2,
        )
        padded_shape[axis] = (
            math.ceil(shape[axis] / total_devices) * total_devices
        )

      # Shard this single data axis across all mesh axes
      spec = [None] * len(shape)
      spec[axis] = tuple(mesh_axis_names)
      partition_spec = jax.sharding.PartitionSpec(*spec)

    else:  # len(axes) == 2
      axis0, axis1 = axes[0], axes[1]
      total_devices = math.prod(mesh_axis_sizes.values())

      def _compute_candidate(spec_list, shard_plan):
        """Compute (waste, spec, padded) for a shard_plan: list of (mesh_label, data_axis).

        mesh_label is either a single mesh axis name or a tuple of all mesh axis
        names.
        """
        p = list(shape)
        waste = 0
        for mesh_label, d_axis in shard_plan:
          if isinstance(mesh_label, tuple):
            ms = math.prod(mesh_axis_sizes[n] for n in mesh_label)
          else:
            ms = mesh_axis_sizes[mesh_label]
          s = shape[d_axis]
          if s % ms != 0:
            waste += (math.ceil(s / ms) * ms) - s
            p[d_axis] = math.ceil(s / ms) * ms
        return waste, spec_list, tuple(p)

      # Candidate A: mesh_x -> axis0, mesh_y -> axis1
      spec_a = [None] * len(shape)
      spec_a[axis0] = mesh_axis_names[0]
      spec_a[axis1] = mesh_axis_names[1]
      cand_a = _compute_candidate(
          spec_a, [(mesh_axis_names[0], axis0), (mesh_axis_names[1], axis1)]
      )

      # Candidate B: mesh_x -> axis1, mesh_y -> axis0
      spec_b = [None] * len(shape)
      spec_b[axis1] = mesh_axis_names[0]
      spec_b[axis0] = mesh_axis_names[1]
      cand_b = _compute_candidate(
          spec_b, [(mesh_axis_names[0], axis1), (mesh_axis_names[1], axis0)]
      )

      # Candidate C: all mesh axes -> axis0 only
      spec_c = [None] * len(shape)
      spec_c[axis0] = tuple(mesh_axis_names)
      cand_c = _compute_candidate(spec_c, [(tuple(mesh_axis_names), axis0)])

      # Candidate D: all mesh axes -> axis1 only
      spec_d = [None] * len(shape)
      spec_d[axis1] = tuple(mesh_axis_names)
      cand_d = _compute_candidate(spec_d, [(tuple(mesh_axis_names), axis1)])

      candidates = [cand_a, cand_b, cand_c, cand_d]

      # Filter out candidates where a sharded axis is too small
      def _is_viable(spec_list):
        for i, s in enumerate(spec_list):
          if s is None:
            continue
          if isinstance(s, tuple):
            ms = math.prod(mesh_axis_sizes[n] for n in s)
          else:
            ms = mesh_axis_sizes[s]
          if shape[i] < ms * _MIN_ELEMS_PER_DEVICE:
            return False
        return True

      viable = [(w, sp, pp) for w, sp, pp in candidates if _is_viable(sp)]
      if not viable:
        # All too small — replicate
        spec = [None] * len(shape)
        partition_spec = jax.sharding.PartitionSpec(*spec)
        return jax.sharding.NamedSharding(mesh, partition_spec), tuple(shape)

      best_waste, best_spec, best_padded = min(viable, key=lambda x: x[0])
      spec = best_spec
      padded_shape = list(best_padded)

      if best_waste > 0:
        warnings.warn(
            f"Shape {shape} requires padding for sharding, padded to"
            f" {tuple(padded_shape)}"
        )

      partition_spec = jax.sharding.PartitionSpec(*spec)

    return jax.sharding.NamedSharding(mesh, partition_spec), tuple(padded_shape)


def jax_jit_lower_compile(func: Callable, *args, **kwargs) -> Callable:
  # in_shardings = tuple(
  #     arg.sharding if isinstance(arg, jax.ShapeDtypeStruct) and getattr(arg, "sharding", None) is not None
  #     else None
  #     for arg in args
  # )
  # if any(s is not None for s in in_shardings):
  #   assert
  #   return jax.jit(func, in_shardings=in_shardings).lower(*args).compile()
  return jax.jit(func).lower(*args).compile()


def store_jax_kernel(func: Callable, *args, **kwargs) -> None:
  name = kwargs.get("name")
  path = os.path.join(_serialized_jax_kernel_dir, f"{name}.jax")

  serialized_jax_kernel = export.export(jax.jit(func))(*args).serialize()
  with open(path, "wb") as f:
    f.write(serialized_jax_kernel)
  print(f"stored jax kernel to {path}")


def load_jax_kernel(
    name: str, alternative_callable: Optional[Callable] = None
) -> Optional[Callable]:
  path = os.path.join(_serialized_jax_kernel_dir, f"{name}.jax")
  if not os.path.exists(path):
    return alternative_callable
  with open(path, "rb") as f:
    serialized_jax_kernel = export.deserialize(bytearray(f.read()))
  print(f"loaded jax kernel from {path}")
  return serialized_jax_kernel.call


def store_jax_executable(func: Callable, *args, **kwargs) -> None:
  return store_jax_kernel(func, *args, **kwargs)


def load_jax_executable(name: str) -> Optional[Callable]:
  return load_jax_kernel(name)


# =============================================================================
# Hashing utilities
# =============================================================================

# 62-character alphabet: digits + lowercase + uppercase
_HASH_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_HASH_BASE = len(_HASH_CHARS)  # 62


@lru_cache(maxsize=1)
def _get_hash_length() -> int:
  """Read hash_length from configurations.toml, defaulting to 16."""
  import toml

  config_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "configurations.toml"
  )
  with open(config_path, "r", encoding="utf-8") as f:
    config = toml.load(f)
  return int(config.get("hash_length", 16))


def _serialize(arg: Any) -> str:
  """Produce a deterministic, type-tagged string for any common Python value.

  Handles arbitrarily nested lists and tuples recursively.
  """
  if isinstance(arg, bool):
    return f"bool:{arg}"
  if isinstance(arg, int):
    return f"int:{arg}"
  if isinstance(arg, float):
    return f"float:{repr(arg)}"
  if isinstance(arg, str):
    # embed length so "ab","c" != "a","bc"
    return f"str{len(arg)}:{arg}"
  if isinstance(arg, bytes):
    return f"bytes:{arg.hex()}"
  if isinstance(arg, (list, tuple)):
    tag = "list" if isinstance(arg, list) else "tuple"
    inner = ",".join(_serialize(x) for x in arg)
    return f"{tag}[{len(arg)}:{inner}]"
  if isinstance(arg, dict):
    items = ";".join(
        f"{_serialize(k)}->{_serialize(v)}"
        for k, v in sorted(arg.items(), key=lambda kv: repr(kv[0]))
    )
    return f"dict{{{len(arg)}:{items}}}"
  # Fallback for other types (e.g. numpy scalars, custom objects)
  return f"{type(arg).__name__}:{repr(arg)}"


def hash_args(*args: Any) -> str:
  """Hash any number of Python values into a fixed-length alphanumeric string.

  The output length is controlled by ``hash_length`` in ``configurations.toml``.
  Characters are drawn from ``[0-9a-zA-Z]`` (base-62), giving 62^length possible
  values — e.g. 16 characters yield ~4.7 × 10²⁸ distinct hashes.  The result is
  safe to embed directly in a file name.

  Supported argument types: ``int``, ``float``, ``bool``, ``str``, ``bytes``,
  ``list``, ``tuple`` (arbitrarily nested), ``dict``, and any type with a
  stable ``repr``.

  Args:
      *args: Values to hash.

  Returns:
      A fixed-length alphanumeric hash string.

  Example:
      >>> hash_args(42, "hello", [1, 2, 3])
      'aB3x9Kp2mNqR7tYz'
  """
  length = _hash_length
  payload = "|".join(_serialize(a) for a in args)
  digest = hashlib.blake2b(payload.encode("utf-8")).digest()  # 64 bytes

  # Base-62 encode the big-endian integer
  num = int.from_bytes(digest, "big")
  chars: list[str] = []
  while num:
    chars.append(_HASH_CHARS[num % _HASH_BASE])
    num //= _HASH_BASE

  # blake2b gives ~86 base-62 digits from 64 bytes — always enough for any
  # reasonable hash_length without padding
  return "".join(reversed(chars))[:length]


# =============================================================================
# Sharding utilities
# =============================================================================
def create_sharding():
  """Create default batch and replicated shardings for the current device mesh."""
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
  shd.set_mesh(mesh)

  partition_spec = jax.sharding.PartitionSpec
  return mesh, partition_spec
