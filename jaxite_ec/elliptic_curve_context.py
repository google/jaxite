import abc
from concurrent.futures import ProcessPoolExecutor
import functools
import multiprocessing
import os
import warnings

import jax
import jax.numpy as jnp
from jaxite.jaxite_ec import finite_field_context
from jaxite.jaxite_ec import utils
import numpy as np

FiniteFieldContextBase = finite_field_context.FiniteFieldContextBase
abstractmethod = abc.abstractmethod
ABC = abc.ABC


# Use 'forkserver' to avoid JAX multithreading + fork deadlock
_MP_CONTEXT = multiprocessing.get_context("forkserver")

JaxParameters = utils.JaxParameters
JaxKernelContextBase = utils.JaxKernelContextBase
hash_args = utils.hash_args
pad_jax_array = utils.pad_jax_array
store_jax_executable = utils.store_jax_kernel
load_jax_executable = utils.load_jax_kernel
jax_jit_lower_compile = utils.jax_jit_lower_compile
jax.config.update("jax_enable_x64", True)


class EllipticCurveContextBase(ABC):
  """Abstract base class defining the interface for finite field operations.

  Subclasses must implement all abstract methods to provide concrete
  finite field arithmetic operations.
  """

  @abstractmethod
  def __init__(self, parameters: dict):
    """Initialize the finite field context.

    Args:
        parameters: Configuration dictionary containing field parameters.
    """
    self.parameters = parameters
    self.prime = parameters.get("prime", None)
    assert self.prime is not None, "prime must be provided"
    self.zero_point = None
    ff_ctx_class = parameters.get("finite_field_context_class", None)
    assert (
        ff_ctx_class is not None
    ), "finite_field_context_class must be provided"
    self.ff_ctx: FiniteFieldContextBase = ff_ctx_class(
        parameters.get("finite_field_parameters", {})
    )

  @abstractmethod
  def to_computational_format(self, a) -> jnp.ndarray:
    """Convert input to the internal computational representation.

    Args:
        a: Input value in standard format.

    Returns:
        Value converted to computational format (e.g., Montgomery form).
    """
    pass

  @abstractmethod
  def to_original_format(self, a):
    """Convert from computational format back to standard representation.

    Args:
        a: Value in computational format.

    Returns:
        Value in standard integer representation.
    """
    pass

  @abstractmethod
  def point_add(self, a: jnp.ndarray, b: jnp.ndarray):
    """Perform point addition: (a + b)

    Args:
        a: First operand in computational format.
        b: Second operand in computational format.

    Returns:
        Product in computational format.
    """
    pass

  @abstractmethod
  def point_double(self, a: jnp.ndarray):
    """Perform point doubling: (2 * a)

    Args:
        a: First operand in computational format.

    Returns:
        Product in computational format.
    """
    pass

  def get_finite_field_context(self) -> FiniteFieldContextBase:
    return self.ff_ctx

  def _modular_multiply(self, a: int, b: int) -> int:
    return (a * b) % self.prime

  def _modular_reduce(self, a: int) -> int:
    return a % self.prime

  def _modular_divide(self, a: int, b: int) -> int:
    assert b != 0, "ec divide: b is zero"
    b_inv = pow(b, self.prime - 2, self.prime)
    return (a * b_inv) % self.prime


class CPUWeierstrassAffineContext(EllipticCurveContextBase):
  """CPU implementation of Weierstrass affine curve operations.

  This class provides CPU-based implementations for point addition and doubling
  on a Weierstrass curve in affine coordinates.
  This class is only for private functional testing, not for production use.
  """

  def __init__(self, parameters: dict):
    super().__init__(parameters)
    # warnings.warn("CPUWeierstrassAffineContext is only for private functional testing, not for production use",UserWarning, stacklevel=2)

    # Curve configuration
    self.a = parameters["a"]
    self.b = parameters["b"]

  def point_add(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    raise NotImplementedError(
        "CPUWeierstrassAffineContext: point_add is not implemented"
    )

  def point_double(self, point: jnp.ndarray) -> jnp.ndarray:
    raise NotImplementedError(
        "CPUWeierstrassAffineContext: point_double is not implemented"
    )

  def to_computational_format(self, a: list) -> jnp.ndarray:
    raise NotImplementedError(
        "CPUWeierstrassAffineContext: to_computational_format is not"
        " implemented"
    )

  def to_original_format(self, a: jnp.ndarray) -> list:
    raise NotImplementedError(
        "CPUWeierstrassAffineContext: to_original_format is not implemented"
    )

  def _point_add(self, point_a: list, point_b: list) -> list:
    def single_point_add(point_a: list, point_b: list) -> list[int]:
      x1, y1 = point_a
      x2, y2 = point_b
      slope = self._modular_divide(
          self._modular_reduce(y2 - y1), self._modular_reduce(x2 - x1)
      )
      x3 = self._modular_reduce(self._modular_multiply(slope, slope) - x1 - x2)
      y3 = self._modular_reduce(
          self._modular_multiply(slope, self._modular_reduce(x1 - x3)) - y1
      )
      return [x3, y3]

    list_depth = utils.nested_list_depth(point_a)
    if list_depth == 1:
      return single_point_add(point_a, point_b)
    elif list_depth == 2:
      return [
          single_point_add(point_a_i, point_b_i)
          for point_a_i, point_b_i in zip(point_a, point_b)
      ]
    else:
      raise ValueError(
          f"Invalid list depth {list_depth} of input for point addition"
      )

  def _point_double(self, point: list) -> list:
    def single_point_double(point: list) -> list[int]:
      x, y = point
      slope = self._modular_divide(
          self._modular_reduce(3 * x * x + self.a), self._modular_reduce(2 * y)
      )
      x3 = self._modular_reduce(self._modular_multiply(slope, slope) - 2 * x)
      y3 = self._modular_reduce(self._modular_multiply(slope, x - x3) - y)
      return [x3, y3]

    list_depth = utils.nested_list_depth(point)
    if list_depth == 1:
      return single_point_double(point)
    elif list_depth == 2:
      return [single_point_double(point_i) for point_i in point]
    else:
      raise ValueError("Invalid list depth of input for point doubling")


class ExtendedTwistedEdwardsContextBase(EllipticCurveContextBase):

  def __init__(self, parameters: dict):
    super().__init__(parameters)

    # Curve configuration
    self.a = parameters["a"]
    self.twist_d = parameters["twist_d"]
    self.alpha = parameters["alpha"]
    self.s = parameters["s"]
    self.A = parameters["MA"]
    self.B = parameters["MB"]
    self.t = parameters["t"]
    self.k = self.twist_d + self.twist_d
    self.zero_point = [0, 1, 1, 0]

  def _twist(self, coordinates: list[int]) -> list[int]:
    assert (
        len(coordinates) == 2
    ), "Twisted Edwards coordinates must be of length 2"
    x, y = coordinates
    # Convert to montgomery (Notel it is ec montgomery not field montgomery)
    xm = self._modular_reduce(self.s * (x - self.alpha))
    ym = self._modular_reduce(self.s * y)
    # Convert to edwards
    if ym == 0:
      raise ValueError("ec twist: ym is zero")
    xt = self._modular_divide(xm, ym)

    yt_denom = xm + 1
    if yt_denom == 0:
      raise ValueError("ec twist: yt_denom is zero")
    yt = self._modular_divide(xm - 1, yt_denom)

    xt = self._modular_multiply(xt, self.t)
    return [xt, yt]

  def _untwist(self, coordinates: list[int]) -> list[int]:
    assert (
        len(coordinates) == 2
    ), "Twisted Edwards coordinates must be of length 2"
    xt, yt = coordinates
    xt = self._modular_divide(xt, self.t)
    # Convert to montgomery
    xm = self._modular_divide((1 + yt), (1 - yt))
    ym = self._modular_divide((1 + yt), self._modular_multiply((1 - yt), xt))
    # Convert to weierstrass
    x = self._modular_reduce(
        self._modular_divide(xm, self.B)
        + self._modular_divide(self.A, self._modular_multiply(3, self.B))
    )
    y = self._modular_divide(ym, self.B)
    return [x, y]

  def _convert_to_edwards_affine(self, coordinates: list[int]) -> list[int]:
    assert (
        len(coordinates) == 4
    ), "Twisted Edwards coordinates must be of length 2"
    x, y, z, t = coordinates
    z_inv = self._modular_divide(1, z)
    x = self._modular_multiply(x, z_inv)
    y = self._modular_multiply(y, z_inv)
    return [x, y]

  def _convert_to_extended_twisted_edwards(
      self, coordinates: list[int]
  ) -> list[int]:
    assert (
        len(coordinates) == 2
    ), "Twisted Edwards coordinates must be of length 2"
    xt, yt = self._twist(coordinates)
    return [xt, yt, 1, self._modular_multiply(xt, yt)]

  def _convert_to_weierstrass_affine(self, coordinates: list[int]) -> list[int]:
    assert (
        len(coordinates) == 4
    ), "Extended Twisted Edwards coordinates must be of length 4"
    affine_coords = self._convert_to_edwards_affine(coordinates)
    untwisted_coords = self._untwist(affine_coords)
    return untwisted_coords


class ExtendedTwistedEdwardsContext(
    ExtendedTwistedEdwardsContextBase, JaxKernelContextBase
):

  def __init__(self, parameters: dict):
    super().__init__(parameters)
    JaxKernelContextBase.__init__(self)
    self.jax_parameters = JaxParameters()
    self._init_jax_parameters()

  def to_computational_format(self, a: list) -> jnp.ndarray:
    list_depth = utils.nested_list_depth(a)
    # NOTE: the dimension is (batch, coordinates)
    if list_depth == 1:
      twisted_coords = self._convert_to_extended_twisted_edwards(a)
    elif list_depth == 2:
      twisted_coords = [
          self._convert_to_extended_twisted_edwards(a_i) for a_i in a
      ]
    else:
      raise ValueError(
          "Invalid list depth of input for converting to extended twisted"
          " edwards coordinates"
      )
    result = self.ff_ctx.to_computational_format(twisted_coords)
    if list_depth == 1:
      result = jnp.broadcast_to(result, (result.shape[0], 1, result.shape[1]))
    elif list_depth == 2:
      result = result.transpose(1, 0, 2)
    # NOTE: the computational format dim is (coordinates, batch, precision)
    if self.use_sharding:
      named_sharding, padded_shape = self.create_named_sharding(
          shape=result.shape, axes=[1]
      )
      result = pad_jax_array(result, padded_shape)
      return result.to_device(named_sharding)
    else:
      return result.to_device(jax.devices()[0])

  def to_original_format(self, a: jnp.ndarray) -> list:
    dim = a.ndim
    # NOTE: the computational format dim is (coordinates, batch, precision)
    if dim == 3:
      a = a.transpose(
          1, 0, 2
      )  # (coordinates, batch, precision) -> (batch, coordinates, precision)
    a = self.ff_ctx.to_original_format(a)
    if dim == 2:
      affine_coords = self._convert_to_weierstrass_affine(a)
    elif dim == 3:
      affine_coords = [self._convert_to_weierstrass_affine(a_i) for a_i in a]
    else:
      raise ValueError(
          "Invalid dimension of input for converting to weierstrass affine"
          " coordinates"
      )
    return affine_coords

  def _init_jax_parameters(self):
    self.jax_parameters.set_parameter(
        twist_d=self.ff_ctx.to_computational_format(self.twist_d),
    )

  def _point_add(
      self, point_a: jnp.ndarray, point_b: jnp.ndarray
  ) -> jnp.ndarray:
    twist_d = self.jax_parameters.twist_d

    inputsl = point_a
    inputsr = point_b
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsr)
    a, b, d, c = jnp.vsplit(outputs, 4)
    # print(a.shape, b.shape, d.shape, c.shape)

    pax, pay, _, _ = jnp.vsplit(point_a, 4)
    pbx, pby, _, _ = jnp.vsplit(point_b, 4)

    e1 = self.ff_ctx._modular_add(pax, pay)
    e2 = self.ff_ctx._modular_add(pbx, pby)
    twist_d_here = jnp.broadcast_to(
        twist_d.reshape(-1, twist_d.shape[0]), c.shape
    )
    if self.use_sharding:
      twist_d_here = jax.sharding.reshard(twist_d_here, jax.typeof(e2).sharding)
    inputsl = jnp.concatenate((e1, c), axis=0)
    inputsr = jnp.concatenate((e2, twist_d_here), axis=0)
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsr)
    e3, c = jnp.vsplit(outputs, 2)

    e = self.ff_ctx._modular_subtract(self.ff_ctx._modular_subtract(e3, a), b)
    f = self.ff_ctx._modular_subtract(d, c)
    g = self.ff_ctx._modular_add(d, c)
    h = self.ff_ctx._modular_add(a, b)

    inputsl = jnp.concatenate((e, g, f, e), axis=0)
    inputsr = jnp.concatenate((f, h, g, h), axis=0)
    # print(inputsl.shape, inputsr.shape)
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsr)
    return outputs.reshape(4, -1, outputs.shape[-1])

  def _point_double(self, point: jnp.ndarray) -> jnp.ndarray:
    x, y, z, t = point

    et = self.ff_ctx._modular_multiply(x, y)
    inputsl = jnp.vstack((x, y, z, et))
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsl)
    a, b, ct, et2 = jnp.vsplit(outputs, 4)

    h = self.ff_ctx._modular_negate(self.ff_ctx._modular_add(a, b))
    e = self.ff_ctx._modular_add(et2, h)
    g = self.ff_ctx._modular_subtract(b, a)
    f = self.ff_ctx._modular_subtract(g, self.ff_ctx._modular_add(ct, ct))

    inputsl = jnp.vstack((e, g, f, e))
    inputsr = jnp.vstack((f, h, g, h))
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsr)
    return outputs.reshape(4, -1, outputs.shape[-1])

  def point_add(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:

    if self.use_compiled_kernels:
      kernel_hash = hash_args(a.shape, a.dtype.__str__())
      return self.compiled_kernels[kernel_hash]["point_add"](a, b)
    else:
      return self._point_add(a, b)

  # TODO: fix the functional bug in point_double
  def point_double(self, point: jnp.ndarray) -> jnp.ndarray:
    raise ValueError("point_double has logic bug")
    shape_dtype_struct = jax.ShapeDtypeStruct(point.shape, point.dtype)
    if self.use_compiled_kernels:
      return self.compiled_kernels[shape_dtype_struct.__hash__()][
          "point_double"
      ](point)
    else:
      return self._point_double(point)

  def _get_shape_dtype_structs(
      self, parameters: dict
  ) -> list[jax.ShapeDtypeStruct]:
    batch_size = parameters["batch_size"]
    num_moduli = self.jax_parameters.twist_d.shape[0]
    point_shape = (4, batch_size, num_moduli)
    if self.use_sharding:
      named_sharding, padded_shape = self.create_named_sharding(
          shape=point_shape, axes=[1]
      )
      return [
          jax.ShapeDtypeStruct(
              padded_shape, jnp.uint32, sharding=named_sharding
          )
      ]
    return [jax.ShapeDtypeStruct(point_shape, jnp.uint32)]

  def context_hash(self) -> str:
    return hash_args(
        self.__class__.__name__,
        self.ff_ctx.context_hash(),
        self.a,
        self.twist_d,
        self.alpha,
        self.s,
        self.A,
        self.B,
        self.t,
        self.use_sharding,
    )

  def serialize(self, parameters: dict):
    shape_dtype_structs = self._get_shape_dtype_structs(parameters)
    kernel_hash = hash_args(self.context_hash(), parameters)
    class_name = self.__class__.__name__

    store_jax_executable(
        self._point_add,
        shape_dtype_structs[0],
        shape_dtype_structs[0],
        name=f"{class_name}_point_add_{kernel_hash}",
    )
    store_jax_executable(
        self._point_double,
        shape_dtype_structs[0],
        name=f"{class_name}_point_double_{kernel_hash}",
    )

  def compile(self, parameters: dict):
    shape_dtype_structs = self._get_shape_dtype_structs(parameters)
    kernel_hash = hash_args(self.context_hash(), parameters)
    class_name = self.__class__.__name__

    point_add_kernel = load_jax_executable(
        f"{class_name}_point_add_{kernel_hash}"
    )
    point_double_kernel = load_jax_executable(
        f"{class_name}_point_double_{kernel_hash}"
    )

    if None in [point_add_kernel, point_double_kernel]:
      warnings.warn(
          f"Not found stored serialized compiled kernels, compiling...",
          UserWarning,
          stacklevel=2,
      )

    kernel_hash = hash_args(
        shape_dtype_structs[0].shape, shape_dtype_structs[0].dtype.__str__()
    )
    self.compiled_kernels[kernel_hash] = {
        "point_add": (
            point_add_kernel
            if point_add_kernel is not None
            else jax_jit_lower_compile(
                self._point_add, shape_dtype_structs[0], shape_dtype_structs[0]
            )
        ),
        "point_double": (
            point_double_kernel
            if point_double_kernel is not None
            else jax_jit_lower_compile(
                self._point_double, shape_dtype_structs[0]
            )
        ),
    }
    self.use_compiled_kernels = True


def _twist_extend_and_rns_worker(
    point, s, alpha, prime, prime_m2, t, rns_moduli, radix_bits
):
  """Module-level worker: twist + extend + RNS conversion using gmpy2 (must be picklable)."""
  import gmpy2

  x, y = gmpy2.mpz(point[0]), gmpy2.mpz(point[1])
  xm = (s * (x - alpha)) % prime
  ym = (s * y) % prime
  if ym == 0:
    raise ValueError("ec twist: ym is zero")
  xt = (xm * gmpy2.powmod(ym, prime_m2, prime)) % prime
  yt_denom = xm + 1
  if yt_denom == 0:
    raise ValueError("ec twist: yt_denom is zero")
  yt = ((xm - 1) * gmpy2.powmod(yt_denom, prime_m2, prime)) % prime
  xt = (xt * t) % prime
  coords = [int(xt), int(yt), 1, int((xt * yt) % prime)]
  # RNS conversion inline: (a % m) << radix_bits) % m for each coordinate
  return [[(((c % m) << radix_bits) % m) for m in rns_moduli] for c in coords]


class ExtendedTwistedEdwardsNDContext(
    ExtendedTwistedEdwardsContextBase, JaxKernelContextBase
):
  """Extended Twisted Edwards context supporting arbitrary batch dimensions.

  Computational format layout: (coordinates=4, *batch_dims, precision)
  where batch_dims can be any number of dimensions, e.g.:
    - (4, batch, precision)                   -- 1D batch
    - (4, batch1, batch2, precision)          -- 2D batch
    - (4, batch1, batch2, batch3, precision)  -- 3D batch

  Input points are nested lists of [x, y] affine Weierstrass coordinates.
  Nesting depth determines batch dimensions:
    - [x, y]                       → (4, 1, precision)
    - [[x,y], ...]                 → (4, batch1, precision)
    - [[[x,y], ...], ...]          → (4, batch1, batch2, precision)
  """

  def __init__(self, parameters: dict):
    super().__init__(parameters)
    JaxKernelContextBase.__init__(self)
    self.jax_parameters = JaxParameters()
    self._init_jax_parameters()

  def to_computational_format(self, a: list) -> jnp.ndarray:
    list_depth = utils.nested_list_depth(a)
    if list_depth < 1:
      raise ValueError(f"Invalid list depth {list_depth} for point conversion")

    # Use parallel processing with gmpy2 for large flat batches of points (depth==2)
    # Fuses twist + extend + RNS conversion into one parallel step
    _PARALLEL_THRESHOLD = 2048
    if list_depth == 2 and len(a) >= _PARALLEL_THRESHOLD:
      import gmpy2

      ff_ctx = self.ff_ctx
      worker = functools.partial(
          _twist_extend_and_rns_worker,
          s=gmpy2.mpz(self.s),
          alpha=gmpy2.mpz(self.alpha),
          prime=gmpy2.mpz(self.prime),
          prime_m2=gmpy2.mpz(self.prime - 2),
          t=gmpy2.mpz(self.t),
          rns_moduli=ff_ctx.rns_moduli,
          radix_bits=ff_ctx.radix_bits,
      )
      num_workers = min(64, os.cpu_count() or 1, max(1, len(a) // 256))
      with ProcessPoolExecutor(
          max_workers=num_workers, mp_context=_MP_CONTEXT
      ) as pool:
        rns_coords = list(
            pool.map(worker, a, chunksize=max(1, len(a) // num_workers))
        )
      # rns_coords: list of (4, moduli_num) per point → (N, 4, moduli_num) array
      result = jnp.array(
          np.array(rns_coords, dtype=np.uint32), dtype=jnp.uint32
      )
      # (N, 4, moduli_num) → (4, N, moduli_num)
      result = result.transpose(1, 0, 2)
    else:

      def recursive_twist(lst, depth):
        if depth == 1:
          return self._convert_to_extended_twisted_edwards(lst)
        return [recursive_twist(item, depth - 1) for item in lst]

      twisted_coords = recursive_twist(a, list_depth)

      result = self.ff_ctx.to_computational_format(twisted_coords)

      if list_depth == 1:
        # (4, precision) → (4, 1, precision)
        result = jnp.expand_dims(result, axis=1)
      else:
        # (*batch_dims, 4, precision) → (4, *batch_dims, precision)
        ndim = result.ndim
        perm = (ndim - 2,) + tuple(range(ndim - 2)) + (ndim - 1,)
        result = result.transpose(perm)

    if self.use_sharding:
      shard_axes = list(range(1, min(3, result.ndim - 1)))
      if not shard_axes:
        shard_axes = [1]
      named_sharding, padded_shape = self.create_named_sharding(
          shape=result.shape, axes=shard_axes
      )
      result = pad_jax_array(result, padded_shape)
      return result.to_device(named_sharding)
    else:
      return result.to_device(jax.devices()[0])

  def to_original_format(self, a: jnp.ndarray) -> list:
    ndim = a.ndim
    if ndim < 2:
      raise ValueError(f"Expected at least 2D array, got {ndim}D")

    if ndim == 2:
      a_orig = self.ff_ctx.to_original_format(a)
      return self._convert_to_weierstrass_affine(a_orig)

    # (4, *batch_dims, precision) → (*batch_dims, 4, precision)
    perm = tuple(range(1, ndim - 1)) + (0, ndim - 1)
    a = a.transpose(perm)
    a_orig = self.ff_ctx.to_original_format(a)

    batch_depth = ndim - 2

    def recursive_untwist(lst, depth):
      if depth == 0:
        return self._convert_to_weierstrass_affine(lst)
      return [recursive_untwist(item, depth - 1) for item in lst]

    return recursive_untwist(a_orig, batch_depth)

  def _init_jax_parameters(self):
    self.jax_parameters.set_parameter(
        twist_d=self.ff_ctx.to_computational_format(self.twist_d),
    )

  def _point_add(
      self, point_a: jnp.ndarray, point_b: jnp.ndarray
  ) -> jnp.ndarray:
    twist_d = self.jax_parameters.twist_d

    inputsl = point_a
    inputsr = point_b
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsr)
    a, b, d, c = jnp.vsplit(outputs, 4)

    pax, pay, _, _ = jnp.vsplit(point_a, 4)
    pbx, pby, _, _ = jnp.vsplit(point_b, 4)

    e1 = self.ff_ctx._modular_add(pax, pay)
    e2 = self.ff_ctx._modular_add(pbx, pby)
    twist_d_here = jnp.broadcast_to(twist_d, c.shape)
    try:
      _e2_sh = jax.typeof(e2).sharding
      if _e2_sh is not None:
        twist_d_here = jax.sharding.reshard(twist_d_here, _e2_sh)
    except Exception:
      pass
    inputsl = jnp.concatenate((e1, c), axis=0)
    inputsr = jnp.concatenate((e2, twist_d_here), axis=0)
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsr)
    e3, c = jnp.vsplit(outputs, 2)

    e = self.ff_ctx._modular_subtract(self.ff_ctx._modular_subtract(e3, a), b)
    f = self.ff_ctx._modular_subtract(d, c)
    g = self.ff_ctx._modular_add(d, c)
    h = self.ff_ctx._modular_add(a, b)

    inputsl = jnp.concatenate((e, g, f, e), axis=0)
    inputsr = jnp.concatenate((f, h, g, h), axis=0)
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsr)
    return outputs

  def _point_double(self, point: jnp.ndarray) -> jnp.ndarray:
    original_shape = point.shape
    x, y, z, t = point

    et = self.ff_ctx._modular_multiply(x, y)
    inputsl = jnp.vstack((x, y, z, et))
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsl)
    a, b, ct, et2 = jnp.vsplit(outputs, 4)

    h = self.ff_ctx._modular_negate(self.ff_ctx._modular_add(a, b))
    e = self.ff_ctx._modular_add(et2, h)
    g = self.ff_ctx._modular_subtract(b, a)
    f = self.ff_ctx._modular_subtract(g, self.ff_ctx._modular_add(ct, ct))

    inputsl = jnp.vstack((e, g, f, e))
    inputsr = jnp.vstack((f, h, g, h))
    outputs = self.ff_ctx._modular_multiply(inputsl, inputsr)
    return outputs.reshape(original_shape)

  def point_add(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    if self.use_compiled_kernels:
      kernel_hash = hash_args(a.shape, a.dtype.__str__())
      return self.compiled_kernels[kernel_hash]["point_add"](a, b)
    else:
      return self._point_add(a, b)

  # TODO: fix the functional bug in point_double
  def point_double(self, point: jnp.ndarray) -> jnp.ndarray:
    raise ValueError("point_double has logic bug")
    shape_dtype_struct = jax.ShapeDtypeStruct(point.shape, point.dtype)
    if self.use_compiled_kernels:
      return self.compiled_kernels[shape_dtype_struct.__hash__()][
          "point_double"
      ](point)
    else:
      return self._point_double(point)

  def _get_shape_dtype_structs(
      self, parameters: dict
  ) -> list[jax.ShapeDtypeStruct]:
    batch_shape = parameters.get("batch_shape", None)
    if batch_shape is None:
      batch_shape = (parameters["batch_size"],)
    num_moduli = self.jax_parameters.twist_d.shape[0]
    point_shape = (4,) + tuple(batch_shape) + (num_moduli,)
    if self.use_sharding:
      shard_axes = list(range(1, min(3, len(point_shape) - 1)))
      if not shard_axes:
        shard_axes = [1]
      named_sharding, padded_shape = self.create_named_sharding(
          shape=point_shape, axes=shard_axes
      )
      return [
          jax.ShapeDtypeStruct(
              padded_shape, jnp.uint32, sharding=named_sharding
          )
      ]
    return [jax.ShapeDtypeStruct(point_shape, jnp.uint32)]

  def context_hash(self) -> str:
    return hash_args(
        self.__class__.__name__,
        self.ff_ctx.context_hash(),
        self.a,
        self.twist_d,
        self.alpha,
        self.s,
        self.A,
        self.B,
        self.t,
        self.use_sharding,
    )

  def serialize(self, parameters: dict):
    shape_dtype_structs = self._get_shape_dtype_structs(parameters)
    kernel_hash = hash_args(self.context_hash(), parameters)
    class_name = self.__class__.__name__

    store_jax_executable(
        self._point_add,
        shape_dtype_structs[0],
        shape_dtype_structs[0],
        name=f"{class_name}_point_add_{kernel_hash}",
    )
    store_jax_executable(
        self._point_double,
        shape_dtype_structs[0],
        name=f"{class_name}_point_double_{kernel_hash}",
    )

  def compile(self, parameters: dict):
    shape_dtype_structs = self._get_shape_dtype_structs(parameters)
    kernel_hash = hash_args(self.context_hash(), parameters)
    class_name = self.__class__.__name__

    point_add_kernel = load_jax_executable(
        f"{class_name}_point_add_{kernel_hash}"
    )
    point_double_kernel = load_jax_executable(
        f"{class_name}_point_double_{kernel_hash}"
    )

    if None in [point_add_kernel, point_double_kernel]:
      warnings.warn(
          f"Not found stored serialized compiled kernels, compiling...",
          UserWarning,
          stacklevel=2,
      )

    kernel_hash = hash_args(
        shape_dtype_structs[0].shape, shape_dtype_structs[0].dtype.__str__()
    )
    self.compiled_kernels[kernel_hash] = {
        "point_add": (
            point_add_kernel
            if point_add_kernel is not None
            else jax_jit_lower_compile(
                self._point_add, shape_dtype_structs[0], shape_dtype_structs[0]
            )
        ),
        "point_double": (
            point_double_kernel
            if point_double_kernel is not None
            else jax_jit_lower_compile(
                self._point_double, shape_dtype_structs[0]
            )
        ),
    }
    self.use_compiled_kernels = True
