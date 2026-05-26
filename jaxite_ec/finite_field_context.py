from abc import ABC, abstractmethod
import math
from typing import Any, List, Union
import warnings
import jax
import jax.numpy as jnp
from jaxite.jaxite_ec import utils
import numpy as np

JaxKernelContextBase = utils.JaxKernelContextBase
JaxParameters = utils.JaxParameters
hash_args = utils.hash_args
jax_jit_lower_compile = utils.jax_jit_lower_compile
load_jax_executable = utils.load_jax_executable
pad_jax_array = utils.pad_jax_array
store_jax_executable = utils.store_jax_executable

jax.config.update("jax_enable_x64", True)


class FiniteFieldContextBase(ABC):
  """Abstract base class defining the interface for finite field operations.

  Subclasses must implement all abstract methods to provide concrete
  finite field arithmetic operations.
  """

  prime: Any = None
  parameters: Any = None
  rns_moduli: Any = None
  radix_bits: Any = None

  @abstractmethod
  def __init__(self, parameters: dict):
    """Initialize the finite field context.

    Args:
        parameters: Configuration dictionary containing field parameters.
    """
    self.prime = parameters.get("prime", None)
    assert self.prime is not None, "prime must be provided"
    self.parameters = parameters

  @abstractmethod
  def to_computational_format(self, a) -> jnp.ndarray:
    """Convert input to the internal computational representation.

    Args:
        a: Input value in standard format.

    Returns:
        Value converted to computational format (e.g., Montgomery form).
    """
    pass

  def _modular_multiply(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return a

  def _modular_add(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return a

  def _modular_subtract(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return a

  def _modular_negate(self, a: jnp.ndarray) -> jnp.ndarray:
    return a

  def _modular_reduce(self, a: jnp.ndarray) -> jnp.ndarray:
    return a

  @abstractmethod
  def to_original_format(self, a) -> Union[int, List[Any]]:
    """Convert from computational format back to standard representation.

    Args:
        a: Value in computational format.

    Returns:
        Value in standard integer representation.
    """
    pass

  @abstractmethod
  def context_hash(self) -> str:
    pass

  @abstractmethod
  def modular_multiply(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Perform modular multiplication: (a * b) mod p.

    Args:
        a: First operand in computational format.
        b: Second operand in computational format.

    Returns:
        Product in computational format.
    """
    pass

  @abstractmethod
  def modular_add(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Perform modular addition: (a + b) mod p.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        Sum reduced modulo p.
    """
    pass

  @abstractmethod
  def modular_subtract(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Perform modular subtraction: (a - b) mod p.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        Difference reduced modulo p.
    """
    pass

  @abstractmethod
  def modular_negate(self, a: jnp.ndarray) -> jnp.ndarray:
    """Perform modular negation: -a mod p.

    Args:
        a: Operand.

    Returns:
        Negation reduced modulo p.
    """
    pass

  @abstractmethod
  def modular_reduce(self, a: jnp.ndarray) -> jnp.ndarray:
    """Reduce value modulo the field prime.

    Args:
        a: Value to reduce (may be larger than prime).

    Returns:
        Value reduced to [0, p).
    """
    pass


class RNSContextBase(FiniteFieldContextBase):

  def __init__(self, parameters: dict):
    super().__init__(parameters)
    self.rns_moduli = parameters.get("rns_moduli", [])
    assert len(self.rns_moduli) != 0, "rns_moduli must be non-empty"
    self.total_modulus = math.prod(self.rns_moduli)
    assert (
        self.total_modulus > self.prime
    ), "RNS total modulus must be greater than prime"
    self.precision_bits = parameters.get(
        "precision_bits", 0
    )  # Default precision bits
    assert self.precision_bits != 0, "precision bits must be non-zero"

    self.crt_factors = self._compute_crt_factors(self.rns_moduli)
    # RNSContextBase._check_parameters(self)

  def _check_parameters(self):
    pass  # disable the warning for now
    sum_moduli_bits = math.ceil(math.log2(sum(self.rns_moduli)))
    if self.precision_bits < sum_moduli_bits:
      warnings.warn(
          "precision_bits is less than sum of moduli_bits, precision_bits:"
          f" {self.precision_bits}, sum of moduli_bits: {sum_moduli_bits}."
          + "This may cause precision loss.",
          UserWarning,
          stacklevel=2,
      )

    max_modulus_bits = max(math.ceil(math.log2(m)) for m in self.rns_moduli)
    required_precision = max_modulus_bits + math.log2(len(self.rns_moduli)) + 1
    if self.precision_bits < required_precision:
      warnings.warn(
          "precision_bits is less than required precision, precision_bits:"
          f" {self.precision_bits}, required precision: {required_precision}."
          + "This may cause precision loss.",
          UserWarning,
          stacklevel=2,
      )

  def _compute_crt_factors(self, moduli: list[int]):
    ms = [self.total_modulus // m for m in moduli]
    ms_inv = [
        utils.modular_inverse(ms[i], moduli[i]) for i in range(len(moduli))
    ]
    return [
        (ms[i] * ms_inv[i]) % self.total_modulus for i in range(len(moduli))
    ]

  def _elementwise_add(self, a: list[int], b: list[int]):
    assert len(a) == len(b), "a and b must have the same length"
    return [a[i] + b[i] for i in range(len(a))]

  def _elementwise_subtract(self, a: list[int], b: list[int]):
    assert len(a) == len(b), "a and b must have the same length"
    return [a[i] - b[i] for i in range(len(a))]

  def _elementwise_multiply(self, a: list[int], b: list[int]):
    assert len(a) == len(b), "a and b must have the same length"
    return [a[i] * b[i] for i in range(len(a))]

  def _elementwise_reduce(self, z: list[int], m: list[int]):
    assert len(z) == len(m), "z and m must have the same length"
    return [z[i] % m[i] for i in range(len(m))]

  def _elementwise_left_shift(self, a: list[int], b: Union[list[int], int]):
    if isinstance(b, list):
      assert len(a) == len(b), "a and b must have the same length"
      return [a[i] << b[i] for i in range(len(a))]
    else:
      return [a[i] << b for i in range(len(a))]

  def _elementwise_right_shift(self, a: list[int], b: Union[list[int], int]):
    if isinstance(b, list):
      assert len(a) == len(b), "a and b must have the same length"
      return [a[i] >> b[i] for i in range(len(a))]
    else:
      return [a[i] >> b for i in range(len(a))]

  def _elementwise_and(self, a: list[int], b: Union[list[int], int]):
    if isinstance(b, list):
      assert len(a) == len(b), "a and b must have the same length"
      return [a[i] & b[i] for i in range(len(a))]
    else:
      return [a[i] & b for i in range(len(a))]

  def _rns_decompose(self, x: int, moduli: list[int]):
    return [(x % m) for m in moduli]

  def _get_crns_vector_I_before_reducing(self, moduli_m: list[int], y):
    vector_i = []
    modular_M = math.prod(moduli_m)
    for i, m_i in enumerate(moduli_m):
      # M_y_over_m_i = (modular_M * y) // m_i
      M_y_over_m_i = (modular_M) // m_i
      # logging.info(f"M_y_over_m_i: {M_y_over_m_i}")
      inv_M_y_over_m_i = utils.modular_inverse(M_y_over_m_i % m_i, m_i)
      # logging.info(f"inv_M_y_over_m_i: {inv_M_y_over_m_i}")
      I_i = inv_M_y_over_m_i * M_y_over_m_i * y
      vector_i.append(I_i)
    return vector_i

  def get_moduli_num(self) -> int:
    return len(self.rns_moduli)

  def _crns(
      self,
      x: list[int],
      matrix_E: list[list[int]],
      vector_f_T: list[int],
      vector_g: list[int],
      moduli: list[int],
      u: int,
  ):
    """CRNS Computation based on Algorithm steps 9-12

    Args:
        x: Input vector x_M in RNS representation
        matrix_E: Precomputed matrix E from crns_precompute
        vector_f_T: Precomputed vector f^T from crns_precompute
        vector_g: Precomputed vector g from crns_precompute
        moduli: RNS moduli (should be self.rns_moduli_n)

    Returns:
        x_N: Result vector in RNS representation modulo N
    """
    # Step 9: E, f^T, g = CRNSPRECOMPUTATION(M, y, N, z, u) - already done
    # (matrix_E, vector_f_T, vector_g are passed as parameters)

    # Step 10: v = x_M · f^T (Dot product; can be parallelized with x_M · E)
    v = sum(x[i] * vector_f_T[i] for i in range(len(x)))

    # Step 11: k = ⌊v/2^u⌋ (Bitshifting computes fixed-point quotient)
    k = v >> u  # Right shift by u bits is equivalent to floor division by 2^u

    # Step 12: return x_N = x_M · E + k·g (vector-matrix multiply, scalar-vector multiply, compute ICRT)
    x_N = []
    for j in range(len(moduli)):
      # Compute x_M · E for column j
      dot_product = sum(x[i] * matrix_E[i][j] for i in range(len(x)))
      dot_product = dot_product % moduli[j]
      x_N.append(dot_product)

    for j in range(len(moduli)):
      # Add k * g_j and take modulo n_j
      # result = (x_N[j] + k * vector_g[j]) % moduli[j]
      result = x_N[j] + k * vector_g[j]
      x_N[j] = result
    return x_N


class DRNSlazyContextBase(RNSContextBase):

  def __init__(self, parameters: dict):
    super().__init__(parameters)
    self.radix_bits = parameters.get("radix_bits", 0)
    assert self.radix_bits != 0, " radix bits must be non-zero"
    self.moduli_inv_on_radix = [
        utils.modular_inverse(m, 1 << self.radix_bits) for m in self.rns_moduli
    ]
    self.crns_y, self.crns_z = self._precompute_crns_parameters()
    self.matrix_E, self.vector_f_T, self.vector_g = self._crns_precompute(
        self.rns_moduli,
        self.prime,
        self.crns_y,
        self.crns_z,
        self.precision_bits,
    )

  def _check_parameters(self):
    pass  # disable the warning for now
    if (self.prime << self.w) > self.total_modulus:
      warnings.warn(
          "Total modulus is not enough to hold prime in DRNSlazy, total"
          f" modulus: {self.total_modulus}, prime: {self.prime}."
          + "This may cause finite field overflow.",
          UserWarning,
          stacklevel=2,
      )

  def _precompute_crns_parameters(self):
    radix_inv = utils.modular_inverse(2**self.radix_bits, self.total_modulus)
    return radix_inv, 1 << (2 * self.radix_bits)

  def _crns_precompute(
      self,
      moduli: list[int],
      prime: int,
      y: int,
      z: int,
      u: int,
  ):
    modular = math.prod(moduli)
    vector_i = []
    for i, m_i in enumerate(moduli):
      M_over_m_i = modular // m_i
      inv_M_over_m_i = utils.modular_inverse(M_over_m_i % m_i, m_i)
      I_i = (inv_M_over_m_i * M_over_m_i * y) % modular
      I_i = I_i
      vector_i.append(I_i)

    matrix_E = []
    for i, I_i in enumerate(vector_i):
      E_row = []
      for j, n_j in enumerate(moduli):
        E_ij = (z * (I_i % prime)) % n_j
        E_row.append(E_ij)
      matrix_E.append(E_row)

    vector_f_T = []
    for I_i in vector_i:
      f_i_T = math.ceil((I_i * (1 << u)) / modular)
      vector_f_T.append(f_i_T)

    vector_g = []
    for n_j in moduli:
      g_j = (-z * (modular % prime)) % n_j
      vector_g.append(g_j)

    return matrix_E, vector_f_T, vector_g

  def _elementwise_montgomery_reduce(
      self, z: list[int], m: list[int], m_inv: list[int]
  ):
    assert len(z) == len(m), "z and m must have the same length"
    assert len(z) == len(m_inv), "z and m_inv must have the same length"
    mask = (1 << self.radix_bits) - 1
    z_low = self._elementwise_and(z, mask)
    z_high = self._elementwise_right_shift(z, self.radix_bits)
    q = self._elementwise_and(self._elementwise_multiply(z_low, m_inv), mask)
    h = self._elementwise_right_shift(
        self._elementwise_multiply(q, m), self.radix_bits
    )
    t = self._elementwise_subtract(z_high, h)
    t = self._elementwise_add(t, m)
    return t


class DRNSlazyContext(DRNSlazyContextBase, JaxKernelContextBase):

  def __init__(self, parameters: dict):
    super().__init__(parameters)
    JaxKernelContextBase.__init__(self)
    self.jax_parameters = JaxParameters()
    self._init_jax_parameters()

  def to_computational_format(self, a: Union[int, list]) -> jnp.ndarray:
    moduli = self.rns_moduli
    radix_bits = self.radix_bits

    def individual_convert(a: int) -> list:
      return [(((a % m) << radix_bits) % m) for m in moduli]

    def recursive_convert(a):
      if isinstance(a, int):
        return individual_convert(a)
      return [recursive_convert(a_i) for a_i in a]

    converted_list = recursive_convert(a)
    converted_a = jnp.array(
        np.array(converted_list, dtype=np.uint32), dtype=jnp.uint32
    )
    if self.use_sharding:
      named_sharding, padded_shape = self.create_named_sharding(
          shape=converted_a.shape, axes=[0]
      )
      converted_a = pad_jax_array(converted_a, padded_shape)
      return converted_a.to_device(named_sharding)
    else:
      return converted_a.to_device(jax.devices()[0])

  def to_original_format(self, a: jnp.ndarray) -> Union[int, list]:  # type: ignore
    def individual_convert(a: list[int]) -> int:
      a = self._elementwise_montgomery_reduce(
          a, self.rns_moduli, self.moduli_inv_on_radix
      )
      r = 0
      for i in range(len(a)):
        r = (r + a[i] * self.crt_factors[i]) % self.total_modulus
      return r % self.prime

    def recursive_convert(a):
      if a.ndim == 1:
        return individual_convert(a.tolist())
      return [recursive_convert(a_i) for a_i in a]

    return recursive_convert(a)

  def _get_shape_dtype_structs(
      self, parameters: dict
  ) -> list[jax.ShapeDtypeStruct]:
    batch_shape = parameters["batch_shape"]
    oprand_shape = batch_shape + (len(self.rns_moduli),)
    if self.use_sharding:
      named_sharding, padded_shape = self.create_named_sharding(
          shape=oprand_shape, axes=[0]
      )
      return [
          jax.ShapeDtypeStruct(
              padded_shape, jnp.uint32, sharding=named_sharding
          )
      ]
    return [jax.ShapeDtypeStruct(oprand_shape, jnp.uint32)]

  def context_hash(self) -> str:
    return hash_args(
        self.__class__.__name__,
        self.prime,
        self.rns_moduli,
        self.precision_bits,
        self.radix_bits,
        self.use_sharding,
    )

  def serialize(self, parameters: dict):
    shape_dtype_structs = self._get_shape_dtype_structs(parameters)
    kernel_hash = hash_args(self.context_hash(), parameters)
    class_name = self.__class__.__name__

    store_jax_executable(
        self._modular_multiply,
        shape_dtype_structs[0],
        shape_dtype_structs[0],
        name=f"{class_name}_modular_multiply_{kernel_hash}",
    )
    store_jax_executable(
        self._modular_add,
        shape_dtype_structs[0],
        shape_dtype_structs[0],
        name=f"{class_name}_modular_add_{kernel_hash}",
    )
    store_jax_executable(
        self._modular_subtract,
        shape_dtype_structs[0],
        shape_dtype_structs[0],
        name=f"{class_name}_modular_subtract_{kernel_hash}",
    )
    store_jax_executable(
        self._modular_reduce,
        shape_dtype_structs[0],
        name=f"{class_name}_modular_reduce_{kernel_hash}",
    )
    store_jax_executable(
        self._modular_negate,
        shape_dtype_structs[0],
        name=f"{class_name}_modular_negate_{kernel_hash}",
    )

  def compile(self, parameters: dict):
    shape_dtype_structs = self._get_shape_dtype_structs(parameters)
    kernel_hash = hash_args(self.context_hash(), parameters)
    class_name = self.__class__.__name__

    modular_multiply_kernel = load_jax_executable(
        f"{class_name}_modular_multiply_{kernel_hash}"
    )
    modular_add_kernel = load_jax_executable(
        f"{class_name}_modular_add_{kernel_hash}"
    )
    modular_subtract_kernel = load_jax_executable(
        f"{class_name}_modular_subtract_{kernel_hash}"
    )
    modular_reduce_kernel = load_jax_executable(
        f"{class_name}_modular_reduce_{kernel_hash}"
    )
    modular_negate_kernel = load_jax_executable(
        f"{class_name}_modular_negate_{kernel_hash}"
    )

    if None in [
        modular_multiply_kernel,
        modular_add_kernel,
        modular_subtract_kernel,
        modular_reduce_kernel,
        modular_negate_kernel,
    ]:
      # if not self.use_sharding:
      warnings.warn(
          f"Not found stored serialized compiled kernels, compiling...",
          UserWarning,
          stacklevel=2,
      )

    kernel_hash = hash_args(
        shape_dtype_structs[0].shape, shape_dtype_structs[0].dtype.__str__()
    )
    # print(f"kernel_hash: {kernel_hash}")

    self.compiled_kernels[kernel_hash] = {
        "modular_multiply": (
            modular_multiply_kernel
            if modular_multiply_kernel is not None
            else jax_jit_lower_compile(
                self._modular_multiply,
                shape_dtype_structs[0],
                shape_dtype_structs[0],
            )
        ),
        "modular_add": (
            modular_add_kernel
            if modular_add_kernel is not None
            else jax_jit_lower_compile(
                self._modular_add,
                shape_dtype_structs[0],
                shape_dtype_structs[0],
            )
        ),
        "modular_subtract": (
            modular_subtract_kernel
            if modular_subtract_kernel is not None
            else jax_jit_lower_compile(
                self._modular_subtract,
                shape_dtype_structs[0],
                shape_dtype_structs[0],
            )
        ),
        "modular_reduce": (
            modular_reduce_kernel
            if modular_reduce_kernel is not None
            else jax_jit_lower_compile(
                self._modular_reduce, shape_dtype_structs[0]
            )
        ),
        "modular_negate": (
            modular_negate_kernel
            if modular_negate_kernel is not None
            else jax_jit_lower_compile(
                self._modular_negate, shape_dtype_structs[0]
            )
        ),
    }
    self.use_compiled_kernels = True

  def _jax_crns_precompute(
      self, moduli: list[int], prime: int, y: int, z: int, u: int
  ):
    rns_moduli_bytes = 4
    vector_I = self._get_crns_vector_I_before_reducing(moduli, y)
    modular = math.prod(moduli)

    vector_I_byteshifted = []
    for value_i in vector_I:
      value_i_byteshifted = [
          (value_i << (8 * byte_idx)) % modular
          for byte_idx in range(rns_moduli_bytes)
      ]
      vector_I_byteshifted = vector_I_byteshifted + value_i_byteshifted

    matrix_E = []
    for value_i_byteshifted in vector_I_byteshifted:
      value_i_byteshifted = z * (value_i_byteshifted % prime)
      matrix_E.append(self._rns_decompose(value_i_byteshifted, moduli))

    matrix_E_np = np.array(matrix_E, dtype=np.uint32).reshape(-1, len(moduli))
    matrix_E_np_u8 = matrix_E_np.view(np.uint8)

    vector_f = []
    for value_i_byteshifted in vector_I_byteshifted:
      value_i_byteshifted_f = math.ceil(
          (value_i_byteshifted * (1 << u)) / modular
      )
      vector_f.append(value_i_byteshifted_f)
    vector_f_T_np = np.array(vector_f, dtype=np.uint32).reshape(-1, 1)
    vector_f_T_np_u8 = vector_f_T_np.view(np.uint8)

    vector_g = self._rns_decompose(-z * (modular % prime), moduli)
    vector_g_np = np.array(vector_g, dtype=np.uint32).reshape(1, -1)

    matrix_E_with_f_T_np = np.hstack((matrix_E_np_u8, vector_f_T_np_u8))
    matrix_E_with_f_T_np = matrix_E_with_f_T_np.reshape(
        len(moduli), 4, len(moduli) + 1, 4
    )  # NOTE: reshape is special part for the optimization
    return matrix_E_with_f_T_np.tolist(), vector_g_np.tolist()

  def _init_jax_parameters(self):
    half_word_mask = 0xFFFF
    half_word_bits = 16
    word_bits = 32
    word_mask = (1 << word_bits) - 1
    rns_moduli_low = [m & half_word_mask for m in self.rns_moduli]
    rns_moduli_high = [m >> half_word_bits for m in self.rns_moduli]
    rns_moduli_inv_word = [
        utils.modular_inverse(m, 2**word_bits) for m in self.rns_moduli
    ]
    crns_precision = self.precision_bits
    crns_matrix_E_with_f_T, crns_vector_g = self._jax_crns_precompute(
        self.rns_moduli,
        self.prime,
        self.crns_y,
        self.crns_z,
        self.precision_bits,
    )
    num_moduli = len(self.rns_moduli)
    moduli_sub = self.to_computational_format(
        256 * num_moduli * 4 * 2 * self.prime
    )

    self.jax_parameters.set_parameter(
        word_mask=word_mask,
        half_word_mask=half_word_mask,
        half_word_bits=half_word_bits,
        word_bits=word_bits,
        rns_moduli=jnp.array(self.rns_moduli, dtype=jnp.uint64),
        rns_moduli_low=jnp.array(rns_moduli_low, dtype=jnp.uint16),
        rns_moduli_high=jnp.array(rns_moduli_high, dtype=jnp.uint16),
        rns_moduli_inv_word=jnp.array(rns_moduli_inv_word, dtype=jnp.uint32),
        crns_precision=jnp.array(crns_precision, dtype=jnp.uint32),
        crns_stacked_mat_E_with_f_T=jnp.array(
            crns_matrix_E_with_f_T, dtype=jnp.uint8
        ),
        crns_vector_g=jnp.array(crns_vector_g, dtype=jnp.uint32),
        rns_moduli_sub=jnp.array(moduli_sub, dtype=jnp.uint32),
        rns_moduli_negate=jnp.array(self.rns_moduli, dtype=jnp.uint32),
    )

  def _jax_montgomery_reduce(self, z: jax.Array) -> jax.Array:

    # Computation
    z_low = z.astype(jnp.uint32)
    z_high = (z >> self.jax_parameters.word_bits).astype(jnp.uint32)
    t = (
        z_low * self.jax_parameters.rns_moduli_inv_word
    ) & self.jax_parameters.word_mask
    t_low = t & self.jax_parameters.half_word_mask
    t_high = (
        t >> self.jax_parameters.half_word_bits
    ) & self.jax_parameters.half_word_mask

    prod_high = (
        t_high * self.jax_parameters.rns_moduli_high
    )  # This contributes directly to upper 32 bits
    prod_mid_high = (
        t_high * self.jax_parameters.rns_moduli_low
    )  # Upper 16 bits go to upper 32 bits
    prod_mid_low = (
        t_low * self.jax_parameters.rns_moduli_high
    )  # Upper 16 bits go to upper 32 bits
    prod_low = (
        t_low * self.jax_parameters.rns_moduli_low
    )  # Upper 16 bits contribute to middle part
    mid_low = (
        (prod_mid_high & self.jax_parameters.half_word_mask)
        + (prod_mid_low & self.jax_parameters.half_word_mask)
        + (prod_low >> self.jax_parameters.half_word_bits)
    )
    mid_high = (
        (prod_mid_high >> self.jax_parameters.half_word_bits)
        + (prod_mid_low >> self.jax_parameters.half_word_bits)
        + (mid_low >> self.jax_parameters.half_word_bits)
    )

    # Final upper 32 bits
    t_final = prod_high + mid_high
    b = z_high + self.jax_parameters.rns_moduli - t_final
    return b.astype(jnp.uint32)

  def _jax_crns(self, z: jax.Array) -> jax.Array:
    shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint64)
    precision_mask = jnp.array(
        (1 << self.jax_parameters.crns_precision) - 1, dtype=jnp.uint32
    )
    num_moduli_n = self.jax_parameters.crns_vector_g.shape[1]

    einsum_subscripts = "...kq,kqnp->...np"
    # computation
    x_u8 = jax.lax.bitcast_convert_type(z, jnp.uint8)

    # x_reconstruction_with_v = jnp.matmul(x_u8, stacked_mat_E_with_f_T, preferred_element_type=jnp.uint32)
    x_reconstruction_with_v = jnp.einsum(
        einsum_subscripts,
        x_u8,
        self.jax_parameters.crns_stacked_mat_E_with_f_T,
        preferred_element_type=jnp.uint32,
    )
    x_reconstruction_with_v_u64 = jnp.sum(
        x_reconstruction_with_v.astype(jnp.uint64) << shift_factors, axis=(-1,)
    )

    x_n, vector_v = jnp.split(
        x_reconstruction_with_v_u64, [num_moduli_n], axis=-1
    )
    vector_v = (vector_v >> self.jax_parameters.crns_precision) & precision_mask

    x_n = x_n + jnp.multiply(vector_v, self.jax_parameters.crns_vector_g)
    return x_n

  def _modular_multiply(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    z = jnp.multiply(a.astype(jnp.uint64), b.astype(jnp.uint64))
    z_reduced = self._jax_montgomery_reduce(z)
    z_rns_reduced = self._jax_crns(
        z_reduced
    )  # could be skipped for small prime
    z_reduced = self._jax_montgomery_reduce(
        z_rns_reduced
    )  # could be skipped for small prime (paired with the above)
    return z_reduced

  def _modular_add(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return a + b

  def _modular_subtract(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return self._modular_negate(b) + a

  def _modular_reduce(self, a: jnp.ndarray) -> jnp.ndarray:
    z_rns_reduced = self._jax_crns(a)
    z_reduced = self._jax_montgomery_reduce(z_rns_reduced)
    return z_reduced

  def _modular_negate(self, a: jax.Array) -> jax.Array:
    return jnp.add(
        jnp.subtract(self.jax_parameters.rns_moduli_negate, a),
        self.jax_parameters.rns_moduli_sub,
    )

  def modular_multiply(self, a: jax.Array, b: jax.Array) -> jax.Array:
    kernel_hash = hash_args(a.shape, a.dtype.__str__())
    if self.use_compiled_kernels:
      print(f"using compiled kernel for modular_multiply: {kernel_hash}")
      return self.compiled_kernels[kernel_hash]["modular_multiply"](a, b)
    else:
      return self._modular_multiply(a, b)

  def modular_add(self, a: jax.Array, b: jax.Array) -> jax.Array:
    kernel_hash = hash_args(a.shape, a.dtype.__str__())
    if self.use_compiled_kernels:
      return self.compiled_kernels[kernel_hash]["modular_add"](a, b)
    else:
      return self._modular_add(a, b)

  def modular_subtract(self, a: jax.Array, b: jax.Array) -> jax.Array:
    kernel_hash = hash_args(a.shape, a.dtype.__str__())
    if self.use_compiled_kernels:
      return self.compiled_kernels[kernel_hash]["modular_subtract"](a, b)
    else:
      return self._modular_subtract(a, b)

  def modular_reduce(self, a: jax.Array) -> jax.Array:
    kernel_hash = hash_args(a.shape, a.dtype.__str__())
    if self.use_compiled_kernels:
      return self.compiled_kernels[kernel_hash]["modular_reduce"](a)
    else:
      return self._modular_reduce(a)

  def modular_negate(self, a: jax.Array) -> jax.Array:
    kernel_hash = hash_args(a.shape, a.dtype.__str__())
    if self.use_compiled_kernels:
      return self.compiled_kernels[kernel_hash]["modular_negate"](a)
    else:
      return self._modular_negate(a)


# =============================================================================
# Lazy matrix reduction context
# =============================================================================


def _lazy_check_carry(value_c: jax.Array) -> jax.Array:
  """Check whether any 32-bit chunk holds a value exceeding 32 bits."""
  return jnp.any(jnp.not_equal(jnp.right_shift(value_c, jnp.uint64(32)), 0))


def _lazy_carry_propagate(value_c: jax.Array) -> jax.Array:
  """Propagate carries between adjacent 32-bit chunks."""
  n = value_c.shape[-1]
  roll_mat = jnp.array(
      [0, 1] + ([0] * n + [1]) * (n - 2) + [1] + [0] * (n - 1),
      dtype=jnp.uint16,
  ).reshape(n, n)
  low = jnp.bitwise_and(value_c, jnp.uint64(0xFFFFFFFF))
  high = jnp.right_shift(value_c, jnp.uint64(32)).astype(jnp.uint16)
  high = jnp.matmul(high, roll_mat, preferred_element_type=jnp.uint32).astype(
      jnp.uint16
  )
  return jnp.add(low, high.astype(jnp.uint64))


class LazyContextBase(FiniteFieldContextBase):
  """Base class for lazy matrix modular reduction.

  Represents field elements as little-endian arrays of uint32 chunks
  (base 2^32).  The number of chunks is chunk_num_u8 // 4 + 1 to
  provide one extra chunk of headroom for intermediate overflow.
  """

  def __init__(self, parameters: dict):
    super().__init__(parameters)
    raw_chunk_num_u8 = parameters.get(
        "chunk_num_u8", math.ceil(int(self.prime).bit_length() / 8)
    )
    # The byte pipeline in ``_mul_to_u8`` emits a ``4 * 2 * chunk_num_u32``-byte
    # buffer, and ``_modular_multiply`` slices ``high = vc[:, n8:2*n8+4]``
    # against a ``(n8+4, n8)`` lazy matrix.  The slice only fits when
    # ``chunk_num_u8`` is a multiple of 4; round up so the invariant holds
    # for primes with arbitrary bit-length.
    self.chunk_num_u8 = ((raw_chunk_num_u8 + 3) // 4) * 4
    self.chunk_num_u32 = self.chunk_num_u8 // 4 + 1
    self.rns_moduli = self.chunk_num_u32
    self.word_mask = (1 << 32) - 1
    self.modulus_lazy_mat = self._construct_lazy_matrix()
    self.prime_chunk = tuple(self._int_to_array(self.prime, self.chunk_num_u32))

  @staticmethod
  def _int_to_array(x: int, size: int, base: int = 32) -> list:
    mask = (1 << base) - 1
    elems = []
    while x > 0:
      elems.append(int(x & mask))
      x >>= base
    return elems[:size] + [0] * max(0, size - len(elems))

  @staticmethod
  def _array_to_int(arr, base: int = 32) -> int:
    result = 0
    for i, v in enumerate(arr):
      result |= int(v) << (i * base)
    return result

  def _construct_lazy_matrix(self):
    """Build the lazy reduction matrix.

    Row i = (256^(chunk_num_u8 + i)) % prime, expressed as chunk_num_u8
    little-endian uint8 chunks.  Shape: (chunk_num_u8 + 4, chunk_num_u8).
    """
    n = self.chunk_num_u8
    return tuple(
        tuple(self._int_to_array(pow(256, n + i, self.prime), n, base=8))
        for i in range(n + 4)
    )


class CROSSLazyContext(LazyContextBase, JaxKernelContextBase):
  """Lazy matrix modular multiplication context for JAX."""

  def __init__(self, parameters: dict):
    super().__init__(parameters)
    JaxKernelContextBase.__init__(self)
    self._lazy_mat_jnp = jnp.array(self.modulus_lazy_mat, dtype=jnp.uint16)
    self._prime_jnp = jnp.array(self.prime_chunk, dtype=jnp.uint32)

  # ---------- format conversion ----------

  def to_computational_format(self, a) -> jnp.ndarray:
    def _convert(x: int) -> jnp.ndarray:
      return jnp.array(
          self._int_to_array(x % self.prime, self.chunk_num_u32),
          dtype=jnp.uint32,
      )

    def _recurse(x):
      if isinstance(x, int):
        return _convert(x)
      return jnp.array([_recurse(xi) for xi in x], dtype=jnp.uint32)

    converted_a = _recurse(a)
    if self.use_sharding:
      named_sharding, padded_shape = self.create_named_sharding(
          shape=converted_a.shape, axes=[0]
      )
      converted_a = pad_jax_array(converted_a, padded_shape)
      return converted_a.to_device(named_sharding)
    else:
      return converted_a.to_device(jax.devices()[0])

  def to_original_format(self, a: jnp.ndarray):
    def _convert(arr) -> int:
      return self._array_to_int(arr.tolist()) % self.prime

    def _recurse(x):
      if x.ndim == 1:
        return _convert(x)
      return [_recurse(xi) for xi in x]

    return _recurse(a)

  # ---------- internal JAX helpers ----------

  @staticmethod
  def _conv_1d(a: jax.Array, b: jax.Array) -> jax.Array:
    a = jax.lax.bitcast_convert_type(a, jnp.uint8).reshape(-1)
    b = jax.lax.bitcast_convert_type(b, jnp.uint8).reshape(-1)
    if jax.default_backend() == "gpu":
      # cuDNN's integer conv only supports s8, not u8, so route through
      # float32 on GPU. u8*u8 sums fit exactly in float32's 24-bit mantissa.
      res = jnp.convolve(a.astype(jnp.float32), b.astype(jnp.float32))
      return res.astype(jnp.uint32)
    return jnp.convolve(a, b, preferred_element_type=jnp.uint32)

  def _rechunkify(self, x: jax.Array, n_u16: int, n_u32: int) -> jax.Array:
    """Merge adjacent uint8 coefficients into uint16, then uint32 chunks."""
    shift_u16 = jnp.array([[0, 8]] * n_u16, dtype=jnp.uint8)
    shift_u32 = jnp.array([[0, 16]] * n_u32, dtype=jnp.uint8)
    shape = x.shape[:-1] + (-1, 2) if x.ndim == 2 else (-1, 2)
    x = jnp.sum(jnp.left_shift(x.reshape(shape), shift_u16), axis=-1)
    x = jnp.sum(
        jnp.left_shift(x.reshape(shape).astype(jnp.uint64), shift_u32), axis=-1
    )
    return x

  def _mul_to_u8(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Multiply two chunk arrays; return product as a flat uint8 array."""
    n = self.chunk_num_u32
    batch = a.shape[0]
    res = jax.vmap(self._conv_1d)(a, b)
    res = jnp.pad(res, ((0, 0), (0, 1)))
    res = self._rechunkify(res, 4 * n, 2 * n)
    res = jax.lax.while_loop(_lazy_check_carry, _lazy_carry_propagate, res)
    return jax.lax.bitcast_convert_type(
        res.astype(jnp.uint32), jnp.uint8
    ).reshape(batch, -1)

  def _sub_raw(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Compute a - b assuming a >= b (batched, shape [batch, chunk_num_u32])."""
    n = self.chunk_num_u32
    borrow_low = jnp.array(
        [self.word_mask + 1] * (n - 1) + [0], dtype=jnp.uint64
    )
    borrow_high = jnp.array([0] + [1] * (n - 2) + [0], dtype=jnp.uint64)
    c = jnp.subtract(
        jnp.add(a.astype(jnp.uint64), borrow_low), b.astype(jnp.uint64)
    )
    c = jnp.subtract(c, borrow_high)
    c = jax.lax.while_loop(_lazy_check_carry, _lazy_carry_propagate, c)
    c = c.at[:, n - 1].set(c[:, n - 1] - 1)
    return c.astype(jnp.uint32)

  def _compare(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Return per-batch sign: >=0 means a>=b, <0 means a<b."""
    sign = jnp.sign(a.astype(jnp.int64) - b.astype(jnp.int64))
    weights = jnp.array(
        [2**i for i in range(self.chunk_num_u32)], dtype=jnp.int32
    )
    return jnp.sum(sign * weights, axis=-1)

  # ---------- core modular operations ----------

  def _modular_multiply(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    n8 = self.chunk_num_u8
    batch = a.shape[0]
    vc = self._mul_to_u8(a, b)
    low = vc[:, :n8]
    high = vc[:, n8 : n8 * 2 + 4]
    reduced = jnp.matmul(
        high.astype(jnp.uint16),
        self._lazy_mat_jnp,
        preferred_element_type=jnp.uint32,
    )
    vc2 = jnp.add(low.astype(jnp.uint32), reduced)
    vc2 = self._rechunkify(vc2, n8 // 2, n8 // 4)
    vc2 = jnp.pad(vc2, ((0, 0), (0, 1)))
    vc2 = jax.lax.while_loop(_lazy_check_carry, _lazy_carry_propagate, vc2)
    return vc2.astype(jnp.uint32)[:, : self.chunk_num_u32]

  def _modular_add(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    c = jax.lax.while_loop(
        _lazy_check_carry,
        _lazy_carry_propagate,
        jnp.add(a.astype(jnp.uint64), b.astype(jnp.uint64)),
    )
    return c.astype(jnp.uint32)

  def _modular_negate(self, a: jnp.ndarray) -> jnp.ndarray:
    p = jnp.broadcast_to(self._prime_jnp, a.shape)
    neg = self._sub_raw(p, a)
    is_zero = jnp.all(a == 0, axis=-1, keepdims=True)
    return jnp.where(is_zero, a, neg)

  def _modular_subtract(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return self._modular_add(a, self._modular_negate(b))

  def _modular_reduce(self, a: jnp.ndarray) -> jnp.ndarray:
    p = jnp.broadcast_to(self._prime_jnp, a.shape)
    cond = jnp.greater_equal(self._compare(a, p), 0).reshape(-1, 1)
    return jnp.where(cond, self._sub_raw(a, p), a)

  # ---------- public interface ----------

  def modular_multiply(self, a: jax.Array, b: jax.Array) -> jax.Array:
    kernel_hash = hash_args(a.shape, a.dtype.__str__())
    if self.use_compiled_kernels:
      print(f"using compiled kernel for modular_multiply: {kernel_hash}")
      return self.compiled_kernels[kernel_hash]["modular_multiply"](a, b)
    else:
      return self._modular_multiply(a, b)

  def modular_add(self, a: jax.Array, b: jax.Array) -> jax.Array:
    kernel_hash = hash_args(a.shape, a.dtype.__str__())
    if self.use_compiled_kernels:
      return self.compiled_kernels[kernel_hash]["modular_add"](a, b)
    else:
      return self._modular_add(a, b)

  def modular_subtract(self, a: jax.Array, b: jax.Array) -> jax.Array:
    kernel_hash = hash_args(a.shape, a.dtype.__str__())
    if self.use_compiled_kernels:
      return self.compiled_kernels[kernel_hash]["modular_subtract"](a, b)
    else:
      return self._modular_subtract(a, b)

  def modular_reduce(self, a: jax.Array) -> jax.Array:
    kernel_hash = hash_args(a.shape, a.dtype.__str__())
    if self.use_compiled_kernels:
      return self.compiled_kernels[kernel_hash]["modular_reduce"](a)
    else:
      return self._modular_reduce(a)

  def modular_negate(self, a: jax.Array) -> jax.Array:
    kernel_hash = hash_args(a.shape, a.dtype.__str__())
    if self.use_compiled_kernels:
      return self.compiled_kernels[kernel_hash]["modular_negate"](a)
    else:
      return self._modular_negate(a)

  def context_hash(self) -> str:
    return hash_args(
        self.__class__.__name__,
        self.prime,
        self.chunk_num_u8,
        self.use_sharding,
    )

  def _get_shape_dtype_structs(
      self, parameters: dict
  ) -> list[jax.ShapeDtypeStruct]:
    batch_shape = parameters["batch_shape"]
    operand_shape = batch_shape + (self.chunk_num_u32,)
    if self.use_sharding:
      named_sharding, padded_shape = self.create_named_sharding(
          shape=operand_shape, axes=[0]
      )
      return [
          jax.ShapeDtypeStruct(
              padded_shape, jnp.uint32, sharding=named_sharding
          )
      ]
    return [jax.ShapeDtypeStruct(operand_shape, jnp.uint32)]

  def serialize(self, parameters):  # pytype: disable=signature-mismatch
    shape_dtype_structs = self._get_shape_dtype_structs(parameters)
    kernel_hash = hash_args(self.context_hash(), parameters)
    class_name = self.__class__.__name__

    store_jax_executable(
        self._modular_multiply,
        shape_dtype_structs[0],
        shape_dtype_structs[0],
        name=f"{class_name}_modular_multiply_{kernel_hash}",
    )
    store_jax_executable(
        self._modular_add,
        shape_dtype_structs[0],
        shape_dtype_structs[0],
        name=f"{class_name}_modular_add_{kernel_hash}",
    )
    store_jax_executable(
        self._modular_subtract,
        shape_dtype_structs[0],
        shape_dtype_structs[0],
        name=f"{class_name}_modular_subtract_{kernel_hash}",
    )
    store_jax_executable(
        self._modular_reduce,
        shape_dtype_structs[0],
        name=f"{class_name}_modular_reduce_{kernel_hash}",
    )
    store_jax_executable(
        self._modular_negate,
        shape_dtype_structs[0],
        name=f"{class_name}_modular_negate_{kernel_hash}",
    )

  def compile(self, parameters):  # pytype: disable=signature-mismatch
    shape_dtype_structs = self._get_shape_dtype_structs(parameters)
    kernel_hash = hash_args(self.context_hash(), parameters)
    class_name = self.__class__.__name__

    modular_multiply_kernel = load_jax_executable(
        f"{class_name}_modular_multiply_{kernel_hash}"
    )
    modular_add_kernel = load_jax_executable(
        f"{class_name}_modular_add_{kernel_hash}"
    )
    modular_subtract_kernel = load_jax_executable(
        f"{class_name}_modular_subtract_{kernel_hash}"
    )
    modular_reduce_kernel = load_jax_executable(
        f"{class_name}_modular_reduce_{kernel_hash}"
    )
    modular_negate_kernel = load_jax_executable(
        f"{class_name}_modular_negate_{kernel_hash}"
    )

    if None in [
        modular_multiply_kernel,
        modular_add_kernel,
        modular_subtract_kernel,
        modular_reduce_kernel,
        modular_negate_kernel,
    ]:
      # if not self.use_sharding:
      warnings.warn(
          f"Not found stored serialized compiled kernels, compiling...",
          UserWarning,
          stacklevel=2,
      )

    kernel_hash = hash_args(
        shape_dtype_structs[0].shape, shape_dtype_structs[0].dtype.__str__()
    )

    self.compiled_kernels[kernel_hash] = {
        "modular_multiply": (
            modular_multiply_kernel
            if modular_multiply_kernel is not None
            else jax_jit_lower_compile(
                self._modular_multiply,
                shape_dtype_structs[0],
                shape_dtype_structs[0],
            )
        ),
        "modular_add": (
            modular_add_kernel
            if modular_add_kernel is not None
            else jax_jit_lower_compile(
                self._modular_add,
                shape_dtype_structs[0],
                shape_dtype_structs[0],
            )
        ),
        "modular_subtract": (
            modular_subtract_kernel
            if modular_subtract_kernel is not None
            else jax_jit_lower_compile(
                self._modular_subtract,
                shape_dtype_structs[0],
                shape_dtype_structs[0],
            )
        ),
        "modular_reduce": (
            modular_reduce_kernel
            if modular_reduce_kernel is not None
            else jax_jit_lower_compile(
                self._modular_reduce, shape_dtype_structs[0]
            )
        ),
        "modular_negate": (
            modular_negate_kernel
            if modular_negate_kernel is not None
            else jax_jit_lower_compile(
                self._modular_negate, shape_dtype_structs[0]
            )
        ),
    }
    self.use_compiled_kernels = True
