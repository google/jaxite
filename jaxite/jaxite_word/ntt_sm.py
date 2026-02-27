import concurrent.futures
import jax
import jax.numpy as jnp
import jaxite.jaxite_word.finite_field as ff_context
import jaxite.jaxite_word.util as util
import numpy as np


########################
# Common Functions
########################
def matmul_bat_einsum(lhs: jax.Array, rhs: jax.Array, subscripts: str):
  """Basis Aligned Transformation (BAT) based matrix multiplication

  Args:
      lhs (jax.Array): input
      rhs (jax.Array): twiddle factor matrix
      subscripts (str): einsum subscripts

  Returns:
      jax.Array: result
  """
  # preprocess
  lhs = jax.lax.bitcast_convert_type(lhs, new_dtype=jnp.uint8)
  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)

  # computation
  i8_products = jnp.einsum(
      subscripts, lhs, rhs, preferred_element_type=jnp.uint32
  )
  return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(-1,))


def matmul_conv_flexible_kernel(
    x: jnp.ndarray, y: jnp.ndarray, subscripts: tuple[str, str, str]
) -> jnp.ndarray:
  assert x.dtype == jnp.uint32
  assert y.dtype == jnp.uint32

  lhs: jax.Array = jax.lax.bitcast_convert_type(x, new_dtype=jnp.uint8)  # bnmp
  rhs: jax.Array = jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint8)  # nk1q
  # https://github.com/google/jax/issues/11483
  rhs = jax.lax.rev(rhs, [2])

  if "NVIDIA" in jax.devices()[0].device_kind:
    u8_products = jax.lax.conv_general_dilated(
        lhs.astype(
            jnp.int16
        ),  # NVIDIA GPU does not support uint8 as input type
        rhs.astype(
            jnp.int16
        ),  # NVIDIA GPU does not support uint8 as input type
        window_strides=(1,),
        padding=((3, 3),),
        dimension_numbers=subscripts,
        preferred_element_type=jnp.float32,  # NVIDIA GPU does not support uint32 as output type
    )
  else:
    u8_products = jax.lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides=(1,),
        padding=((3, 3),),
        dimension_numbers=subscripts,
        preferred_element_type=jnp.uint32,
    )

  shift_factors = jnp.array([0, 8, 16, 24, 32, 40, 48], dtype=jnp.uint32)
  return jnp.sum(u8_products.astype(jnp.uint64) << shift_factors, axis=(2,))


########################
# Parameter Generation Functions
########################
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
  # Vectorized modular exponentiation via exponent bit-decomposition
  r_idx = np.arange(rows, dtype=np.int64)[:, None]
  c_idx = np.arange(cols, dtype=np.int64)[None, :]
  exponents = r_idx * c_idx  # shape (rows, cols)
  twiddle_matrix = np.zeros((rows, cols), dtype=int)

  def compute_row(r):
    for c in range(cols):
      twiddle_matrix[r, c] = pow(int(omega), int(exponents[r, c]), int(q))

  with concurrent.futures.ThreadPoolExecutor() as executor:
    list(executor.map(compute_row, range(rows)))
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
  twiddle_matrix_inv = np.zeros((rows, cols), dtype=int)
  for r in range(rows):
    for c in range(cols):
      twiddle_matrix_inv[r, c] = pow(int(omega), int(-r * c), int(q))
  return twiddle_matrix_inv


########################
# NTT Context with different modular reduction methods
########################
class NTTContextBase:
  """Base class for NTT Context with different modular reduction methods

  This class implements the numpy version of three-step NTT algorithm.
  Args:
      moduli: The modulus.
      transform_length: The transform length.
      parameters: The parameters.

  Returns:
      The NTT Context.
  """

  def __init__(self, moduli: int, parameters: dict):
    self.ff_ctx = parameters.get("finite_field_context", ff_context.BarrettContext(moduli))
    self.num_bytes = 4

    self.moduli = moduli
    self.parameters = parameters
    assert self.moduli < 2**31, "moduli must be less than 2**32"
    self.r = parameters.get("r", 0)
    self.c = parameters.get("c", 0)
    assert self.r != 0, "r must be non-zero"
    assert self.c != 0, "c must be non-zero"
    self.transform_length = self.r * self.c
    self.psi = util.root_of_unity(2 * self.transform_length, self.moduli)
    self.omega = (self.psi**2) % self.moduli

    self.ntt_tf_step1, self.ntt_tf_step2, self.ntt_tf_step3 = (
        self.ntt_coefficients_precompute()
    )
    self.intt_tf_step1, self.intt_tf_step2, self.intt_tf_step3 = (
        self.intt_coefficients_precompute()
    )

    self.memory_aligned_transformation()
    self.ntt_tf_bat_mat_comp_step1 = self.basis_aligned_transformation(
        self.to_computation_format(self.ntt_tf_mat_step1)
    )
    self.ntt_tf_mat_comp_step2 = self.to_computation_format(
        self.ntt_tf_mat_step2
    ).astype(jnp.uint64)
    self.ntt_tf_bat_mat_comp_step3 = self.basis_aligned_transformation(
        self.to_computation_format(self.ntt_tf_mat_step3)
    )
    self.intt_tf_bat_mat_comp_step1 = self.basis_aligned_transformation(
        self.to_computation_format(self.intt_tf_mat_step1)
    )
    self.intt_tf_mat_comp_step2 = self.to_computation_format(
        self.intt_tf_mat_step2
    ).astype(jnp.uint64)
    self.intt_tf_bat_mat_comp_step3 = self.basis_aligned_transformation(
        self.to_computation_format(self.intt_tf_mat_step3)
    )

  ########################
  # Offline Functions
  ########################
  def ntt_coefficients_precompute(self):
    omega_col = pow(self.omega, self.c, self.moduli)
    omega_row = pow(self.omega, self.r, self.moduli)
    tf_step1 = gen_twiddle_matrix(self.r, self.r, self.moduli, omega_col)
    tf_step2 = gen_twiddle_matrix(self.r, self.c, self.moduli, self.omega)
    tf_step3 = gen_twiddle_matrix(self.c, self.c, self.moduli, omega_row)
    return tf_step1, tf_step2, tf_step3

  def intt_coefficients_precompute(self):
    omega_col = pow(self.omega, self.c, self.moduli)
    omega_row = pow(self.omega, self.r, self.moduli)
    inv_omega_col = pow(omega_col, -1, self.moduli)
    inv_omega_row = pow(omega_row, -1, self.moduli)
    intt_tf_step1 = gen_twiddle_matrix(
        self.c, self.c, self.moduli, inv_omega_row
    )
    intt_tf_step2 = gen_twiddle_matrix_inv(
        self.r, self.c, self.moduli, self.omega
    )
    # Precompute col_inv * step2 to merge the two multiplication steps in intt
    col_inv = pow(self.c, -1, self.moduli)
    row_inv = pow(self.r, -1, self.moduli)
    intt_tf_step2 = (intt_tf_step2 * col_inv) % self.moduli
    intt_tf_step3 = gen_twiddle_matrix(
        self.r, self.r, self.moduli, inv_omega_col
    )
    intt_tf_step3 = (intt_tf_step3 * row_inv) % self.moduli
    return intt_tf_step1, intt_tf_step2, intt_tf_step3

  def to_computation_format(self, a: np.ndarray):
    return self.ff_ctx.to_computation_format(a)

  def to_original_format(self, a: np.ndarray):
    return self.ff_ctx.to_original_format(a)

  def basis_aligned_transformation(self, matrix: np.ndarray):
    n_row, n_col = matrix.shape  # might not be the same as self.r and self.c
    matrix_u64 = matrix.astype(np.uint64)
    matrix_u64_byteshifted = np.array(
        [matrix_u64 << (8 * byte_idx) for byte_idx in range(self.num_bytes)],
        dtype=np.uint64,
    )
    # shape is (4, rows, cols)
    matrix_u64_byteshifted_mod_modulus = (
        matrix_u64_byteshifted % self.moduli
    ).astype(np.uint32)
    # shape is (4, rows, cols, bytes=4)
    matrix_u8 = jax.lax.bitcast_convert_type(
        matrix_u64_byteshifted_mod_modulus, jnp.uint8
    ).transpose(1, 0, 2, 3)
    return matrix_u8

  def memory_aligned_transformation(self):
    """Memory Aligned Transformation (MAT)

    Must run after gen_twiddle_matrix()
    """

    def get_bit_reverse_perm(n):
      """Generates a list of indices for bit-reversal permutation of size n."""
      if n <= 0:
        return []
      bits = n.bit_length() - 1
      perm = [0] * n
      for i in range(n):
        # Reverse bits of i
        r = 0
        temp = i
        for _ in range(bits):
          r = (r << 1) | (temp & 1)
          temp >>= 1
        perm[i] = r
      return perm

    perm_r = get_bit_reverse_perm(self.r)
    perm_c = get_bit_reverse_perm(self.c)
    self.ntt_tf_mat_step1 = self.ntt_tf_step1[perm_r, :]
    self.ntt_tf_mat_step2 = self.ntt_tf_step2[perm_r, :]
    self.ntt_tf_mat_step3 = self.ntt_tf_step3[:, perm_c]
    self.intt_tf_mat_step1 = self.intt_tf_step1[perm_c, :]
    self.intt_tf_mat_step2 = self.intt_tf_step2[perm_r, :]
    self.intt_tf_mat_step3 = self.intt_tf_step3[:, perm_r]

  def ntt_three_step_reference(self, x):
    """3-step NTT algorithm reference implementation

    Args:
        x: The input vector.

    Returns:
        The NTT result.
    """
    assert (
        len(x) == self.transform_length
    ), "x must have length transform_length"
    twist_factor = self.twist_factor
    tf_step1 = self.ntt_tf_step1.astype(np.uint64)
    step2 = self.ntt_tf_step2.astype(np.uint64)
    tf_step3 = self.ntt_tf_step3.astype(np.uint64)
    x = np.array(x, dtype=np.uint64)

    x_twisted = np.mod(x * twist_factor, self.moduli)
    x_matrix = x_twisted.reshape((self.r, self.c))
    y = np.mod(np.matmul(tf_step1, x_matrix), self.moduli)
    y = np.mod(y * step2, self.moduli)
    z = np.mod(np.matmul(y, tf_step3), self.moduli)
    x = z.flatten()
    return x.tolist()

  def intt_three_step_reference(self, x):
    """3-step Inverse NTT algorithm reference implementation

    Args:
        x: The input vector.

    Returns:
        The Inverse NTT result.
    """
    assert (
        len(x) == self.transform_length
    ), "x must have length transform_length"
    tf_step1 = self.intt_tf_step1.astype(np.uint64)
    step2 = self.intt_tf_step2.astype(np.uint64)
    tf_step3 = self.intt_tf_step3.astype(np.uint64)
    x = np.array(x, dtype=np.uint64)

    z = x.reshape((self.c, self.r))
    y = np.mod(np.matmul(z, tf_step1), self.moduli)
    y = np.mod(y * step2, self.moduli)  # step2 includes col_inv
    a = np.mod(np.matmul(tf_step3, y), self.moduli)  # tf_step3 includes row_inv
    x_recovered = np.array(a).flatten()
    x = np.mod(x_recovered * self.untwist_factor, self.moduli)

    return x.tolist()

  ########################
  # Online Functions
  ########################
  def ntt(self, v: jax.Array):
    """NTT with modular u32

    B = Batch size, R = self.r, C = self.c
    Q = 4 (number of bytes per element)

    Args:
        v: - is u32 array of shape (B, R, C) - will be casted into u8 array of
          shape (B, R, C, Q)
        ntt_bat_tf_step1: - is u8 array of shape (R, 4, R, 4)
        ntt_tf_step2: - is u32 array of shape (R, C)
        ntt_bat_tf_step3: - is u8 array of shape (C, 4, C, 4)

    Returns:
        - is u32 array of shape (B, R, C)
        - output
    """
    result_step1 = matmul_bat_einsum(
        v, self.ntt_tf_bat_mat_comp_step1, "brcq,zqrp->bzcp"
    )
    result_step1_reduced = self.ff_ctx.modular_reduction(result_step1)
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.ntt_tf_mat_comp_step2
    )
    result_step2_reduced = self.ff_ctx.modular_reduction(result_step2)
    result_step3 = matmul_bat_einsum(
        result_step2_reduced, self.ntt_tf_bat_mat_comp_step3, "brcq,cqnp->brnp"
    )
    result_step3_reduced = self.ff_ctx.modular_reduction(result_step3)
    return result_step3_reduced

  def intt(self, v: jax.Array):
    """INTT with modular u32

    B = Batch size, R = self.r, C = self.c
    Q = 4 (number of bytes per element)

    Args:
        v: - is u32 array of shape (B, R, C) - will be casted into u8 array of
          shape (B, R, C, Q)
        intt_bat_tf_step1: - is u8 array of shape (C, 4, C, 4)
        intt_tf_step2: - is u32 array of shape (R, C)
        intt_bat_tf_step3: - is u8 array of shape (R, 4, R, 4)

    Returns:
        - is u32 array of shape (B, R, C)
        - output
    """
    result_step1 = matmul_bat_einsum(
        v, self.intt_tf_bat_mat_comp_step1, "brcq,cqlp->brlp"
    )
    result_step1_reduced = self.ff_ctx.modular_reduction(result_step1)
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.intt_tf_mat_comp_step2
    )
    result_step2_reduced = self.ff_ctx.modular_reduction(result_step2)
    result_step3 = matmul_bat_einsum(
        result_step2_reduced, self.intt_tf_bat_mat_comp_step3, "brcq,lqrp->blcp"
    )
    result_step3_reduced = self.ff_ctx.modular_reduction(result_step3)
    return result_step3_reduced


class NTTBarrettContext(NTTContextBase):

  def __init__(self, moduli: int, parameters: dict):
    super().__init__(moduli, parameters)
    if type(self.moduli) is int:
      self.moduli = [self.moduli]
    assert self.ff_ctx is not None, "finite_field_context must be provided"
    assert (
        self.moduli == self.ff_ctx.moduli
    ), "moduli must be the same as the moduli of the finite_field_context"


class NTTMontgomeryContext(NTTContextBase):

  def __init__(self, moduli: int, parameters: dict):
    super().__init__(moduli, parameters)
    if type(self.moduli) is int:
      self.moduli = [self.moduli]
    assert self.ff_ctx is not None, "finite_field_context must be provided"
    assert (
        self.moduli == self.ff_ctx.moduli
    ), "moduli must be the same as the moduli of the finite_field_context"


class NTTBATLazyContext(NTTContextBase):

  def __init__(self, moduli: int, parameters: dict):
    super().__init__(moduli, parameters)
    if type(self.moduli) is int:
      self.moduli = [self.moduli]
    assert self.ff_ctx is not None, "finite_field_context must be provided"
    assert (
        self.moduli == self.ff_ctx.moduli
    ), "moduli must be the same as the moduli of the finite_field_context"
    self.ff_ctx_full = ff_context.BarrettContext(moduli)

  ########################
  # Online Functions
  ########################
  def ntt(self, v: jax.Array):
    """NTT with modular u32

    B = Batch size, R = self.r, C = self.c
    Q = 4 (number of bytes per element)

    Args:
        v: - is u32 array of shape (B, R, C) - will be casted into u8 array of
          shape (B, R, C, Q)
        ntt_bat_tf_mat_comp_step1: - is u8 array of shape (R, 4, R, 4)
        ntt_tf_mat_comp_step2: - is u32 array of shape (R, C)
        ntt_bat_tf_mat_comp_step3: - is u8 array of shape (C, 4, C, 4)

    Returns:
        - is u32 array of shape (B, R, C)
        - output
    """
    result_step1 = matmul_bat_einsum(
        v, self.ntt_tf_bat_mat_comp_step1, "brcq,zqrp->bzcp"
    )
    result_step1_reduced = self.ff_ctx.modular_reduction(result_step1)
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.ntt_tf_mat_comp_step2
    )
    result_step2_reduced = self.ff_ctx_full.modular_reduction(result_step2)
    result_step3 = matmul_bat_einsum(
        result_step2_reduced, self.ntt_tf_bat_mat_comp_step3, "brcq,cqnp->brnp"
    )
    result_step3_reduced = self.ff_ctx_full.modular_reduction(result_step3)
    return result_step3_reduced

  def intt(self, v: jax.Array):
    """INTT with modular u32

    B = Batch size, R = self.r, C = self.c
    Q = 4 (number of bytes per element)

    Args:
        v: - is u32 array of shape (B, C, R) - will be casted into u8 array of
          shape (B, C, R, Q)
        intt_bat_tf_mat_comp_step1: - is u8 array of shape (C, 4, C, 4)
        intt_tf_mat_comp_step2: - is u32 array of shape (R, C) # Step 1
          multiplication changes its order
        intt_bat_tf_mat_comp_step3: - is u8 array of shape (R, 4, R, 4)

    Returns:
        - is u32 array of shape (B, R, C)
        - output
    """
    result_step1 = matmul_bat_einsum(
        v, self.intt_tf_bat_mat_comp_step1, "brcq,cqlp->brlp"
    )
    result_step1_reduced = self.ff_ctx.modular_reduction(result_step1)
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.intt_tf_mat_comp_step2
    )
    result_step2_reduced = self.ff_ctx_full.modular_reduction(result_step2)
    result_step3 = matmul_bat_einsum(
        result_step2_reduced, self.intt_tf_bat_mat_comp_step3, "brcq,lqrp->blcp"
    )
    result_step3_reduced = self.ff_ctx_full.modular_reduction(result_step3)
    return result_step3_reduced


class NTTShoupContext(NTTContextBase):
  """NTT with Shoup's Modular Reduction

  Note that Shoup's Reduction is NOT compatible with Basis Aligned
  Transformation (BAT).
  We use 1-d convolution to perform matrix multiplication for Shoup.
  """

  def __init__(self, moduli: int, parameters: dict):
    super().__init__(moduli, parameters)
    if type(self.moduli) is int:
      self.moduli = [self.moduli]
    assert self.ff_ctx is not None, "finite_field_context must be provided"
    assert (
        self.moduli == self.ff_ctx.moduli
    ), "moduli must be the same as the moduli of the finite_field_context"
    self.ntt_tf_mat_step1 = self.to_computation_format(
        self.ntt_tf_mat_step1
    ).astype(jnp.uint32)
    self.ntt_tf_mat_step2 = self.to_computation_format(
        self.ntt_tf_mat_step2
    ).astype(jnp.uint64)
    self.ntt_tf_mat_step3 = self.to_computation_format(
        self.ntt_tf_mat_step3
    ).astype(jnp.uint32)
    self.intt_tf_mat_step1 = self.to_computation_format(
        self.intt_tf_mat_step1
    ).astype(jnp.uint32)
    self.intt_tf_mat_step2 = self.to_computation_format(
        self.intt_tf_mat_step2
    ).astype(jnp.uint64)
    self.intt_tf_mat_step3 = self.to_computation_format(
        self.intt_tf_mat_step3
    ).astype(jnp.uint32)

    self.ntt_tf_mat_step1_shoup = self.to_shoup_computation_format(
        self.ntt_tf_mat_step1
    ).astype(jnp.uint32)
    self.ntt_tf_mat_step2_shoup = self.to_shoup_computation_format(
        self.ntt_tf_mat_step2
    ).astype(jnp.uint64)
    self.ntt_tf_mat_step3_shoup = self.to_shoup_computation_format(
        self.ntt_tf_mat_step3
    ).astype(jnp.uint32)
    self.intt_tf_mat_step1_shoup = self.to_shoup_computation_format(
        self.intt_tf_mat_step1
    ).astype(jnp.uint32)
    self.intt_tf_mat_step2_shoup = self.to_shoup_computation_format(
        self.intt_tf_mat_step2
    ).astype(jnp.uint64)
    self.intt_tf_mat_step3_shoup = self.to_shoup_computation_format(
        self.intt_tf_mat_step3
    ).astype(jnp.uint32)

  def to_shoup_computation_format(self, a: np.ndarray):
    shape = a.shape
    a = a.flatten()
    a_list = a.tolist()
    a_computation_format = [
        self.ff_ctx.precompute_constant_operand(a_i) for a_i in a_list
    ]
    a_computation_format = np.array(a_computation_format, dtype=np.uint64)
    a_computation_format = a_computation_format.reshape(*shape)
    return a_computation_format

  def ntt(self, v: jax.Array):
    """NTT with modular u32

    Args:
        v: - is u32 array of shape (B, R, C) - input

    Returns:
        - is u32 array of shape (B, R, C)
        - output
    """
    result_step1 = matmul_conv_flexible_kernel(
        self.ntt_tf_mat_step1, v, ("NCW", "IOW", "NCW")
    )
    result_step1_shoup = matmul_conv_flexible_kernel(
        self.ntt_tf_mat_step1_shoup, v, ("NCW", "IOW", "NCW")
    )
    result_step1_reduced = self.ff_ctx.modular_reduction(
        result_step1, result_step1_shoup
    )
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.ntt_tf_mat_step2
    )
    result_step2_shoup = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.ntt_tf_mat_step2_shoup
    )
    result_step2_reduced = self.ff_ctx.modular_reduction(
        result_step2, result_step2_shoup
    )
    result_step3 = matmul_conv_flexible_kernel(
        result_step2_reduced, self.ntt_tf_mat_step3, ("NCW", "IOW", "CNW")
    )
    result_step3_shoup = matmul_conv_flexible_kernel(
        result_step2_reduced, self.ntt_tf_mat_step3_shoup, ("NCW", "IOW", "CNW")
    )
    result_step3_reduced = self.ff_ctx.modular_reduction(
        result_step3, result_step3_shoup
    )
    result_step3_reduced = result_step3_reduced.T
    return result_step3_reduced.astype(jnp.uint32)

  def intt(self, v: jax.Array):
    """INTT with modular u32

    Args:
        v: - is u32 array of shape (B, R, C) - input

    Returns:
        - is u32 array of shape (B, R, C)
        - output
    """
    # computation
    v = v.T
    result_step1 = matmul_conv_flexible_kernel(
        v, self.intt_tf_mat_step1, ("CNW", "IOW", "NCW")
    )
    result_step1_shoup = matmul_conv_flexible_kernel(
        v, self.intt_tf_mat_step1_shoup, ("CNW", "IOW", "NCW")
    )
    result_step1_reduced = self.ff_ctx.modular_reduction(
        result_step1, result_step1_shoup
    )
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.intt_tf_mat_step2
    )
    result_step2_shoup = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.intt_tf_mat_step2_shoup
    )
    result_step2_reduced = self.ff_ctx.modular_reduction(
        result_step2, result_step2_shoup
    )
    result_step3 = matmul_conv_flexible_kernel(
        self.intt_tf_mat_step3, result_step2_reduced, ("NCW", "IOW", "NCW")
    )
    result_step3_shoup = matmul_conv_flexible_kernel(
        self.intt_tf_mat_step3_shoup,
        result_step2_reduced,
        ("NCW", "IOW", "NCW"),
    )
    result_step3_reduced = self.ff_ctx.modular_reduction(
        result_step3, result_step3_shoup
    )
    return result_step3_reduced
