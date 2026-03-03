"""This script is specifically designed for NTT/INTT used for ciphertext.

Main difference to ntt_o.py is this script supports (1) multiple moduli (2)
multiple batch (3) distributed sharding
"""

import concurrent.futures
from typing import List, Union
import jaxite.jaxite_word.finite_field as ff_context
import jax
import jax.numpy as jnp
import numpy as np
import jaxite.jaxite_word.util as util

def _is_nvidia():
  return "NVIDIA" in jax.devices()[0].device_kind


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

  if _is_nvidia():
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
class NTTCiphertextContextBase:
  """Base class for NTT Context with different modular reduction methods

  This class implements the numpy version of three-step NTT algorithm.
  Args:
      moduli: The modulus.
      transform_length: The transform length.
      parameters: The parameters.

  Returns:
      The NTT Context.
  """

  def __init__(
      self, moduli: Union[int, List[int]], parameters: dict, perf_test=False
  ):
    self.ff_ctx = parameters.get("finite_field_context", None)
    self.num_bytes = 4

    if type(moduli) == int:
      moduli = [moduli]
    self.moduli = moduli
    self.parameters = parameters
    assert all(q < 2**31 for q in moduli), "moduli must be less than 2**32"
    self.r = parameters.get("r", 0)
    self.c = parameters.get("c", 0)
    assert self.r != 0, "r must be non-zero"
    assert self.c != 0, "c must be non-zero"
    self.transform_length = self.r * self.c
    self.psi_list = [
        util.root_of_unity(2 * self.transform_length, q) for q in self.moduli
    ]
    self.omega_list = [
        (psi**2) % q for psi, q in zip(self.psi_list, self.moduli)
    ]
    if perf_test:
      # Use random data for performance testing to avoid expensive precomputation
      key = jax.random.PRNGKey(0)
      self.ntt_bat_tf_step1 = jax.random.bits(
          key, (self.r, 4, self.r, 4, len(moduli)), dtype=jnp.uint8
      )
      self.ntt_tf_step2 = jax.random.bits(
          key, (self.r, self.c, len(moduli)), dtype=jnp.uint64
      )
      self.ntt_bat_tf_step3 = jax.random.bits(
          key, (self.c, 4, self.c, 4, len(moduli)), dtype=jnp.uint8
      )
      self.intt_bat_tf_step1 = jax.random.bits(
          key, (self.c, 4, self.c, 4, len(moduli)), dtype=jnp.uint8
      )
      self.intt_tf_step2 = jax.random.bits(
          key, (self.r, self.c, len(moduli)), dtype=jnp.uint64
      )
      self.intt_bat_tf_step3 = jax.random.bits(
          key, (self.r, 4, self.r, 4, len(moduli)), dtype=jnp.uint8
      )

      if isinstance(self, NTTCiphertextShoupContext):
        self.ntt_tf_step1 = jax.random.bits(
            key, (self.r, self.r, len(moduli)), dtype=jnp.uint32
        )
        self.ntt_tf_step3 = jax.random.bits(
            key, (self.c, self.c, len(moduli)), dtype=jnp.uint32
        )
        self.intt_tf_step1 = jax.random.bits(
            key, (self.c, self.c, len(moduli)), dtype=jnp.uint32
        )
        self.intt_tf_step3 = jax.random.bits(
            key, (self.r, self.r, len(moduli)), dtype=jnp.uint32
        )

        self.ntt_tf_step1_shoup = jax.random.bits(
            key, (self.r, self.r, len(moduli)), dtype=jnp.uint32
        )
        self.ntt_tf_step2_shoup = jax.random.bits(
            key, (self.r, self.c, len(moduli)), dtype=jnp.uint64
        )
        self.ntt_tf_step3_shoup = jax.random.bits(
            key, (self.c, self.c, len(moduli)), dtype=jnp.uint32
        )
        self.intt_tf_step1_shoup = jax.random.bits(
            key, (self.c, self.c, len(moduli)), dtype=jnp.uint32
        )
        self.intt_tf_step2_shoup = jax.random.bits(
            key, (self.r, self.c, len(moduli)), dtype=jnp.uint64
        )
        self.intt_tf_step3_shoup = jax.random.bits(
            key, (self.r, self.r, len(moduli)), dtype=jnp.uint32
        )
    else:
      self.memory_aligned_transformation()
      self.ntt_tf_step1, self.ntt_tf_step2, self.ntt_tf_step3 = (
          self.ntt_coefficients_precompute()
      )
      self.intt_tf_step1, self.intt_tf_step2, self.intt_tf_step3 = (
          self.intt_coefficients_precompute()
      )
      self.ntt_bat_tf_step1 = self.basis_aligned_transformation(
          self.to_computation_format(self.ntt_tf_step1)
      )
      self.ntt_tf_step2 = self.to_computation_format(self.ntt_tf_step2).astype(
          jnp.uint64
      )
      self.ntt_bat_tf_step3 = self.basis_aligned_transformation(
          self.to_computation_format(self.ntt_tf_step3)
      )
      self.intt_bat_tf_step1 = self.basis_aligned_transformation(
          self.to_computation_format(self.intt_tf_step1)
      )
      self.intt_tf_step2 = self.to_computation_format(
          self.intt_tf_step2
      ).astype(jnp.uint64)
      self.intt_bat_tf_step3 = self.basis_aligned_transformation(
          self.to_computation_format(self.intt_tf_step3)
      )

  ########################
  # Offline Functions
  ########################
  def ntt_coefficients_precompute(self):
    """R = self.r, C = self.c, M = len(self.moduli)

    - ntt_tf_step1: shape (R, R, M), u32
    - ntt_tf_step2: shape (R, C, M), u32
    - ntt_tf_step3: shape (C, C, M), u32
    """
    tf_step1_list, tf_step2_list, tf_step3_list = [], [], []
    for i, modulus in enumerate(self.moduli):
      omega_col = pow(self.omega_list[i], self.c, modulus)
      omega_row = pow(self.omega_list[i], self.r, modulus)
      tf_step1_one_modulus = gen_twiddle_matrix(
          self.r, self.r, modulus, omega_col
      )
      tf_step2_one_modulus = gen_twiddle_matrix(
          self.r, self.c, modulus, self.omega_list[i]
      )
      tf_step3_one_modulus = gen_twiddle_matrix(
          self.c, self.c, modulus, omega_row
      )
      tf_step1_one_modulus = tf_step1_one_modulus[
          self.perm_r, :
      ]  # Memory Aligned Transformation
      tf_step2_one_modulus = tf_step2_one_modulus[
          self.perm_r, :
      ]  # Memory Aligned Transformation
      tf_step3_one_modulus = tf_step3_one_modulus[
          :, self.perm_c
      ]  # Memory Aligned Transformation
      tf_step1_list.append(tf_step1_one_modulus)
      tf_step2_list.append(tf_step2_one_modulus)
      tf_step3_list.append(tf_step3_one_modulus)
    tf_step1 = jnp.array(tf_step1_list, dtype=jnp.uint32).transpose(
        1, 2, 0
    )  # Make moduli the last dimension
    tf_step2 = jnp.array(tf_step2_list, dtype=jnp.uint32).transpose(
        1, 2, 0
    )  # Make moduli the last dimension
    tf_step3 = jnp.array(tf_step3_list, dtype=jnp.uint32).transpose(
        1, 2, 0
    )  # Make moduli the last dimension
    return tf_step1, tf_step2, tf_step3

  def intt_coefficients_precompute(self):
    """R = self.r, C = self.c, M = len(self.moduli)

    - intt_tf_step1: shape (C, C, M), u32
    - intt_tf_step2: shape (R, C, M), u32
    - intt_tf_step3: shape (R, R, M), u32
    """
    intt_tf_step1_list, intt_tf_step2_list, intt_tf_step3_list = [], [], []
    for i, modulus in enumerate(self.moduli):
      omega_col = pow(self.omega_list[i], self.c, modulus)
      omega_row = pow(self.omega_list[i], self.r, modulus)
      inv_omega_col = pow(omega_col, -1, modulus)
      inv_omega_row = pow(omega_row, -1, modulus)
      intt_tf_step1_one_modulus = gen_twiddle_matrix(
          self.c, self.c, modulus, inv_omega_row
      )
      intt_tf_step2_one_modulus = gen_twiddle_matrix_inv(
          self.r, self.c, modulus, self.omega_list[i]
      )
      intt_tf_step3_one_modulus = gen_twiddle_matrix(
          self.r, self.r, modulus, inv_omega_col
      )
      intt_tf_step1_one_modulus = intt_tf_step1_one_modulus[
          self.perm_c, :
      ]  # Memory Aligned Transformation
      intt_tf_step2_one_modulus = intt_tf_step2_one_modulus[
          self.perm_r, :
      ]  # Memory Aligned Transformation
      intt_tf_step3_one_modulus = intt_tf_step3_one_modulus[
          :, self.perm_r
      ]  # Memory Aligned Transformation
      col_inv = pow(self.c, -1, modulus)
      row_inv = pow(self.r, -1, modulus)
      intt_tf_step2_one_modulus = (
          intt_tf_step2_one_modulus * col_inv
      ) % modulus
      intt_tf_step3_one_modulus = (
          intt_tf_step3_one_modulus * row_inv
      ) % modulus
      intt_tf_step1_list.append(intt_tf_step1_one_modulus)
      intt_tf_step2_list.append(intt_tf_step2_one_modulus)
      intt_tf_step3_list.append(intt_tf_step3_one_modulus)
    intt_tf_step1 = jnp.array(intt_tf_step1_list, dtype=jnp.uint32).transpose(
        1, 2, 0
    )  # Make moduli the last dimension
    intt_tf_step2 = jnp.array(intt_tf_step2_list, dtype=jnp.uint32).transpose(
        1, 2, 0
    )  # Make moduli the last dimension
    intt_tf_step3 = jnp.array(intt_tf_step3_list, dtype=jnp.uint32).transpose(
        1, 2, 0
    )  # Make moduli the last dimension
    return intt_tf_step1, intt_tf_step2, intt_tf_step3

  def to_computation_format(self, a: Union[np.ndarray, jax.Array]):
    assert self.ff_ctx is not None
    return self.ff_ctx.to_computation_format(a.astype(jnp.uint64)).astype(
        jnp.uint32
    )

  def to_original_format(self, a: Union[np.ndarray, jax.Array]):
    assert self.ff_ctx is not None
    return self.ff_ctx.to_original_format(a.astype(jnp.uint64)).astype(
        jnp.uint32
    )

  def basis_aligned_transformation(self, matrix: np.ndarray):
    matrix_u64 = matrix.astype(np.uint64)
    matrix_u64_byteshifted = np.array(
        [matrix_u64 << (8 * byte_idx) for byte_idx in range(self.num_bytes)],
        dtype=np.uint64,
    )
    # shape is (4, rows, cols, moduli)
    matrix_u64_byteshifted_mod_modulus = (
        matrix_u64_byteshifted % jnp.array(self.moduli, dtype=np.uint64)
    ).astype(np.uint32)
    # shape is (4, rows, cols, moduli, bytes=4)
    matrix_u8 = jax.lax.bitcast_convert_type(
        matrix_u64_byteshifted_mod_modulus, jnp.uint8
    ).transpose(1, 0, 2, 4, 3)
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

    self.perm_r = get_bit_reverse_perm(self.r)
    self.perm_c = get_bit_reverse_perm(self.c)

  def get_jax_parameters(self):
    assert self.ff_ctx is not None
    return {
        "ntt_bat_tf_step1": util.to_tuple(self.ntt_bat_tf_step1),
        "ntt_tf_step2": util.to_tuple(self.ntt_tf_step2),
        "ntt_bat_tf_step3": util.to_tuple(self.ntt_bat_tf_step3),
        "intt_bat_tf_step1": util.to_tuple(self.intt_bat_tf_step1),
        "intt_tf_step2": util.to_tuple(self.intt_tf_step2),
        "intt_bat_tf_step3": util.to_tuple(self.intt_bat_tf_step3),
        "finite_field_parameters": self.ff_ctx.get_jax_parameters(),
        "rows": self.r,
        "cols": self.c,
    }

  ########################
  # Online Functions
  ########################
  def ntt_limb(self, v: jax.Array, limb_index: int):
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
        v, self.ntt_bat_tf_step1[..., limb_index], "brcq,zqrp->bzcp"
    )
    assert self.ff_ctx is not None
    result_step1_reduced = self.ff_ctx.modular_reduction_single_modulus(
        result_step1, limb_index
    )
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64),
        self.ntt_tf_step2[..., limb_index],
    )
    result_step2_reduced = self.ff_ctx.modular_reduction_single_modulus(
        result_step2, limb_index
    )
    result_step3 = matmul_bat_einsum(
        result_step2_reduced,
        self.ntt_bat_tf_step3[..., limb_index],
        "brcq,cqnp->brnp",
    )
    result_step3_reduced = self.ff_ctx.modular_reduction_single_modulus(
        result_step3, limb_index
    )
    return result_step3_reduced

  def intt_limb(self, v: jax.Array, limb_index: int):
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
        v, self.intt_bat_tf_step1[..., limb_index], "brcq,cqlp->brlp"
    )
    assert self.ff_ctx is not None
    result_step1_reduced = self.ff_ctx.modular_reduction_single_modulus(
        result_step1, limb_index
    )
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64),
        self.intt_tf_step2[..., limb_index],
    )
    result_step2_reduced = self.ff_ctx.modular_reduction_single_modulus(
        result_step2, limb_index
    )
    result_step3 = matmul_bat_einsum(
        result_step2_reduced,
        self.intt_bat_tf_step3[..., limb_index],
        "brcq,lqrp->blcp",
    )
    result_step3_reduced = self.ff_ctx.modular_reduction_single_modulus(
        result_step3, limb_index
    )
    return result_step3_reduced

  def ntt(self, v: jax.Array):
    """NTT with modular u32

    B = Batch size, R = self.r, C = self.c
    Q = 4 (number of bytes per element)
    M = len(self.moduli)

    Args:
        v: - is u32 array of shape (B, R, C, M) - will be casted into u8 array
          of shape (B, R, C, M, Q)
        ntt_bat_tf_step1: - is u8 array of shape (R, 4, R, 4, M)
        ntt_tf_step2: - is u32 array of shape (R, C, M)
        ntt_bat_tf_step3: - is u8 array of shape (C, 4, C, 4, M)

    Returns:
        - is u32 array of shape (B, R, C, M)
        - output
    """
    result_step1 = matmul_bat_einsum(
        v, self.ntt_bat_tf_step1, "brcmq,zqrpm->bzcmp"
    )  # "mqkp,bknq->bmnp"; "bkncq,mqkpc->bmncp"
    assert self.ff_ctx is not None
    result_step1_reduced = self.ff_ctx.modular_reduction(result_step1)
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.ntt_tf_step2
    )
    result_step2_reduced = self.ff_ctx.modular_reduction(result_step2)
    result_step3 = matmul_bat_einsum(
        result_step2_reduced, self.ntt_bat_tf_step3, "brcmq,cqnpm->brnmp"
    )  # "bmkq,kqnp->bnmp" "bmkcq,kqnpc->bnmcp"
    result_step3_reduced = self.ff_ctx.modular_reduction(result_step3)
    return result_step3_reduced

  def intt(self, v: jax.Array):
    """INTT with modular u32

    B = Batch size, R = self.r, C = self.c
    Q = 4 (number of bytes per element)
    M = len(self.moduli)

    Args:
        v: - is u32 array of shape (B, R, C, M) - will be casted into u8 array
          of shape (B, R, C, M, Q)
        intt_bat_tf_step1: - is u8 array of shape (C, 4, C, 4, M)
        intt_tf_step2: - is u32 array of shape (R, C, M)
        intt_bat_tf_step3: - is u8 array of shape (R, 4, R, 4, M)

    Returns:
        - is u32 array of shape (B, R, C, M)
        - output
    """
    result_step1 = matmul_bat_einsum(
        v, self.intt_bat_tf_step1, "brcmq,cqlpm->brlmp"
    )
    assert self.ff_ctx is not None
    result_step1_reduced = self.ff_ctx.modular_reduction(result_step1)
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.intt_tf_step2
    )
    result_step2_reduced = self.ff_ctx.modular_reduction(result_step2)
    result_step3 = matmul_bat_einsum(
        result_step2_reduced, self.intt_bat_tf_step3, "brcmq,lqrpm->blcmp"
    )
    result_step3_reduced = self.ff_ctx.modular_reduction(result_step3)
    return result_step3_reduced

  ########################
  # Modulus Dropping Functions
  ########################
  def drop_last_modulus(self):
    self.ntt_bat_tf_step1 = self.ntt_bat_tf_step1[..., :-1]
    self.ntt_tf_step2 = self.ntt_tf_step2[..., :-1]
    self.ntt_bat_tf_step3 = self.ntt_bat_tf_step3[..., :-1]
    self.intt_bat_tf_step1 = self.intt_bat_tf_step1[..., :-1]
    self.intt_tf_step2 = self.intt_tf_step2[..., :-1]
    self.intt_bat_tf_step3 = self.intt_bat_tf_step3[..., :-1]
    assert self.ff_ctx is not None
    self.ff_ctx.drop_last_modulus()


class NTTCiphertextBarrettContext(NTTCiphertextContextBase):

  def __init__(self, moduli: int, parameters: dict, perf_test=False):
    super().__init__(moduli, parameters, perf_test=perf_test)
    if type(self.moduli) is int:
      self.moduli = [self.moduli]
    if self.ff_ctx is None:
      self.ff_ctx = ff_context.BarrettContext(moduli)
    assert self.ff_ctx is not None, "finite_field_context must be provided"
    assert (
        self.moduli == self.ff_ctx.moduli
    ), "moduli must be the same as the moduli of the finite_field_context"


class NTTCiphertextMontgomeryContext(NTTCiphertextContextBase):

  def __init__(self, moduli: int, parameters: dict, perf_test=False):
    super().__init__(moduli, parameters, perf_test=perf_test)
    if type(self.moduli) is int:
      self.moduli = [self.moduli]
    if self.ff_ctx is None:
      self.ff_ctx = ff_context.MontgomeryContext(moduli)
    assert self.ff_ctx is not None, "finite_field_context must be provided"
    assert (
        self.moduli == self.ff_ctx.moduli
    ), "moduli must be the same as the moduli of the finite_field_context"


class NTTCiphertextShoupContext(NTTCiphertextContextBase):
  """NTT with Shoup's Modular Reduction

  Note that Shoup's Reduction is NOT compatible with Basis Aligned
  Transformation (BAT).
  We use 1-d convolution to perform matrix multiplication for Shoup.
  """

  def __init__(self, moduli: int, parameters: dict, perf_test=False):
    super().__init__(moduli, parameters, perf_test=perf_test)
    if type(self.moduli) is int:
      self.moduli = [self.moduli]
    if self.ff_ctx is None:
      self.ff_ctx = ff_context.ShoupContext(moduli)
    assert self.ff_ctx is not None, "finite_field_context must be provided"
    assert (
        self.moduli == self.ff_ctx.moduli
    ), "moduli must be the same as the moduli of the finite_field_context"

    if not perf_test:
      self.ntt_bat_tf_step1 = self.to_computation_format(
          self.ntt_tf_step1
      ).astype(jnp.uint32)
      self.ntt_tf_step2 = self.to_computation_format(self.ntt_tf_step2).astype(
          jnp.uint64
      )
      self.ntt_bat_tf_step3 = self.to_computation_format(
          self.ntt_tf_step3
      ).astype(jnp.uint32)
      self.intt_bat_tf_step1 = self.to_computation_format(
          self.intt_tf_step1
      ).astype(jnp.uint32)
      self.intt_tf_step2 = self.to_computation_format(
          self.intt_tf_step2
      ).astype(jnp.uint64)
      self.intt_bat_tf_step3 = self.to_computation_format(
          self.intt_tf_step3
      ).astype(jnp.uint32)

      self.ntt_tf_step1_shoup = self.to_shoup_computation_format(
          self.ntt_tf_step1
      ).astype(jnp.uint32)
      self.ntt_tf_step2_shoup = self.to_shoup_computation_format(
          self.ntt_tf_step2
      ).astype(jnp.uint64)
      self.ntt_tf_step3_shoup = self.to_shoup_computation_format(
          self.ntt_tf_step3
      ).astype(jnp.uint32)
      self.intt_tf_step1_shoup = self.to_shoup_computation_format(
          self.intt_tf_step1
      ).astype(jnp.uint32)
      self.intt_tf_step2_shoup = self.to_shoup_computation_format(
          self.intt_tf_step2
      ).astype(jnp.uint64)
      self.intt_tf_step3_shoup = self.to_shoup_computation_format(
          self.intt_tf_step3
      ).astype(jnp.uint32)

  def to_computation_format(self, a: Union[np.ndarray, jax.Array]):
    return self.ff_ctx.to_computation_format(a.astype(jnp.uint64))

  def to_shoup_computation_format(self, a: Union[np.ndarray, jax.Array]):
    return self.ff_ctx.precompute_constant_operand(a.astype(jnp.uint64))

  def to_original_format(self, a: Union[np.ndarray, jax.Array]):
    return self.ff_ctx.to_original_format(a.astype(jnp.uint64))

  def get_jax_parameters(self):
    return {
        "ntt_tf_step1": util.to_tuple(self.ntt_tf_step1),
        "ntt_tf_step2": util.to_tuple(self.ntt_tf_step2),
        "ntt_tf_step3": util.to_tuple(self.ntt_tf_step3),
        "intt_tf_step1": util.to_tuple(self.intt_tf_step1),
        "intt_tf_step2": util.to_tuple(self.intt_tf_step2),
        "intt_tf_step3": util.to_tuple(self.intt_tf_step3),
        "finite_field_parameters": self.ff_ctx.get_jax_parameters(),
        "rows": self.r,
        "cols": self.c,
        "ntt_tf_step1_shoup": util.to_tuple(self.ntt_tf_step1_shoup),
        "ntt_tf_step2_shoup": util.to_tuple(self.ntt_tf_step2_shoup),
        "ntt_tf_step3_shoup": util.to_tuple(self.ntt_tf_step3_shoup),
        "intt_tf_step1_shoup": util.to_tuple(self.intt_tf_step1_shoup),
        "intt_tf_step2_shoup": util.to_tuple(self.intt_tf_step2_shoup),
        "intt_tf_step3_shoup": util.to_tuple(self.intt_tf_step3_shoup),
    }

  def ntt(self, v: jax.Array):
    """NTT with modular u32

    Args:
        v: - is u32 array of shape (B, R, C) - input

    Returns:
        - is u32 array of shape (B, R, C)
        - output
    """
    conv_over_rns = jax.vmap(
        lambda x, y: matmul_conv_flexible_kernel(x, y, ("NCW", "IOW", "NCW")),
        in_axes=(-1, -1),
        out_axes=-1,
    )
    batched_conv_step1 = jax.vmap(
        lambda x, y_b: conv_over_rns(x, y_b),
        in_axes=(None, 0),  # x is shared across B, y_b iterates over axis 0
        out_axes=0,
    )

    conv_over_rns_step3 = jax.vmap(
        lambda x, y: matmul_conv_flexible_kernel(x, y, ("NCW", "IOW", "CNW")),
        in_axes=(-1, -1),
        out_axes=-1,
    )
    batched_conv_step3 = jax.vmap(
        lambda x_b, y: conv_over_rns_step3(x_b, y),
        in_axes=(0, None),
        out_axes=0,
    )
    result_step1 = batched_conv_step1(self.ntt_tf_step1, v)
    result_step1_shoup = batched_conv_step1(self.ntt_tf_step1_shoup, v)
    result_step1_reduced = self.ff_ctx.modular_reduction(
        result_step1, result_step1_shoup
    )
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.ntt_tf_step2
    )
    result_step2_shoup = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.ntt_tf_step2_shoup
    )
    result_step2_reduced = self.ff_ctx.modular_reduction(
        result_step2, result_step2_shoup
    )
    result_step3 = batched_conv_step3(result_step2_reduced, self.ntt_tf_step3)
    result_step3_shoup = batched_conv_step3(
        result_step2_reduced, self.ntt_tf_step3_shoup
    )
    result_step3_reduced = self.ff_ctx.modular_reduction(
        result_step3, result_step3_shoup
    )
    result_step3_reduced = result_step3_reduced.transpose(0, 2, 1, 3)
    return result_step3_reduced

  def intt(self, v: jax.Array):
    """INTT with modular u32

    Args:
        v: - is u32 array of shape (B, R, C) - input

    Returns:
        - is u32 array of shape (B, R, C)
        - output
    """
    # computation
    conv_over_rns = jax.vmap(
        lambda x, y: matmul_conv_flexible_kernel(x, y, ("CNW", "IOW", "NCW")),
        in_axes=(-1, -1),
        out_axes=-1,
    )
    batched_conv_step1 = jax.vmap(
        lambda x_b, y: conv_over_rns(x_b, y),
        in_axes=(0, None),  # x is shared across B, y_b iterates over axis 0
        out_axes=0,
    )

    conv_over_rns_step3 = jax.vmap(
        lambda x, y: matmul_conv_flexible_kernel(x, y, ("NCW", "IOW", "NCW")),
        in_axes=(-1, -1),
        out_axes=-1,
    )
    batched_conv_step3 = jax.vmap(
        lambda x, y_b: conv_over_rns_step3(x, y_b),
        in_axes=(None, 0),
        out_axes=0,
    )
    v = v.transpose(0, 2, 1, 3)
    result_step1 = batched_conv_step1(v, self.intt_tf_step1)
    result_step1_shoup = batched_conv_step1(v, self.intt_tf_step1_shoup)
    result_step1_reduced = self.ff_ctx.modular_reduction(
        result_step1, result_step1_shoup
    )
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.intt_tf_step2
    )
    result_step2_shoup = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.intt_tf_step2_shoup
    )
    result_step2_reduced = self.ff_ctx.modular_reduction(
        result_step2, result_step2_shoup
    )
    result_step3 = batched_conv_step3(self.intt_tf_step3, result_step2_reduced)
    result_step3_shoup = batched_conv_step3(
        self.intt_tf_step3_shoup, result_step2_reduced
    )
    result_step3_reduced = self.ff_ctx.modular_reduction(
        result_step3, result_step3_shoup
    )
    return result_step3_reduced


class NTTCiphertextBATLazyContext(NTTCiphertextContextBase):

  def __init__(self, moduli: int, parameters: dict, perf_test=False):
    super().__init__(moduli, parameters, perf_test=perf_test)
    if type(self.moduli) is int:
      self.moduli = [self.moduli]
    if self.ff_ctx is None:
      self.ff_ctx = ff_context.BarrettContext(moduli)
    assert self.ff_ctx is not None, "finite_field_context must be provided"
    assert (
        self.moduli == self.ff_ctx.moduli
    ), "moduli must be the same as the moduli of the finite_field_context"
    self.ff_ctx_bat_lazy = ff_context.BATLazyContext(moduli)

  def ntt(self, v: jax.Array):
    """NTT with modular u32

    B = Batch size, R = self.r, C = self.c
    Q = 4 (number of bytes per element)
    M = len(self.moduli)

    Args:
        v: - is u32 array of shape (B, R, C, M) - will be casted into u8 array
          of shape (B, R, C, M, Q)
        ntt_bat_tf_step1: - is u8 array of shape (R, 4, R, 4, M)
        ntt_tf_step2: - is u32 array of shape (R, C, M)
        ntt_bat_tf_step3: - is u8 array of shape (C, 4, C, 4, M)

    Returns:
        - is u32 array of shape (B, R, C, M)
        - output
    """
    result_step1 = matmul_bat_einsum(
        v, self.ntt_bat_tf_step1, "brcmq,zqrpm->bzcmp"
    )  # "mqkp,bknq->bmnp"; "bkncq,mqkpc->bmncp"
    result_step1_reduced = self.ff_ctx_bat_lazy.modular_reduction(result_step1)
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.ntt_tf_step2
    )
    result_step2_reduced = self.ff_ctx.modular_reduction(result_step2)
    result_step3 = matmul_bat_einsum(
        result_step2_reduced.astype(jnp.uint32),
        self.ntt_bat_tf_step3,
        "brcmq,cqnpm->brnmp",
    )  # "bmkq,kqnp->bnmp" "bmkcq,kqnpc->bnmcp"
    result_step3_reduced = self.ff_ctx.modular_reduction(result_step3)
    return result_step3_reduced

  def intt(self, v: jax.Array):
    """INTT with modular u32

    B = Batch size, R = self.r, C = self.c
    Q = 4 (number of bytes per element)
    M = len(self.moduli)

    Args:
        v: - is u32 array of shape (B, R, C, M) - will be casted into u8 array
          of shape (B, R, C, M, Q)
        intt_bat_tf_step1: - is u8 array of shape (C, 4, C, 4, M)
        intt_tf_step2: - is u32 array of shape (R, C, M)
        intt_bat_tf_step3: - is u8 array of shape (R, 4, R, 4, M)

    Returns:
        - is u32 array of shape (B, R, C, M)
        - output
    """
    result_step1 = matmul_bat_einsum(
        v, self.intt_bat_tf_step1, "brcmq,cqlpm->brlmp"
    )
    result_step1_reduced = self.ff_ctx_bat_lazy.modular_reduction(result_step1)
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.intt_tf_step2
    )
    result_step2_reduced = self.ff_ctx.modular_reduction(result_step2)
    result_step3 = matmul_bat_einsum(
        result_step2_reduced, self.intt_bat_tf_step3, "brcmq,lqrpm->blcmp"
    )
    result_step3_reduced = self.ff_ctx.modular_reduction(result_step3)
    return result_step3_reduced
