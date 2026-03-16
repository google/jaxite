"""Name: JAX Finite Field Context Integration

Name Template: <Framework><Representation><Reduction><Strategy>Context<Base>
    - <Framework>:  (JAX accelerator backend).
    - <Representation>: [Optional]
        - Empty: Standard scalar.
        - RNS: Residue Number System.
        - DRNS: Digitized RNS.
        - RD: Radix Decomposition (Big Integer simulation).
    - <Reduction>: [Optional]
        - Montgomery: Montgomery reduction.
        - Barrett: Barrett reduction.
        - Shoup: Shoup reduction.
    - <Strategy>: [Optional]
        - MultipleModuli: vectorized over moduli.
        - Lazy: Lazy reduction.
        - Opt/Opt2: Optimization levels or specific variants.
    - Context: Class suffix.
    - <Base>: [Optional] Abstract base class.

Explanation: This module adapts the generic finite field contexts for use with
JAX. It inherits from the base contexts in `finite_field_context.py` and adds
functionality to precompute and format parameters (such as modular inverses, RNS
matrices, and bit-shifted constants) into JAX-compatible arrays. It serves as
the configuration bridge between the mathematical specifications and the JAX
kernels.
"""

import math
from typing import List, Union
import jax
import jax.numpy as jnp
import jaxite.jaxite_word.util as util

jax.config.update("jax_enable_x64", True)


########################
# Base Context Class
########################
class FiniteFieldContextBase:

  def __init__(self, moduli: int):
    self.moduli = moduli

  def to_computation_format(self, a: jnp.ndarray):
    return a

  def to_original_format(self, a: jnp.ndarray):
    return a

  def precompute_constant_operand(self, a: int):
    # return [(a * (1 << self.w)) // m for m in self.moduli] # The algorithm being performed
    return a

  def get_jax_parameters(self):
    return {}

  def modular_reduction(self, a: jnp.ndarray) -> jnp.ndarray:
    raise NotImplementedError("Subclasses must implement this method")

  def drop_last_modulus(self):
    raise NotImplementedError("Subclasses must implement this method")


########################
# Montgomery Modulus Reduction Context
########################
class MontgomeryContext(FiniteFieldContextBase):

  def __init__(self, moduli: Union[List[int], int]):
    super().__init__(moduli)
    self.moduli = moduli
    if type(self.moduli) is int:
      self.moduli = [self.moduli]
    self.w = 32
    self.w_inv = [util.modinv(1 << self.w, m) for m in self.moduli]
    self.w_inv_reduction = jnp.array(self.w_inv, jnp.uint64)

    self.moduli_reduction = jnp.array(self.moduli, jnp.uint64)

    self.moduli_inv_32 = [util.modinv(m, 2**32) for m in self.moduli]
    self.moduli_low16 = [m & 0xFFFF for m in self.moduli]
    self.moduli_high16 = [m >> 16 for m in self.moduli]

    self.q = jnp.array(self.moduli, dtype=jnp.uint32)
    self.q_low = jnp.array(self.moduli_low16, dtype=jnp.uint32)
    self.q_high = jnp.array(self.moduli_high16, dtype=jnp.uint32)
    self.q_inv_32 = jnp.array(self.moduli_inv_32, dtype=jnp.uint32)

  def to_computation_format(self, a: jnp.ndarray):
    # return [(a * (1 << self.w)) % m for m in self.moduli] # The algorithm being performed
    return ((a << self.w) % self.moduli_reduction).astype(jnp.uint32)

  def to_original_format(self, a: jnp.ndarray):
    return (a * self.w_inv_reduction) % self.moduli_reduction

  def get_jax_parameters(self):
    return {
        "moduli": util.to_tuple(self.moduli),
        "moduli_inv_32": util.to_tuple(self.moduli_inv_32),
        "moduli_low": util.to_tuple(self.moduli_low16),
        "moduli_high": util.to_tuple(self.moduli_high16),
    }

  def modular_reduction(self, z: jnp.ndarray) -> jnp.ndarray:
    """Montgomery reduction from u64 to u32 optimized version using only 32-bit operations

    Args:
        z: - is u64 array of shape (B, M) - input

    parameters:
        moduli:
            - Tuple parameters constants
            - is u32 array of shape (M)
            - modular or moduli
        moduli_low:
            - Tuple parameters constants
            - is u32 array of shape (M)
            - low 16 bits of modular or moduli
        moduli_high:
            - Tuple parameters constants
            - is u32 array of shape (M)
            - high 16 bits of modular or moduli
        moduli_inv_32:
            - Tuple parameters constants
            - is u32 array of shape (M)
            - modular inverse of q mod 2^32
    Returns:
        - is u32 array of shape (B, M)
        - output
        - reduced value
    """

    # Local constants
    MASK32 = 0xFFFFFFFF
    MASK16 = 0xFFFF
    SHIFT16 = 16
    SHIFT32 = 32
    # Ensure dimensions for broadcasting
    q = self.q
    q_low = self.q_low
    q_high = self.q_high
    q_inv_32 = self.q_inv_32

    # Computation
    z_low = z.astype(jnp.uint32)
    z_high = (z >> SHIFT32).astype(jnp.uint32)
    t = (z_low * q_inv_32) & MASK32
    t_low = t & MASK16
    t_high = (t >> SHIFT16) & MASK16

    prod_high = t_high * q_high  # This contributes directly to upper 32 bits
    prod_mid_high = t_high * q_low  # Upper 16 bits go to upper 32 bits
    prod_mid_low = t_low * q_high  # Upper 16 bits go to upper 32 bits
    prod_low = t_low * q_low  # Upper 16 bits contribute to middle part
    mid_low = (
        (prod_mid_high & MASK16)
        + (prod_mid_low & MASK16)
        + (prod_low >> SHIFT16)
    )
    mid_high = (
        (prod_mid_high >> SHIFT16)
        + (prod_mid_low >> SHIFT16)
        + (mid_low >> SHIFT16)
    )

    # Final upper 32 bits
    t_final = prod_high + mid_high
    b = z_high + q - t_final
    # Ensure strict reduction
    # b = jnp.where(b >= q, b - q, b).astype(jnp.uint32)
    return b.astype(jnp.uint32)

  def drop_last_modulus(self):
    # self.moduli_reduction, self.moduli_inv_32, self.moduli_low16, self.moduli_high16 are not updated here.
    # Because they are not used in the reduction.
    # self.moduli = self.moduli[:-1]
    self.moduli_reduction = self.moduli_reduction[:-1]
    self.q = self.q[:-1]
    self.q_low = self.q_low[:-1]
    self.q_high = self.q_high[:-1]
    self.q_inv_32 = self.q_inv_32[:-1]


########################
# Barrett Modulus Reduction Context
########################
class BarrettContext(FiniteFieldContextBase):

  def __init__(self, moduli: Union[List[int], int]):
    super().__init__(moduli)
    self.moduli = moduli
    if type(self.moduli) is int:
      self.moduli = [self.moduli]

    self.barrett_s = [2 * math.ceil(math.log2(m)) for m in self.moduli]
    self.barrett_w = [min(s, 32) for s in self.barrett_s]
    self.barrett_s_w = [s - w for s, w in zip(self.barrett_s, self.barrett_w)]
    self.barrett_m = [
        math.floor(2**s / m) for s, m in zip(self.barrett_s, self.moduli)
    ]
    # used for run-time reduction
    self.m = jnp.array(self.barrett_m, dtype=jnp.uint64)
    self.moduli_reduction = jnp.array(self.moduli, dtype=jnp.uint64)
    self.w = jnp.array(self.barrett_w, dtype=jnp.uint16)
    self.s_w = jnp.array(self.barrett_s_w, dtype=jnp.uint16)

  def to_computation_format(self, a):
    return a

  def to_original_format(self, a):
    return a

  def get_jax_parameters(self):
    return {
        "barrett_m": util.to_tuple(self.barrett_m),
        "moduli": util.to_tuple(self.moduli),
        "barrett_w": util.to_tuple(self.barrett_w),
        "barrett_s_w": util.to_tuple(self.barrett_s_w),
    }

  def modular_reduction(self, z: jnp.ndarray) -> jnp.ndarray:
    """Vectorized implementation of the Barrett reduction.

    Works for modulus `q` less than 31 bits.

    This implementation sets the internal shift width `w` to `min(s, 32)` so it
    works with small modulus `moduli < 2^16`.

    Args:
        z: The input value.
        moduli: The RNS moduli.
        s_w: The bit width of moduli.
        w: The internal shift width.
        m: The precomputed value for Barrett reduction.

    Returns:
        The result of the Barrett reduction.
    """
    m = self.m
    moduli = self.moduli_reduction
    w = self.w
    s_w = self.s_w

    z1 = z & 0xFFFFFFFF
    z2 = z >> w
    t = ((z1 * m) >> w) + (z2 * m)
    t = t >> s_w
    z = z - t * moduli
    pred = z >= moduli
    return jnp.where(pred, z - moduli, z).astype(jnp.uint32)
    # return (z - moduli * pred).astype(jnp.uint32)

  def modular_reduction_single_modulus(
      self, z: jnp.ndarray, modulus_index: int
  ) -> jnp.ndarray:
    """Vectorized implementation of the Barrett reduction.

    Works for modulus `q` less than 31 bits.

    This implementation sets the internal shift width `w` to `min(s, 32)` so it
    works with small modulus `moduli < 2^16`.

    Args:
        z: The input value.
        moduli: The RNS moduli.
        s_w: The bit width of moduli.
        w: The internal shift width.
        m: The precomputed value for Barrett reduction.

    Returns:
        The result of the Barrett reduction.
    """
    m = self.m[modulus_index]
    moduli = self.moduli_reduction[modulus_index]
    w = self.w[modulus_index]
    s_w = self.s_w[modulus_index]

    z1 = z.astype(jnp.uint32)
    z2 = (z >> w).astype(jnp.uint32)
    t = ((z1 * m) >> w) + (z2 * m)
    t = t >> s_w
    z = z - t * moduli
    pred = z >= moduli
    return jnp.where(pred, z - moduli, z).astype(jnp.uint32)
    # return (z - moduli * pred).astype(jnp.uint32)

  def drop_last_modulus(self):
    # self.barrett_s, self.barrett_w, self.barrett_s_w, self.barrett_m are not updated here.
    # Because they are not used in the reduction.
    # self.moduli = self.moduli[:-1]
    self.m = self.m[:-1]
    self.moduli_reduction = self.moduli_reduction[:-1]
    self.w = self.w[:-1]
    self.s_w = self.s_w[:-1]


########################
# Shoup Modulus Reduction Context
########################
class ShoupContext(FiniteFieldContextBase):

  def __init__(self, moduli: Union[List[int], int]):
    super().__init__(moduli)
    self.moduli = moduli
    if type(self.moduli) is int:
      self.moduli = [self.moduli]
    self.moduli_reduction = jnp.array(self.moduli, jnp.uint64)
    self.q = jnp.array(self.moduli, dtype=jnp.uint64)
    self.w = 32

  def to_computation_format(self, a: jnp.ndarray):
    # return [(a % m) for m in self.moduli] # The algorithm being performed
    return (a % self.moduli_reduction).astype(jnp.uint32)

  def to_original_format(self, a: jnp.ndarray):
    return (a % self.moduli_reduction).astype(jnp.uint32)

  def precompute_constant_operand(self, a: int):
    # return [(a * (1 << self.w)) // m for m in self.moduli] # The algorithm being performed
    return (a << self.w) // self.moduli_reduction

  def get_jax_parameters(self):
    return {
        "moduli": util.to_tuple(self.moduli),
    }

  def modular_reduction(self, z: jnp.ndarray, z_s: jnp.ndarray) -> jnp.ndarray:
    """Shoup's reduction from u64 to u32

    Args:
        z: - is u64 array of shape (B, M) - input - z = a * b
        z_s: - is u64 array of shape (B, M) - input - z_s = a * b_s - b_s is b
          in Shoup's precomputation format

    parameters:
        moduli:
            - Tuple parameters constants
            - is u32 array of shape (M)
            - modular or moduli
    Returns:
        - is u32 array of shape (B, M)
        - output
        - reduced value
    """
    t = z_s >> 32
    u = z - t * self.q
    # Ensure strict reduction
    # u = jnp.where(u >= self.q, u - self.q, u).astype(jnp.uint32)
    return u.astype(jnp.uint32)

  def drop_last_modulus(self):
    # self.moduli is not updated here.
    # Because it is used in the precomputation.
    # self.moduli = self.moduli[:-1]
    self.moduli_reduction = self.moduli_reduction[:-1]
    self.q = self.q[:-1]


########################
# BAT Lazy Reduction Context
########################
class BATLazyContext(FiniteFieldContextBase):

  def __init__(self, moduli: Union[List[int], int]):
    super().__init__(moduli)
    self.moduli = moduli
    if type(self.moduli) is int:
      self.moduli = [self.moduli]

    # L=4 bytes (for 32-bit modulus)
    self.L = 4

    # Precompute R matrix for each modulus
    # R_i,j corresponds to the j-th byte of (256^(i+L) mod q)
    # Dimensions: (M, 4, 4) because we have 4 high-bytes (B) and 4 result-bytes (L)
    moduli_arr = jnp.array(self.moduli, dtype=jnp.uint64)

    # 1. Vectorize 'i' loop (bytes 4, 5, 6, 7): Compute r_val = 256^(i+4) % m
    shifts_i = jnp.arange(4, 8, dtype=jnp.uint64) * 8
    # Broadcast shape: (1, 4) vs (M, 1) -> (M, 4)
    r_vals = (jnp.array(1, dtype=jnp.uint64) << shifts_i[None, :]) % moduli_arr[
        :, None
    ]

    # 2. Vectorize 'j' loop: Split r_vals into 4 bytes (little endian)
    shifts_j = jnp.arange(4, dtype=jnp.uint64) * 8
    # Result: (M, 4, 4)
    self.R = ((r_vals[:, :, None] >> shifts_j[None, None, :]) & 0xFF).astype(
        jnp.uint8
    )
    self.moduli_reduction = jnp.array(self.moduli, jnp.uint64)

  def to_computation_format(self, a: jnp.ndarray):
    return a

  def to_original_format(self, a: jnp.ndarray):
    return (a % self.moduli_reduction).astype(jnp.uint32)

  def get_jax_parameters(self):
    return {"moduli": util.to_tuple(self.moduli), "R": self.R}

  def modular_reduction(self, z: jnp.ndarray) -> jnp.ndarray:
    """BAT Lazy Reduction from u64 to u32

    Implements: result = B @ R + A
    where z is split into Lower Part A (bytes 0-3) and Higher Part B (bytes
    4-7).

    Args:
        z: u64 array of shape (..., M) if RNS, or arbitrary shape if single
          modulus.

    Returns:
        u32 array (Partially reduced)
    """
    # 1. Extract bytes from z using bitcast
    # This treats the 64-bit integers as vectors of 8 bytes (Little Endian)
    z_bytes = jax.lax.bitcast_convert_type(
        z.astype(jnp.uint64), new_dtype=jnp.uint8
    )

    # 2. Split into Lower Part A (bytes 0-3) and Higher Part B (bytes 4-7)
    # A_bytes, B_bytes each have shape (..., 4) where ... matches z's shape.
    A_bytes, B_bytes = jnp.split(z_bytes, 2, axis=-1)

    # 3. Perform Matrix Multiplication: LazyReductionResult = B @ R + A
    # Logic:
    # - If we have a single modulus (M=1), we assume ALL input elements should be
    #   reduced by this same modulus, regardless of input shape dimensions.
    # - If we have multiple moduli (M>1), we assume the LAST dimension of input
    #   corresponds to the moduli dimension M.

    # Unified implementation for both Single Modulus and RNS
    # Use einsum for automatic broadcasting and hardware-efficient 8-bit matmul
    # - Single Modulus: B (..., 4) @ R_squeezed (4, 4) -> (..., 4)
    # - RNS: B (..., M, 4) @ R (M, 4, 4) -> (..., M, 4)
    # Note: jnp.squeeze ensures R is (4, 4) when M=1, matching the "Single Modulus" lack of M-dim.
    # We perform the input in 8-bit and accumulate in 32-bit for TPU efficiency.
    matmul_res = jnp.einsum(
        "...i,...ij->...j", B_bytes, self.R, preferred_element_type=jnp.uint32
    )

    # 4. Add Lower Part A
    result_bytes = matmul_res + A_bytes

    # 5. Reconstruct integer
    shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
    result = jnp.sum(
        result_bytes.astype(jnp.uint64) << shift_factors, axis=(-1,)
    )

    return result

  def drop_last_modulus(self):
    self.moduli = self.moduli[:-1]
    self.R = self.R[:-1]
