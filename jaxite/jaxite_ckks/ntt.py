"""NTT kernels for CKKS on TPU."""

import abc
import dataclasses
from typing import Iterable

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import barrett
from jaxite.jaxite_ckks import math as ckks_math

# Enable 64-bit precision for large integer arithmetic
jax.config.update("jax_enable_x64", True)

ABC = abc.ABC
abstractmethod = abc.abstractmethod


class NTTBase(ABC):
  """Abstract base class for NTT/INTT kernels."""

  @abstractmethod
  def precompute_constants(
      self,
      moduli: Iterable[int],
      r: int,
      c: int,
  ):
    """Generates precomputed constants needed for NTT."""

  @abstractmethod
  def ntt(self, v: jnp.ndarray) -> jnp.ndarray:
    """Performs an NTT.

    Args:
      v: the input array to perform NTT on. Shape (..., R, C, M).

    Returns: the transformed vector.
    """

  @abstractmethod
  def intt(self, v: jnp.ndarray) -> jnp.ndarray:
    """Performs an INTT.

    Args:
      v: the input array to perform INTT on. Shape (..., R, C, M).

    Returns: the transformed vector.
    """


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class NTTBarrettConstants:
  """Precomputed constants for Barrett-based NTT."""

  ntt_bat_tf_step1: jax.Array
  ntt_tf_step2: jax.Array
  ntt_bat_tf_step3: jax.Array
  intt_bat_tf_step1: jax.Array
  intt_tf_step2: jax.Array
  intt_bat_tf_step3: jax.Array
  barrett_constants: barrett.BarrettConstants
  r: int
  c: int
  moduli: jax.Array

  def tree_flatten(self):
    children = (
        self.ntt_bat_tf_step1,
        self.ntt_tf_step2,
        self.ntt_bat_tf_step3,
        self.intt_bat_tf_step1,
        self.intt_tf_step2,
        self.intt_bat_tf_step3,
        self.barrett_constants,
        self.moduli,
    )
    aux_data = (self.r, self.c)
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children[:7], *aux_data, children[7])

  def __hash__(self):
    return id(self)


def _matmul_bat_einsum(
    lhs: jax.Array, rhs: jax.Array, subscripts: str
) -> jax.Array:
  """Basis Aligned Transformation (BAT) based matrix multiplication.

  Args:
    lhs: input
    rhs: twiddle factor matrix
    subscripts: einsum subscripts

  Returns:
    The matrix multiplication result.
  """
  lhs_u8 = jax.lax.bitcast_convert_type(lhs, jnp.uint8)
  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
  i8_products = jnp.einsum(
      subscripts, lhs_u8, rhs, preferred_element_type=jnp.uint32
  )
  return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(-1,))


def _basis_aligned_transformation(
    matrix: jnp.ndarray, moduli: list[int]
) -> jnp.ndarray:
  """Prepares a matrix for Basis Aligned Transformation (BAT)."""
  matrix_u64 = matrix.astype(jnp.uint64)
  num_bytes = 4
  matrix_u64_byteshifted = jnp.array(
      [matrix_u64 << (8 * byte_idx) for byte_idx in range(num_bytes)],
      dtype=jnp.uint64,
  )
  moduli_arr = jnp.array(moduli, dtype=jnp.uint64)
  matrix_u64_byteshifted_mod_modulus = (
      matrix_u64_byteshifted % moduli_arr
  ).astype(jnp.uint32)
  # Output shape: (4, ..., moduli, 4)
  matrix_u8 = jax.lax.bitcast_convert_type(
      matrix_u64_byteshifted_mod_modulus, jnp.uint8
  )
  return matrix_u8


@jax.tree_util.register_pytree_node_class
class NTTBarrett(NTTBase):
  """Kernel for NTT with Barrett reduction."""

  def __init__(self):
    self.constants: NTTBarrettConstants = None

  def tree_flatten(self):
    children = (self.constants,)
    aux_data = None
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    obj = cls()
    obj.constants = children[0]
    return obj

  def __hash__(self):
    return id(self)

  def precompute_constants(
      self,
      moduli: Iterable[int],
      r: int,
      c: int,
  ):
    moduli = list(moduli)
    transform_length = r * c

    psi_list = [
        ckks_math.root_of_unity(2 * transform_length, q) for q in moduli
    ]
    omega_list = [pow(psi, 2, q) for psi, q in zip(psi_list, moduli)]

    perm_r = ckks_math.get_bit_reverse_perm(r)
    perm_c = ckks_math.get_bit_reverse_perm(c)

    # NTT Precompute
    ntt_tf_step1_list, ntt_tf_step2_list, ntt_tf_step3_list = [], [], []
    for i, modulus in enumerate(moduli):
      omega = omega_list[i]
      omega_col = pow(omega, c, modulus)
      omega_row = pow(omega, r, modulus)

      tf1 = ckks_math.gen_twiddle_matrix(r, r, modulus, omega_col)
      tf2 = ckks_math.gen_twiddle_matrix(r, c, modulus, omega)
      tf3 = ckks_math.gen_twiddle_matrix(c, c, modulus, omega_row)

      # Memory Aligned Transformation (MAT)
      # From CROSS: https://arxiv.org/abs/2501.07047
      tf1 = tf1[perm_r, :]
      tf2 = tf2[perm_r, :]
      tf3 = tf3[:, perm_c]

      ntt_tf_step1_list.append(tf1)
      ntt_tf_step2_list.append(tf2)
      ntt_tf_step3_list.append(tf3)

    ntt_tf_step1 = jnp.array(ntt_tf_step1_list).transpose(1, 2, 0)
    ntt_tf_step2 = jnp.array(ntt_tf_step2_list).transpose(1, 2, 0)
    ntt_tf_step3 = jnp.array(ntt_tf_step3_list).transpose(1, 2, 0)

    # INTT Precompute
    intt_tf_step1_list, intt_tf_step2_list, intt_tf_step3_list = [], [], []
    for i, modulus in enumerate(moduli):
      omega = omega_list[i]
      omega_col = pow(omega, c, modulus)
      omega_row = pow(omega, r, modulus)
      inv_omega_col = pow(omega_col, -1, modulus)
      inv_omega_row = pow(omega_row, -1, modulus)

      itf1 = ckks_math.gen_twiddle_matrix(c, c, modulus, inv_omega_row)
      itf2 = ckks_math.gen_twiddle_matrix_inv(r, c, modulus, omega)
      itf3 = ckks_math.gen_twiddle_matrix(r, r, modulus, inv_omega_col)

      # MAT
      # From CROSS: https://arxiv.org/abs/2501.07047
      itf1 = itf1[perm_c, :]
      itf2 = itf2[perm_r, :]
      itf3 = itf3[:, perm_r]

      # Scaling
      col_inv = pow(c, -1, modulus)
      row_inv = pow(r, -1, modulus)
      itf2 = (itf2 * col_inv) % modulus
      itf3 = (itf3 * row_inv) % modulus

      intt_tf_step1_list.append(itf1)
      intt_tf_step2_list.append(itf2)
      intt_tf_step3_list.append(itf3)

    intt_tf_step1 = jnp.array(intt_tf_step1_list).transpose(1, 2, 0)
    intt_tf_step2 = jnp.array(intt_tf_step2_list).transpose(1, 2, 0)
    intt_tf_step3 = jnp.array(intt_tf_step3_list).transpose(1, 2, 0)

    def to_bat(tf, moduli):
      # tf: (R, R, M)
      raw_bat = _basis_aligned_transformation(tf, moduli)
      # raw_bat shape: (4_byte_shift, rows, cols, moduli, 4_u8_bytes)
      # We want (rows, 4_byte_shift, cols, 4_u8_bytes, moduli)
      # matching subscripts q=shift, p=u8
      return raw_bat.transpose(1, 0, 2, 4, 3)

    self.constants = NTTBarrettConstants(
        ntt_bat_tf_step1=to_bat(ntt_tf_step1, moduli),
        ntt_tf_step2=ntt_tf_step2.astype(jnp.uint64),
        ntt_bat_tf_step3=to_bat(ntt_tf_step3, moduli),
        intt_bat_tf_step1=to_bat(intt_tf_step1, moduli),
        intt_tf_step2=intt_tf_step2.astype(jnp.uint64),
        intt_bat_tf_step3=to_bat(intt_tf_step3, moduli),
        barrett_constants=barrett.precompute_barrett_constants(moduli),
        r=r,
        c=c,
        moduli=jnp.array(moduli, dtype=jnp.uint32),
    )

  @jax.jit
  def ntt(self, v: jnp.ndarray) -> jnp.ndarray:
    """Performs the forward NTT using BAT optimization."""
    # Step 1: Sum over rows. lhs: "...rcmq", rhs: "zqrpm"
    # q=u8 (lhs axis 4), q=shift (rhs axis 1). Summed.
    # z=target row (axis 0), r=source row (axis 2). Sum over r.
    # p=u8 target (axis 3). Becomes axis 4 of result.
    res1 = _matmul_bat_einsum(
        v, self.constants.ntt_bat_tf_step1, "...rcmq,zqrpm->...zcmp"
    )
    res1 = barrett.modular_reduction(res1, self.constants.barrett_constants)

    res2 = res1.astype(jnp.uint64) * self.constants.ntt_tf_step2
    res2 = barrett.modular_reduction(res2, self.constants.barrett_constants)

    # Step 3: Sum over cols. lhs: "...rcmq", rhs: "cqnpm"
    # c=source col (axis 2), n=target col (axis 2 of rhs? NO, axis 0).
    # Wait, in to_bat, axis 0 is rows, axis 2 is cols.
    # For ntt_tf_step3, axis 0 is source, axis 1 is target.
    # So to_bat axis 0 is source, axis 2 is target.
    # Subscripts: "cqnpm" -> c=0 (source), n=2 (target).
    res3 = _matmul_bat_einsum(
        res2, self.constants.ntt_bat_tf_step3, "...rcmq,cqnpm->...rnmp"
    )
    return barrett.modular_reduction(res3, self.constants.barrett_constants)

  @jax.jit
  def intt(self, v: jnp.ndarray) -> jnp.ndarray:
    """Performs the inverse NTT using BAT optimization."""
    # Step 1: Sum over cols. lhs: "...rcmq", rhs: "cqlpm"
    # itf1 axis 0 is source, axis 1 is target.
    # to_bat axis 0 is source, axis 2 is target.
    # Subscripts: "cqlpm" -> c=0 (source), l=2 (target).
    res1 = _matmul_bat_einsum(
        v, self.constants.intt_bat_tf_step1, "...rcmq,cqlpm->...rlmp"
    )
    res1 = barrett.modular_reduction(res1, self.constants.barrett_constants)

    res2 = res1.astype(jnp.uint64) * self.constants.intt_tf_step2
    res2 = barrett.modular_reduction(res2, self.constants.barrett_constants)

    # Step 3: Sum over rows. lhs: "...rcmq", rhs: "lqrpm"
    # itf3 axis 0 is target, axis 1 is source.
    # to_bat axis 0 is target, axis 2 is source.
    # Subscripts: "lqrpm" -> l=0 (target), r=2 (source).
    res3 = _matmul_bat_einsum(
        res2, self.constants.intt_bat_tf_step3, "...rcmq,lqrpm->...lcmp"
    )
    return barrett.modular_reduction(res3, self.constants.barrett_constants)
