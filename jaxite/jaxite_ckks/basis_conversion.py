"""Basis conversion kernels for CKKS on TPU."""

import abc
import dataclasses
from typing import Iterable

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import barrett
from jaxite.jaxite_ckks import rns_utils

# Enable 64-bit precision for large integer arithmetic
jax.config.update("jax_enable_x64", True)

ABC = abc.ABC
abstractmethod = abc.abstractmethod


class BasisConversion(ABC):
  """Abstract base class for basis conversion kernels."""

  @abstractmethod
  def precompute_constants(
      self,
      modulus_chain: Iterable[int],
      control_indices: Iterable[Iterable[Iterable[int]]],
  ):
    """Generates precomputed constants needed for basis conversion.

    Args:
      modulus_chain: the list of moduli to support
      control_indices: a list of sets of basis conversions to support,
        in the form of (from_indices, to_indices). I.e., if one element of the
        control_indices list is the pair ([0, 1], [2, 3]), then constants will
        be generated to support a basis conversion from
        from_basis = [modulus_chain[0], modulus_chain[1]] to
        to_basis = [modulus_chain[2], modulus_chain[3]]. This is needed to
        support basis conversions at different CKKS levels, and the
        control_index argument of basis_change specifies which basis conversion
        to perform by indexing into this list.
    """

  @abstractmethod
  def basis_change(
      self, in_tower: jnp.ndarray, control_index: int = 0
  ) -> jnp.ndarray:
    """Performs a basis conversion.

    Args:
      in_tower: the input to perform basis_conversion on, assumed to be
        in the "from_basis" specified by control_index.
      control_index: the speficiation of which basis conversion to perform,
        cf. the docstring for control_indices on precompute_constants.

    Returns: the value of in_tower in the "to_basis" specified by
      control_index.
    """


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class BarrettBasisConversionConstants:
  """Precomputed constants for Barrett-based basis conversion."""

  q_hat_inv_mod_q: jax.Array
  q_hat_mod_p_bat: jax.Array
  origin_barrett: barrett.BarrettConstants
  target_barrett: barrett.BarrettConstants

  def tree_flatten(self):
    children = (
        self.q_hat_inv_mod_q,
        self.q_hat_mod_p_bat,
        self.origin_barrett,
        self.target_barrett,
    )
    aux_data = None
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children)

  def __hash__(self):
    return id(self)


def matmul_bat_einsum(
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
  lhs_u8 = lhs.view(jnp.uint8)
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
class BasisConversionBarrett(BasisConversion):
  """Kernel for Basis Conversion with Barrett reduction."""

  def __init__(self):
    self.precomputed_constants: list[BarrettBasisConversionConstants] = []

  def tree_flatten(self):
    children = (self.precomputed_constants,)
    aux_data = None
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    obj = cls()
    obj.precomputed_constants = children[0]
    return obj

  def __hash__(self):
    return id(self)

  def precompute_constants(
      self,
      modulus_chain: Iterable[int],
      control_indices: Iterable[Iterable[Iterable[int]]],
  ):
    modulus_chain = list(modulus_chain)
    self.precomputed_constants = []
    for original_idx, target_idx in control_indices:
      original_moduli = [modulus_chain[i] for i in original_idx]
      target_moduli = [modulus_chain[i] for i in target_idx]

      q_hat_inv_mod_q = jnp.array(
          rns_utils.compute_q_hat_inv_mod_q(original_moduli), dtype=jnp.uint64
      )
      q_hat_mod_p = jnp.array(
          rns_utils.compute_q_hat_mod_p(original_moduli, target_moduli),
          dtype=jnp.uint64,
      )

      # BAT Preprocessing
      q_hat_mod_p_bat_raw = _basis_aligned_transformation(
          q_hat_mod_p, target_moduli
      )
      q_hat_mod_p_bat = q_hat_mod_p_bat_raw.transpose(1, 0, 2, 3).reshape(
          -1, q_hat_mod_p_bat_raw.shape[2], 4
      )

      constants = BarrettBasisConversionConstants(
          q_hat_inv_mod_q=q_hat_inv_mod_q,
          q_hat_mod_p_bat=q_hat_mod_p_bat,
          origin_barrett=barrett.precompute_barrett_constants(original_moduli),
          target_barrett=barrett.precompute_barrett_constants(target_moduli),
      )
      self.precomputed_constants.append(constants)

  @jax.jit(static_argnames="control_index")
  def basis_change(
      self, in_tower: jnp.ndarray, control_index: int = 0
  ) -> jnp.ndarray:
    """Performs the approximate basis change using BAT optimization."""
    constants = self.precomputed_constants[control_index]
    in_tower = jnp.asarray(in_tower, dtype=jnp.uint64)

    # Step 1: Compute c_unreduced = in_tower * QHatInvModq
    # Ensure constants.q_hat_inv_mod_q broadcasts over leading dimensions.
    # q_hat_inv_mod_q has shape (sizeQ,). in_tower has shape (..., sizeQ).
    c_unreduced = in_tower * constants.q_hat_inv_mod_q

    # Step 2: Modular Reduction
    c = barrett.modular_reduction(c_unreduced, constants.origin_barrett)

    # Step 3: BAT-based matrix multiplication
    summed_terms = matmul_bat_einsum(
        c, constants.q_hat_mod_p_bat, "...q,qpb->...pb"
    )

    # Step 4: Final Modular Reduction
    out_tower = barrett.modular_reduction(
        summed_terms, constants.target_barrett
    )
    return out_tower
