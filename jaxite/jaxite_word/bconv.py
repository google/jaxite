"""BConv: Basis Conversion class for JAX-based homomorphic encryption.

This module provides the BConv class and subclasses which handle basis extension
and modulus switching
using efficient modular reduction. It is designed to work with
vectorized operations on JAX arrays.
"""

import jax
import jax.numpy as jnp
from jaxite.jaxite_word import finite_field as ff_context
import jaxite.jaxite_word.util as util

# Maintain 64-bit precision for large integer arithmetic
jax.config.update("jax_enable_x64", True)


def _is_nvidia():
  return "NVIDIA" in jax.devices()[0].device_kind


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
  lhs = lhs.view(jnp.uint8)

  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)

  # computation
  i8_products = jnp.einsum(
      subscripts, lhs, rhs, preferred_element_type=jnp.uint32
  )
  return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(-1,))


class BConv:

  def __init__(self, overall_moduli):
    """Initialize the BConv object.

    Args:
        overall_moduli: A list or tuple of integers representing the all
          available moduli.
    """
    self.overall_moduli = overall_moduli
    # Lists to store configurations for each control index
    self.original_moduli = []
    self.target_moduli = []
    self.ff_ctx_origin = []
    self.ff_ctx_target = []

    # Lists to store precomputed constants for each control index
    self.QHatInvModq = []
    self.QHatModp = []
    self.QHatModpBAT = []

  def _create_contexts(self, original_moduli, target_moduli):
    """Initialize the finite field contexts. Must be implemented by subclasses.

    Returns: (ff_ctx_origin, ff_ctx_target)
    """
    raise NotImplementedError

  def _generate_constants_single(
      self, original_moduli, target_moduli, ff_ctx_origin, ff_ctx_target
  ):
    """Generates constants for a single configuration."""
    # compute_QHatInvModq_QHatModp returns lists, we convert them to JAX arrays
    # with appropriate shapes for broadcasting.
    QHatInvModq_list, QHatModp_list = util.compute_QHatInvModq_QHatModp(
        original_moduli, target_moduli
    )

    # QHatInvModq: Inverse of (Q/q_i) mod q_i
    # Shape: (sizeQ,) -> JAX array
    QHatInvModq = jnp.array(QHatInvModq_list, dtype=jnp.uint64)
    QHatInvModq = ff_ctx_origin.to_computation_format(QHatInvModq)

    # QHatModp: (Q/q_i) mod p_j
    # Shape: (sizeQ, sizeP) -> JAX array
    QHatModp = jnp.array(QHatModp_list, dtype=jnp.uint64)
    QHatModp = ff_ctx_target.to_computation_format(QHatModp)

    # BAT Preprocessing
    # QHatModpBAT
    # Input QHatModp: (sizeQ, sizeP)
    # _basis_aligned_transformation -> (4, sizeQ, sizeP, 4) (dims: a, q, p, b)
    # We want to match input (..., d, q, a) -> output (..., d, p, b)
    # Transpose to (q, a, p, b) -> (q*a, p, b) for einsum "...dq, qpb -> ...dpb"
    QHatModpBAT_raw = self._basis_aligned_transformation(
        QHatModp, target_moduli
    )
    QHatModpBAT = QHatModpBAT_raw.transpose(1, 0, 2, 3).reshape(
        -1, QHatModpBAT_raw.shape[2], 4
    )

    return QHatInvModq, QHatModp, QHatModpBAT

  def control_gen(self, control_indices_list, perf_test=False):
    """Generates and stores precomputed constants QHatInvModq and QHatModp necessary for

    the basis change operation.

    Args:
        control_indices_list: A sequence of (original_index, target_index)
          tuples/lists.
                              original_index: Indices of original_moduli in
                                overall_moduli.
                              target_index: Indices of target_moduli in
                                overall_moduli.
    """
    # Clear existing lists
    self.original_moduli = []
    self.target_moduli = []
    self.ff_ctx_origin = []
    self.ff_ctx_target = []
    self.QHatInvModq = []
    self.QHatModp = []
    self.QHatModpBAT = []

    for original_index, target_index in control_indices_list:
      omParams = [self.overall_moduli[i] for i in original_index]
      tmParams = [self.overall_moduli[i] for i in target_index]

      self.original_moduli.append(omParams)
      self.target_moduli.append(tmParams)

      ctx_origin, ctx_target = self._create_contexts(omParams, tmParams)
      self.ff_ctx_origin.append(ctx_origin)
      self.ff_ctx_target.append(ctx_target)

      if perf_test:
        sizeQ = len(omParams)
        sizeP = len(tmParams)

        # Mocking QHatInvModq: (sizeQ,)
        QHatInvModq = util.random_parameters(
            (sizeQ,), omParams, dtype=jnp.uint64
        )

        # Mocking QHatModp: (sizeQ, sizeP)
        QHatModp = util.random_parameters(
            (sizeQ, sizeP), tmParams, dtype=jnp.uint64
        )

        # Mocking QHatModpBAT: (sizeQ * 4, sizeP, 4)
        # It's uint8, so just random bytes
        QHatModpBAT = jnp.zeros((sizeQ * 4, sizeP, 4), dtype=jnp.uint8)

        self.QHatInvModq.append(QHatInvModq)
        self.QHatModp.append(QHatModp)
        self.QHatModpBAT.append(QHatModpBAT)
      else:
        QHatInvModq, QHatModp, QHatModpBAT = self._generate_constants_single(
            omParams, tmParams, ctx_origin, ctx_target
        )

        self.QHatInvModq.append(QHatInvModq)
        self.QHatModp.append(QHatModp)
        self.QHatModpBAT.append(QHatModpBAT)

  def _basis_aligned_transformation(self, matrix: jnp.ndarray, moduli):
    """Prepares a matrix for Basis Aligned Transformation (BAT).

    Adapted from ntt_mm.py. Assumes matrix last dimension corresponds to
    'moduli'.
    """
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

  # @functools.partial(jax.jit, static_argnames=("self",))
  def basis_change(
      self, in_tower: jnp.ndarray, control_index: int = 0
  ) -> jnp.ndarray:
    """Performs the approximate basis change from original_moduli to target_moduli.

    Input:
        in_tower: Coefficients in original basis.
                  Shape: (..., ring_dim, sizeQ)
        control_index: Index of the control set to use.

    Output:
        out_tower: Coefficients in new basis.
                   Shape: (..., ring_dim, sizeP)
    """
    # Ensure inputs are correctly typed
    in_tower = jnp.asarray(in_tower, dtype=jnp.uint64)

    # Retrieve constants and contexts for this control index
    QHatInvModq = self.QHatInvModq[control_index]
    QHatModp = self.QHatModp[control_index]
    ff_ctx_origin = self.ff_ctx_origin[control_index]
    ff_ctx_target = self.ff_ctx_target[control_index]

    # Step 1: Compute c_unreduced = in_tower * QHatInvModq
    c_unreduced = in_tower * QHatInvModq

    # Step 2: Modular Reduction on c_unreduced using original moduli context
    c = ff_ctx_origin.modular_reduction(c_unreduced)

    # Base Term: c * QHatModp
    # Shape: (..., d, p)
    if _is_nvidia():
      summed_terms = jnp.einsum(
          "...dq,qp->...dp",
          c.astype(jnp.uint32),
          QHatModp.astype(jnp.uint32),
          preferred_element_type=jnp.uint64,
      )
    else:
      products = (
          c[..., None].astype(jnp.uint64) * QHatModp[None, ...]
      )  # Need to convert it into BAT based implementation
      summed_terms = jnp.sum(products, axis=-2)

    # Step 4: Final Modular Reduction using target moduli context
    out_tower = ff_ctx_target.modular_reduction(summed_terms)

    return out_tower

  # @functools.partial(jax.jit, static_argnames=("self",))
  def basis_change_bat(
      self, in_tower: jnp.ndarray, control_index: int = 0
  ) -> jnp.ndarray:
    """Performs the approximate basis change using BAT optimization.

    Currently does not support modulus switching.

    Input:
        in_tower: Coefficients in original basis.
                  Shape: (..., ring_dim, sizeQ)
        control_index: Index of the control set to use.

    Output:
        out_tower: Coefficients in new basis.
                   Shape: (..., ring_dim, sizeP)
    """

    # Ensure inputs are u64 for BAT
    # Note: We assume inputs fit in u64 (< 2^64)
    in_tower_u64 = jnp.asarray(in_tower, dtype=jnp.uint64)

    # Retrieve constants and contexts
    QHatInvModq = self.QHatInvModq[control_index]
    QHatModpBAT = self.QHatModpBAT[control_index]
    ff_ctx_origin = self.ff_ctx_origin[control_index]
    ff_ctx_target = self.ff_ctx_target[control_index]

    # Step 1: Compute c_unreduced = in_tower * QHatInvModqBAT
    c_unreduced = in_tower_u64 * QHatInvModq

    # Step 2: Modular Reduction
    c = ff_ctx_origin.modular_reduction(c_unreduced).astype(jnp.uint32)

    # QHatModpBAT: (q*a, p, b)
    summed_terms = matmul_bat_einsum(c, QHatModpBAT, "...q,qpb->...pb")

    # Step 4: Final Modular Reduction
    out_tower = ff_ctx_target.modular_reduction(summed_terms)

    return out_tower


class BConvBarrett(BConv):

  def _create_contexts(self, original_moduli, target_moduli):
    return (
        ff_context.BarrettContext(moduli=original_moduli),
        ff_context.BarrettContext(moduli=target_moduli),
    )


class BConvMontgomery(BConv):

  def _create_contexts(self, original_moduli, target_moduli):
    return (
        ff_context.MontgomeryContext(moduli=original_moduli),
        ff_context.MontgomeryContext(moduli=target_moduli),
    )

  def basis_change(
      self, in_tower: jnp.ndarray, control_index: int = 0
  ) -> jnp.ndarray:
    in_tower = jnp.asarray(in_tower, dtype=jnp.uint64)

    QHatInvModq = self.QHatInvModq[control_index]
    QHatModp = self.QHatModp[control_index]
    ff_ctx_origin = self.ff_ctx_origin[control_index]
    ff_ctx_target = self.ff_ctx_target[control_index]

    c_unreduced = in_tower * QHatInvModq
    c = ff_ctx_origin.modular_reduction(c_unreduced)

    # Domain conversion for Montgomery
    c_reduced_back = ff_ctx_origin.to_original_format(c)
    c = ff_ctx_target.to_computation_format(c_reduced_back)

    if _is_nvidia():
      summed_terms = jnp.einsum(
          "...dq,qp->...dp",
          c.astype(jnp.uint32),
          QHatModp.astype(jnp.uint32),
          preferred_element_type=jnp.uint64,
      )
    else:
      products = (
          c[..., None].astype(jnp.uint64) * QHatModp[None, ...]
      )  # Need to convert it into BAT based implementation
      summed_terms = jnp.sum(products, axis=-2)

    out_tower = ff_ctx_target.modular_reduction(summed_terms)
    return out_tower


class BConvShoup(BConv):

  def __init__(self, overall_moduli):
    super().__init__(overall_moduli)
    self.QHatInvModq_shoup = []
    self.QHatModp_shoup = []

  def _create_contexts(self, original_moduli, target_moduli):
    return (
        ff_context.ShoupContext(moduli=original_moduli),
        ff_context.ShoupContext(moduli=target_moduli),
    )

  def control_gen(self, control_indices_list, perf_test=False):
    super().control_gen(control_indices_list)
    # Clear shoup lists
    self.QHatInvModq_shoup = []
    self.QHatModp_shoup = []

    for i in range(len(control_indices_list)):
      QHatInvModq = self.QHatInvModq[i]
      QHatModp = self.QHatModp[i]
      ff_ctx_origin = self.ff_ctx_origin[i]
      ff_ctx_target = self.ff_ctx_target[i]

      QHatInvModq_shoup = ff_ctx_origin.precompute_constant_operand(QHatInvModq)
      QHatModp_shoup = ff_ctx_target.precompute_constant_operand(QHatModp)

      self.QHatInvModq_shoup.append(QHatInvModq_shoup)
      self.QHatModp_shoup.append(QHatModp_shoup)

  def basis_change(
      self, in_tower: jnp.ndarray, control_index: int = 0
  ) -> jnp.ndarray:
    in_tower = jnp.asarray(in_tower, dtype=jnp.uint64)

    QHatInvModq = self.QHatInvModq[control_index]
    QHatInvModq_shoup = self.QHatInvModq_shoup[control_index]
    QHatModp = self.QHatModp[control_index]
    QHatModp_shoup = self.QHatModp_shoup[control_index]
    ff_ctx_origin = self.ff_ctx_origin[control_index]
    ff_ctx_target = self.ff_ctx_target[control_index]

    c_unreduced = in_tower * QHatInvModq

    # Use dual-operand reduction
    c_unreduced_shoup = in_tower * QHatInvModq_shoup
    c = ff_ctx_origin.modular_reduction(c_unreduced, c_unreduced_shoup)

    if _is_nvidia():
      summed_terms = jnp.einsum(
          "...dq,qp->...dp",
          c.astype(jnp.uint32),
          QHatModp.astype(jnp.uint32),
          preferred_element_type=jnp.uint64,
      )
    else:
      products = (
          c[..., None].astype(jnp.uint64) * QHatModp[None, ...]
      )  # Need to convert it into BAT based implementation
      summed_terms = jnp.sum(products, axis=-2)

    summed_terms_shoup = jnp.einsum(
        "...dq,qp->...dp",
        c.astype(jnp.uint32),
        QHatModp_shoup.astype(jnp.uint32),
        preferred_element_type=jnp.uint64,
    )
    out_tower = ff_ctx_target.modular_reduction(
        summed_terms, summed_terms_shoup
    )
    return out_tower


class BConvBATLazy(BConv):

  def _create_contexts(self, original_moduli, target_moduli):
    return (
        ff_context.BATLazyContext(moduli=original_moduli),
        ff_context.BATLazyContext(moduli=target_moduli),
    )

  def basis_change(
      self, in_tower: jnp.ndarray, control_index: int = 0
  ) -> jnp.ndarray:
    in_tower = jnp.asarray(in_tower, dtype=jnp.uint64)

    QHatInvModq = self.QHatInvModq[control_index]
    QHatModp = self.QHatModp[control_index]
    ff_ctx_origin = self.ff_ctx_origin[control_index]
    ff_ctx_target = self.ff_ctx_target[control_index]

    c_unreduced = in_tower * QHatInvModq
    c = ff_ctx_origin.modular_reduction(c_unreduced)

    # Force strict reduction for BATLazy correctness
    c = ff_ctx_origin.to_original_format(c)

    if _is_nvidia():
      summed_terms = jnp.einsum(
          "...dq,qp->...dp",
          c.astype(jnp.uint32),
          QHatModp.astype(jnp.uint32),
          preferred_element_type=jnp.uint64,
      )
    else:
      products = (
          c[..., None].astype(jnp.uint64) * QHatModp[None, ...]
      )  # Need to convert it into BAT based implementation
      summed_terms = jnp.sum(products, axis=-2)

    out_tower = ff_ctx_target.modular_reduction(summed_terms)
    return out_tower

  def basis_change_bat(
      self, in_tower: jnp.ndarray, control_index: int = 0
  ) -> jnp.ndarray:
    in_tower = jnp.asarray(in_tower, dtype=jnp.uint64)

    QHatInvModq = self.QHatInvModq[control_index]
    QHatModpBAT = self.QHatModpBAT[control_index]
    ff_ctx_origin = self.ff_ctx_origin[control_index]
    ff_ctx_target = self.ff_ctx_target[control_index]

    c_unreduced = in_tower * QHatInvModq
    c = ff_ctx_origin.modular_reduction(c_unreduced).astype(jnp.uint32)

    # Enforce strict reduction
    c = ff_ctx_origin.to_original_format(c)

    summed_terms = matmul_bat_einsum(c, QHatModpBAT, "...q,qpb->...pb")
    out_tower = ff_ctx_target.modular_reduction(summed_terms)
    return out_tower
