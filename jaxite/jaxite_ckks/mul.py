"""Multiplication and Relinearization kernels for CKKS."""

import dataclasses
from typing import Callable, Iterable, Optional
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import barrett
from jaxite.jaxite_ckks import basis_conversion
from jaxite.jaxite_ckks import ntt
from jaxite.jaxite_ckks import rescale
from jaxite.jaxite_ckks import types
import numpy as np

Ciphertext = types.Ciphertext


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class EvaluationKeys:
  """Evaluation keys for relinearization."""

  a: jax.Array  # Shape (dnum, degree, num_moduli)
  b: jax.Array
  moduli: jax.Array

  def tree_flatten(self):
    """Flatten EvaluationKey into its children and auxiliary data."""
    return (self.a, self.b, self.moduli), ()

  @classmethod
  def tree_unflatten(cls, _, children):
    """Reconstruct EvaluationKeys from auxiliary data and children."""
    return cls(*children)


@jax.tree_util.register_pytree_node_class
class Mul:
  """Kernel for ciphertext multiplication and relinearization.

  Design Note:
  This class is currently specialized for a single level (a specific set of
  moduli) rather than handling multiple levels dynamically.

  Trade-offs:
  - Specializing keeps the device memory footprint small by not loading
    constants for all levels at once. This is important for memory-constrained
    accelerators like TPUs.
  - The cost is that separate instances of `Mul` are needed per level.
  """

  def __init__(
      self,
      rescaler: Optional[rescale.Rescale] = None,
      bconv: Optional[basis_conversion.BasisConversionBarrett] = None,
      ntt_current: Optional[ntt.NTTBarrett] = None,
      ntt_extend: Optional[ntt.NTTBarrett] = None,
      ntt_factory: Optional[Callable[[], ntt.NTTBarrett]] = None,
      full_ntt: Optional[ntt.NTTBarrett] = None,
  ):
    self.is_initialized = False
    self._injected_rescaler = rescaler
    self._injected_bconv = bconv
    self._injected_ntt_current = ntt_current
    self._injected_ntt_extend = ntt_extend
    self._injected_ntt_factory = ntt_factory
    self._injected_full_ntt = full_ntt

  @staticmethod
  def compute_control_indices(
      drop_last_moduli: list[int], extend_moduli: list[int], dnum: int
  ) -> list[list[list[int]]]:
    """Computes control indices for key switching.

    Args:
      drop_last_moduli: The moduli that are dropped.
      extend_moduli: The moduli that are extended.
      dnum: The number of parts for key switching.

    Returns:
      A list of control indices for each part, where each part contains a list
      of selected tower indices and a list of non-selected tower indices.
    """
    size_q_drop_last = len(drop_last_moduli)
    alpha = (size_q_drop_last + dnum - 1) // dnum
    ks_num_parts_ql = (size_q_drop_last + alpha - 1) // alpha

    original_moduli_extract_index = []
    for i in range(size_q_drop_last):
      if i % alpha == 0:
        original_moduli_extract_index.append([i])
      else:
        original_moduli_extract_index[-1].append(i)

    control_indices_list = []

    idx_cur_last_tower = len(drop_last_moduli)
    overall_size_p_in = len(extend_moduli)
    extend_indices = list(
        range(idx_cur_last_tower, idx_cur_last_tower + overall_size_p_in)
    )
    rotate_indices = list(range(idx_cur_last_tower))
    control_indices_list.append([extend_indices, rotate_indices])

    for part in range(ks_num_parts_ql):
      select_tower_overall_index = original_moduli_extract_index[part]
      non_select_tower_overall_index = [
          i
          for i in range(len(drop_last_moduli) + len(extend_moduli))
          if i not in select_tower_overall_index
      ]
      control_indices_list.append(
          [select_tower_overall_index, non_select_tower_overall_index]
      )

    return control_indices_list

  def precompute_constants(
      self,
      original_moduli: Iterable[int],
      extend_moduli: Iterable[int],
      dnum: int,
      r: int,
      c: int,
      composite_degree: int = 1,
  ):
    """Precomputes constants needed for multiplication and relinearization."""
    self.original_moduli = list(original_moduli)
    self.extend_moduli = list(extend_moduli)
    self.dnum = dnum
    self.r = r
    self.c = c
    self.degree = r * c
    self.composite_degree = composite_degree

    self.drop_last_moduli = self.original_moduli[:-composite_degree]
    # Ensure all elements in drop_last_moduli and extend_moduli are strictly
    # less than 2**31 to avoid overflow in both `tensor_multiply` and
    # `relinearize`.
    if not (
        all(m < 2**31 for m in self.drop_last_moduli)
        and all(m < 2**31 for m in self.extend_moduli)
    ):
      raise ValueError("Moduli must be < 2**31")
    self.drop_last_extend_moduli = self.drop_last_moduli + self.extend_moduli

    # 1. Rescale setup
    self.rescaler = self._injected_rescaler or rescale.Rescale()
    # This check determines if rescaler is initialized, and differs from bconv
    # due to API differences.
    if self.rescaler.moduli is None:
      self.rescaler.precompute_constants(
          self.original_moduli, composite_degree, r, c
      )

    # 2. Basis Conversion setup for Key Switch
    control_indices_list = Mul.compute_control_indices(
        self.drop_last_moduli, self.extend_moduli, self.dnum
    )

    size_q_drop_last = len(self.drop_last_moduli)
    alpha = (size_q_drop_last + self.dnum - 1) // self.dnum
    self.ks_alpha = alpha
    self.ks_num_parts_ql = (size_q_drop_last + alpha - 1) // alpha

    ks_select_tower_index_overall = []
    ks_non_select_tower_index_overall = []
    ks_restore_indices = []

    for part in range(self.ks_num_parts_ql):
      select_tower_overall_index, non_select_tower_overall_index = (
          control_indices_list[part + 1]
      )
      concat_order = select_tower_overall_index + non_select_tower_overall_index
      restore_index = [0] * len(concat_order)
      for pos, val in enumerate(concat_order):
        restore_index[val] = pos
      ks_select_tower_index_overall.append(
          jnp.array(select_tower_overall_index, jnp.uint16)
      )
      ks_non_select_tower_index_overall.append(
          jnp.array(non_select_tower_overall_index, jnp.uint16)
      )
      ks_restore_indices.append(jnp.array(restore_index, jnp.uint16))

    self.ks_select_tower_index_overall = ks_select_tower_index_overall
    self.ks_non_select_tower_index_overall = ks_non_select_tower_index_overall
    self.ks_restore_indices = ks_restore_indices

    self.bconv = (
        self._injected_bconv or basis_conversion.BasisConversionBarrett()
    )
    # This check determines if bconv is initialized, and differs from rescaler
    # due to API differences.
    if not self.bconv.precomputed_constants:
      self.bconv.precompute_constants(
          self.drop_last_extend_moduli, control_indices_list
      )

    # 3. NTT kernels for key switch parts
    # We need full NTT constants to slice from
    if (
        self._injected_full_ntt is not None
        and self._injected_full_ntt.constants is not None
    ):
      self.full_ntt_constants = self._injected_full_ntt.constants
    else:
      full_ntt = (
          self._injected_ntt_factory()
          if self._injected_ntt_factory
          else ntt.NTTBarrett()
      )
      full_ntt.precompute_constants(self.drop_last_extend_moduli, r, c)
      self.full_ntt_constants = full_ntt.constants

    if self._injected_ntt_current is not None:
      self.ntt_current = self._injected_ntt_current
    else:
      self.ntt_current = (
          self._injected_ntt_factory()
          if self._injected_ntt_factory
          else ntt.NTTBarrett()
      )
    # .constants are always overwritten on injected NTT kernels to ensure they
    # match the specific moduli slices computed for this Mul instance.
    self.ntt_current.constants = self.full_ntt_constants.slice_moduli(
        slice(0, len(self.drop_last_moduli))
    )

    if self._injected_ntt_extend is not None:
      self.ntt_extend = self._injected_ntt_extend
    else:
      self.ntt_extend = (
          self._injected_ntt_factory()
          if self._injected_ntt_factory
          else ntt.NTTBarrett()
      )
    # .constants are always overwritten on injected NTT kernels to ensure they
    # match the specific moduli slices computed for this Mul instance.
    self.ntt_extend.constants = self.full_ntt_constants.slice_moduli(
        slice(
            len(self.drop_last_moduli),
            len(self.drop_last_moduli) + len(self.extend_moduli),
        )
    )

    self.drop_last_extend_moduli_jax = jnp.array(
        self.drop_last_extend_moduli, dtype=jnp.uint64
    )

    self.ks_ntt_kernels = []
    for part in range(self.ks_num_parts_ql):
      target_indices = ks_non_select_tower_index_overall[part]
      ntt_part = (
          self._injected_ntt_factory()
          if self._injected_ntt_factory
          else ntt.NTTBarrett()
      )
      # Constants are always overwritten to ensure they match the specific
      # moduli slices computed for this Mul instance.
      ntt_part.constants = self.full_ntt_constants.slice_moduli(
          np.array(target_indices)
      )
      self.ks_ntt_kernels.append(ntt_part)

    # 4. PInvModq and q_correction for approx mod down
    p_prod = 1  # pylint: disable=invalid-name
    for moduli in self.extend_moduli:
      p_prod *= moduli
    p_inv_mod_q_approx_down = [
        pow(p_prod, -1, q) for q in self.drop_last_moduli
    ]
    # pylint: disable=invalid-name
    self.PInvModq = jnp.array(p_inv_mod_q_approx_down, dtype=jnp.uint64)

    self.q_correction = jnp.array(self.drop_last_moduli, dtype=jnp.uint64)
    self.q_correction_uint32 = jnp.array(
        self.drop_last_moduli, dtype=jnp.uint32
    )

    self.barrett_constants = barrett.precompute_barrett_constants(
        self.drop_last_moduli
    )
    self.full_barrett_constants = barrett.precompute_barrett_constants(
        self.drop_last_extend_moduli
    )

    self.is_initialized = True

  def tensor_multiply(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
    """Performs tensor multiplication of two ciphertexts after rescaling."""
    if not self.is_initialized:
      raise ValueError("Constants must be precomputed first.")

    # Clone to avoid in-place modification side effects in JAX
    ct1_scaled = Ciphertext(ct1.data, ct1.moduli)
    ct2_scaled = Ciphertext(ct2.data, ct2.moduli)

    self.rescaler.rescale(ct1_scaled)
    self.rescaler.rescale(ct2_scaled)

    a0 = ct1_scaled.data[0].astype(jnp.uint64)
    a1 = ct1_scaled.data[1].astype(jnp.uint64)
    b0 = ct2_scaled.data[0].astype(jnp.uint64)
    b1 = ct2_scaled.data[1].astype(jnp.uint64)

    moduli = ct1_scaled.moduli.astype(jnp.uint64)

    mul0 = (a0 * b0) % moduli
    mul2 = (a1 * b1) % moduli
    mul1 = (a0 * b1 + a1 * b0) % moduli

    data = jnp.stack([mul0, mul1, mul2]).astype(jnp.uint32)
    return Ciphertext(data, ct1_scaled.moduli)

  def relinearize(
      self, ct_3elem: Ciphertext, evk: EvaluationKeys
  ) -> Ciphertext:
    """Performs relinearization on a 3-element ciphertext.

    Relinearization reduces a 3-element ciphertext (resulting from
    multiplication) back to a 2-element ciphertext. It involves a key switch
    operation on the third element, followed by an approximate modulus down
    operation to reduce noise.

    The algorithm proceeds in two main phases: Key Switch and Approximate
    Modulus Down.

    Key Switch:
    *   Domain Conversion: The third element is converted from the NTT domain
        back to the coefficient domain.
    *   Decomposition and Basis Extension: The data is processed in partitions.
        For each part, a subset of towers is selected and extended to the full
        basis using a basis change operation.
    *   The extended parts are converted back to the NTT domain.
    *   Key Application: The reconstructed towers are multiplied with the
        evaluation keys.
    *   Accumulation: The products are accumulated across all parts. While
        individual products are reduced to prevent overflow, the reduction of
        the accumulated sum is deferred and performed only once after all parts
        have been accumulated.

    Approximate Modulus Down:
    *   Extract and Convert: The result of the key switch is split. The part
        corresponding to the extended basis is converted to the coefficient
        domain.
    *   Basis Change: It is mapped back to the original basis via a basis
        change operation.
    *   Back to NTT: The result is converted back to the NTT domain.
    *   Final Combination: This mapped part is subtracted from the original
        basis part of the key switch result, multiplied by the inverse of the
        product of special primes, and added back to the first two elements of
        the original ciphertext to produce the final 2-element ciphertext.

    Args:
      ct_3elem: A 3-element ciphertext resulting from multiplication.
      evk: The evaluation keys used for relinearization.

    Returns:
      The relinearized 2-element ciphertext.

    Raises:
      ValueError: If constants are not precomputed first.
    """
    if not self.is_initialized:
      raise ValueError("Constants must be precomputed first.")

    ct_post_mult = ct_3elem.data[..., :2, :, :]
    last_ele_post_mult = ct_3elem.data[
        ..., 2:3, :, :
    ]  # Shape (..., 1, degree, num_moduli)

    # Key Switch
    # Convert last_ele_post_mult to coeff form
    last_ele_reshaped = last_ele_post_mult.reshape(
        *last_ele_post_mult.shape[:-2], self.r, self.c, -1
    )
    last_ele_coeffs = self.ntt_current.intt(
        last_ele_reshaped.astype(jnp.uint32)
    )
    last_ele_coeffs = last_ele_coeffs.reshape(
        *last_ele_coeffs.shape[:-3], self.degree, -1
    )

    ks_res0 = None
    ks_res1 = None

    for part in range(self.ks_num_parts_ql):
      select_idxs = self.ks_select_tower_index_overall[part]
      # Extract selected towers in coeff form
      part_ct_clone_coef = jnp.take(last_ele_coeffs, select_idxs, axis=-1)

      # Basis change to non-selected towers + extend towers
      # control_index = part + 1 (since index 0 is for approx mod down)
      part_ct_clone_eval = self.bconv.basis_change(
          part_ct_clone_coef, control_index=part + 1
      )

      # Convert basis-changed part to NTT form
      ntt_part = self.ks_ntt_kernels[part]
      part_ct_clone_eval_reshaped = part_ct_clone_eval.reshape(
          *part_ct_clone_eval.shape[:-2], self.r, self.c, -1
      )
      parts_ct_compl_multi_moduli = ntt_part.ntt(
          part_ct_clone_eval_reshaped.astype(jnp.uint32)
      )
      parts_ct_compl_multi_moduli = parts_ct_compl_multi_moduli.reshape(
          *parts_ct_compl_multi_moduli.shape[:-3], self.degree, -1
      )

      # Concatenate original NTT form and basis-changed NTT form
      # original NTT form is just the selected towers from last_ele_post_mult
      original_ntt_part = jnp.take(last_ele_post_mult, select_idxs, axis=-1)

      parts_ct_ext_cur_part = jnp.concatenate(
          [original_ntt_part, parts_ct_compl_multi_moduli], axis=-1
      )

      # Restore original moduli order
      parts_ct_ext_cur_part = jnp.take(
          parts_ct_ext_cur_part, self.ks_restore_indices[part], axis=-1
      )

      # Multiply with evaluation keys
      evk_a_part = evk.a[part]
      evk_b_part = evk.b[part]

      prod_b = parts_ct_ext_cur_part.astype(jnp.uint64) * evk_b_part.astype(
          jnp.uint64
      )
      prod_a = parts_ct_ext_cur_part.astype(jnp.uint64) * evk_a_part.astype(
          jnp.uint64
      )

      # Reduce to prevent overflow
      prod_b = barrett.modular_reduction(prod_b, self.full_barrett_constants)
      prod_a = barrett.modular_reduction(prod_a, self.full_barrett_constants)

      if ks_res0 is None:
        ks_res0 = prod_b.astype(jnp.uint64)
        ks_res1 = prod_a.astype(jnp.uint64)
      else:
        ks_res0 = ks_res0 + prod_b.astype(jnp.uint64)
        ks_res1 = ks_res1 + prod_a.astype(jnp.uint64)

    # Apply modulo reduction once after the loop
    ks_res0 = ks_res0 % self.drop_last_extend_moduli_jax
    ks_res1 = ks_res1 % self.drop_last_extend_moduli_jax

    keyswitch_core_res = jnp.concatenate(
        [ks_res0, ks_res1], axis=-3
    )  # Shape (..., 2, degree, num_moduli)

    # Approximate Modulus Down
    idx_cur_last_tower = len(self.drop_last_moduli)
    overall_size_p_in = len(self.extend_moduli)

    # Extract P towers part for basis conversion
    p_part = keyswitch_core_res[
        ..., idx_cur_last_tower : (idx_cur_last_tower + overall_size_p_in)
    ]

    # Convert to coeff form
    p_part_reshaped = p_part.reshape(*p_part.shape[:-3], 2, self.r, self.c, -1)
    p_part_coeffs = self.ntt_extend.intt(p_part_reshaped.astype(jnp.uint32))
    p_part_coeffs = p_part_coeffs.reshape(
        *p_part_coeffs.shape[:-3], self.degree, -1
    )

    # Basis change from P to Q (drop_last)
    # control_index=0 is for approx mod down
    ct_new_basis_coef = self.bconv.basis_change(p_part_coeffs, control_index=0)

    # Convert back to NTT form
    ct_new_basis_coef_reshaped = ct_new_basis_coef.reshape(
        *ct_new_basis_coef.shape[:-3], 2, self.r, self.c, -1
    )
    ct_new_basis_ntt = self.ntt_current.ntt(
        ct_new_basis_coef_reshaped.astype(jnp.uint32)
    )
    ct_new_basis_ntt = ct_new_basis_ntt.reshape(
        *ct_new_basis_ntt.shape[:-3], self.degree, -1
    )

    # Q part of keyswitch result
    q_part = keyswitch_core_res[..., :idx_cur_last_tower]

    # Compute (q_part - ct_new_basis_ntt) * PInvModq
    moduli_jax = self.q_correction_uint32.reshape(
        (1,) * (q_part.ndim - 1) + (-1,)
    )

    sub_result = jnp.where(
        q_part < ct_new_basis_ntt,
        q_part + moduli_jax - ct_new_basis_ntt,
        q_part - ct_new_basis_ntt,
    )

    pinv_broadcast = self.PInvModq.reshape((1,) * (q_part.ndim - 1) + (-1,))
    approx_mod_down = (
        sub_result.astype(jnp.uint64) * pinv_broadcast
    ) % moduli_jax

    # Add to ct_post_mult
    result = ct_post_mult + approx_mod_down.astype(jnp.uint32)
    val = jnp.where(result >= moduli_jax, result - moduli_jax, result)

    return Ciphertext(val, ct_3elem.moduli)

  # Mul instances should be initialized before being flattened as PyTrees,
  # otherwise injected arguments may be lost.
  def tree_flatten(self):
    """Flattens the Mul instance into children and auxiliary data for JAX PyTree."""
    if not self.is_initialized:
      return ((None,) * 15, (False,))

    children = (
        self.rescaler,
        self.bconv,
        self.ks_select_tower_index_overall,
        self.ks_non_select_tower_index_overall,
        self.ks_restore_indices,
        self.full_ntt_constants,
        self.ks_ntt_kernels,
        self.PInvModq,
        self.q_correction,
        self.barrett_constants,
        self.full_barrett_constants,
        self.ntt_current,
        self.ntt_extend,
        self.drop_last_extend_moduli_jax,
        self.q_correction_uint32,
    )
    aux_data = (
        True,
        tuple(self.original_moduli),
        tuple(self.extend_moduli),
        self.dnum,
        self.r,
        self.c,
        self.composite_degree,
        self.ks_alpha,
        self.ks_num_parts_ql,
    )
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """Reconstructs a Mul instance from auxiliary data and children."""
    obj = cls()
    is_initialized = aux_data[0]
    if not is_initialized:
      return obj

    (
        obj.rescaler,
        obj.bconv,
        obj.ks_select_tower_index_overall,
        obj.ks_non_select_tower_index_overall,
        obj.ks_restore_indices,
        obj.full_ntt_constants,
        obj.ks_ntt_kernels,
        obj.PInvModq,
        obj.q_correction,
        obj.barrett_constants,
        obj.full_barrett_constants,
        obj.ntt_current,
        obj.ntt_extend,
        obj.drop_last_extend_moduli_jax,
        obj.q_correction_uint32,
    ) = children

    (
        _,
        original_moduli,
        extend_moduli,
        obj.dnum,
        obj.r,
        obj.c,
        obj.composite_degree,
        obj.ks_alpha,
        obj.ks_num_parts_ql,
    ) = aux_data
    obj.original_moduli = list(original_moduli)
    obj.extend_moduli = list(extend_moduli)

    obj.drop_last_moduli = obj.original_moduli[: -obj.composite_degree]
    obj.drop_last_extend_moduli = obj.drop_last_moduli + obj.extend_moduli
    obj.degree = obj.r * obj.c
    obj.is_initialized = True
    return obj
