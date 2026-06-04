"""Rescale kernel for CKKS."""

import math
from typing import Iterable
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import ntt
from jaxite.jaxite_ckks import types

Ciphertext = types.Ciphertext


def _gamma_beta_calculation(moduli_list: list[int]):
  """Computes gamma and beta parameters for approximate modulus switching.

  Given a list of moduli [q_0, q_1, ..., q_L], where q_L is the modulus to be
  dropped, let Q = prod_{i=0}^{L-1} q_i be the product of the remaining moduli.

  The rescaling operation computes y = (x - x_L) / q_L, where x_L is the
  centered reduction of x modulo q_L. Modulo each remaining q_i, this is:

    y = (c_i - x_L) * q_L^-1 = c_i * beta_i + x_L * gamma_i (mod q_i)

  where:
    beta_i = q_L^-1 (mod q_i)
    gamma_i = -q_L^-1 (mod q_i)

  Args:
      moduli_list: a list of moduli

  Returns:
      Two arrays containing gamma and beta constants.
  """
  assert len(moduli_list) > 1, "moduli_list must have at least 2 moduli"
  q_l = moduli_list[-1]
  q_prod = math.prod(moduli_list[:-1])
  q_inv_mod_ql = pow(q_prod, -1, q_l)

  # To compute gamma_i, we compute a shared integer gamma_common:
  #
  #   gamma_common = (Q * (Q^-1 mod q_L) - 1) / q_L
  #
  # which satisfies gamma_common = -q_L^-1 (mod q_i) for all i < L because
  # Q = 0 (mod q_i).
  gamma_common = (q_prod * q_inv_mod_ql - 1) // q_l

  gammas = []
  betas = []
  for i in range(len(moduli_list) - 1):
    mod_i = moduli_list[i]
    gamma_i = gamma_common % mod_i
    beta_i = pow(q_l, -1, mod_i)
    gammas.append(gamma_i)
    betas.append(beta_i)
  return jnp.array(gammas, jnp.uint64), jnp.array(betas, jnp.uint64)


@jax.tree_util.register_pytree_node_class
class Rescale:
  """Kernel for in-place ciphertext rescaling.

  This kernel only supports rescaling the last limb of the ciphertext's modulus
  chain. Both single-rescaling and double-rescaling are supported.

  Attributes:
      moduli: the full set of moduli.
      num_rescales: an integer describing the number of rescales to do with each
        application of the kernel. E.g., 1 for single-rescaling and 2 for
        double-rescaling.
      r: r and c are factors of the polynomial modulus degree N used for
        reshaping tensors for TPU data alignment.
      c: see r above.
      gammas_stacked: a set of pre-computed gamma factors for each level, see
        the docs on _gamma_beta_calculation.
      betas_stacked: a set of pre-computed beta factors for each level, see the
        docs on _gamma_beta_calculation.
      thresholds: a list of values (q+1)//2 representing the midpoint of each
        modulus to apply a shift to a centered coset representative mod q.
      ntt_last_limb_kernels: a list of NTT kernels sliced to support applying
        NTT to the last RNS limbs.
      ntt_remaining_limbs_kernels: a list of NTT kernels sliced to support
        applying NTT to the remaining limbs.
  """

  def __init__(self):
    self.moduli = None
    self.num_rescales = 1
    self.r = None
    self.c = None
    self.gammas_stacked = None
    self.betas_stacked = None
    self.thresholds = None
    self.ntt_last_limb_kernels = []
    self.ntt_remaining_limbs_kernels = []

  def precompute_constants(
      self, moduli: Iterable[int], num_rescales: int, r: int, c: int
  ):
    """Precompute constants for rescale computation."""
    self.moduli = list(moduli)
    self.num_rescales = num_rescales
    self.r = r
    self.c = c
    num_moduli = len(self.moduli)

    all_gammas = []
    all_betas = []
    self.thresholds = []
    self.ntt_last_limb_kernels = []
    self.ntt_remaining_limbs_kernels = []

    full_ntt = ntt.NTTBarrett()
    full_ntt.precompute_constants(self.moduli, self.r, self.c)
    full_constants = full_ntt.constants

    current_moduli = self.moduli
    for _ in range(num_rescales):
      n = len(current_moduli)
      gammas, betas = _gamma_beta_calculation(current_moduli)

      # Pad to match full moduli size for easy stacking
      padded_gammas = (
          jnp.zeros((num_moduli - 1,), dtype=jnp.uint64).at[: n - 1].set(gammas)
      )
      padded_betas = (
          jnp.zeros((num_moduli - 1,), dtype=jnp.uint64).at[: n - 1].set(betas)
      )

      all_gammas.append(padded_gammas)
      all_betas.append(padded_betas)

      # Threshold for centered reduction
      last_modulus = current_moduli[-1]
      self.thresholds.append((last_modulus + 1) // 2)

      # Slice NTT constants instead of recomputing
      ntt_last = ntt.NTTBarrett()
      ntt_last.constants = full_constants.slice_moduli(slice(n - 1, n))
      self.ntt_last_limb_kernels.append(ntt_last)

      ntt_remaining = ntt.NTTBarrett()
      ntt_remaining.constants = full_constants.slice_moduli(slice(0, n - 1))
      self.ntt_remaining_limbs_kernels.append(ntt_remaining)

      current_moduli = current_moduli[:-1]

    self.gammas_stacked = jnp.stack(all_gammas)
    self.betas_stacked = jnp.stack(all_betas)
    self.thresholds = jnp.array(self.thresholds, dtype=jnp.uint64)

  def rescale(self, ciphertext: Ciphertext):
    """Performs in-place rescaling on the ciphertext."""
    if self.moduli is None:
      raise ValueError("Constants must be precomputed first.")

    data = ciphertext.data
    num_elements, degree, num_moduli = data.shape
    assert degree == self.r * self.c

    current_moduli = self.moduli

    for iter_idx in range(self.num_rescales):
      n = len(current_moduli)
      last_idx = n - 1

      gammas = self.gammas_stacked[iter_idx, :last_idx]
      betas = self.betas_stacked[iter_idx, :last_idx]
      threshold = self.thresholds[iter_idx]
      last_modulus = current_moduli[-1]
      remaining_moduli = current_moduli[:-1]

      # Extract last limb
      last_limb = data[..., -1:]  # Shape (num_elements, degree, 1)

      # Reshape for NTT
      last_limb_reshaped = last_limb.reshape(num_elements, self.r, self.c, 1)

      # Convert to coefficient form
      ntt_kernel = self.ntt_last_limb_kernels[iter_idx]
      last_limb_coeffs = ntt_kernel.intt(last_limb_reshaped.astype(jnp.uint32))
      last_limb_coeffs = last_limb_coeffs.reshape(num_elements, degree, 1)

      # Centered reduction simulation
      switched = jnp.where(
          last_limb_coeffs < threshold,
          last_limb_coeffs,
          jnp.array(remaining_moduli, jnp.int64).reshape(1, 1, -1)
          - int(last_modulus)
          + last_limb_coeffs.astype(jnp.int64),
      )

      # Multiply by gammas
      twisted = switched.astype(jnp.uint64) * gammas.reshape(1, 1, -1)

      # Reduce and convert back to NTT
      reduced_twisted = (
          twisted
          % jnp.array(remaining_moduli, dtype=jnp.uint64).reshape(1, 1, -1)
      ).astype(jnp.uint32)

      ntt_kernel_work = self.ntt_remaining_limbs_kernels[iter_idx]
      reduced_twisted_reshaped = reduced_twisted.reshape(
          num_elements, self.r, self.c, -1
      )
      ntt_result = ntt_kernel_work.ntt(reduced_twisted_reshaped)

      # Scale remaining limbs and add ntt_result
      remaining_data = data[..., :-1]
      scaled_data = remaining_data.astype(jnp.uint64) * betas.reshape(1, 1, -1)

      # Reshape ntt_result back to (num_elements, degree, num_moduli-1)
      ntt_result_reshaped = ntt_result.reshape(num_elements, degree, -1)

      res = scaled_data + ntt_result_reshaped

      # Final reduction for this iteration
      data = (
          res % jnp.array(remaining_moduli, dtype=jnp.uint64).reshape(1, 1, -1)
      ).astype(jnp.uint32)

      current_moduli = remaining_moduli

    # Update ciphertext object
    ciphertext.data = data
    ciphertext.moduli = jnp.array(current_moduli, dtype=jnp.uint64)

  def tree_flatten(self):
    children = (
        self.gammas_stacked,
        self.betas_stacked,
        self.thresholds,
        self.ntt_last_limb_kernels,
        self.ntt_remaining_limbs_kernels,
    )
    aux_data = (self.moduli, self.num_rescales, self.r, self.c)
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    obj = cls()
    obj.moduli, obj.num_rescales, obj.r, obj.c = aux_data
    (
        obj.gammas_stacked,
        obj.betas_stacked,
        obj.thresholds,
        obj.ntt_last_limb_kernels,
        obj.ntt_remaining_limbs_kernels,
    ) = children
    return obj
