# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rescale kernel for CKKS."""

import math
from typing import Iterable

import jax
from jax.experimental import pallas as pl
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


def make_rescale_total_t_kernel(
    L: int,
    num_rescales: int,
    num_elements: int,
    r: int,
    c: int,
    n_rem: int,
    interpret: bool = False,
):
  """Returns a Pallas kernel for accumulating total_t.

  This kernel computes the `final_t` component for the rescaling operation
  by iteratively applying modular arithmetic operations over multiple rescale
  steps. The grid processes (element, row) combinations, looping over columns
  and remaining limbs internally to comply with TPU hardware vector dimension
  constraints.

  Args:
    L: Total number of moduli.
    num_rescales: The number of rescale operations to perform.
    num_elements: Number of elements in the batch.
    r: Reshaped dimension r.
    c: Reshaped dimension c.
    n_rem: Number of remaining moduli.
    interpret: If True, executes in interpreter mode (useful for CPU).

  Returns:
    A Pallas kernel callable that computes the `final_t` values.
  """
  out_shape = jax.ShapeDtypeStruct((num_elements, r, c, n_rem), jnp.uint64)
  grid = (num_elements, r)

  def _rescale_total_t_kernel(
      z_values_ref,
      gammas_stacked_ref,
      betas_stacked_ref,
      moduli_ref,
      thresholds_ref,
      final_t_ref,
  ):
    # Loop over all columns in the row
    for col_idx in range(c):
      # Load precomputed z_values locally for this elements (e, r, col_idx).
      z_values_local = [
          jnp.zeros(1, dtype=jnp.uint64) for _ in range(num_rescales)
      ]
      for k in range(num_rescales):
        z_values_local[k] = z_values_ref[0, 0, col_idx, k].astype(jnp.uint64)

      # Loop over all remaining modulus limbs
      for m_idx in range(n_rem):
        total_t_val = jnp.zeros(1, dtype=jnp.uint64)
        x_i_val = z_values_local[0]  # x_0 = z_0 for current (e, r, col_idx)

        rem_m_limb = moduli_ref[m_idx]

        # The sequential loop over rescale steps (i)
        for i in range(num_rescales):
          # Load constants for step 'i' from the full precomputed arrays
          current_mod_idx_q_L = L - 1 - i
          qi_q_L = moduli_ref[current_mod_idx_q_L]

          gi_rem_val = gammas_stacked_ref[i, m_idx].astype(jnp.uint64)
          bi_rem_val = betas_stacked_ref[i, m_idx].astype(jnp.uint64)
          threshold_val = thresholds_ref[i]

          # Compute c_i and t_i_rem (Equation 1 in CKKS Rescale)
          ci_rem_val = (
              rem_m_limb - (qi_q_L * gi_rem_val % rem_m_limb)
          ) % rem_m_limb
          is_large = (x_i_val >= threshold_val).astype(jnp.uint64)
          ti_rem_val = (
              x_i_val * gi_rem_val + is_large * ci_rem_val
          ) % rem_m_limb

          # Update total_t_val (Equation 2 in CKKS Rescale)
          total_t_val = (total_t_val * bi_rem_val + ti_rem_val) % rem_m_limb

          # Calculate new_x for the next iteration if applicable (Equation 3 in CKKS Rescale)
          if i < num_rescales - 1:
            next_z_val = z_values_local[i + 1]
            next_mod_idx_q = L - 2 - i

            qi_next_q = moduli_ref[next_mod_idx_q]
            gi_next_for_x = gammas_stacked_ref[i, next_mod_idx_q].astype(
                jnp.uint64
            )
            bi_next_for_x = betas_stacked_ref[i, next_mod_idx_q].astype(
                jnp.uint64
            )

            ci_next_for_x = (
                qi_next_q - (qi_q_L * gi_next_for_x % qi_next_q)
            ) % qi_next_q

            x_i_val = (
                next_z_val * bi_next_for_x
                + (x_i_val * gi_next_for_x + is_large * ci_next_for_x)
            ) % qi_next_q

        final_t_ref[0, 0, col_idx, m_idx] = total_t_val[0]

  return pl.pallas_call(
      _rescale_total_t_kernel,
      out_shape=out_shape,
      grid=grid,
      in_specs=[
          # Precomputed z_values (INTTs of dropped limbs)
          pl.BlockSpec(
              index_map=lambda e, r: (e, r, 0, 0),
              block_shape=(1, 1, c, num_rescales),
          ),  # z_values_ref
          pl.BlockSpec(
              index_map=lambda e, r: (0, 0),
              block_shape=(num_rescales, L - 1),
          ),  # gammas_stacked_ref
          pl.BlockSpec(
              index_map=lambda e, r: (0, 0),
              block_shape=(num_rescales, L - 1),
          ),  # betas_stacked_ref
          pl.BlockSpec(
              index_map=lambda e, r: (0,), block_shape=(L,)
          ),  # moduli_ref
          pl.BlockSpec(
              index_map=lambda e, r: (0,), block_shape=(num_rescales,)
          ),  # thresholds_ref
      ],
      out_specs=pl.BlockSpec(
          index_map=lambda e, r: (e, r, 0, 0),
          block_shape=(1, 1, c, n_rem),
      ),  # final_t_ref
      name="rescale_total_t_kernel",
      interpret=interpret,
  )


def make_rescale_total_beta_kernel(
    L: int, num_rescales: int, n_rem: int, interpret: bool = False
):
  """Returns a Pallas kernel for accumulating total_beta.

  This kernel computes the `final_beta` component for the rescaling operation.
  A single program executes the loop for all remaining moduli in sequence to
  comply
  with TPU hardware vector dimension constraints.

  Args:
    L: Total number of moduli.
    num_rescales: The number of rescale operations to perform.
    n_rem: Number of remaining moduli.
    interpret: If True, executes in interpreter mode (useful for CPU).

  Returns:
    A Pallas kernel callable that computes the `final_beta` values.
  """
  out_shape = jax.ShapeDtypeStruct((n_rem,), jnp.uint64)
  grid = (1,)

  def _rescale_total_beta_kernel(betas_stacked_ref, moduli_ref, final_beta_ref):
    for m_idx in range(n_rem):
      total_beta_val = jnp.ones(1, dtype=jnp.uint64)
      rem_m_limb = moduli_ref[m_idx]

      # The total_beta calculation only depends on bi_rem and rem_m,
      # not x_i or is_large.
      for i in range(num_rescales):
        bi_rem_val = betas_stacked_ref[i, m_idx].astype(jnp.uint64)
        total_beta_val = (total_beta_val * bi_rem_val) % rem_m_limb

      final_beta_ref[m_idx] = total_beta_val[0]

  return pl.pallas_call(
      _rescale_total_beta_kernel,
      out_shape=out_shape,
      grid=grid,
      in_specs=[
          pl.BlockSpec(
              index_map=lambda _: (0, 0), block_shape=(num_rescales, L - 1)
          ),  # betas_stacked_ref
          pl.BlockSpec(
              index_map=lambda _: (0,), block_shape=(L,)
          ),  # moduli_ref
      ],
      out_specs=pl.BlockSpec(
          index_map=lambda _: (0,), block_shape=(n_rem,)
      ),  # final_beta_ref
      name="rescale_total_beta_kernel",
      interpret=interpret,
  )


@jax.tree_util.register_pytree_node_class
class Rescale:
  """Kernel for ciphertext rescaling.

  This kernel only supports rescaling the last limb of the ciphertext's modulus
  chain. Supports multi-step rescaling (single, double, and multi-step).

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
  moduli: list[int] | None
  num_rescales: int
  r: int
  c: int
  gammas_stacked: jax.Array | None
  betas_stacked: jax.Array | None
  thresholds: jax.Array | None
  ntt_last_limb_kernels: list[ntt.NTTBarrett]
  ntt_remaining_limbs_kernels: list[ntt.NTTBarrett]

  def __init__(self):
    self.moduli = None
    self.num_rescales = 1
    self.r = 0
    self.c = 0
    self.gammas_stacked = None
    self.betas_stacked = None
    self.thresholds = None
    self.ntt_last_limb_kernels = []
    self.ntt_remaining_limbs_kernels = []

  def precompute_constants(
      self, moduli: Iterable[int], num_rescales: int, r: int, c: int
  ):
    """Precompute constants for rescale computation."""
    print("    [rescale] Starting precompute_constants...", flush=True)
    self.moduli = list(moduli)
    self.num_rescales = num_rescales
    self.r = r
    self.c = c
    num_moduli = len(self.moduli)

    all_gammas = []
    all_betas = []
    thresholds = []
    self.ntt_last_limb_kernels = []
    self.ntt_remaining_limbs_kernels = []

    print("    [rescale] Precomputing full NTT...", flush=True)
    full_ntt = ntt.NTTBarrett()
    full_ntt.precompute_constants(self.moduli, self.r, self.c)
    full_constants = full_ntt.constants
    print("    [rescale] Full NTT constants precomputed.", flush=True)

    current_moduli = self.moduli
    for i in range(num_rescales):
      print(f"    [rescale] Loop iteration {i}...", flush=True)
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
      thresholds.append((last_modulus + 1) // 2)

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
    self.thresholds = jnp.array(thresholds, dtype=jnp.uint64)

  def rescale(self, ciphertext: Ciphertext) -> Ciphertext:
    """Performs rescaling on the ciphertext (not in-place)."""
    if (
        self.moduli is None
        or self.thresholds is None
        or self.gammas_stacked is None
        or self.betas_stacked is None
    ):
      raise ValueError("Constants must be precomputed first.")

    num_elements = ciphertext.data.shape[0]
    L = ciphertext.moduli.shape[0]
    n_rem = L - self.num_rescales
    degree = self.r * self.c
    data_2d = ciphertext.data.reshape(num_elements, self.r, self.c, L)
    moduli = ciphertext.moduli.astype(jnp.uint64)

    # Prefetch the necessary last_limbs into Vmem to hide HBM latency.
    last_limbs = data_2d[..., L - self.num_rescales : L]
    remaining_moduli = ciphertext.moduli[:n_rem]
    thresholds = self.thresholds.astype(jnp.uint64)

    if self.num_rescales == 1:
      # Optimized float64 Barrett reduction for K=1
      iter_idx = 0
      last_idx = L - 1
      last_limb = last_limbs[..., 0:1]
      rem_limbs = data_2d[..., :-1]

      mod_u32 = remaining_moduli.astype(jnp.uint32).reshape(1, 1, 1, -1)
      mod_i32 = mod_u32.astype(jnp.int32)
      mod_f64 = mod_u32.astype(jnp.float64)
      inv_mod_f64 = 1.0 / mod_f64

      g_u32 = (
          self.gammas_stacked[iter_idx, :last_idx]
          .astype(jnp.uint32)
          .reshape(1, 1, 1, -1)
      )
      g_f64 = g_u32.astype(jnp.float64)
      b_u32 = (
          self.betas_stacked[iter_idx, :last_idx]
          .astype(jnp.uint32)
          .reshape(1, 1, 1, -1)
      )
      b_f64 = b_u32.astype(jnp.float64)

      # 1. Transform last limb to coefficient domain.
      last_limb_coeffs = self.ntt_last_limb_kernels[iter_idx].intt(
          last_limb.astype(jnp.uint32)
      )

      # 2. Scale and center using fused 64-bit Barrett reduction logic.
      is_large = (last_limb_coeffs >= self.thresholds[iter_idx]).astype(
          jnp.uint32
      )
      a1_f64 = last_limb_coeffs.astype(jnp.float64)
      c1_f64 = is_large.astype(jnp.float64)

      q1 = jnp.floor((a1_f64 * g_f64 + c1_f64) * inv_mod_f64).astype(jnp.uint32)
      r1 = (
          last_limb_coeffs.astype(jnp.uint32) * g_u32 + is_large - q1 * mod_u32
      ).astype(jnp.int32)
      q1c = jnp.floor(r1.astype(jnp.float64) * inv_mod_f64).astype(jnp.int32)
      red1 = r1 - q1c * mod_i32
      red1 = jnp.where(red1 < 0, red1 + mod_i32, red1)
      red1 = jnp.where(red1 >= mod_i32, red1 - mod_i32, red1)
      reduced_twisted = red1.astype(jnp.uint32)

      # 3. Transform result back to the NTT domain.
      ntt_result = self.ntt_remaining_limbs_kernels[iter_idx].ntt(
          reduced_twisted
      )

      # 4. Rescale remaining limbs in the NTT domain using the same fused Barrett logic.
      a2_f64 = rem_limbs.astype(jnp.float64)
      c2_f64 = ntt_result.astype(jnp.float64)

      q2 = jnp.floor((a2_f64 * b_f64 + c2_f64) * inv_mod_f64).astype(jnp.uint32)
      r2 = (
          rem_limbs.astype(jnp.uint32) * b_u32
          + ntt_result.astype(jnp.uint32)
          - q2 * mod_u32
      ).astype(jnp.int32)
      q2c = jnp.floor(r2.astype(jnp.float64) * inv_mod_f64).astype(jnp.int32)
      red2 = r2 - q2c * mod_i32
      red2 = jnp.where(red2 < 0, red2 + mod_i32, red2)
      red2 = jnp.where(red2 >= mod_i32, red2 - mod_i32, red2)
      new_data_rc = red2.astype(jnp.uint32)

      return types.Ciphertext(
          data=new_data_rc.reshape(num_elements, degree, n_rem),
          moduli=remaining_moduli,
      )

    if self.num_rescales == 2:
      # Unrolled K=2 implementation
      q_L1, q_L2 = moduli[L - 1], moduli[L - 2]
      g0 = self.gammas_stacked[0].astype(jnp.uint64)
      b0 = self.betas_stacked[0].astype(jnp.uint64)
      g1 = self.gammas_stacked[1].astype(jnp.uint64)
      b1 = self.betas_stacked[1].astype(jnp.uint64)
      z0 = (
          self.ntt_last_limb_kernels[0]
          .intt(last_limbs[..., 1:2].astype(jnp.uint32))
          .astype(jnp.uint64)
      )
      z1 = (
          self.ntt_last_limb_kernels[1]
          .intt(last_limbs[..., 0:1].astype(jnp.uint32))
          .astype(jnp.uint64)
      )
      z0_flat, z1_flat = z0[..., 0], z1[..., 0]
      is_large0 = z0_flat >= thresholds[0]
      c0_L2 = (q_L2 - (q_L1 * g0[n_rem] % q_L2)) % q_L2
      t0_L2 = (
          z0_flat * g0[n_rem] + is_large0.astype(jnp.uint64) * c0_L2
      ) % q_L2
      x1 = (z1_flat * b0[n_rem] + t0_L2) % q_L2
      is_large1 = x1 >= thresholds[1]
      total_t_list = []
      for i in range(n_rem):
        qi = moduli[i]
        c0_i = (qi - (q_L1 * g0[i] % qi)) % qi
        c1_i = (qi - (q_L2 * g1[i] % qi)) % qi
        t0_i = (z0_flat * g0[i] + jnp.where(is_large0, c0_i, 0)) % qi
        total_t_list.append(
            (t0_i * b1[i] + x1 * g1[i] + is_large1.astype(jnp.uint64) * c1_i)
            % qi
        )
      t_ntt = (
          self.ntt_remaining_limbs_kernels[1]
          .ntt(jnp.stack(total_t_list, axis=-1).astype(jnp.uint32))
          .astype(jnp.uint64)
      )
      new_data_list = []
      for i in range(n_rem):
        qi = moduli[i]
        beta_i = (b0[i] * b1[i]) % qi
        new_data_list.append(
            (data_2d[..., i].astype(jnp.uint64) * beta_i + t_ntt[..., i]) % qi
        )
      new_data = jnp.stack(new_data_list, axis=-1)
      return types.Ciphertext(
          data=new_data.astype(jnp.uint32).reshape(num_elements, degree, n_rem),
          moduli=ciphertext.moduli[:n_rem],
      )

    if self.num_rescales > 2:
      return self._rescale_fused_jax(ciphertext)

    raise ValueError("Invalid number of rescales.")

  def _rescale_sequential(self, ciphertext: Ciphertext) -> Ciphertext:
    """Old sequential fallback implementation (used for benchmarking)."""
    assert self.gammas_stacked is not None
    assert self.betas_stacked is not None
    assert self.thresholds is not None

    num_elements = ciphertext.data.shape[0]
    degree = self.r * self.c
    current_data = ciphertext.data
    current_moduli = ciphertext.moduli

    for iter_idx in range(self.num_rescales):
      n = len(current_moduli)
      last_idx = n - 1
      n_rem = n - 1

      data_2d = current_data.reshape(num_elements, self.r, self.c, n)

      last_limb = data_2d[..., -1:]
      rem_limbs = data_2d[..., :-1]
      remaining_moduli = current_moduli[:-1]

      mod_u32 = remaining_moduli.astype(jnp.uint32).reshape(1, 1, 1, -1)
      mod_i32 = mod_u32.astype(jnp.int32)
      mod_f64 = mod_u32.astype(jnp.float64)
      inv_mod_f64 = 1.0 / mod_f64

      g_u32 = (
          self.gammas_stacked[iter_idx, :last_idx]
          .astype(jnp.uint32)
          .reshape(1, 1, 1, -1)
      )
      g_f64 = g_u32.astype(jnp.float64)
      b_u32 = (
          self.betas_stacked[iter_idx, :last_idx]
          .astype(jnp.uint32)
          .reshape(1, 1, 1, -1)
      )
      b_f64 = b_u32.astype(jnp.float64)

      last_limb_coeffs = self.ntt_last_limb_kernels[iter_idx].intt(
          last_limb.astype(jnp.uint32)
      )

      is_large = (last_limb_coeffs >= self.thresholds[iter_idx]).astype(
          jnp.uint32
      )
      a1_f64 = last_limb_coeffs.astype(jnp.float64)
      c1_f64 = is_large.astype(jnp.float64)

      q1 = jnp.floor((a1_f64 * g_f64 + c1_f64) * inv_mod_f64).astype(jnp.uint32)
      r1 = (
          last_limb_coeffs.astype(jnp.uint32) * g_u32 + is_large - q1 * mod_u32
      ).astype(jnp.int32)
      q1c = jnp.floor(r1.astype(jnp.float64) * inv_mod_f64).astype(jnp.int32)
      red1 = r1 - q1c * mod_i32
      red1 = jnp.where(red1 < 0, red1 + mod_i32, red1)
      red1 = jnp.where(red1 >= mod_i32, red1 - mod_i32, red1)
      reduced_twisted = red1.astype(jnp.uint32)

      ntt_result = self.ntt_remaining_limbs_kernels[iter_idx].ntt(
          reduced_twisted
      )

      a2_f64 = rem_limbs.astype(jnp.float64)
      c2_f64 = ntt_result.astype(jnp.float64)

      q2 = jnp.floor((a2_f64 * b_f64 + c2_f64) * inv_mod_f64).astype(jnp.uint32)
      r2 = (
          rem_limbs.astype(jnp.uint32) * b_u32
          + ntt_result.astype(jnp.uint32)
          - q2 * mod_u32
      ).astype(jnp.int32)
      q2c = jnp.floor(r2.astype(jnp.float64) * inv_mod_f64).astype(jnp.int32)
      red2 = r2 - q2c * mod_i32
      red2 = jnp.where(red2 < 0, red2 + mod_i32, red2)
      red2 = jnp.where(red2 >= mod_i32, red2 - mod_i32, red2)
      new_data_rc = red2.astype(jnp.uint32)

      current_data = new_data_rc.reshape(num_elements, degree, n_rem)
      current_moduli = remaining_moduli

    return types.Ciphertext(
        data=current_data,
        moduli=jnp.array(current_moduli, dtype=jnp.uint64),
    )

  def _rescale_fused_jax(self, ciphertext: Ciphertext) -> Ciphertext:
    """Fused JAX implementation for K>2 with parallel Barrett reductions."""
    num_elements = ciphertext.data.shape[0]
    L = ciphertext.moduli.shape[0]
    num_kept = L - self.num_rescales
    degree = self.r * self.c
    data_view = ciphertext.data.reshape(num_elements, self.r, self.c, L)
    initial_moduli = ciphertext.moduli.astype(jnp.uint64)
    moduli_kept = initial_moduli[:num_kept]

    mod_kept_f64 = moduli_kept.astype(jnp.float64).reshape(1, 1, 1, -1)
    inv_mod_kept_f64 = 1.0 / mod_kept_f64
    mod_kept_u32 = moduli_kept.astype(jnp.uint32).reshape(1, 1, 1, -1)
    mod_kept_i32 = moduli_kept.astype(jnp.int32).reshape(1, 1, 1, -1)

    dropped_coeffs = []
    for i in range(self.num_rescales):
      limb = data_view[..., L - 1 - i : L - i]
      limb_coeffs = self.ntt_last_limb_kernels[i].intt(limb.astype(jnp.uint32))
      dropped_coeffs.append(limb_coeffs[..., 0].astype(jnp.uint64))

    total_corr = jnp.zeros(
        (num_elements, self.r, self.c, num_kept), dtype=jnp.uint64
    )

    assert self.gammas_stacked is not None
    assert self.betas_stacked is not None
    assert self.thresholds is not None

    gammas = self.gammas_stacked.astype(jnp.uint64)
    betas = self.betas_stacked.astype(jnp.uint64)
    thresholds = self.thresholds.astype(jnp.uint64)

    def barrett_reduce_single(val, mod):
      mod_f64 = mod.astype(jnp.float64)
      inv_mod_f64 = 1.0 / mod_f64
      mod_u32 = mod.astype(jnp.uint32)
      mod_i32 = mod.astype(jnp.int32)

      val_f64 = val.astype(jnp.float64)
      q = jnp.floor(val_f64 * inv_mod_f64).astype(jnp.uint32)
      r = (val.astype(jnp.uint32) - q * mod_u32).astype(jnp.int32)
      q_c = jnp.floor(r.astype(jnp.float64) * inv_mod_f64).astype(jnp.int32)
      red = r - q_c * mod_i32
      red = jnp.where(red < 0, red + mod_i32, red)
      red = jnp.where(red >= mod_i32, red - mod_i32, red)
      return red.astype(jnp.uint64)

    for i in range(self.num_rescales):
      m_curr = dropped_coeffs[i]
      mask = (m_curr >= thresholds[i]).astype(jnp.uint64)

      g_k = gammas[i, :num_kept].reshape(1, 1, 1, -1)
      b_k = betas[i, :num_kept].reshape(1, 1, 1, -1)

      val = (
          total_corr * b_k
          + m_curr[..., jnp.newaxis] * g_k
          + mask[..., jnp.newaxis]
      )

      val_f64 = val.astype(jnp.float64)
      q = jnp.floor(val_f64 * inv_mod_kept_f64).astype(jnp.uint32)
      r = (val.astype(jnp.uint32) - q * mod_kept_u32).astype(jnp.int32)
      q_c = jnp.floor(r.astype(jnp.float64) * inv_mod_kept_f64).astype(
          jnp.int32
      )
      red = r - q_c * mod_kept_i32
      red = jnp.where(red < 0, red + mod_kept_i32, red)
      red = jnp.where(red >= mod_kept_i32, red - mod_kept_i32, red)
      total_corr = red.astype(jnp.uint64)

      for j in range(i + 1, self.num_rescales):
        idx_m = L - 1 - j
        g_v = gammas[i, idx_m]
        b_v = betas[i, idx_m]
        m_v = initial_moduli[idx_m]

        val_v = m_curr * g_v + mask
        c_v = barrett_reduce_single(val_v, m_v)

        val_next = dropped_coeffs[j] * b_v + c_v
        dropped_coeffs[j] = barrett_reduce_single(val_next, m_v)

    ntt_corr = self.ntt_remaining_limbs_kernels[self.num_rescales - 1].ntt(
        total_corr.astype(jnp.uint32)
    )

    total_scale = jnp.ones((num_kept,), dtype=jnp.uint64)
    for i in range(self.num_rescales):
      total_scale = (total_scale * betas[i, :num_kept]) % moduli_kept

    val_final = data_view[..., :num_kept].astype(
        jnp.uint64
    ) * total_scale.reshape(1, 1, 1, -1) + ntt_corr.astype(jnp.uint64)

    val_final_f64 = val_final.astype(jnp.float64)
    q_f = jnp.floor(val_final_f64 * inv_mod_kept_f64).astype(jnp.uint32)
    r_f = (val_final.astype(jnp.uint32) - q_f * mod_kept_u32).astype(jnp.int32)
    q_fc = jnp.floor(r_f.astype(jnp.float64) * inv_mod_kept_f64).astype(
        jnp.int32
    )
    red_f = r_f - q_fc * mod_kept_i32
    red_f = jnp.where(red_f < 0, red_f + mod_kept_i32, red_f)
    red_f = jnp.where(red_f >= mod_kept_i32, red_f - mod_kept_i32, red_f)
    final_kept = red_f.astype(jnp.uint32)

    return types.Ciphertext(
        data=final_kept.reshape(num_elements, degree, num_kept),
        moduli=ciphertext.moduli[:num_kept],
    )

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


def rescale(ciphertext: Ciphertext, kernel: Rescale) -> Ciphertext:
  """Performs rescaling on the ciphertext using the precomputed kernel constants."""
  return kernel.rescale(ciphertext)
