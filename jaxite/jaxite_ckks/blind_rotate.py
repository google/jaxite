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

"""Blind rotation implementations for CKKS."""

import functools
import math

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import barrett
from jaxite.jaxite_ckks import basis_conversion
from jaxite.jaxite_ckks import blind_rotate_utils
from jaxite.jaxite_ckks import key_switching
from jaxite.jaxite_ckks import mul
from jaxite.jaxite_ckks import ntt
from jaxite.jaxite_ckks import rescale
from jaxite.jaxite_ckks import types
import numpy as np


@jax.tree_util.register_pytree_node_class
class BlindRotation:
  """Kernel for homomorphic blind rotation on TPU."""

  def __init__(self):
    self.bat_switcher: key_switching.BATKeySwitcher = (
        key_switching.BATKeySwitcher()
    )
    self.bc_kernel: basis_conversion.BasisConversionBarrett = (
        basis_conversion.BasisConversionBarrett()
    )
    self.mul_kernel: mul.MulPlaintextCiphertextBarrett = (
        mul.MulPlaintextCiphertextBarrett()
    )
    self.rescale_kernel: rescale.Rescale = rescale.Rescale()
    self.ntt_q: ntt.NTTBarrett = ntt.NTTBarrett()
    self.ntt_p: ntt.NTTBarrett = ntt.NTTBarrett()

    self.q_limbs: jax.Array = np.empty((0,), dtype=np.uint32)  # pyrefly: ignore[bad-assignment]
    self.p_limbs: jax.Array = np.empty((0,), dtype=np.uint32)  # pyrefly: ignore[bad-assignment]
    self.all_moduli: jax.Array = np.empty((0,), dtype=np.uint32)  # pyrefly: ignore[bad-assignment]
    self.q_limbs_u64_expanded: jax.Array = np.empty((1, 1, 0), dtype=np.uint64)  # pyrefly: ignore[bad-assignment]
    self.all_moduli_u64_expanded: jax.Array = np.empty(  # pyrefly: ignore[bad-assignment]
        (1, 1, 0), dtype=np.uint64
    )

  def tree_flatten(self):
    """Flattens the BlindRotation object for JAX PyTree serialization."""
    children = (
        self.bat_switcher,
        self.bc_kernel,
        self.mul_kernel,
        self.rescale_kernel,
        self.ntt_q,
        self.ntt_p,
        self.q_limbs,
        self.p_limbs,
        self.all_moduli,
        self.q_limbs_u64_expanded,
        self.all_moduli_u64_expanded,
    )
    aux_data = None
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """Unflattens the BlindRotation object for JAX PyTree serialization."""
    del aux_data
    obj = cls()
    obj.bat_switcher = children[0]
    obj.bc_kernel = children[1]
    obj.mul_kernel = children[2]
    obj.rescale_kernel = children[3]
    obj.ntt_q = children[4]
    obj.ntt_p = children[5]
    obj.q_limbs = children[6]
    obj.p_limbs = children[7]
    obj.all_moduli = children[8]
    obj.q_limbs_u64_expanded = children[9]
    obj.all_moduli_u64_expanded = children[10]
    return obj

  def precompute_constants(
      self,
      q_limbs: list[int],
      p_limbs: list[int],
      dnum: int,
      r: int,
      c: int,
      num_rescales: int = 1,
  ):
    """Precomputes constants and sub-kernels for blind rotation."""
    all_moduli = q_limbs + p_limbs

    # Store moduli as JAX arrays
    self.q_limbs = jnp.array(q_limbs, dtype=jnp.uint32)
    self.p_limbs = jnp.array(p_limbs, dtype=jnp.uint32)
    self.all_moduli = jnp.array(all_moduli, dtype=jnp.uint32)

    self.q_limbs_u64_expanded = jnp.array(q_limbs, dtype=jnp.uint64).reshape(
        1, 1, -1
    )
    self.all_moduli_u64_expanded = jnp.array(
        all_moduli, dtype=jnp.uint64
    ).reshape(1, 1, -1)

    # 1. Precompute NTT kernels
    self.ntt_q = ntt.NTTBarrett()
    self.ntt_q.precompute_constants(q_limbs, r, c)
    self.ntt_p = ntt.NTTBarrett()
    self.ntt_p.precompute_constants(p_limbs, r, c)

    # 2. Precompute BasisConversion constants
    limbs_per_part = math.ceil(len(q_limbs) / dnum)
    bc_pairs = []
    for i in range(dnum):
      start_idx = i * limbs_per_part
      end_idx = min(start_idx + limbs_per_part, len(q_limbs))
      in_indices = list(range(start_idx, end_idx))
      out_indices = [j for j in range(len(all_moduli)) if j not in in_indices]
      bc_pairs.append((in_indices, out_indices))
    self.bc_kernel = basis_conversion.BasisConversionBarrett()
    self.bc_kernel.precompute_constants(all_moduli, bc_pairs)

    self.mul_kernel = mul.MulPlaintextCiphertextBarrett()
    self.mul_kernel.precompute_constants(all_moduli)

    # 4. Precompute Rescale constants
    self.rescale_kernel = rescale.Rescale()
    self.rescale_kernel.precompute_constants(
        all_moduli, num_rescales=num_rescales, r=r, c=c
    )

    # Precompute BATKeySwitcher constants
    self.bat_switcher = key_switching.BATKeySwitcher()
    self.bat_switcher.precompute_constants(
        q_limbs,
        p_limbs,
        r,
        c,
        bc_kernel=self.bc_kernel,
        mul_kernel=self.mul_kernel,
        rescale_kernel=self.rescale_kernel,
    )

  @functools.partial(jax.jit, static_argnums=(0, 4))
  def brot_mux(
      self,
      ct_in: types.Ciphertext,
      mux_key_bat: list[tuple[jax.Array, jax.Array]],
      permutations: list[jax.Array],
      control_index: int = 0,
  ) -> types.Ciphertext:
    """Homomorphic Blind Rotation using Mux Method with BAT."""
    q_limbs_list = ct_in.moduli
    q_limbs_array = jnp.array(q_limbs_list, dtype=jnp.uint32)
    num_moduli = q_limbs_array.shape[0]

    if num_moduli == 1:
      q_for_reduction = q_limbs_array[0]
      ct_current_data = jnp.squeeze(ct_in.data, axis=-1)

      def _ks_helper_single_mod(ct_data_squeezed, key):
        ct_temp = types.Ciphertext(
            data=jnp.expand_dims(ct_data_squeezed, axis=-1), moduli=q_limbs_list
        )
        res_data_expanded = self.bat_switcher.key_switch(
            ct=ct_temp,
            key_matrix_bat=key,
            control_index=control_index,
        ).data
        return jnp.squeeze(res_data_expanded, axis=-1)

      vmapped_ks = jax.vmap(_ks_helper_single_mod, in_axes=(0, 0))

    else:
      q_for_reduction = q_limbs_array.reshape(1, 1, -1)
      ct_current_data = ct_in.data

      def _ks_helper_multi_mod(ct_data_full, key):
        ct_temp = types.Ciphertext(data=ct_data_full, moduli=q_limbs_list)
        return self.bat_switcher.key_switch(
            ct=ct_temp,
            key_matrix_bat=key,
            control_index=control_index,
        ).data

      vmapped_ks = jax.vmap(_ks_helper_multi_mod, in_axes=(0, 0))

    stacked_bat_key_jk_0 = jnp.stack([pair[0] for pair in mux_key_bat])
    stacked_bat_key_not_jk_1 = jnp.stack([pair[1] for pair in mux_key_bat])
    stacked_permutations = jnp.stack(permutations)

    num_iterations = len(mux_key_bat)

    for i in range(num_iterations):
      bat_key0_slice = stacked_bat_key_jk_0[i]
      bat_key1_slice = stacked_bat_key_not_jk_1[i]
      perm_jk_0_slice = stacked_permutations[i]

      rotated_data = jnp.take(ct_current_data, perm_jk_0_slice, axis=1)

      stacked_input = jnp.stack([rotated_data, ct_current_data], axis=0)
      bat_keys_iter = jnp.stack([bat_key0_slice, bat_key1_slice], axis=0)
      cts_out_data = vmapped_ks(stacked_input, bat_keys_iter)

      sum_data = jnp.sum(cts_out_data.astype(jnp.uint64), axis=0)
      ct_current_data = jnp.where(
          sum_data >= q_for_reduction, sum_data - q_for_reduction, sum_data
      ).astype(jnp.uint32)

    if num_moduli == 1:
      final_data = jnp.expand_dims(ct_current_data, axis=-1)
    else:
      final_data = ct_current_data

    return types.Ciphertext(data=final_data, moduli=q_limbs_list)

  def brot_cm(
      self,
      cmkey_j: list[types.Ciphertext],
      pt_rot_mu_all: list[types.Plaintext],
  ) -> types.Ciphertext:
    """Homomorphic Blind Rotation using the Column Method (BRotCM)."""
    if len(cmkey_j) != len(pt_rot_mu_all):
      raise ValueError("Lengths of cmkey_j and pt_rot_mu_all must match.")

    if cmkey_j[0].moduli.shape != pt_rot_mu_all[0].moduli.shape:
      raise ValueError("Moduli shapes of cmkey_j and pt_rot_mu_all must match.")

    # Algorithm 2, Step 2: pt <- Ecd(mu) and Step 4: pt_rot <- pt(X^{5^i})
    # Note: pt_rot_mu_all contains the pre-rotated plaintexts.
    ct_data = jnp.stack([ct.data for ct in cmkey_j])
    pt_data = jnp.stack([pt.data for pt in pt_rot_mu_all])
    pt_data_expanded = jnp.expand_dims(pt_data, axis=1)

    # Algorithm 2, Step 1: ct <- (0, 0)
    # Stacking and batching the additions into a single sum.
    batch_ct = types.Ciphertext(data=ct_data, moduli=cmkey_j[0].moduli)
    batch_pt = types.Plaintext(
        data=pt_data_expanded, moduli=pt_rot_mu_all[0].moduli
    )

    # Algorithm 2, Step 5: ct <- ct + pt_rot * CM_key_i
    # Perform batch multiplication
    batch_ct_mul = self.mul_kernel.mul(batch_ct, batch_pt)

    # Accumulate the products along the batch axis (0) in uint64 to prevent
    # overflow.
    sum_data = jnp.sum(batch_ct_mul.data.astype(jnp.uint64), axis=0)

    # Perform a single modular reduction on the accumulated sum
    reduced_data = barrett.modular_reduction(
        sum_data, self.mul_kernel.barrett_constants
    )

    ct_out = types.Ciphertext(
        data=reduced_data.astype(jnp.uint32),
        moduli=cmkey_j[0].moduli,
    )

    # Algorithm 2, Step 7: return Rescale_P(ct)
    return self.rescale_kernel.rescale(ct_out)

  @functools.partial(jax.jit, static_argnums=(0, 5, 6))
  def brot_hybrid(
      self,
      pts: list[types.Plaintext] | tuple[types.Plaintext, ...],
      cmkey_hybrid: list[list[types.Ciphertext]],
      mmkey_hybrid_bat: list[tuple[jax.Array, jax.Array]],
      permutations: list[jax.Array],
      theta: int,
      control_index: int = 0,
  ) -> types.Ciphertext:
    """Homomorphic Blind Rotation using Hybrid Method with BAT."""
    degree = pts[0].data.shape[0]
    gs = jnp.array(
        [pow(5, -j0, 2 * degree) for j0 in range(theta)], dtype=jnp.uint32
    )

    stacked_cmkey_data = jnp.stack([
        jnp.stack([ct.data for ct in inner_list]) for inner_list in cmkey_hybrid
    ])

    num_moduli = pts[0].data.shape[-1]
    total_terms = len(pts) * theta

    if total_terms <= 16:
      summed_data = jnp.zeros((2, degree, num_moduli), dtype=jnp.uint64)
      for k, pt in enumerate(pts):
        pt_data_u64 = pt.data.astype(jnp.uint64)
        for t in range(theta):
          pt_rot = blind_rotate_utils.apply_automorphism_ntt(pt_data_u64, gs[t])  # pyrefly: ignore[bad-argument-type]
          summed_data += (
              stacked_cmkey_data[k, t].astype(jnp.uint64)
              * pt_rot[jnp.newaxis, :, :]
          )
      reduced_data = barrett.modular_reduction(
          summed_data, self.mul_kernel.barrett_constants
      )
    else:
      summed_data = jnp.zeros((2, degree, num_moduli), dtype=jnp.uint64)
      for k, pt in enumerate(pts):
        pt_data_u64 = pt.data.astype(jnp.uint64)
        inner_reduced_sum = jnp.zeros((2, degree, num_moduli), dtype=jnp.uint64)
        for t_chunk in range(0, theta, 16):
          chunk_sum = jnp.zeros((2, degree, num_moduli), dtype=jnp.uint64)
          for t in range(t_chunk, min(t_chunk + 16, theta)):
            pt_rot = blind_rotate_utils.apply_automorphism_ntt(
                pt_data_u64, gs[t]  # pyrefly: ignore[bad-argument-type]
            )
            chunk_sum += (
                stacked_cmkey_data[k, t].astype(jnp.uint64)
                * pt_rot[jnp.newaxis, :, :]
            )
          chunk_reduced = barrett.modular_reduction(
              chunk_sum, self.mul_kernel.barrett_constants
          )
          inner_reduced_sum += chunk_reduced.astype(jnp.uint64)
        summed_data += inner_reduced_sum

      reduced_data = barrett.modular_reduction(
          summed_data, self.mul_kernel.barrett_constants
      )

    ct_giant = types.Ciphertext(
        data=reduced_data.astype(jnp.uint32), moduli=self.all_moduli
    )

    ct_giant = self.rescale_kernel.rescale(ct_giant)

    return self.brot_mux(
        ct_in=ct_giant,
        mux_key_bat=mmkey_hybrid_bat,
        permutations=permutations,
        control_index=control_index,
    )
