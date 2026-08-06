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

  key_switcher: key_switching.KeySwitcher = key_switching.KeySwitcher()
  bc_kernel: basis_conversion.BasisConversionBarrett = (
      basis_conversion.BasisConversionBarrett()
  )
  mul_kernel: mul.MulPlaintextCiphertextBarrett = (
      mul.MulPlaintextCiphertextBarrett()
  )
  rescale_kernel: rescale.Rescale = rescale.Rescale()
  ntt_q: ntt.NTTBarrett = ntt.NTTBarrett()
  ntt_p: ntt.NTTBarrett = ntt.NTTBarrett()

  q_limbs: jax.Array = np.empty((0,), dtype=np.uint32)  # pytype: disable=annotation-type-mismatch
  p_limbs: jax.Array = np.empty((0,), dtype=np.uint32)  # pytype: disable=annotation-type-mismatch
  all_moduli: jax.Array = np.empty((0,), dtype=np.uint32)  # pytype: disable=annotation-type-mismatch
  q_limbs_u64_expanded: jax.Array = np.empty((1, 1, 0), dtype=np.uint64)  # pytype: disable=annotation-type-mismatch
  all_moduli_u64_expanded: jax.Array = np.empty((1, 1, 0), dtype=np.uint64)  # pytype: disable=annotation-type-mismatch

  def tree_flatten(self):
    """Flattens the BlindRotation object for JAX PyTree serialization."""
    children = (
        self.key_switcher,
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
    obj.key_switcher = children[0]
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

    # 5. Precompute KeySwitcher constants
    self.key_switcher = key_switching.KeySwitcher()
    self.key_switcher.precompute_constants(
        q_limbs,
        p_limbs,
        dnum,
        r,
        c,
        bc_kernel=self.bc_kernel,
        mul_kernel=self.mul_kernel,
    )

  def hmuxrot(
      self,
      ct: types.Ciphertext,
      hmrkey: types.HMuxRotKey,
      automorphism_indices: jax.Array,
      control_index: int = 0,
  ) -> types.Ciphertext:
    """Evaluates HMuxRot^(j)(hmrkey_beta, ct) using precomputed indices (Algorithm 5 of SHIP paper)."""
    alpha_rot = blind_rotate_utils.apply_automorphism_ntt_with_indices(
        ct.data[1], automorphism_indices
    )
    beta_rot = blind_rotate_utils.apply_automorphism_ntt_with_indices(
        ct.data[0], automorphism_indices
    )

    # Format inputs for KeySwitcher: standard key switching switches the c1
    # component, so we place alpha_rot/beta_rot in the c1 slot.
    zeros = jnp.zeros_like(alpha_rot)
    ct_stacked_data = jnp.stack(
        [jnp.stack([zeros, alpha_rot]), jnp.stack([zeros, beta_rot])]
    )
    ct_stacked_moduli = jnp.stack([ct.moduli, ct.moduli])
    ct_stacked = types.Ciphertext(
        data=ct_stacked_data, moduli=ct_stacked_moduli
    )

    # Wrap HMuxRotKey parts in EvaluationKeys compatible with KeySwitcher.
    # hmrkey.key0.data has shape (2, degree, num_QP)
    # hmrkey.key0.data[1:2] has shape (1, degree, num_QP)
    stacked_a = jnp.stack([hmrkey.key0.data[1:2], hmrkey.key1.data[1:2]])
    stacked_b = jnp.stack([hmrkey.key0.data[0:1], hmrkey.key1.data[0:1]])
    ksk_stacked_moduli = jnp.stack([hmrkey.key0.moduli, hmrkey.key1.moduli])
    ksk_stacked = types.EvaluationKeys(
        a=stacked_a,
        b=stacked_b,
        moduli=ksk_stacked_moduli,
    )

    vmapped_key_switch = jax.vmap(
        lambda ct_i, ksk_i: self.key_switcher.key_switch(
            ct=ct_i,
            ksk=ksk_i,
            start_control_index=control_index,
        ),
        in_axes=(0, 0),
    )
    ct_prods = vmapped_key_switch(ct_stacked, ksk_stacked)

    sum_data = ct_prods.data[0].astype(jnp.uint64) + ct_prods.data[1].astype(
        jnp.uint64
    )
    sum_reduced = sum_data % self.all_moduli_u64_expanded

    ct_sum = types.Ciphertext(
        data=sum_reduced.astype(jnp.uint32), moduli=self.all_moduli
    )

    self.rescale_kernel.rescale(ct_sum)
    return ct_sum

  def brot_mux(
      self,
      ct_in: types.Ciphertext,
      mux_key: types.MuxRotationKey,
      control_index: int = 0,
  ) -> types.Ciphertext:
    """Homomorphic Blind Rotation using the Mux Method (BRotMux).

    Sequentially applies the MUX-based conditional rotation for each bit of the
    rotation index j.
    Computes: Rot_j(mu) mod Q, where mu is the cleartext of ct_in, and j is the
    secret rotation index represented by the bits of mux_key.
    Reference: https://eprint.iacr.org/2025/784 Algorithm 3

    Args:
      ct_in: The input ciphertext under Q.
      mux_key: The MuxRotationKey containing the keys for each bit.
      control_index: The control index for basis conversion Q -> P.

    Returns:
      A Ciphertext under Q representing the rotated ciphertext.
    """
    degree = ct_in.data.shape[1]
    identity_indices = jnp.arange(degree, dtype=jnp.uint32)

    keys_jk_0 = [pair[0] for pair in mux_key.keys]
    keys_not_jk_1 = [pair[1] for pair in mux_key.keys]

    stacked_keys_jk_0 = jax.tree.map(lambda *args: jnp.stack(args), *keys_jk_0)
    stacked_keys_not_jk_1 = jax.tree.map(
        lambda *args: jnp.stack(args), *keys_not_jk_1
    )
    stacked_permutations = jnp.stack(mux_key.permutations)

    def scan_body(ct_out, x):
      hmrkey_jk_0, hmrkey_not_jk_1, perm_jk_0 = x

      # Stack keys and permutations
      stacked_hmrkeys = jax.tree.map(
          lambda k0, k1: jnp.stack([k0, k1]), hmrkey_jk_0, hmrkey_not_jk_1
      )
      stacked_perms = jnp.stack([perm_jk_0, identity_indices])

      # Algorithm 3, Steps 3-4: ct0 <- HMuxRot(hmrkey_jk_0, ct_out) and
      # ct1 <- HMuxRot(hmrkey_not_jk_1, ct_out).
      # Evaluated in parallel using jax.vmap.
      vmapped_hmuxrot = jax.vmap(
          lambda hmrkey_i, perm_i: self.hmuxrot(
              ct=ct_out,
              hmrkey=hmrkey_i,
              automorphism_indices=perm_i,
              control_index=control_index,
          ),
          in_axes=(0, 0),
      )

      cts = vmapped_hmuxrot(stacked_hmrkeys, stacked_perms)

      # Algorithm 3, Step 5: ct_out <- ct0 + ct1
      sum_data = cts.data[0].astype(jnp.uint64) + cts.data[1].astype(jnp.uint64)
      sum_reduced = jnp.where(
          sum_data >= self.q_limbs_u64_expanded,
          sum_data - self.q_limbs_u64_expanded,
          sum_data,
      )
      new_ct_out = types.Ciphertext(
          data=sum_reduced.astype(jnp.uint32), moduli=ct_out.moduli
      )
      return new_ct_out, None

    scan_inputs = (
        stacked_keys_jk_0,
        stacked_keys_not_jk_1,
        stacked_permutations,
    )
    # Algorithm 3, Step 2: Loop over the bits of the rotation index
    # (using jax.lax.scan)
    ct_final, _ = jax.lax.scan(scan_body, ct_in, scan_inputs)
    return ct_final

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
    self.rescale_kernel.rescale(ct_out)
    return ct_out

  def brot_hybrid(
      self,
      pts: list[types.Plaintext] | tuple[types.Plaintext, ...],
      cmkey_hybrid: list[list[types.Ciphertext]],
      mmkey_hybrid: types.MuxRotationKey,
      theta: int,
      control_index: int = 0,
  ) -> types.Ciphertext:
    """Homomorphic Blind Rotation using the Hybrid Method (BRotHybrid)."""
    if len(pts) != 4:
      raise ValueError("pts must contain exactly 4 Plaintexts.")
    if len(cmkey_hybrid) != 4:
      raise ValueError(
          "cmkey_hybrid must contain exactly 4 lists of Ciphertexts."
      )

    degree = pts[0].data.shape[0]

    # Algorithm 4, Step 1: ct <- (0, 0) (accumulated in parallel below)
    # Stack inputs
    # pts_data: shape (4, degree, num_moduli)
    pts_data = jnp.stack([pt.data for pt in pts])
    # cmkey_data: shape (4, theta, 2, degree, num_moduli)
    cmkey_data = jnp.stack(
        [jnp.stack([ct.data for ct in cmkeys]) for cmkeys in cmkey_hybrid]
    )

    # Note: Swapped baby/giant step layout (Approach B, Section 5.4 of
    # paper).
    # Algorithm 4, Step 3 (swapped): for j0 (baby step) from 0 to
    # theta - 1 do
    # Compute automorphisms for all k in parallel
    gs = jnp.array(
        [int(pow(5, -j0, 2 * degree)) for j0 in range(theta)],
        dtype=jnp.uint32,
    )

    # pt_rot_all_k: shape (4, theta, degree, num_moduli)
    # pt_rot_all_k: pt_k(X^{5^{-j0}}) for all k and j0
    def get_pt_rot(pt_data):
      return jax.vmap(
          blind_rotate_utils.apply_automorphism_ntt, in_axes=(None, 0)
      )(pt_data, gs)

    pt_rot_all_k = jax.vmap(get_pt_rot)(pts_data)
    # pt_rot_all_k_expanded: shape (4, theta, 1, degree, num_moduli)
    pt_rot_all_k_expanded = jnp.expand_dims(pt_rot_all_k, axis=2)

    # Algorithm 4, Step 4 (swapped): Multiply pt_k(X^{5^{-j0}}) * CM_key_j0
    # We perform this multiplication for all k and j0 in parallel
    # (without intermediate reduction).
    # prod: shape (4, theta, 2, degree, num_moduli) in uint64
    prod = cmkey_data.astype(jnp.uint64) * pt_rot_all_k_expanded.astype(
        jnp.uint64
    )

    # Sum over theta (baby steps) for each k
    # sum_theta: shape (4, 2, degree, num_moduli)
    sum_theta = jnp.sum(prod, axis=1)

    # Perform modular reduction on the accumulated sum for each k
    # reduced_theta: shape (4, 2, degree, num_moduli) in uint32
    reduced_theta = barrett.modular_reduction(
        sum_theta, self.mul_kernel.barrett_constants
    )

    # Algorithm 4, Step 5: Sum over k
    # summed_data: shape (2, degree, num_moduli) in uint64
    summed_data = jnp.sum(reduced_theta.astype(jnp.uint64), axis=0)

    # Final modular reduction to get ct_giant
    reduced_data = barrett.modular_reduction(
        summed_data, self.mul_kernel.barrett_constants
    )

    ct_giant = types.Ciphertext(
        data=reduced_data.astype(jnp.uint32),
        moduli=self.all_moduli,
    )

    # Algorithm 4, Step 7: ct <- Rescale_P(ct)
    # Rescale by P to go back to modulus Q
    self.rescale_kernel.rescale(ct_giant)

    # Note: Swapped baby/giant step layout (Approach B, Section 5.4 of
    # paper).
    # Algorithm 4, Step 8 (swapped): return BRotMux(mmkey, ct) for giant
    # steps index j1 * theta, using stride = theta.
    # Run Mux Method conditional rotations on the giant-step output,
    # with stride=theta.
    return self.brot_mux(
        ct_in=ct_giant,
        mux_key=mmkey_hybrid,
        control_index=control_index,
    )
