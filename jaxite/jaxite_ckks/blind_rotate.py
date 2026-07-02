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


@jax.tree_util.register_pytree_node_class
class BlindRotation:
  """Kernel for homomorphic blind rotation on TPU."""

  def __init__(self):
    self.key_switcher = key_switching.KeySwitcher()
    self.bc_kernel = basis_conversion.BasisConversionBarrett()
    self.mul_kernel = mul.MulPlaintextCiphertextBarrett(None)  # pyrefly: ignore[bad-argument-type]
    self.rescale_kernel = rescale.Rescale()
    self.ntt_q = ntt.NTTBarrett()
    self.ntt_p = ntt.NTTBarrett()

  def tree_flatten(self):
    children = (
        self.key_switcher,
        self.bc_kernel,
        self.mul_kernel,
        self.rescale_kernel,
        self.ntt_q,
        self.ntt_p,
    )
    aux_data = None
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    obj = cls()
    obj.key_switcher = children[0]
    obj.bc_kernel = children[1]
    obj.mul_kernel = children[2]
    obj.rescale_kernel = children[3]
    obj.ntt_q = children[4]
    obj.ntt_p = children[5]
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

    # 1. Precompute NTT kernels
    self.ntt_q.precompute_constants(q_limbs, r, c)
    self.ntt_p.precompute_constants(p_limbs, r, c)

    # 2. Precompute KeySwitcher constants
    self.key_switcher.precompute_constants(q_limbs, p_limbs, dnum, r, c)

    # 3. Precompute BasisConversion constants
    limbs_per_part = math.ceil(len(q_limbs) / dnum)
    bc_pairs = []
    for i in range(dnum):
      start_idx = i * limbs_per_part
      end_idx = min(start_idx + limbs_per_part, len(q_limbs))
      in_indices = list(range(start_idx, end_idx))
      out_indices = [j for j in range(len(all_moduli)) if j not in in_indices]
      bc_pairs.append((in_indices, out_indices))
    self.bc_kernel.precompute_constants(all_moduli, bc_pairs)

    # 4. Precompute Mul constants
    mul_constants = barrett.precompute_barrett_constants(all_moduli)
    self.mul_kernel = mul.MulPlaintextCiphertextBarrett(mul_constants)

    # 5. Precompute Rescale constants
    self.rescale_kernel.precompute_constants(
        all_moduli, num_rescales=num_rescales, r=r, c=c
    )

  def hmuxrot(
      self,
      ct: types.Ciphertext,
      hmrkey: types.HMuxRotKey,
      j: int,
      p_limbs: jax.Array,
      control_index: int = 0,
  ) -> types.Ciphertext:
    """Evaluates HMuxRot^(j)(hmrkey_beta, ct)."""
    g = pow(5, -j, 2 * ct.data.shape[1])
    alpha_rot = blind_rotate_utils.apply_automorphism_ntt(ct.data[1], g)
    beta_rot = blind_rotate_utils.apply_automorphism_ntt(ct.data[0], g)

    # Format inputs for KeySwitcher: standard key switching switches the c1
    # component, so we place alpha_rot/beta_rot in the c1 slot.
    zeros = jnp.zeros_like(alpha_rot)
    ct_c1 = types.Ciphertext(
        data=jnp.stack([zeros, alpha_rot]), moduli=ct.moduli
    )
    ct_c0 = types.Ciphertext(
        data=jnp.stack([zeros, beta_rot]), moduli=ct.moduli
    )

    # Wrap HMuxRotKey parts in EvaluationKeys compatible with KeySwitcher.
    ksk0 = types.EvaluationKeys(
        a=hmrkey.key0.data[1:2],
        b=hmrkey.key0.data[0:1],
        moduli=hmrkey.key0.moduli,
    )
    ksk1 = types.EvaluationKeys(
        a=hmrkey.key1.data[1:2],
        b=hmrkey.key1.data[0:1],
        moduli=hmrkey.key1.moduli,
    )

    ct_prod0 = self.key_switcher.key_switch(
        ct=ct_c1,
        ksk=ksk0,
        p_limbs=p_limbs,
        bc_kernel=self.bc_kernel,
        mul_kernel=self.mul_kernel,
        start_control_index=control_index,
    )
    ct_prod1 = self.key_switcher.key_switch(
        ct=ct_c0,
        ksk=ksk1,
        p_limbs=p_limbs,
        bc_kernel=self.bc_kernel,
        mul_kernel=self.mul_kernel,
        start_control_index=control_index,
    )

    all_moduli = ct_prod0.moduli
    all_moduli_u64 = all_moduli.astype(jnp.uint64).reshape(1, 1, -1)

    sum_data = ct_prod0.data.astype(jnp.uint64) + ct_prod1.data.astype(
        jnp.uint64
    )
    sum_reduced = sum_data % all_moduli_u64

    ct_sum = types.Ciphertext(
        data=sum_reduced.astype(jnp.uint32), moduli=all_moduli
    )

    self.rescale_kernel.rescale(ct_sum)
    return ct_sum

  def brot_mux(
      self,
      ct_in: types.Ciphertext,
      mux_key: types.MuxRotationKey,
      p_limbs: jax.Array,
      control_index: int = 0,
      stride: int = 1,
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
      p_limbs: The limbs of the auxiliary modulus P.
      control_index: The control index for basis conversion Q -> P.
      stride: The stride to scale the rotation amount by. Defaults to 1.

    Returns:
      A Ciphertext under Q representing the rotated ciphertext.
    """
    # Algorithm 3, Step 1: ct_out <- ct
    ct_out = ct_in

    # Algorithm 3, Step 2: for k from 0 to n - 1 do
    for k, (hmrkey_jk_0, hmrkey_not_jk_1) in enumerate(mux_key.keys):
      # Algorithm 3, Step 3: ct0 <- HMuxRot_{2^k * stride}(hmrkey_{j_k}, ct_out)
      ct0 = self.hmuxrot(
          ct=ct_out,
          hmrkey=hmrkey_jk_0,
          j=(2**k) * stride,
          p_limbs=p_limbs,
          control_index=control_index,
      )
      # Algorithm 3, Step 4: ct1 <- HMuxRot_0(hmrkey_{1 - j_k}, ct_out)
      ct1 = self.hmuxrot(
          ct=ct_out,
          hmrkey=hmrkey_not_jk_1,
          j=0,
          p_limbs=p_limbs,
          control_index=control_index,
      )
      # Algorithm 3, Step 5: ct_out <- ct0 + ct1
      moduli_expanded = jnp.array(ct0.moduli, dtype=jnp.uint64).reshape(
          1, 1, -1
      )
      sum_data = ct0.data.astype(jnp.uint64) + ct1.data.astype(jnp.uint64)
      sum_reduced = jnp.where(
          sum_data >= moduli_expanded, sum_data - moduli_expanded, sum_data
      )
      ct_out = types.Ciphertext(
          data=sum_reduced.astype(jnp.uint32), moduli=ct0.moduli
      )

    # Algorithm 3, Step 7: return ct_out
    return ct_out

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
