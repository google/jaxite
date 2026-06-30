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

"""Key switching utilities for CKKS ciphertexts."""

import math
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import basis_conversion
from jaxite.jaxite_ckks import mul
from jaxite.jaxite_ckks import ntt
from jaxite.jaxite_ckks import types


@jax.tree_util.register_pytree_node_class
class KeySwitcher:
  """Kernel for homomorphic key switching on TPU."""

  def __init__(self):
    self.ntt_kernels_q: list[ntt.NTTBarrett] = []
    self.ntt_kernels_out: list[ntt.NTTBarrett] = []

  def precompute_constants(
      self,
      q_limbs: list[int],
      p_limbs: list[int],
      dnum: int,
      r: int,
      c: int,
  ):
    """Precomputes NTT kernels for key switching modular partitions."""
    limbs_per_part = math.ceil(len(q_limbs) / dnum)
    all_moduli = q_limbs + p_limbs

    self.ntt_kernels_q = []
    self.ntt_kernels_out = []

    for i in range(dnum):
      start_idx = i * limbs_per_part
      end_idx = min(start_idx + limbs_per_part, len(q_limbs))

      # q_part
      q_part = q_limbs[start_idx:end_idx]
      ntt_q = ntt.NTTBarrett()
      ntt_q.precompute_constants(q_part, r, c)
      self.ntt_kernels_q.append(ntt_q)

      # out_moduli (all moduli except q_part)
      in_indices = list(range(start_idx, end_idx))
      out_moduli = [
          all_moduli[j] for j in range(len(all_moduli)) if j not in in_indices
      ]
      ntt_out = ntt.NTTBarrett()
      ntt_out.precompute_constants(out_moduli, r, c)
      self.ntt_kernels_out.append(ntt_out)

  def tree_flatten(self):
    children = (self.ntt_kernels_q, self.ntt_kernels_out)
    aux_data = None
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    obj = cls()
    obj.ntt_kernels_q = children[0]
    obj.ntt_kernels_out = children[1]
    return obj

  def key_switch(
      self,
      ct: types.Ciphertext,
      ksk: types.EvaluationKeys,
      p_limbs: jax.Array,
      bc_kernel: basis_conversion.BasisConversionBarrett,
      mul_kernel: mul.MulPlaintextCiphertextBarrett,
      start_control_index: int,
  ) -> types.Ciphertext:
    """Switch ciphertext from source key to destination key modulo QP."""
    c0 = ct.data[0]
    c1 = ct.data[1]
    q_limbs = ct.moduli
    num_q = q_limbs.shape[0]
    degree = c1.shape[0]
    dnum = len(self.ntt_kernels_q)
    limbs_per_part = math.ceil(num_q / dnum)
    all_moduli = jnp.concatenate([q_limbs, p_limbs]).astype(jnp.uint32)
    num_qp = all_moduli.shape[0]

    c0_ks = jnp.zeros((degree, num_qp), dtype=jnp.uint64)
    c1_ks = jnp.zeros((degree, num_qp), dtype=jnp.uint64)
    all_moduli_u64 = all_moduli.astype(jnp.uint64).reshape(1, -1)

    for i in range(dnum):
      start_idx = i * limbs_per_part
      end_idx = min(start_idx + limbs_per_part, num_q)
      q_part = q_limbs[start_idx:end_idx]
      num_q_part = q_part.shape[0]

      in_indices = list(range(start_idx, end_idx))
      out_indices = [j for j in range(num_qp) if j not in in_indices]
      in_indices_arr = jnp.array(in_indices, dtype=jnp.int32)
      out_indices_arr = jnp.array(out_indices, dtype=jnp.int32)
      out_moduli = all_moduli[out_indices_arr]
      num_out_moduli = out_moduli.shape[0]

      # Extract partition and convert to coefficient domain
      c1_part = c1[:, start_idx:end_idx]
      c1_part_reshaped = c1_part.reshape(
          1,
          self.ntt_kernels_q[i].constants.r,
          self.ntt_kernels_q[i].constants.c,
          num_q_part,
      )
      c1_part_coeffs = self.ntt_kernels_q[i].intt(
          c1_part_reshaped.astype(jnp.uint32)
      )
      c1_part_coeffs = c1_part_coeffs.reshape(degree, num_q_part)

      # Basis change to out_moduli
      control_index_loop = start_control_index + i
      c1_part_out_coeffs = bc_kernel.basis_change(
          c1_part_coeffs, control_index=control_index_loop
      )

      # Convert back to NTT domain modulo out_moduli
      c1_part_out_coeffs_reshaped = c1_part_out_coeffs.reshape(
          1,
          self.ntt_kernels_out[i].constants.r,
          self.ntt_kernels_out[i].constants.c,
          num_out_moduli,
      )
      c1_part_out_ntt = self.ntt_kernels_out[i].ntt(
          c1_part_out_coeffs_reshaped.astype(jnp.uint32)
      )
      c1_part_out = c1_part_out_ntt.reshape(degree, num_out_moduli)

      # Merge into full all_moduli representation
      c1_part_qp = jnp.zeros((degree, num_qp), dtype=jnp.uint32)
      c1_part_qp = c1_part_qp.at[:, in_indices_arr].set(c1_part)
      c1_part_qp = c1_part_qp.at[:, out_indices_arr].set(c1_part_out)

      # Multiply by partition key modulo all_moduli
      ksk_b_part = ksk.b[i]
      ksk_a_part = ksk.a[i]

      c0_ks_part = mul_kernel.mul(
          types.Plaintext(data=ksk_b_part, moduli=all_moduli),
          types.Plaintext(data=c1_part_qp, moduli=all_moduli),
      )
      c1_ks_part = mul_kernel.mul(
          types.Plaintext(data=ksk_a_part, moduli=all_moduli),
          types.Plaintext(data=c1_part_qp, moduli=all_moduli),
      )

      # Sum modulo all_moduli
      c0_ks = (c0_ks + c0_ks_part.data.astype(jnp.uint64)) % all_moduli_u64
      c1_ks = (c1_ks + c1_ks_part.data.astype(jnp.uint64)) % all_moduli_u64

    # Scale c0 by P
    p_mod_q = jnp.ones_like(q_limbs, dtype=jnp.uint64)
    for p in p_limbs:
      p_mod_q = (p_mod_q * (p % q_limbs).astype(jnp.uint64)) % q_limbs.astype(
          jnp.uint64
      )
    c0_scaled_q = (
        c0.astype(jnp.uint64) * p_mod_q.astype(jnp.uint64).reshape(1, -1)
    ) % q_limbs.astype(jnp.uint64).reshape(1, -1)
    c0_scaled_p = jnp.zeros((degree, p_limbs.shape[0]), dtype=jnp.uint32)
    c0_scaled_qp = jnp.concatenate(
        [c0_scaled_q.astype(jnp.uint32), c0_scaled_p], axis=-1
    )

    # Compute c0_prime = P * c0 + c0_ks
    c0_prime = (c0_scaled_qp.astype(jnp.uint64) + c0_ks) % all_moduli_u64
    c1_prime = c1_ks

    return types.Ciphertext(
        data=jnp.stack(
            [c0_prime.astype(jnp.uint32), c1_prime.astype(jnp.uint32)]
        ),
        moduli=all_moduli,
    )
