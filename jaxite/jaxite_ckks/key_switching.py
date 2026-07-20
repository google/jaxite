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
from jaxite.jaxite_ckks import barrett
from jaxite.jaxite_ckks import basis_conversion
from jaxite.jaxite_ckks import blind_rotate_utils
from jaxite.jaxite_ckks import math as ckks_math
from jaxite.jaxite_ckks import mul
from jaxite.jaxite_ckks import ntt
from jaxite.jaxite_ckks import rescale
from jaxite.jaxite_ckks import types


@jax.tree_util.register_pytree_node_class
class KeySwitcher:
  """Kernel for homomorphic key switching on TPU."""

  ntt_kernels_q: list[ntt.NTTBarrett]
  ntt_kernels_out: list[ntt.NTTBarrett]
  p_limbs: jax.Array
  bc_kernel: basis_conversion.BasisConversionBarrett
  mul_kernel: mul.MulPlaintextCiphertextBarrett

  def precompute_constants(
      self,
      q_limbs: list[int],
      p_limbs: list[int],
      dnum: int,
      r: int,
      c: int,
      bc_kernel: basis_conversion.BasisConversionBarrett,
      mul_kernel: mul.MulPlaintextCiphertextBarrett,
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

    self.p_limbs = jnp.array(p_limbs, dtype=jnp.uint32)
    self.bc_kernel = bc_kernel
    self.mul_kernel = mul_kernel

  def tree_flatten(self):
    children = (
        self.ntt_kernels_q,
        self.ntt_kernels_out,
        self.p_limbs,
        self.bc_kernel,
        self.mul_kernel,
    )
    aux_data = None
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    obj = cls()
    obj.ntt_kernels_q = children[0]
    obj.ntt_kernels_out = children[1]
    obj.p_limbs = children[2]
    obj.bc_kernel = children[3]
    obj.mul_kernel = children[4]
    return obj

  def key_switch(
      self,
      ct: types.Ciphertext,
      ksk: types.EvaluationKeys,
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
    all_moduli = jnp.concatenate([q_limbs, self.p_limbs]).astype(jnp.uint32)
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
      r_q = self.ntt_kernels_q[i].constants.r
      c_q = self.ntt_kernels_q[i].constants.c
      _, num_blocks_q = ckks_math.compute_tpu_block_sizes(degree, r_q * c_q)
      c1_part_reshaped = c1_part.reshape(
          num_blocks_q,
          r_q,
          c_q,
          num_q_part,
      )
      c1_part_coeffs = self.ntt_kernels_q[i].intt(
          c1_part_reshaped.astype(jnp.uint32)
      )
      c1_part_coeffs = c1_part_coeffs.reshape(degree, num_q_part)

      # Basis change to out_moduli
      control_index_loop = start_control_index + i
      c1_part_out_coeffs = self.bc_kernel.basis_change(
          c1_part_coeffs, control_index=control_index_loop
      )

      # Convert back to NTT domain modulo out_moduli
      r_out = self.ntt_kernels_out[i].constants.r
      c_out = self.ntt_kernels_out[i].constants.c
      _, num_blocks_out = ckks_math.compute_tpu_block_sizes(
          degree, r_out * c_out
      )
      c1_part_out_coeffs_reshaped = c1_part_out_coeffs.reshape(
          num_blocks_out,
          r_out,
          c_out,
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

      c0_ks_part = self.mul_kernel.mul(
          types.Plaintext(data=ksk_b_part, moduli=all_moduli),  # pyrefly: ignore[bad-argument-type]
          types.Plaintext(data=c1_part_qp, moduli=all_moduli),
      )
      c1_ks_part = self.mul_kernel.mul(
          types.Plaintext(data=ksk_a_part, moduli=all_moduli),  # pyrefly: ignore[bad-argument-type]
          types.Plaintext(data=c1_part_qp, moduli=all_moduli),
      )

      # Sum modulo all_moduli
      c0_ks = (c0_ks + c0_ks_part.data.astype(jnp.uint64)) % all_moduli_u64
      c1_ks = (c1_ks + c1_ks_part.data.astype(jnp.uint64)) % all_moduli_u64

    # Scale c0 by P
    p_mod_q = jnp.ones_like(q_limbs, dtype=jnp.uint64)
    for p in self.p_limbs:
      p_mod_q = (p_mod_q * (p % q_limbs).astype(jnp.uint64)) % q_limbs.astype(
          jnp.uint64
      )
    c0_scaled_q = (
        c0.astype(jnp.uint64) * p_mod_q.astype(jnp.uint64).reshape(1, -1)
    ) % q_limbs.astype(jnp.uint64).reshape(1, -1)
    c0_scaled_p = jnp.zeros((degree, self.p_limbs.shape[0]), dtype=jnp.uint32)
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


@jax.tree_util.register_pytree_node_class
class BATKeySwitcher:
  """Kernel for BAT-based key switching on TPU."""

  ntt_q: ntt.NTTBarrett
  ntt_p: ntt.NTTBarrett
  p_limbs: jax.Array
  r: int
  c: int
  block_size: int
  bc_kernel: basis_conversion.BasisConversionBarrett
  mul_kernel: mul.MulPlaintextCiphertextBarrett
  rescale_kernel: rescale.Rescale

  def precompute_constants(
      self,
      q_limbs: list[int],
      p_limbs: list[int],
      r: int,
      c: int,
      bc_kernel: basis_conversion.BasisConversionBarrett,
      mul_kernel: mul.MulPlaintextCiphertextBarrett,
      rescale_kernel: rescale.Rescale,
      block_size: int = 128,
  ) -> None:
    """Precomputes NTT kernels for BAT key switching."""
    self.ntt_q = ntt.NTTBarrett()
    self.ntt_q.precompute_constants(q_limbs, r, c)
    self.ntt_p = ntt.NTTBarrett()
    self.ntt_p.precompute_constants(p_limbs, r, c)
    self.p_limbs = jnp.array(p_limbs, dtype=jnp.uint32)
    self.r = r
    self.c = c
    self.block_size = block_size
    self.bc_kernel = bc_kernel
    self.mul_kernel = mul_kernel
    self.rescale_kernel = rescale_kernel

  def transform_key_to_bat(
      self, ksk: types.EvaluationKeys, key1: types.Ciphertext
  ) -> jax.Array:
    """Transforms standard KSK and key1 into BAT key representation.

    Args:
      ksk: The standard key switching key (EvaluationKeys) of shape (1, degree,
        num_moduli) for a and b.
      key1: The encryption of P (Ciphertext) of shape (2, degree, num_moduli).

    Returns:
      The transformed key matrix of shape (num_blocks, block_size, num_moduli,
      2,
      2, 4, 4) in uint8.
    """
    if ksk.b.shape[0] != 1:
      raise ValueError("BAT key switching only supports dnum = 1.")

    key0_data = jnp.stack([ksk.b[0], ksk.a[0]])
    key1_data = key1.data

    # Stack to shape (2, 2, degree, num_moduli) and transpose to (degree, num_moduli, 2, 2)
    stacked = jnp.stack([key0_data, key1_data], axis=1)
    key_matrix = jnp.transpose(stacked, (2, 3, 0, 1))

    matrix_u64 = jnp.array(key_matrix, dtype=jnp.uint64)
    num_bytes = 4
    matrix_u64_byteshifted = jnp.array(
        [matrix_u64 << (8 * byte_idx) for byte_idx in range(num_bytes)],
        dtype=jnp.uint64,
    )
    moduli_expanded = jnp.array(ksk.moduli, dtype=jnp.uint64).reshape(
        1, 1, -1, 1, 1
    )
    matrix_u64_byteshifted_mod_modulus = (
        matrix_u64_byteshifted % moduli_expanded
    ).astype(jnp.uint32)
    matrix_u8 = jax.lax.bitcast_convert_type(
        matrix_u64_byteshifted_mod_modulus, jnp.uint8
    )
    matrix_u8_transposed = jnp.transpose(matrix_u8, (1, 2, 3, 4, 0, 5))
    degree = key_matrix.shape[0]
    block_size_actual, num_blocks = ckks_math.compute_tpu_block_sizes(
        degree, self.block_size
    )
    return matrix_u8_transposed.reshape(
        num_blocks, block_size_actual, *matrix_u8_transposed.shape[1:]
    )

  def key_switch(
      self,
      ct: types.Ciphertext,
      key_matrix_bat: jax.Array,
      control_index: int,
  ) -> types.Ciphertext:
    """Switch ciphertext from source key to destination key using BAT."""
    c0 = ct.data[0]
    c1 = ct.data[1]

    c0_ct = types.Ciphertext(data=jnp.expand_dims(c0, axis=0), moduli=ct.moduli)
    c1_ct = types.Ciphertext(data=jnp.expand_dims(c1, axis=0), moduli=ct.moduli)

    c0_lifted = blind_rotate_utils.lift_ciphertext(
        c0_ct,
        self.bc_kernel,
        control_index,
        self.p_limbs,
        self.ntt_q,
        self.ntt_p,
        self.r,
        self.c,
    )
    c1_lifted = blind_rotate_utils.lift_ciphertext(
        c1_ct,
        self.bc_kernel,
        control_index,
        self.p_limbs,
        self.ntt_q,
        self.ntt_p,
        self.r,
        self.c,
    )

    c0_lifted_data = jnp.squeeze(c0_lifted.data, axis=0)
    c1_lifted_data = jnp.squeeze(c1_lifted.data, axis=0)

    vector_v = jnp.stack([c1_lifted_data, c0_lifted_data], axis=-1)

    prod = self._matvec_product(vector_v, key_matrix_bat)

    reduced = barrett.modular_reduction(prod, self.mul_kernel.barrett_constants)

    ct_out = types.Ciphertext(data=reduced, moduli=c0_lifted.moduli)

    self.rescale_kernel.rescale(ct_out)
    return ct_out

  def _matvec_product(
      self, vector_v: jax.Array, key_matrix_bat: jax.Array
  ) -> jax.Array:
    """Computes BAT-based matrix-vector product for 2x2 key multiplication."""
    degree = vector_v.shape[-3]
    num_moduli = vector_v.shape[-2]

    block_size, num_blocks = ckks_math.compute_tpu_block_sizes(
        degree, self.block_size
    )

    v_reshaped = vector_v.reshape(
        *vector_v.shape[:-3], num_blocks, block_size, num_moduli, 2
    )

    v_u8 = jax.lax.bitcast_convert_type(v_reshaped, jnp.uint8)

    i8_products = jnp.einsum(
        "...ikjvq,ikjuvqp->...ikjup",
        v_u8,
        key_matrix_bat,
        preferred_element_type=jnp.uint32,
    )

    shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
    summed = jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=-1)
    summed_flat = summed.reshape(*summed.shape[:-4], degree, num_moduli, 2)
    return jnp.moveaxis(summed_flat, -1, 0)

  def tree_flatten(self):
    """Flattens the BATKeySwitcher into children and auxiliary data for JAX PyTree."""
    children = (
        self.ntt_q,
        self.ntt_p,
        self.p_limbs,
        self.bc_kernel,
        self.mul_kernel,
        self.rescale_kernel,
    )
    aux_data = (self.r, self.c, self.block_size)
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """Unflattens the BATKeySwitcher from children and auxiliary data for JAX PyTree."""
    obj = cls()
    obj.ntt_q = children[0]
    obj.ntt_p = children[1]
    obj.p_limbs = children[2]
    obj.bc_kernel = children[3]
    obj.mul_kernel = children[4]
    obj.rescale_kernel = children[5]
    obj.r = aux_data[0]
    obj.c = aux_data[1]
    obj.block_size = aux_data[2]
    return obj
