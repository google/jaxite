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

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import barrett
from jaxite.jaxite_ckks import basis_conversion
from jaxite.jaxite_ckks import bat_utils
from jaxite.jaxite_ckks import blind_rotate_utils
from jaxite.jaxite_ckks import mul
from jaxite.jaxite_ckks import rescale
from jaxite.jaxite_ckks import types
import numpy as np


def hmuxrot(
    ct: types.Ciphertext,
    hmrkey: types.HMuxRotKey,
    j: int,
    bc_kernel: basis_conversion.BasisConversionBarrett,
    control_index: int,
    p_limbs: jax.Array,
    mul_kernel: mul.MulPlaintextCiphertextBarrett,
    rescale_kernel: rescale.Rescale,
) -> types.Ciphertext:
  """Evaluates HMuxRot^(j)(hmrkey_beta, ct).

  Computes: P^-1 * [ a(X^{5^{-j}}) * hmrkey_0 + b(X^{5^{-j}}) * hmrkey_1 ] mod Q

  Args:
    ct: The input ciphertext (a, b) under Q.
    hmrkey: The HMuxRot key under PQ.
    j: The rotation index.
    bc_kernel: The basis conversion kernel.
    control_index: The control index for basis conversion Q -> P.
    p_limbs: The limbs of P.
    mul_kernel: The multiplication kernel under PQ.
    rescale_kernel: The rescaling kernel.

  Returns:
    The resulting ciphertext under Q.
  """
  g = pow(5, -j, 2 * ct.data.shape[1])
  a_rot = blind_rotate_utils.apply_automorphism_ntt(ct.data[1], g)
  b_rot = blind_rotate_utils.apply_automorphism_ntt(ct.data[0], g)

  a_rot_ct = types.Ciphertext(
      data=jnp.expand_dims(a_rot, axis=0), moduli=ct.moduli
  )
  b_rot_ct = types.Ciphertext(
      data=jnp.expand_dims(b_rot, axis=0), moduli=ct.moduli
  )

  a_lifted_ct = blind_rotate_utils.lift_ciphertext(
      a_rot_ct, bc_kernel, control_index, p_limbs
  )
  b_lifted_ct = blind_rotate_utils.lift_ciphertext(
      b_rot_ct, bc_kernel, control_index, p_limbs
  )

  a_lifted_pt = types.Plaintext(
      data=jnp.squeeze(a_lifted_ct.data, axis=0), moduli=a_lifted_ct.moduli
  )
  b_lifted_pt = types.Plaintext(
      data=jnp.squeeze(b_lifted_ct.data, axis=0), moduli=b_lifted_ct.moduli
  )

  # Stack a_lifted_pt and b_lifted_pt into vector_v of shape
  # (degree, num_moduli, 2)
  vector_v = jnp.stack([a_lifted_pt.data, b_lifted_pt.data], axis=-1)

  # Compute matrix multiplication using the BAT matrix-vector multiplication
  # kernel
  prod = bat_utils.matmul_bat_key_vector(vector_v, hmrkey.key_matrix_bat)

  # Perform modular reduction
  reduced = barrett.modular_reduction(prod, mul_kernel.barrett_constants)

  ctout = types.Ciphertext(data=reduced, moduli=a_lifted_pt.moduli)

  rescale_kernel.rescale(ctout)

  return ctout


def brot_mux(
    ct_in: types.Ciphertext,
    mux_key: types.MuxRotationKey,
    p_limbs: jax.Array,
    bc_kernel: basis_conversion.BasisConversionBarrett,
    control_index: int,
    mul_kernel: mul.MulPlaintextCiphertextBarrett,
    rescale_kernel: rescale.Rescale,
) -> types.Ciphertext:
  """Homomorphic Blind Rotation using the Mux Method (BRotMux).

  Sequentially applies the MUX-based conditional rotation for each bit of the
  rotation index, resulting in a right-rotation of ct_in by the secret index.

  Args:
    ct_in: The input ciphertext under Q.
    mux_key: The MuxRotationKey containing the keys for each bit.
    p_limbs: The limbs of the auxiliary modulus P.
    bc_kernel: The basis conversion kernel.
    control_index: The control index for basis conversion Q -> P.
    mul_kernel: The multiplication kernel under PQ.
    rescale_kernel: The rescaling kernel.

  Returns:
    A Ciphertext under Q representing the rotated ciphertext.
  """
  ct_out = ct_in
  for k, (hmrkey_jk_0, hmrkey_not_jk_1) in enumerate(mux_key.keys):
    ct0 = hmuxrot(
        ct=ct_out,
        hmrkey=hmrkey_jk_0,
        j=2**k,
        bc_kernel=bc_kernel,
        control_index=control_index,
        p_limbs=p_limbs,
        mul_kernel=mul_kernel,
        rescale_kernel=rescale_kernel,
    )
    ct1 = hmuxrot(
        ct=ct_out,
        hmrkey=hmrkey_not_jk_1,
        j=0,
        bc_kernel=bc_kernel,
        control_index=control_index,
        p_limbs=p_limbs,
        mul_kernel=mul_kernel,
        rescale_kernel=rescale_kernel,
    )
    moduli_expanded = jnp.array(ct0.moduli, dtype=jnp.uint64).reshape(1, 1, -1)
    sum_data = ct0.data.astype(jnp.uint64) + ct1.data.astype(jnp.uint64)
    sum_reduced = jnp.where(
        sum_data >= moduli_expanded, sum_data - moduli_expanded, sum_data
    )
    ct_out = types.Ciphertext(
        data=sum_reduced.astype(jnp.uint32), moduli=ct0.moduli
    )

  return ct_out


def brot_cm(
    cmkey_j: list[types.Ciphertext],
    pt_rot_mu_all: list[types.Plaintext],
    mul_kernel: mul.MulPlaintextCiphertextBarrett,
    rescale_kernel: rescale.Rescale,
) -> types.Ciphertext:
  """Homomorphic Blind Rotation using the Column Method (BRotCM).

  Computes: sum_{i in I} Ecd(Rot_i(mu)) * cmkey_i^(j)
  where cmkey_i^(j) encrypts 1 if i == j else 0.
  The result decrypts to Rot_j(mu).

  To ensure JAX compatibility and performance, this function expects
  pre-encoded rotated plaintexts. The function also expects the plaintexts to be
  encoded with the auxiliary modulus PQ (and the cmkeys under PQ) for level
  saving.

  Args:
    cmkey_j: A list of Ciphertexts representing the column keys for secret index
      j. Length must be equal to the number of slots (num_slots).
    pt_rot_mu_all: A list of Plaintexts containing the encoded rotated versions
      of the message mu for each index i in the set I (corresponding to the
      indices of cmkey_j). Length must be equal to num_slots.
    mul_kernel: The multiplication kernel to use.
    rescale_kernel: The rescaling kernel.

  Returns:
    A Ciphertext encrypting the rotated slots.
  """
  if len(cmkey_j) != len(pt_rot_mu_all):
    raise ValueError("Lengths of cmkey_j and pt_rot_mu_all must match.")

  if not np.array_equal(cmkey_j[0].moduli, pt_rot_mu_all[0].moduli):
    raise ValueError("Moduli of cmkey_j and pt_rot_mu_all must match.")

  ct_data = jnp.stack([ct.data for ct in cmkey_j])

  pt_data = jnp.stack([pt.data for pt in pt_rot_mu_all])

  pt_data_expanded = jnp.expand_dims(pt_data, axis=1)

  batch_ct = types.Ciphertext(data=ct_data, moduli=cmkey_j[0].moduli)
  batch_pt = types.Plaintext(
      data=pt_data_expanded, moduli=pt_rot_mu_all[0].moduli
  )

  # Perform batch multiplication
  batch_ct_mul = mul_kernel.mul(batch_ct, batch_pt)

  # Accumulate the products along the batch axis (0) in uint64 to prevent
  # overflow.
  sum_data = jnp.sum(batch_ct_mul.data.astype(jnp.uint64), axis=0)

  # Perform a single modular reduction on the accumulated sum
  reduced_data = barrett.modular_reduction(
      sum_data, mul_kernel.barrett_constants
  )

  ct_out = types.Ciphertext(
      data=reduced_data.astype(jnp.uint32),
      moduli=cmkey_j[0].moduli,
  )

  rescale_kernel.rescale(ct_out)
  return ct_out
