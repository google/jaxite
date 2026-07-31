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

"""Homomorphic ciphertext rotation for CKKS."""

import math
import jax
from jaxite.jaxite_ckks import basis_conversion
from jaxite.jaxite_ckks import blind_rotate_utils
from jaxite.jaxite_ckks import key_switching
from jaxite.jaxite_ckks import mul
from jaxite.jaxite_ckks import rescale
from jaxite.jaxite_ckks import types


@jax.tree_util.register_pytree_node_class
class Rotate:
  """Kernel for homomorphic ciphertext rotation on TPU."""

  def __init__(self):
    self.key_switcher = key_switching.KeySwitcher()
    self.bc_kernel = basis_conversion.BasisConversionBarrett()
    self.mul_kernel = mul.MulPlaintextCiphertextBarrett()
    self.rescale_kernel = rescale.Rescale()

  def tree_flatten(self):
    children = (
        self.key_switcher,
        self.bc_kernel,
        self.mul_kernel,
        self.rescale_kernel,
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
    """Precomputes constants and sub-kernels for rotation."""
    all_moduli = q_limbs + p_limbs

    # 1. Precompute BasisConversion constants
    limbs_per_part = math.ceil(len(q_limbs) / dnum)
    bc_pairs = []
    for i in range(dnum):
      start_idx = i * limbs_per_part
      end_idx = min(start_idx + limbs_per_part, len(q_limbs))
      in_indices = list(range(start_idx, end_idx))
      out_indices = [j for j in range(len(all_moduli)) if j not in in_indices]
      bc_pairs.append((in_indices, out_indices))
    self.bc_kernel.precompute_constants(all_moduli, bc_pairs)

    # 2. Precompute Mul constants
    self.mul_kernel.precompute_constants(all_moduli)

    # 3. Precompute Rescale constants
    self.rescale_kernel.precompute_constants(
        all_moduli, num_rescales=num_rescales, r=r, c=c
    )

    # 4. Precompute KeySwitcher constants
    self.key_switcher.precompute_constants(
        q_limbs,
        p_limbs,
        dnum,
        r,
        c,
        bc_kernel=self.bc_kernel,
        mul_kernel=self.mul_kernel,
    )

  def rotate(
      self,
      ct: types.Ciphertext,
      rot_key: types.EvaluationKeys,
      j: int,
      control_index: int = 0,
  ) -> types.Ciphertext:
    """Homomorphically rotates a CKKS ciphertext by j slots.

    Args:
      ct: The input ciphertext (c0, c1) under Q.
      rot_key: The key switching key from s(X^g) to s(X).
      j: The rotation amount (number of slots).
      control_index: The control index for basis conversion Q -> P.

    Returns:
      A Ciphertext under Q representing the rotated ciphertext.
    """
    degree = ct.data.shape[1]

    # Step 1: Apply automorphism X -> X^g in the NTT domain
    g = pow(5, j, 2 * degree)
    ct_rot_data = blind_rotate_utils.apply_automorphism_ntt(ct.data, g)
    ct_rot = types.Ciphertext(data=ct_rot_data, moduli=ct.moduli)

    # Step 2: Key Switch: convert ct_rot from Q to P under original key s(X)
    ct_switched = self.key_switcher.key_switch(
        ct=ct_rot,
        ksk=rot_key,
        start_control_index=control_index,
    )

    # Step 3: Rescale by P to drop auxiliary modulus and divide by P
    self.rescale_kernel.rescale(ct_switched)

    return ct_switched
