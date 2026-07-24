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

"""Conjugation utilities for CKKS ciphertexts."""

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import basis_conversion
from jaxite.jaxite_ckks import key_switching
from jaxite.jaxite_ckks import mul
from jaxite.jaxite_ckks import rescale
from jaxite.jaxite_ckks import types


@jax.tree_util.register_pytree_node_class
class Conjugation:
  """Kernel for homomorphic conjugation on TPU."""

  key_switcher: key_switching.KeySwitcher
  rescale_kernel: rescale.Rescale
  conj_key: types.EvaluationKeys
  start_control_index: int

  def precompute_constants(
      self,
      q_limbs: list[int],
      p_limbs: list[int],
      dnum: int,
      r: int,
      c: int,
      bc_kernel: basis_conversion.BasisConversionBarrett,
      mul_kernel: mul.MulPlaintextCiphertextBarrett,
      rescale_kernel: rescale.Rescale,
      conj_key: types.EvaluationKeys,
      start_control_index: int,
  ):
    self.key_switcher = key_switching.KeySwitcher()
    self.key_switcher.precompute_constants(
        q_limbs,
        p_limbs,
        dnum,
        r,
        c,
        bc_kernel=bc_kernel,
        mul_kernel=mul_kernel,
    )
    self.rescale_kernel = rescale_kernel
    self.conj_key = conj_key
    self.start_control_index = start_control_index

  def tree_flatten(self):
    children = (self.key_switcher, self.rescale_kernel, self.conj_key)
    aux_data = (self.start_control_index,)
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    obj = cls()
    obj.key_switcher = children[0]
    obj.rescale_kernel = children[1]
    obj.conj_key = children[2]
    obj.start_control_index = aux_data[0]
    return obj

  def conjugate(self, ct: types.Ciphertext) -> types.Ciphertext:
    """Homomorphically conjugates a CKKS ciphertext."""
    # 1. Apply automorphism X -> X^-1 by flipping along the degree dimension
    ct_conj = types.Ciphertext(
        data=jnp.flip(ct.data, axis=1),
        moduli=ct.moduli,
    )

    # 2. Key Switch: convert ct_conj from Q to P
    ct_prime = self.key_switcher.key_switch(
        ct=ct_conj,
        ksk=self.conj_key,
        start_control_index=self.start_control_index,
    )

    # 3. Rescale by P to drop auxiliary modulus and divide by P
    self.rescale_kernel.rescale(ct_prime)
    return ct_prime
