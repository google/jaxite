"""Utility functions for homomorphic blind rotation."""

import math
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import basis_conversion
from jaxite.jaxite_ckks import ntt
from jaxite.jaxite_ckks import types


def apply_automorphism_ntt(data: jax.Array, g: int) -> jax.Array:
  """Applies the automorphism X -> X^g to a polynomial in the NTT domain.

  Handles the bit-reversed layout of jaxite's NTT representation.

  Args:
    data: The polynomial in NTT domain. Shape (..., degree, num_moduli).
    g: The automorphism generator (must be odd).

  Returns:
    The permuted polynomial in NTT domain with the same shape.
  """
  degree = data.shape[-2]
  bits = int(math.log2(degree))
  indices = jnp.arange(degree, dtype=jnp.uint32)

  # Bit-reverse indices to map the bit-reversed layout of jaxite's NTT
  def bit_reverse(x):
    rev = jnp.zeros_like(x)
    temp = x
    for _ in range(bits):
      rev = (rev << 1) | (temp & 1)
      temp >>= 1
    return rev

  br_indices = bit_reverse(indices)
  g_u32 = jnp.array(g, dtype=jnp.uint32)
  target_roots = (((2 * br_indices + 1) * g_u32 - 1) // 2) % degree
  target_indices = bit_reverse(target_roots)

  return jnp.take(data, target_indices, axis=-2)


def lift_ciphertext(
    ct: types.Ciphertext,
    bc_kernel: basis_conversion.BasisConversionBarrett,
    control_index: int,
    p_limbs: jax.Array,
    ntt_q: ntt.NTTBarrett,
    ntt_p: ntt.NTTBarrett,
    r: int,
    c: int,
) -> types.Ciphertext:
  """Lifts a ciphertext from Q to PQ using basis conversion.

  Args:
    ct: The input ciphertext under Q. Shape (num_elements, degree, num_Q).
    bc_kernel: The precomputed basis conversion kernel from Q to P.
    control_index: The control index specifying the Q -> P conversion.
    p_limbs: The limbs of modulus P.
    ntt_q: The NTT kernel for modulus Q.
    ntt_p: The NTT kernel for modulus P.
    r: The row dimension of the NTT layout.
    c: The column dimension of the NTT layout.

  Returns:
    A lifted Ciphertext under PQ. Shape (num_elements, degree, num_Q + num_P).
  """
  # 1. Reshape ct.data to (num_elements, r, c, num_q) for INTT
  num_elements, degree, num_q = ct.data.shape
  ct_data_reshaped = ct.data.reshape(num_elements, r, c, num_q)

  # 2. Convert ct.data to coefficient domain modulo Q
  ct_coef_q = ntt_q.intt(ct_data_reshaped)
  # Reshape back to (num_elements, degree, num_q)
  ct_coef_q_flat = ct_coef_q.reshape(num_elements, degree, num_q)

  # 3. Do basis conversion in coefficient domain: Q -> P
  data_p_coef = bc_kernel.basis_change(
      ct_coef_q_flat, control_index=control_index
  )

  # 4. Reshape data_p_coef to (num_elements, r, c, num_p) for NTT
  num_p = len(p_limbs)
  data_p_coef_reshaped = data_p_coef.reshape(num_elements, r, c, num_p)

  # 5. Convert data_p_coef to NTT domain modulo P
  data_p_ntt = ntt_p.ntt(data_p_coef_reshaped)
  # Reshape back to (num_elements, degree, num_p)
  data_p_ntt_flat = data_p_ntt.reshape(num_elements, degree, num_p)

  # 6. Concatenate Q (NTT) and P (NTT)
  data_pq = jnp.concatenate([ct.data, data_p_ntt_flat], axis=-1)
  moduli_pq = jnp.concatenate([ct.moduli, jnp.asarray(p_limbs)]).astype(
      jnp.uint32
  )
  return types.Ciphertext(data=data_pq, moduli=moduli_pq)
