"""Blind rotation implementations for CKKS."""

import jax.numpy as jnp
from jaxite.jaxite_ckks import barrett
from jaxite.jaxite_ckks import mul
from jaxite.jaxite_ckks import rescale
from jaxite.jaxite_ckks import types


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

  if not jnp.array_equal(cmkey_j[0].moduli, pt_rot_mu_all[0].moduli):
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
