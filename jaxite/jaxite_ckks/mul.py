"""Multiplication kernels for CKKS."""

import abc
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import barrett
from jaxite.jaxite_ckks import types

ABC = abc.ABC
abstractmethod = abc.abstractmethod


class MulPlaintextCiphertextBase(ABC):
  """Abstract base class for plaintext-ciphertext multiplication kernels."""

  @abstractmethod
  def mul(self, ct: types.Ciphertext, pt: types.Plaintext) -> types.Ciphertext:
    """Multiplies ciphertext with plaintext."""


@jax.tree_util.register_pytree_node_class
class MulPlaintextCiphertextSimple(MulPlaintextCiphertextBase):
  """Kernel for raw plaintext-ciphertext multiplication without reduction.

  Use this when delayed reduction is preferred.
  """

  def __init__(self):
    pass

  def mul(self, ct: types.Ciphertext, pt: types.Plaintext) -> types.Ciphertext:
    if ct.moduli.tolist() != pt.moduli.tolist():
      raise ValueError("Moduli of ciphertext and plaintext must match.")
    # Broadcast pt.data to match ct.data shape (num_elements, degree, num_moduli)
    # pt.data has shape (degree, num_moduli)
    return types.Ciphertext(data=ct.data * pt.data, moduli=ct.moduli)

  def tree_flatten(self):
    return (), None

  @classmethod
  def tree_unflatten(cls, _, _children):
    return cls()


@jax.tree_util.register_pytree_node_class
class MulPlaintextCiphertextBarrett(MulPlaintextCiphertextBase):
  """Kernel for plaintext-ciphertext multiplication using Barrett reduction.

  Note: This kernel will execute on the VPU by doing (barrett_reduce(ct * pt)).
  During the blind rotation (bsk) multiplication with the plaintext, we want the
  VPU executed version.
  CROSS also implements an offline BAT on the plaintexts to utilize the MXU to
  perform a pt-ct multiplication, but using this depends on the context.

  Constraints on inputs:
  1. ct.data and pt.data must contain non-negative integers.
  2. For each corresponding element, the product must be strictly less
     than m^2, where m is the corresponding modulus.
  3. The moduli must be less than 2^31.
  """

  def __init__(self, barrett_constants: barrett.BarrettConstants):
    self.barrett_constants = barrett_constants

  def mul(self, ct: types.Ciphertext, pt: types.Plaintext) -> types.Ciphertext:
    if self.barrett_constants is None:
      raise ValueError("Constants must be precomputed first.")
    if ct.moduli.tolist() != pt.moduli.tolist():
      raise ValueError("Moduli of ciphertext and plaintext must match.")
    # Cast to uint64 to prevent overflow during multiplication
    prod = ct.data.astype(jnp.uint64) * pt.data.astype(jnp.uint64)
    # Barrett reduction expects z and constants.
    # constants.m, constants.moduli, etc. have shape (num_moduli,)
    # prod has shape (num_elements, degree, num_moduli)
    # JAX will broadcast constants to match prod shape during operations in modular_reduction.
    reduced = barrett.modular_reduction(prod, self.barrett_constants)
    # Cast back to uint32 explicitly
    return types.Ciphertext(
        data=reduced.astype(jnp.uint32),
        moduli=ct.moduli,
    )

  def tree_flatten(self):
    return (self.barrett_constants,), None

  @classmethod
  def tree_unflatten(cls, _, children):
    obj = cls(children[0])
    return obj
