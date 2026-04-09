"""Type definitions shared across modules."""

import dataclasses
import jax
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Plaintext:
  """A CKKS Plaintext."""

  data: jax.Array  # Shape (degree, num_moduli)
  moduli: jax.Array

  def tree_flatten(self):
    return (self.data, self.moduli), None

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Ciphertext:
  """A CKKS Ciphertext."""

  data: jax.Array  # Shape (num_elements, degree, num_moduli)
  moduli: jax.Array

  def tree_flatten(self):
    return (self.data, self.moduli), None

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@dataclasses.dataclass(frozen=True)
class PublicKey:
  """A CKKS Public Key."""

  data: np.ndarray  # Shape (2, degree, num_moduli)
  moduli: np.ndarray


@dataclasses.dataclass(frozen=True)
class SecretKey:
  """A CKKS Secret Key."""

  data: np.ndarray  # Shape (degree, num_moduli)
  moduli: np.ndarray
