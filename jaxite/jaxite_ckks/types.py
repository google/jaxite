"""Type definitions shared across modules."""

import dataclasses
import jax
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Plaintext:
  """A CKKS Plaintext."""

  data: jax.Array  # Shape (degree, num_moduli)
  moduli: jax.Array

  def tree_flatten(self):
    return (self.data, self.moduli), None

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(children[0], children[1])


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Ciphertext:
  """A CKKS Ciphertext."""

  data: jax.Array  # Shape (num_elements, degree, num_moduli)
  moduli: jax.Array

  def tree_flatten(self):
    return (self.data, self.moduli), None

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(children[0], children[1])


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


@dataclasses.dataclass(frozen=True)
class EvaluationKeys:
  """CKKS Evaluation Keys."""

  a: jax.Array
  b: jax.Array
  moduli: jax.Array


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class HMuxRotKey:
  """A key used in a single HMuxRot step.

  Consists of two ciphertexts symmetrically encrypted under the destination key
  sk modulo PQ:
    - key0: encrypts P * beta * sk(X^{5^{-j}})
    - key1: encrypts P * beta
  """

  key0: Ciphertext
  key1: Ciphertext

  def tree_flatten(self):
    return (self.key0, self.key1), None

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(children[0], children[1])


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class MuxRotationKey:
  """A set of HMuxRot keys for all bits of a secret rotation index.

  Contains a list of pairs of keys (hmrkey_jk_0, hmrkey_not_jk_1) for each bit k
  from 0 to log2(num_slots) - 1.
  """

  keys: list[tuple[HMuxRotKey, HMuxRotKey]]

  def tree_flatten(self):
    return (self.keys,), None

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(children[0])
