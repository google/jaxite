"""Type definitions shared across modules."""

import dataclasses
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import bat_utils
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
  key_matrix_bat: jax.Array = dataclasses.field(init=False)

  def __post_init__(self):
    # Stack key0.data and key1.data along axis 1 to get
    # (2, 2, degree, num_moduli).
    # Then transposing to (degree, num_moduli, 2, 2)
    stacked = jnp.stack([self.key0.data, self.key1.data], axis=1)
    key_matrix = jnp.transpose(stacked, (2, 3, 0, 1))

    # Precompute BAT representation using bat_utils helper
    key_matrix_bat = bat_utils.basis_aligned_transform_key(
        key_matrix, self.key0.moduli
    )
    object.__setattr__(self, "key_matrix_bat", key_matrix_bat)

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
