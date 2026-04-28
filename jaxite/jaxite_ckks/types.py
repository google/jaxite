"""Type definitions shared across modules."""

import dataclasses
import jax


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
