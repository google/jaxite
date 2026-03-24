"""Barrett modular reduction for JAX-based homomorphic encryption.

For general background see
https://en.wikipedia.org/wiki/Barrett_reduction#Single-word_Barrett_reduction
"""

import dataclasses
import math

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class BarrettConstants:
  """Precomputed constants for Barrett reduction."""

  m: jax.Array
  moduli: jax.Array
  w: jax.Array
  s_w: jax.Array

  def tree_flatten(self):
    children = (self.m, self.moduli, self.w, self.s_w)
    aux_data = None
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


def precompute_barrett_constants(
    moduli: list[int] | int,
) -> BarrettConstants:
  """Precomputes Barrett constants for a list of moduli.

  Args:
    moduli: A list of integers or a single integer representing the moduli.

  Returns:
    A BarrettConstants object containing the precomputed values as JAX arrays.
  """
  if isinstance(moduli, int):
    moduli = [moduli]

  # TODO(#97): Document with a reference
  barrett_s = [2 * math.ceil(math.log2(m)) for m in moduli]
  barrett_w = [min(s, 32) for s in barrett_s]
  barrett_s_w = [s - w for s, w in zip(barrett_s, barrett_w)]
  barrett_m = [math.floor(2**s / m) for s, m in zip(barrett_s, moduli)]

  return BarrettConstants(
      m=jnp.array(barrett_m, dtype=jnp.uint64),
      moduli=jnp.array(moduli, dtype=jnp.uint32),
      w=jnp.array(barrett_w, dtype=jnp.uint16),
      s_w=jnp.array(barrett_s_w, dtype=jnp.uint16),
  )


@jax.jit
def modular_reduction(z: jax.Array, constants: BarrettConstants) -> jax.Array:
  """Vectorized implementation of the Barrett reduction.

  Works for moduli less than 31 bits.

  Args:
    z: The input value(s) to be reduced.
    constants: Precomputed Barrett constants.

  Returns:
    The reduced value(s).
  """
  m = constants.m
  moduli = constants.moduli
  w = constants.w
  s_w = constants.s_w

  # TODO(#97): Document with a reference
  z1 = z & jnp.uint32(0xFFFFFFFF)
  z2 = z >> w
  t = ((z1 * m) >> w) + (z2 * m)
  t = t >> s_w
  z = z - t * moduli
  return jnp.where(z >= moduli, z - moduli, z).astype(jnp.uint32)
