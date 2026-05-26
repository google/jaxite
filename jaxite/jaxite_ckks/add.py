"""Add kernels for CKKS."""

import abc
from typing import Iterable
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import barrett

ABC = abc.ABC
abstractmethod = abc.abstractmethod


class AddBase(ABC):
  """Abstract base class for addition kernels."""

  @abstractmethod
  def precompute_constants(self, moduli: Iterable[int]):
    """Precomputes constants for addition."""

  @abstractmethod
  def add(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Performs addition."""


@jax.tree_util.register_pytree_node_class
class AddSimple(AddBase):
  """Kernel for raw addition without reduction.

  Use this when delayed reduction is preferred, or when adding values that are
  guaranteed not to overflow the data type.
  """

  def __init__(self):
    pass

  def precompute_constants(self, moduli: Iterable[int]):
    pass

  def add(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return a + b

  def tree_flatten(self):
    return (), None

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls()


@jax.tree_util.register_pytree_node_class
class AddModularBarrett(AddBase):
  """Kernel for modular addition using Barrett reduction.

  Use this when general modular addition is needed and inputs might not be
  fully reduced or their sum might exceed 2 * modulus.
  """

  def __init__(self):
    self.barrett_constants = None

  def precompute_constants(self, barrett_constants: barrett.BarrettConstants):
    self.barrett_constants = barrett_constants

  def add(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    if self.barrett_constants is None:
      raise ValueError("Constants must be precomputed first.")
    res = a + b
    return barrett.modular_reduction(res, self.barrett_constants)

  def tree_flatten(self):
    return (self.barrett_constants,), None

  @classmethod
  def tree_unflatten(cls, _, children):
    obj = cls()
    obj.barrett_constants = children[0]
    return obj


@jax.tree_util.register_pytree_node_class
class AddModularSubtract(AddBase):
  """Kernel for modular addition using simple subtraction.

  Use this ONLY when both inputs are guaranteed to be reduced (in range [0,
  modulus-1]), so that their sum is less than 2 * modulus.
  """

  def __init__(self):
    self.moduli = None

  def precompute_constants(self, moduli: Iterable[int]):
    self.moduli = jnp.array(list(moduli), dtype=jnp.uint64)

  def add(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    if self.moduli is None:
      raise ValueError("Constants must be precomputed first.")
    res = a + b
    # Assumes a and b are reduced, so res < 2 * modulus.
    # We need to broadcast moduli to match res shape if needed.
    # Assuming res has shape (..., num_moduli).
    moduli_broadcast = self.moduli.reshape((1,) * (res.ndim - 1) + (-1,))
    return jnp.where(res >= moduli_broadcast, res - moduli_broadcast, res)

  def tree_flatten(self):
    return (self.moduli,), None

  @classmethod
  def tree_unflatten(cls, _, children):
    obj = cls()
    obj.moduli = children[0]
    return obj
