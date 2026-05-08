"""Random generation utilities."""

import abc
import secrets
import numpy as np


class RandomSource(abc.ABC):
  """An interface for random number generation in CKKS."""

  @abc.abstractmethod
  def gen_gaussian_poly(
      self, degree: int, moduli: list[int], sigma: float = 3.19
  ) -> np.ndarray:
    pass

  @abc.abstractmethod
  def gen_ternary_poly(self, degree: int, moduli: list[int]) -> np.ndarray:
    pass

  @abc.abstractmethod
  def gen_uniform_poly(self, degree: int, moduli: list[int]) -> np.ndarray:
    pass


class SecureRandomSource(RandomSource):
  """Random generation utilities for CKKS."""

  def __init__(self):
    self.rng = secrets.SystemRandom()

  def gen_gaussian_poly(
      self, degree: int, moduli: list[int], sigma: float = 3.19
  ) -> np.ndarray:
    coeffs = [round(self.rng.normalvariate(0, sigma)) for _ in range(degree)]
    res = []
    for q in moduli:
      res.append([c % q for c in coeffs])
    return np.array(res, dtype=np.uint64).T

  def gen_ternary_poly(self, degree: int, moduli: list[int]) -> np.ndarray:
    coeffs = [self.rng.choice([0, 1]) for _ in range(degree)]
    res = []
    for q in moduli:
      res.append([c % q for c in coeffs])
    return np.array(res, dtype=np.uint64).T

  def gen_uniform_poly(self, degree: int, moduli: list[int]) -> np.ndarray:
    res = []
    for q in moduli:
      res.append([self.rng.randrange(0, q) for _ in range(degree)])
    return np.array(res, dtype=np.uint64).T


class ZeroNoiseRandomSource(SecureRandomSource):
  """A random source that zeros out the gaussian noise.

  This is meant to be used only in testing, when the test wants to verify exact
  algebraic correctness of operations that normally involve CKKS noise and
  preclude an exact comparison.
  """

  def gen_gaussian_poly(
      self, degree: int, moduli: list[int], sigma: float = 3.19
  ) -> np.ndarray:
    return np.zeros((degree, len(moduli)), dtype=np.uint64)


class TestRandomSource(SecureRandomSource):
  """A random source that can be seeded, for testing."""

  def __init__(self, seed: int):
    import random as std_random
    self.rng = std_random.Random(seed)
