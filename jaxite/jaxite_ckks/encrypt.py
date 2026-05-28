"""Encryption utilities for CKKS."""

import abc
import jax.numpy as jnp
from jaxite.jaxite_ckks import ntt_cpu
from jaxite.jaxite_ckks import random
from jaxite.jaxite_ckks import types
import numpy as np

Plaintext = types.Plaintext
PublicKey = types.PublicKey
SecretKey = types.SecretKey
Ciphertext = types.Ciphertext


ABC = abc.ABC
abstractmethod = abc.abstractmethod


class EncryptBase(ABC):
  """Abstract base class for encryption kernels."""

  @abstractmethod
  def precompute_constants(self, public_key: PublicKey):
    """Precomputes constants for encryption."""

  @abstractmethod
  def encrypt(
      self,
      plaintext: Plaintext,
      random_source: random.RandomSource | None = None,
  ) -> Ciphertext:
    """Encrypts a CKKS plaintext."""


class DecryptBase(ABC):
  """Abstract base class for decryption kernels."""

  @abstractmethod
  def precompute_constants(self, secret_key: SecretKey):
    """Precomputes constants for decryption."""

  @abstractmethod
  def decrypt(self, ciphertext: Ciphertext) -> Plaintext:
    """Decrypts a CKKS plaintext."""


class Encrypt(EncryptBase):
  """Kernel for CKKS encryption."""

  def __init__(self):
    self.public_key = None

  def precompute_constants(self, public_key: PublicKey):
    self.public_key = public_key

  def encrypt(
      self,
      plaintext: Plaintext,
      random_source: random.RandomSource | None = None,
  ) -> Ciphertext:
    if self.public_key is None:
      raise ValueError("Public key must be set via precompute_constants first.")

    random_source = random_source or random.SecureRandomSource()

    degree = plaintext.data.shape[0]
    moduli = plaintext.moduli.tolist()

    v = random_source.gen_ternary_poly(degree, moduli)
    e0 = random_source.gen_gaussian_poly(degree, moduli)
    e1 = random_source.gen_gaussian_poly(degree, moduli)

    v_ntt = ntt_cpu.ntt_negacyclic_poly(np.array(v), moduli)
    e0_ntt = ntt_cpu.ntt_negacyclic_poly(np.array(e0), moduli)
    e1_ntt = ntt_cpu.ntt_negacyclic_poly(np.array(e1), moduli)

    q_u64 = np.array(moduli, dtype=np.uint64).reshape(1, -1)
    pk0 = np.array(self.public_key.data[0])
    pk1 = np.array(self.public_key.data[1])

    # c0 = v*pk0 + e0 + m
    prod0 = (v_ntt * pk0) % q_u64
    c0 = (prod0 + e0_ntt + np.array(plaintext.data)) % q_u64

    # c1 = v*pk1 + e1
    prod1 = (v_ntt * pk1) % q_u64
    c1 = (prod1 + e1_ntt) % q_u64

    c_data = np.stack([c0, c1])

    return Ciphertext(
        jnp.array(c_data, dtype=jnp.uint32),
        jnp.array(self.public_key.moduli, dtype=jnp.uint32),
    )


class Decrypt(DecryptBase):
  """Kernel for CKKS decryption."""

  def __init__(self):
    self.secret_key = None

  def precompute_constants(self, secret_key: SecretKey):
    self.secret_key = secret_key

  def decrypt(self, ciphertext: Ciphertext) -> Plaintext:
    if self.secret_key is None:
      raise ValueError("Secret key must be set via precompute_constants first.")

    moduli = ciphertext.moduli.tolist()
    c0 = np.array(ciphertext.data[0])
    c1 = np.array(ciphertext.data[1])
    s = np.array(self.secret_key.data)[..., : len(moduli)]

    q_u64 = np.array(moduli, dtype=np.uint64).reshape(1, -1)

    # m = c0 + c1*s
    res = (c0 + (c1 * s) % q_u64) % q_u64

    return Plaintext(
        jnp.array(res, dtype=jnp.uint32),
        jnp.array(ciphertext.moduli, dtype=jnp.uint32),
    )
