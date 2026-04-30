"""Encryption utilities for CKKS."""

import jax.numpy as jnp
from jaxite.jaxite_ckks import ntt_cpu
from jaxite.jaxite_ckks import random
from jaxite.jaxite_ckks import types
import numpy as np

Plaintext = types.Plaintext
PublicKey = types.PublicKey
SecretKey = types.SecretKey
Ciphertext = types.Ciphertext


def encrypt(
    plaintext: Plaintext,
    public_key: PublicKey,
    random_source: random.RandomSource | None = None,
) -> Ciphertext:
  """Encrypts a CKKS plaintext using a given public key.

  The computation is performed on CPU because client side operations are not
  expected to have access to a TPU.

  Args:
    plaintext: The Plaintext to encrypt.
    public_key: The PublicKey used for encryption.
    random_source: The RandomSource used for generating randomness.

  Returns:
    The resulting Ciphertext.
  """
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
  pk0 = np.array(public_key.data[0])
  pk1 = np.array(public_key.data[1])

  # c0 = v*pk0 + e0 + m
  prod0 = (v_ntt * pk0) % q_u64
  c0 = (prod0 + e0_ntt + np.array(plaintext.data)) % q_u64

  # c1 = v*pk1 + e1
  prod1 = (v_ntt * pk1) % q_u64
  c1 = (prod1 + e1_ntt) % q_u64

  c_data = np.stack([c0, c1])

  return Ciphertext(jnp.array(c_data), jnp.array(public_key.moduli))


def decrypt(ciphertext: Ciphertext, secret_key: SecretKey) -> Plaintext:
  """Decrypts a CKKS plaintext using a given secret key.

  The computation is performed on CPU because client side operations are not
  expected to have access to a TPU.

  Args:
    ciphertext: The Ciphertext to decrypt.
    secret_key: The SecretKey used for decryption.

  Returns:
    The decrypted Plaintext.
  """
  moduli = ciphertext.moduli.tolist()
  c0 = np.array(ciphertext.data[0])
  c1 = np.array(ciphertext.data[1])
  s = np.array(secret_key.data)

  q_u64 = np.array(moduli, dtype=np.uint64).reshape(1, -1)

  # m = c0 + c1*s
  res = (c0 + (c1 * s) % q_u64) % q_u64

  return Plaintext(jnp.array(res), ciphertext.moduli)
