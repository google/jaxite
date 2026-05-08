"""Key generation utilities for CKKS."""

import jax.numpy as jnp
from jaxite.jaxite_ckks import ntt_cpu
from jaxite.jaxite_ckks import random
from jaxite.jaxite_ckks import types
import numpy as np

PublicKey = types.PublicKey
SecretKey = types.SecretKey


def keygen(
    degree: int,
    moduli: list[int],
    random_source: random.RandomSource | None = None,
) -> tuple[PublicKey, SecretKey]:
  """Generate a public, secret key pair."""
  random_source = random_source or random.SecureRandomSource()

  s = random_source.gen_ternary_poly(degree, moduli)
  a = random_source.gen_uniform_poly(degree, moduli)
  e = random_source.gen_gaussian_poly(degree, moduli)

  s_ntt = ntt_cpu.ntt_negacyclic_poly(s, moduli)
  a_ntt = ntt_cpu.ntt_negacyclic_poly(a, moduli)
  e_ntt = ntt_cpu.ntt_negacyclic_poly(e, moduli)

  q_u64 = np.array(moduli, dtype=np.uint64).reshape(1, -1)
  prod = (a_ntt * s_ntt) % q_u64
  b_ntt = (e_ntt + q_u64 - prod) % q_u64
  pk_data = np.stack([b_ntt, a_ntt])

  return PublicKey(
      pk_data, np.array(moduli, dtype=np.uint64)
  ), SecretKey(s_ntt, np.array(moduli, dtype=np.uint64))
