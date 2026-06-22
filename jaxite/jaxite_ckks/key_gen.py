"""Key generation utilities for CKKS."""

import math

import jax.numpy as jnp
from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import encrypt
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
    hamming_weight: int | None = None,
) -> tuple[PublicKey, SecretKey]:
  """Generate a public, secret key pair."""
  random_source = random_source or random.SecureRandomSource()

  if hamming_weight is not None:
    s = random_source.gen_sparse_binary(degree, hamming_weight, moduli)
  else:
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

  return PublicKey(pk_data, np.array(moduli, dtype=np.uint64)), SecretKey(
      s_ntt, np.array(moduli, dtype=np.uint64)
  )
def extend_secret_key(
    secret_key: SecretKey,
    target_moduli: list[int],
) -> SecretKey:
  """Extends a secret key to a larger set of moduli.

  This is done by converting the key to coefficient domain, tiling the
  coefficients to the target moduli, and converting back to NTT domain.

  Args:
    secret_key: The secret key to extend.
    target_moduli: The larger set of moduli to extend to.

  Returns:
    The extended secret key.
  """
  current_moduli = [int(m) for m in secret_key.moduli]
  if target_moduli[: len(current_moduli)] != current_moduli:
    raise ValueError("target_moduli must start with secret_key.moduli")

  s_coeffs = ntt_cpu.intt_negacyclic_poly(secret_key.data, current_moduli)
  s_coeffs_single = s_coeffs[:, 0]
  s_coeffs_extended = np.tile(
      s_coeffs_single[:, np.newaxis], (1, len(target_moduli))
  )
  s_ext_slots = ntt_cpu.ntt_negacyclic_poly(s_coeffs_extended, target_moduli)
  return SecretKey(s_ext_slots, np.array(target_moduli, dtype=np.uint64))


def compute_scaled_source_key_partition(
    source_key: SecretKey,
    q_limbs: list[int],
    p_val: int,
    all_moduli_len: int,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
  """Computes the scaled source key (P * s_src) for a partition of limbs.

  Args:
    source_key: The source secret key.
    q_limbs: The moduli of the source key.
    p_val: The product of the new moduli (P).
    all_moduli_len: The total number of moduli in the extended key.
    start_idx: The start index of the partition.
    end_idx: The end index of the partition.

  Returns:
    The scaled source key partition.
  """
  degree = source_key.data.shape[0]
  scaled_key = np.zeros((degree, all_moduli_len), dtype=np.uint64)

  if start_idx < end_idx:
    q_slice = np.array(q_limbs[start_idx:end_idx], dtype=np.uint64)
    p_mod_q = np.array([p_val % int(q) for q in q_slice], dtype=np.uint64)
    s_slice = source_key.data[:, start_idx:end_idx]
    res = (s_slice * p_mod_q) % q_slice
    scaled_key[:, start_idx:end_idx] = res

  return scaled_key


def gen_key_switching_key(
    source_key: SecretKey,
    dest_key: SecretKey,
    q_limbs: list[int],
    p_limbs: list[int],
    dnum: int,
    random_source: random.RandomSource | None = None,
) -> types.EvaluationKeys:
  """Generate key switching keys to switch from source_key to dest_key."""
  random_source = random_source or random.SecureRandomSource()

  degree = source_key.data.shape[0]
  num_q = len(q_limbs)
  all_moduli = q_limbs + p_limbs
  all_moduli_u64 = np.array(all_moduli, dtype=np.uint64).reshape(1, -1)

  dest_key_ext = extend_secret_key(dest_key, all_moduli)
  s_dst_ext_slots = dest_key_ext.data

  p_val = math.prod(p_limbs)

  alpha = (num_q + dnum - 1) // dnum
  a_list = []
  b_list = []

  for part in range(dnum):
    a_coeffs = random_source.gen_uniform_poly(degree, all_moduli)
    e_coeffs = random_source.gen_gaussian_poly(degree, all_moduli)

    a_slots = ntt_cpu.ntt_negacyclic_poly(a_coeffs, all_moduli)
    e_slots = ntt_cpu.ntt_negacyclic_poly(e_coeffs, all_moduli)

    start_idx = part * alpha
    end_idx = min((part + 1) * alpha, num_q)
    scaled_key = compute_scaled_source_key_partition(
        source_key=source_key,
        q_limbs=q_limbs,
        p_val=p_val,
        all_moduli_len=len(all_moduli),
        start_idx=start_idx,
        end_idx=end_idx,
    )

    prod = (a_slots * s_dst_ext_slots) % all_moduli_u64
    b_slots = (e_slots + scaled_key + all_moduli_u64 - prod) % all_moduli_u64

    a_list.append(a_slots)
    b_list.append(b_slots)

  return types.EvaluationKeys(
      jnp.array(np.stack(a_list)),
      jnp.array(np.stack(b_list)),
      jnp.array(all_moduli, dtype=jnp.uint64),
  )


def gen_evaluation_key(
    secret_key: SecretKey,
    q_towers: list[int],
    p_towers: list[int],
    dnum: int,
    random_source: random.RandomSource | None = None,
) -> types.EvaluationKeys:
  """Generate evaluation keys for relinearization."""

  # Compute s^2 in NTT domain (this is the source key for relinearization)
  q_moduli_u64 = np.array(q_towers, dtype=np.uint64).reshape(1, -1)
  s2_slots = (secret_key.data * secret_key.data) % q_moduli_u64

  source_key = SecretKey(s2_slots, secret_key.moduli)

  # Call the general function: source is s^2, destination is s
  return gen_key_switching_key(
      source_key=source_key,
      dest_key=secret_key,
      q_limbs=q_towers,
      p_limbs=p_towers,
      dnum=dnum,
      random_source=random_source,
  )


def gen_cm_keys(
    indices: list[int], public_key: PublicKey, scale: float
) -> np.ndarray:
  """Generates column keys."""
  degree = public_key.data.shape[1]
  num_slots = degree // 2
  all_zeroes = [complex(0)] * num_slots
  all_ones = [complex(1)] * num_slots

  encoder = encode.Encode(degree, public_key.moduli.tolist(), scale)
  encryptor = encrypt.Encrypt(public_key)

  plain_0 = encoder.encode(all_zeroes)
  plain_1 = encoder.encode(all_ones)

  n = len(indices)
  cm_keys = np.empty((n, n), dtype=object)

  for i in range(n):
    for j in range(n):
      if i == j:
        cm_keys[i, j] = encryptor.encrypt(plain_1)
      else:
        cm_keys[i, j] = encryptor.encrypt(plain_0)

  return cm_keys
