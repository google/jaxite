"""Key generation utilities for CKKS."""

import math

import jax.numpy as jnp
from jaxite.jaxite_ckks import blind_rotate_utils
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
    sk: SecretKey | None = None,
) -> tuple[PublicKey, SecretKey]:
  """Generate a public, secret key pair."""
  random_source = random_source or random.SecureRandomSource()

  if sk is not None:
    s_ntt = sk.data
  elif hamming_weight is not None:
    s = random_source.gen_sparse_binary(degree, hamming_weight, moduli)
    s_ntt = ntt_cpu.ntt_negacyclic_poly(s, moduli)
  else:
    s = random_source.gen_ternary_poly(degree, moduli)
    s_ntt = ntt_cpu.ntt_negacyclic_poly(s, moduli)

  a = random_source.gen_uniform_poly(degree, moduli)
  e = random_source.gen_gaussian_poly(degree, moduli)

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
    dest_key_ext: SecretKey | None = None,
) -> types.EvaluationKeys:
  """Generate key switching keys to switch from source_key to dest_key."""
  random_source = random_source or random.SecureRandomSource()

  degree = source_key.data.shape[0]
  num_q = len(q_limbs)
  all_moduli = q_limbs + p_limbs
  all_moduli_u64 = np.array(all_moduli, dtype=np.uint64).reshape(1, -1)

  if dest_key_ext is None:
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


def gen_conjugate_key(
    sk: types.SecretKey,
    q_limbs: list[int],
    p_limbs: list[int],
    dnum: int,
    random_source: random.RandomSource | None = None,
) -> types.EvaluationKeys:
  """Generates the key switching key for the conjugate automorphism."""
  # Conjugate of secret key: s(X^-1) is represented by reversing the index
  # of the secret key in NTT domain.
  s_conj_data = np.flip(sk.data, axis=0)
  s_conj = types.SecretKey(s_conj_data, sk.moduli)
  return gen_key_switching_key(
      source_key=s_conj,
      dest_key=sk,
      q_limbs=q_limbs,
      p_limbs=p_limbs,
      dnum=dnum,
      random_source=random_source,
  )


def gen_hmuxrot_key(
    sk: types.SecretKey,
    beta: int,
    j: int,
    q_limbs: list[int],
    p_limbs: list[int],
    random_source: random.RandomSource | None = None,
    sk_ext: types.SecretKey | None = None,
) -> types.HMuxRotKey:
  """Generates an HMuxRotKey symmetrically.

  The key consists of:
    - key0: Symmetrically encrypts target0 = P * beta * sk(X^{5^{-j}}) under sk.
            Constructed as a key-switching key from beta * sk(X^{5^{-j}}) to sk.
    - key1: Symmetrically encrypts target1 = P * beta under sk.

  Args:
    sk: The secret key under Q.
    beta: The selector bit (0 or 1).
    j: The rotation index.
    q_limbs: The limbs of the ciphertext modulus Q.
    p_limbs: The limbs of the auxiliary modulus P.
    random_source: Optional random source. Defaults to SecureRandomSource.
    sk_ext: Optional pre-extended secret key. If not provided, it is computed.

  Returns:
    The generated HMuxRotKey.
  """
  random_source = random_source or random.SecureRandomSource()
  degree = sk.data.shape[0]
  all_moduli = q_limbs + p_limbs
  all_moduli_u64 = np.array(all_moduli, dtype=np.uint64).reshape(1, -1)
  q_limbs_u64 = np.array(q_limbs, dtype=np.uint64).reshape(1, -1)

  if sk_ext is None:
    dest_sk_ext = extend_secret_key(sk, all_moduli)
  else:
    dest_sk_ext = sk_ext

  p_val = math.prod(p_limbs)
  scaled_val0 = (p_val * beta) % all_moduli_u64

  # key0 is a key-switching key from beta * sk(X^{5^{-j}}) to sk.
  g = pow(5, -j, 2 * degree)
  sk_rot_data = blind_rotate_utils.apply_automorphism_ntt(jnp.array(sk.data), g)
  # Scale by beta (modulo Q)
  sk_rot_beta_data = (sk_rot_data * beta) % q_limbs_u64
  sk_rot_beta = types.SecretKey(np.array(sk_rot_beta_data), sk.moduli)

  ksk = gen_key_switching_key(
      source_key=sk_rot_beta,
      dest_key=sk,
      q_limbs=q_limbs,
      p_limbs=p_limbs,
      dnum=1,
      random_source=random_source,
  )
  key0 = types.Ciphertext(
      data=jnp.stack([ksk.b[0], ksk.a[0]]),
      moduli=ksk.moduli,
  )

  # a secret key, we perform direct symmetric encryption inline.
  pk_ext, _ = keygen(degree, all_moduli, random_source, sk=dest_sk_ext)
  encryptor = encrypt.Encrypt(pk_ext)
  target1_data = (
      np.ones((degree, len(all_moduli)), dtype=np.uint64) * scaled_val0
  ) % all_moduli_u64
  plain_target1 = types.Plaintext(
      data=jnp.array(target1_data, dtype=jnp.uint32),
      moduli=jnp.array(all_moduli, dtype=jnp.uint32),
  )
  key1 = encryptor.encrypt(plain_target1, random_source)

  return types.HMuxRotKey(key0, key1)


def gen_mux_rotation_key(
    sk: types.SecretKey,
    secret_bits: list[int],
    q_limbs: list[int],
    p_limbs: list[int],
    stride: int = 1,
    random_source: random.RandomSource | None = None,
) -> types.MuxRotationKey:
  """Generates a MuxRotationKey for the bits of the rotation index.

  For each bit k from 0 to len(secret_bits) - 1, generates:
    - hmrkey_jk_0: HMuxRotKey for beta = secret_bits[k], rotation amount = 2^k *
    stride.
    - hmrkey_not_jk_1: HMuxRotKey for beta = 1 - secret_bits[k], rotation = 0.

  Args:
    sk: The secret key under Q.
    secret_bits: The list of bits representing the secret rotation index.
    q_limbs: The limbs of the ciphertext modulus Q.
    p_limbs: The limbs of the auxiliary modulus P.
    stride: The stride to scale the rotation amount by. Defaults to 1.
    random_source: Optional random source.

  Returns:
    The generated MuxRotationKey.
  """
  all_moduli = q_limbs + p_limbs
  sk_ext = extend_secret_key(sk, all_moduli)

  degree = sk.data.shape[0]
  keys = []
  permutations = []
  for k, bit in enumerate(secret_bits):
    j = (2**k) * stride
    hmrkey_jk_0 = gen_hmuxrot_key(
        sk=sk,
        beta=bit,
        j=j,
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        random_source=random_source,
        sk_ext=sk_ext,
    )
    hmrkey_not_jk_1 = gen_hmuxrot_key(
        sk=sk,
        beta=1 - bit,
        j=0,
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        random_source=random_source,
        sk_ext=sk_ext,
    )
    keys.append((hmrkey_jk_0, hmrkey_not_jk_1))

    g = pow(5, -j, 2 * degree)
    perm = blind_rotate_utils.compute_automorphism_indices(degree, g)
    permutations.append(perm)
  return types.MuxRotationKey(keys, permutations)


def gen_rotation_key(
    sk: types.SecretKey,
    j: int,
    q_limbs: list[int],
    p_limbs: list[int],
    dnum: int,
    random_source: random.RandomSource | None = None,
) -> types.EvaluationKeys:
  """Generates a key switching key to support homomorphic rotation by j slots."""
  degree = sk.data.shape[0]
  g = pow(5, j, 2 * degree)
  sk_rot_data = blind_rotate_utils.apply_automorphism_ntt(jnp.array(sk.data), g)
  sk_rot = types.SecretKey(np.array(sk_rot_data), sk.moduli)
  return gen_key_switching_key(
      source_key=sk_rot,
      dest_key=sk,
      q_limbs=q_limbs,
      p_limbs=p_limbs,
      dnum=dnum,
      random_source=random_source,
  )


def gen_hybrid_key(
    sk: types.SecretKey,
    j: int,
    idx: int,
    s_j: int,
    theta: int,
    q_limbs: list[int],
    p_limbs: list[int],
    random_source: random.RandomSource | None = None,
) -> tuple[list[list[types.Ciphertext]], types.MuxRotationKey]:
  """Generates the hybrid evaluation keys (giant steps and baby steps) for a secret index j."""
  random_source = random_source or random.SecureRandomSource()
  degree = sk.data.shape[0]
  num_slots = degree // 2
  num_giant_steps = num_slots // theta
  all_moduli = q_limbs + p_limbs

  sk_ext = extend_secret_key(sk, all_moduli)

  j_mod = j % num_slots
  j_1 = j_mod // theta
  j_0 = j_mod % theta

  # 1. Generate MuxRotationKey for the giant steps (j_1)
  num_bits = math.ceil(math.log2(num_giant_steps))
  j1_bits = [int((j_1 >> k) & 1) for k in range(num_bits)]
  mmkey_hybrid = gen_mux_rotation_key(
      sk=sk,
      secret_bits=j1_bits,
      q_limbs=q_limbs,
      p_limbs=p_limbs,
      random_source=random_source,
      stride=theta,
  )

  # 2. Compute the 4 masks m^{(k)}
  m_size = num_slots
  mask_base = np.zeros(m_size, dtype=complex)
  j_mod = j % m_size
  mask_base[j_mod:] = 1.0

  s_pos = (1 + s_j) // 2
  s_neg = (1 - s_j) // 2

  # Order the masks to align with the plaintexts order in bootstrapping:
  # - masks[0]: first half, positive (goes with pt1 / pts[0])
  # - masks[1]: first half, negative/conjugate (goes with pt2 / pts[1])
  # - masks[2]: second half, positive (goes with pt3 / pts[2])
  # - masks[3]: second half, negative/conjugate (goes with pt4 / pts[3])
  if idx < m_size:
    masks = [
        s_pos * mask_base,
        s_neg * mask_base,
        s_neg * (1.0 - mask_base),
        s_pos * (1.0 - mask_base),
    ]
  else:
    masks = [
        s_neg * (1.0 - mask_base),
        s_pos * (1.0 - mask_base),
        s_neg * mask_base,
        s_pos * mask_base,
    ]

  # 3. Rotate the masks by -j_1 * theta
  rotated_masks = [np.roll(m_k, -j_1 * theta) for m_k in masks]

  # 4. Encrypt the giant-step keys under PQ
  p_val = math.prod(p_limbs)

  encoder_pq = encode.Encode(degree, all_moduli, float(p_val))

  all_zeroes = [complex(0)] * num_slots
  plain_0 = encoder_pq.encode(all_zeroes)

  pk_ext, _ = keygen(degree, all_moduli, random_source, sk=sk_ext)
  encryptor = encrypt.Encrypt(pk_ext)

  cmkey_hybrid = []
  for k in range(4):
    cmkey_hybrid_k = []
    plain_mask = encoder_pq.encode(rotated_masks[k].tolist())

    for baby_step in range(theta):
      if baby_step == j_0:
        target = plain_mask
      else:
        target = plain_0

      ct = encryptor.encrypt(target, random_source)
      cmkey_hybrid_k.append(ct)
    cmkey_hybrid.append(cmkey_hybrid_k)

  return cmkey_hybrid, mmkey_hybrid
