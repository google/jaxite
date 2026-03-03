"""Global Configuration for running jaxite_word.

The default input data type is 64 bit integer.
"""

import math
import jax
import jax.sharding as shd
import re
import os
import json
import gzip
import jax.numpy as jnp
from typing import Any, Callable, List, Tuple, Union
import copy

gcd = math.gcd

# Capture existing profile run directories so we can identify the new one.
profile_root = os.path.join("./log/xprof", "plugins", "profile")
try:
  pre_existing_dirs = set(os.listdir(profile_root)) if os.path.isdir(profile_root) else set()
except Exception:
  pre_existing_dirs = set()

####################################
# Utility Functions
####################################
def _square_like_mesh_shape(device_count: int) -> Tuple[int, int]:
  """Return a near-square 2D mesh shape that covers all available devices."""
  if device_count <= 0:
    raise ValueError("At least one device is required to build a mesh.")
  sqrt_devices = math.isqrt(device_count)
  for dim0 in range(sqrt_devices, 0, -1):
    if device_count % dim0 == 0:
      return dim0, device_count // dim0
  return 1, device_count


def create_sharding():
  """Create default batch and replicated shardings for the current device mesh."""
  available_devices = jax.devices()
  if not available_devices:
    raise RuntimeError("No devices available for sharding test.")
  if len(available_devices) == 8:
    mesh_shape = (2, 4)
  elif len(available_devices) == 4:
    mesh_shape = (2, 2)
  elif len(available_devices) == 2:
    mesh_shape = (2, 1)
  else:
    mesh_shape = (1, 1)

  mesh = jax.make_mesh(mesh_shape, ('x', 'y'))
  shd.set_mesh(mesh)

  partition_spec = jax.sharding.PartitionSpec
  return mesh, partition_spec


def num_bits(x: int) -> int:
  """Returns the number of bits in x."""
  return x.bit_length() - 1


def is_power_of_two(x: int) -> bool:
  """Returns True if x is a power of two."""
  return x > 0 and (x & (x - 1)) == 0


def to_tuple(a):
  """Create to convert numpy array into tuple."""
  try:
    return tuple(to_tuple(i) for i in a)
  except TypeError:
    return a


def slice_first_k_along_axis0(arrays, k):
  """
  Given an iterable of array-like or sequence objects, return a tuple where
  each element is the slice of the original object taking the first k entries
  along axis 0 (i.e., obj[:k]).

  Example:
    (s_tuple, s_w_tuple, w_tuple, m_tuple) ->
    (s_tuple[:k], s_w_tuple[:k], w_tuple[:k], m_tuple[:k])
  """
  return tuple(arr[:k] for arr in arrays)


def slice_k_to_end_along_axis0(arrays, k):
  """
  Given an iterable of array-like or sequence objects, return a tuple where
  each element is the slice of the original object taking the k to end entries
  along axis 0 (i.e., obj[k:]).

  Example:
    (s_tuple, s_w_tuple, w_tuple, m_tuple) ->
    (s_tuple[k:], s_w_tuple[k:], w_tuple[k:], m_tuple[k:])
  """
  return tuple(arr[k:] for arr in arrays)


def slice_kth_along_axis0(arrays, k):
  """
  Given an iterable of array-like or sequence objects, return a tuple where
  each element is the slice of the original object taking the first k entries
  along axis 0 (i.e., obj[k]).

  Example:
    (s_tuple, s_w_tuple, w_tuple, m_tuple) ->
    (s_tuple[k], s_w_tuple[k], w_tuple[k], m_tuple[k])
  """
  return tuple(arr[k] for arr in arrays)


def slice_0_to_k0_to_k1_along_axis0(arrays, k0, k1):
  """
  Given an iterable of array-like or sequence objects, return a tuple where
  each element is the slice of the original object from k0 to k1 along axis 0 (i.e., obj[k0:k1]).

  Example:
    (s_tuple, s_w_tuple, w_tuple, m_tuple) ->
    (s_tuple[k0:k1], s_w_tuple[k0:k1], w_tuple[k0:k1], m_tuple[k0:k1])
  """
  if isinstance(arrays[0], jnp.ndarray):
    return tuple(jnp.concatenate([x[:k0], x[k1:]]) for x in arrays)
  elif isinstance(arrays, tuple):
    return tuple([arr[:k0] + arr[k1:] for arr in arrays])
  elif isinstance(arrays, list):
    return [arr[:k0] + arr[k1:] for arr in arrays]
  else:
    raise ValueError(f"Unsupported type: {type(arrays)}")


def slice_k0_to_k1_axis0(arrays, k0, k1):
  """
  Given an iterable of array-like or sequence objects, return a tuple where
  each element is the slice of the original object from k0 to k1 along axis 0 (i.e., obj[k0:k1]).

  Example:
    (s_tuple, s_w_tuple, w_tuple, m_tuple) ->
    (s_tuple[k0:k1], s_w_tuple[k0:k1], w_tuple[k0:k1], m_tuple[k0:k1])
  """
  return tuple(arr[k0:k1] for arr in arrays)

####################################
# Math Functions
####################################
def extended_gcd(a, b):
  """Return a tuple of (g, x, y) such that a*x + b*y = g = gcd(a, b)."""
  if b == 0:
    return (a, 1, 0)
  else:
    g, x, y = extended_gcd(b, a % b)
    return (g, y, x - (a // b) * y)


def modinv_manual(x, q):
  """Returns the inverse of x mod q."""
  g, x, _ = extended_gcd(x, q)
  if g != 1:
    raise Exception(f'Modular inverse does not exist for {x} modulo {q}')
  else:
    return x % q


def modinv(x: int, q: int) -> int:
  """Returns the inverse of x mod q."""
  return int(pow(x, -1, q))


def prime_factors(n):
  """Return the set of prime factors of n."""
  factors = set()
  # Divide out factors of 2
  while n % 2 == 0:
    factors.add(2)
    n //= 2
  # Check odd factors from 3 to sqrt(n)
  p = 3
  while p**2 <= n:
    while n % p == 0:
      factors.add(p)
      n //= p
    p += 2
  if n > 1:
    factors.add(n)
  return factors


def find_generator(q):
  """Find a primitive root modulo q.

  Args:
    q (int): The prime modulus.

  Returns:
    A generator of GF(q)^*.

  Raises:
    ValueError: If no generator is found, indicating q is not prime.
  """
  phi = q - 1
  factors = prime_factors(phi)

  # Test candidates from 2 to q-1.
  for g in range(2, q):
    is_generator = all(pow(g, phi // p, q) != 1 for p in factors)
    if is_generator:
      return g
  raise ValueError("No generator found, check that q is prime.")


####################################
# Parameters Generation
####################################
def root_of_unity(m: int, q: int) -> Union[complex, float, int]:
    """Canonical primitive m-th root of unity modulo q that **works with NTT**.

    Args:
      m (int): The order of the root of unity.
      q (int): The prime modulus.

    Returns:
      int: The canonical primitive m-th root of unity modulo q.

    Usage:
      root_of_unity(16, 134219681) # This works with NTT.
      computed_psi = [root_of_unity(m, q) for q in original_modulus]
    """
    assert (q - 1) % m == 0, "q-1 must be divisible by m"
    # Step 1: multiplicative generator of Z_q^*
    g = find_generator(q)
    # Step 2: raise to (q-1)/m to get an m-th root candidate
    r = pow(g, (q - 1) // m, q)
    # Step 3: among r^k with gcd(k,m)=1, pick the minimal value whose order is exactly m
    # For m=2^t, order check is psi^(m/2) == q-1 (i.e., == -1 mod q)
    candidates = []
    half = m // 2
    for k in range(1, m):
        if gcd(k, m) != 1:
            continue
        psi = pow(r, k, q)
        if pow(psi, half, q) == q - 1 and pow(psi, m, q) == 1:
            candidates.append(psi)
    assert candidates, "No primitive m-th root found"
    return min(candidates)


def any_primitive_root_of_unity(n, q):
  """Canonical primitive m-th root of unity modulo q that **may not work with NTT**.

    Args:
      m (int): The order of the root of unity.
      q (int): The prime modulus.

    Returns:
      int: The canonical primitive m-th root of unity modulo q.

    Usage:
      root_of_unity(16, 134219681) # This may not work with NTT.
      computed_psi = [root_of_unity(m, q) for q in original_modulus]
    """
  if (q - 1) % n != 0:
    raise ValueError(
        "n must divide q-1 for a primitive n-th root of unity to exist."
    )

  # Find a generator g of GF(q)^* (a primitive element).
  g = find_generator(q)
  # Compute omega = g^((q-1)/n) mod q.
  exponent = (q - 1) // n
  omega = pow(g, exponent, q)

  # Optional: Verify that omega is indeed of order n.
  if pow(omega, n, q) != 1:
    raise ValueError("Something went wrong: omega^n != 1")
  # Check that no smaller positive exponent gives 1.
  for d in range(1, n):
    if n % d == 0 and pow(omega, d, q) == 1:
      raise ValueError(
          "Found an exponent d < n with omega^d == 1, so omega is not"
          " primitive."
      )

  return omega


def compute_barrett_mu(modulus):
  """Compute the Barrett reduction constant mu for a given modulus m.

  Args:
    modulus (int): The modulus.

  Returns:
    tuple: (mu, k) where mu is the precomputed constant and k is the number of
    digits in base b.
  """
  # k is the smallest integer such that m < b^k.
  b = 2 ** (math.floor(math.log(modulus)) + 1)
  # For m < 2^64, k will be 1.
  k_val = math.floor(math.log(modulus, b)) + 1

  # Compute mu = floor(b^(2k) / m)
  barrett_mu = (b ** (2 * k_val)) // modulus
  return barrett_mu, k_val


def compute_QHatInvModq_QHatModp(original_moduli, target_moduli, perf_test=False):
  """
  Given a list of moduli original_moduli, compute QHatInvModq.
  Input:
    - original_moduli (list[int]):
            The list of primes (moduli) defining the original CRT basis (Q).
    - target_moduli (list[int]):
            The list of primes (moduli) defining the target CRT basis (P).

  For each modulus q_i, compute:
    - Qhat_i = Q // q_i
    - QHatInvModq[i] = modular inverse of Qhat_i modulo q_i
    - QHatModp: Precomputed Q̂ modulo each prime in P. Used in approximate basis switching.
  """
  if perf_test:
      sizeP = len(original_moduli)
      sizeQ = len(target_moduli)
      # Random arrays with matching shapes/dtypes
      PInvModq = random_parameters((sizeQ,), target_moduli, dtype=jnp.uint32).tolist()
      QHatInvModq = random_parameters((sizeP,), target_moduli, dtype=jnp.uint32).tolist()
      QHatModp = random_parameters((sizeP, sizeQ), [min(target_moduli + original_moduli)], dtype=jnp.uint32).tolist()
      return to_tuple((QHatInvModq, QHatModp))
  else:
    Q = 1
    for qi in original_moduli:
      Q *= qi

    QHatInvModq = []
    QHat = []
    for qi in original_moduli:
      Qhat_i = Q // qi
      inv = modinv(Qhat_i, qi)
      QHat.append(Qhat_i)
      QHatInvModq.append(inv)

    QHatModp = []
    for i in range(len(original_moduli)):
      QHatModp_sgl = []
      for j in range(len(target_moduli)):
        QHatModp_sgl.append(QHat[i] % target_moduli[j])
      QHatModp.append(QHatModp_sgl)

  return QHatInvModq, QHatModp


def approx_mod_down_control_generation(current_moduli, target_moduli, perf_test=False):
  if perf_test:
    PInvModq = random_parameters((len(target_moduli),), target_moduli, dtype=jnp.uint32).tolist()
  else:
    P = 1
    for moduli in current_moduli:
      P *= moduli
    PInvModq = [modinv(P, q) for q in target_moduli]
  overall_moduli =  current_moduli + target_moduli
  QHatInvModq, QHatModp = compute_QHatInvModq_QHatModp(current_moduli, target_moduli, perf_test=perf_test)

  return PInvModq, len(overall_moduli) - len(target_moduli), len(overall_moduli) - len(current_moduli), QHatInvModq, QHatModp


def compute_powers_of_psi(ring_dim, moduli, perf_test=False):
  """Computes powers of psi for the given moduli."""
  if perf_test:
    return random_parameters((len(moduli), ring_dim), moduli, dtype=jnp.uint64)
  else:
    psi = [root_of_unity(2 * ring_dim, q) for q in moduli]
    return jnp.array(
        [
            [pow(psi[idx], i, moduli[idx]) for i in range(ring_dim)]
            for idx in range(len(moduli))
        ],
        jnp.uint64,
    )


def is_prime_deterministic(n):
    """
    Deterministic primality test for n < 2^64.
    Uses Trial Division for speed + Deterministic Miller-Rabin for correctness.
    """
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0: return False

    # 1. SPEED OPTIMIZATION: Trial Division
    # Check divisibility by small primes to fail fast on obvious composites.
    # This filters out ~85% of candidates without expensive modular exponentiation.
    small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    for p in small_primes:
        if n == p: return True
        if n % p == 0: return False

    # 2. DETERMINISTIC MILLER-RABIN
    # For n < 2^64, verifying these specific bases guarantees primality.
    # No randomness involved.
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    # Bases required for deterministic check up to 2^64
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    for a in bases:
        if a >= n: break

        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue

        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False # Composite

    return True # Prime


def find_moduli_ntt(total_number, precision, ntt_length):
    """
    Deterministically finds the largest valid NTT moduli.

    Args:
      total_number: Number of moduli to find.
      precision: Bit-width (e.g., 60 for < 2^60).
      ntt_length: The required N-th root of unity (e.g., 1024).
    """
    overall_moduli = []

    # Upper bound
    limit = 2**precision

    # Start search from the largest possible k
    # P = k * ntt_length + 1
    k = (limit - 1) // ntt_length

    while len(overall_moduli) < total_number and k > 0:
        candidate_p = k * ntt_length + 1

        # Check candidate
        if is_prime_deterministic(candidate_p):
            overall_moduli.append(candidate_p)

        k -= 1

    return overall_moduli


def gamma_beta_calculation(moduli_list, perf_test=False):
  if perf_test:
    # Shapes: gammas: (len(moduli_list)-1,), betas: (len(moduli_list)-1,)
    assert len(moduli_list) > 1, "moduli_list must have at least 2 moduli"
    gamma_rand = random_parameters((len(moduli_list)-1,), moduli_list[:-1], dtype=jnp.uint64)
    beta_rand = random_parameters((len(moduli_list)-1,), moduli_list[:-1], dtype=jnp.uint64)
    return jnp.array(gamma_rand, jnp.uint64), jnp.array(beta_rand, jnp.uint64)
  # Compute Q as the product of the moduli for the remaining towers.
  Q = 1
  for m in moduli_list[:-1]:
    Q *= m

  num_towers = len(moduli_list)
  q_l = moduli_list[-1]
  # Compute Q_inv_mod_ql: the inverse of Q modulo q_l.
  Q_inv_mod_ql = modinv(Q, q_l)

  # Compute gamma_common such that:
  # Q * Q_inv_mod_ql = 1 + gamma_common * q_l.
  # Hence, gamma_common = (Q * Q_inv_mod_ql - 1) // q_l.
  gamma_common = (Q * Q_inv_mod_ql - 1) // q_l

  # For each remaining tower compute gamma_i and beta_i.
  gammas = []
  betas = []
  for i in range(num_towers - 1):
    mod_i = moduli_list[i]
    gamma_i = gamma_common % mod_i
    beta_i = modinv(q_l, mod_i)
    gammas.append(gamma_i)
    betas.append(beta_i)
  return jnp.array(gammas, jnp.uint64), jnp.array(betas, jnp.uint64)

####################################
# Random Functions
####################################
def random_batched_ciphertext(shape, modulus_list, dtype=jnp.int32):
  assert len(modulus_list) == shape[-1]
  random_key = jax.random.key(0)
  return jnp.concatenate(
      [
          jax.random.randint(
              random_key,
              shape=(shape[0], shape[1], shape[2], 1),
              minval=0,
              maxval=bound,
              dtype=dtype,
          )
          for bound in modulus_list
      ],
      axis=3,
  )


def random_ciphertext(shape, modulus_list, dtype=jnp.int32):
  assert len(modulus_list) == shape[-1]
  random_key = jax.random.key(0)
  return jnp.concatenate(
      [
          jax.random.randint(
              random_key,
              shape=(shape[0],shape[1], 1),
              minval=0,
              maxval=bound,
              dtype=dtype,
          )
          for bound in modulus_list
      ],
      axis=2,
  )


def random_parameters(shape, modulus_list, dtype=jnp.int32):
  random_key = jax.random.key(0)
  min_modulus = 2**127
  for modulus in modulus_list:
    if modulus < min_modulus:
      min_modulus = modulus
  return jax.random.randint(random_key, shape=shape, minval=0, maxval=min_modulus-1, dtype=dtype)


####################################
# Parse Functions
####################################
def parse_ciphertext_string(input_str):
  """
  Parses the input string into two objects:
    - data: a list of element groups, each a list of evaluations (list of lists of numbers).
            Shape: (num_element, num_eval, num_numbers)
    - modulus: a one-dimensional list of modulus values corresponding to each evaluation index.
                All element groups are assumed to share the same modulus per evaluation.

  Parameters:
      input_str (str): The string containing the input data.

  Returns:
      tuple: (data, modulus) as described.
  """
  data = []
  global_modulus = []  # This will store the modulus once per evaluation index.


  # Process the input line by line.
  for line in input_str.strip().splitlines():
    line = line.strip()

    # Check for an "Element" header.
    if line.startswith("Element"):
      # Start a new element group.
      current_data_group = []
      data.append(current_data_group)

      # Check if there is extra content on the same line after the header.
      header_match = re.match(r'^Element\s+\d+:\s*(.*)', line)
      if header_match:
        remainder = header_match.group(1).strip()
        if remainder:
          # Process an evaluation if it appears on the same line.
          if eval_match := re.match(r'^(\d+):\s*EVAL:\s*\[(.*?)\]\s*modulus:\s*(\d+)', remainder):
            numbers_str = eval_match.group(2)
            mod_val = int(eval_match.group(3))
            numbers = [int(num) for num in numbers_str.split()]
            current_data_group.append(numbers)
            # For the first element group, record the modulus; otherwise, check consistency.
            eval_idx = len(current_data_group) - 1
            if len(data) == 1:
              global_modulus.append(mod_val)
            else:
              if eval_idx < len(global_modulus) and global_modulus[eval_idx] != mod_val:
                raise ValueError(f"Inconsistent modulus at evaluation index {eval_idx}")

    # Otherwise, check if the line is an evaluation line.
    elif eval_match := re.match(r'^(\d+):\s*EVAL:\s*\[(.*?)\]\s*modulus:\s*(\d+)', line):
      numbers_str = eval_match.group(2)
      mod_val = int(eval_match.group(3))
      numbers = [int(num) for num in numbers_str.split()]
      # Holds the current element's data evaluations.
      current_data_group = []
      current_data_group.append(numbers)
      eval_idx = len(current_data_group) - 1
      # For the first element group, record the modulus; for subsequent groups, check consistency.
      if len(data) == 1:
        global_modulus.append(mod_val)
      else:
        if eval_idx < len(global_modulus) and global_modulus[eval_idx] != mod_val:
          raise ValueError(f"Inconsistent modulus at evaluation index {eval_idx}")

  return data, global_modulus


####################################
# Bit Reverse Functions
####################################
def bit_reverse(x, bits):
  """Compute the bit-reversal of integer x with the given number of bits."""
  result = 0
  for i in range(bits):
    if (x >> i) & 1:  # if i-th bit of x is 1
      result |= 1 << (bits - 1 - i)  # set the corresponding reversed bit
  return result


def bit_reverse_array(in_tower):
  x = copy.deepcopy(in_tower)
  bits = len(x).bit_length() - 1
  for i in range(len(x)):
    j = bit_reverse(i, bits)
    if i < j:
      x[i], x[j] = x[j], x[i]
  return x


def bit_reverse_indices(n: int) -> jnp.ndarray:
    """
    Compute an array rev_idx of shape (n,) such that rev_idx[i] is the bit-reversal
    of i over log2(n) bits.
    """
    bits = int(math.log2(n))
    idx = jnp.arange(n)
    # build the reversed index by summing shifted bits
    rev = sum(
        ((idx >> i) & 1) << (bits - 1 - i)
        for i in range(bits)
    )
    return rev



####################################
# Automorphism Functions
####################################
def precompute_auto_map(n: int, k: int) -> List[int]:
    m = n << 1  # cyclOrder
    logm = int(round(math.log2(m)))
    logn = int(round(math.log2(n)))

    precomp: List[int] = [0] * n
    for j in range(n):
        j_tmp = (j << 1) + 1
        t = j_tmp * k
        # ((t % m) >> 1) but written to mirror the C++ bit ops exactly
        idx = (t - ((t >> logm) << logm)) >> 1

        j_rev = bit_reverse(j, logn)
        idx_rev = bit_reverse(idx, logn)
        precomp[j_rev] = idx_rev

    return precomp


def find_automorphism_index_2n_complex(i: int, m: int) -> int:
    """Python translation of nbtheory2.cpp FindAutomorphismIndex2nComplex (243-263).

    Mirrors the C++ logic including early exits, power-of-two validation, and
    modulus via bitmask for m being a power of two.
    """
    if i == 0:
        return 1
    if i == (m - 1):
        return int(i)

    if not is_power_of_two(m):
        raise ValueError("m should be a power of two.")

    # Conjugation automorphism generator
    g0 = pow(5, -1, m) if i < 0 else 5
    g = g0
    i_unsigned = abs(i)
    mask = m - 1
    for _ in range(1, i_unsigned):
        # Equivalent to (g * g0) % m since m is a power of two
        g = (g * g0) & mask
    return int(g)


####################################
# Number Theory Transformation
# Negacyclic NTT is used in CKKS
####################################
def ntt_bit_reverse(a, q, omega):
  """Compute cyclic Number Theoretic Transform of array a modulo q using a given primitive omega of unity."""
  n = len(a)
  # Ensure that omega^n ≡ 1 (mod q) and n divides q-1 for validity.
  # (This should be true if omega is a correct n-th omega of unity.)
  # Bit-reverse the input array indices
  bits = n.bit_length() - 1  # number of bits needed for indexes 0..n-1
  for i in range(n):
    j = bit_reverse(i, bits)
    if i < j:
      a[i], a[j] = a[j], a[i]  # swap to achieve bit-reversed order
  # Cooley-Tukey iterative FFT (NTT)
  length = 2
  while length <= n:
    # Compute twiddle factor step: use omega^(n/length) as the increment
    w_m = pow(omega, n // length, q)
    half = length // 2
    for i in range(0, n, length):  # loop over sub-FFT blocks
      w = 1
      for j in range(i, i + half):  # loop within each block
        u = a[j]
        v = a[j + half] * w % q  # multiply by current twiddle factor
        a[j] = (u + v) % q  # butterfly: combine top part
        a[j + half] = (u - v) % q  # butterfly: combine bottom part
        w = w * w_m % q  # advance twiddle factor for next element
    length *= 2
  return a


def intt_bit_reverse(a, q, omega):
  """Compute the Inverse Number Theoretic Transform of array a modulo p using the given primitive root."""
  n = len(a)
  inv_root = pow(omega, -1, q)  # modular inverse of root
  # Decimation-in-frequency (Gentleman-Sande) butterfly operations
  length = n
  while length >= 2:
    w_m = pow(inv_root, n // length, q)
    half = length // 2
    for i in range(0, n, length):
      w = 1
      for j in range(i, i + half):
        u = a[j]
        v = a[j + half]
        a[j] = (u + v) % q  # combine pairs (top value)
        a[j + half] = (
            ((u - v) % q) * w % q
        )  # combine pairs (bottom), then multiply by twiddle
        w = w * w_m % q  # advance twiddle factor
    length //= 2
  # Bit-reverse the result (to invert the initial bit-reversal
  # permutation in NTT)
  bits = n.bit_length() - 1
  for i in range(n):
    j = bit_reverse(i, bits)
    if i < j:
      a[i], a[j] = a[j], a[i]
  # Divide by n (multiply by n^{-1} mod p) to finish the inverse transform
  inv_n = pow(n, -1, q)
  for i in range(n):
    a[i] = a[i] * inv_n % q
  return a


def ntt_negacyclic_bit_reverse(a, q, psi):
  """Compute the negacyclic NTT of array a (length n) modulo q.

  Args:
    a: list (or 1D array) of integers (length n).
    q: prime modulus.
    psi: an element in GF(q) such that psi^(2*n) = 1 and psi^n = -1 mod q.
          (That is, psi is a primitive 2n-th root of unity; note that then ω =
          psi^2
          is a primitive n-th root of unity.)
    rows: Number of rows in the matrix.
    cols: Number of columns in the matrix.

  Returns:
    The negacyclic NTT of a.

  Process:
    1. Pre-twist: multiply each coefficient a[i] by psi^i.
    2. Compute the vanilla NTT (for example, using ntt_bit_reverse) with ω =
    psi^2.
  """
  n = len(a)
  # Check that psi^n = -1 mod q.
  if pow(psi, n, q) != q - 1:
    raise ValueError(
        "psi is not a valid 2n-th root of unity for negacyclic NTT (psi^n must"
        " equal -1 mod q)."
    )

  # Pre-twisting: multiply a[i] by psi^i.
  a_twisted = [(a[i] * pow(psi, i, q)) % q for i in range(n)]

  # Compute vanilla NTT using ω = psi².
  omega = pow(psi, 2, q)

  return ntt_bit_reverse(a_twisted.copy(), q, omega)


def intt_negacyclic_bit_reverse(a, q, psi):
  """Compute the inverse negacyclic NTT of array a (length n) modulo q.

  Args:
    a   : list (or 1D array) of integers (length n) in the negacyclic evaluation
    domain.
    q   : prime modulus.
    psi : an element in GF(q) such that psi^(2*n) = 1 and psi^n = -1 mod q.
          (That is, psi is a primitive 2n-th root of unity; note that then ω =
          psi^2
          is a primitive n-th root of unity.)
  Returns:
    The original input vector (i.e. the inverse transform).

  Process:
    1. Compute the inverse vanilla NTT using ω = psi².
    2. Post-twist: multiply the result by psi^(–i) for coefficient index i.
  """
  n = len(a)
  omega = pow(psi, 2, q)

  # Compute the inverse vanilla NTT.
  a_inv = intt_bit_reverse(a.copy(), q, omega)

  # Post-twisting: multiply a_inv[i] by psi^(–i).
  psi_inv = pow(psi, -1, q)
  return [(a_inv[i] * pow(psi_inv, i, q)) % q for i in range(n)]


####################################
# Precision Lowering Functions (outside Google)
####################################
def chunk_decomposition(x, chunkwidth=8):
  """Precision-level data conversion.

  Args:
      x: The input data.
      chunkwidth: The chunkwidth.

  Returns:
      The decomposed data.
  """
  dtype = jnp.uint8
  if chunkwidth == 16:
    dtype = jnp.uint16
  elif chunkwidth == 32:
    dtype = jnp.uint32

  elements = []
  mask = (1 << chunkwidth) - 1
  # Mask to extract the lower bits (e.g., 32 bits -> 0xFFFFFFFF)

  # Extract each element from the integer
  while x > 0:
    elements.append(x & mask)  # Extract the lower bits
    x >>= chunkwidth  # Shift to remove the extracted bits

  # Convert the list to a JAX array
  return jnp.array(elements, dtype=dtype)


####################################
# Performance Profiler Functions (outside Google)
####################################
def dump_hlo_from_lowered(lowered: Any, out_dir: str, out_filename: str) -> str:
  """
  Extract HLO (or StableHLO) textual IR from a lowered computation and write it to a file.
  Returns the full output file path.
  """
  # Prefer XLA HLO; fall back to StableHLO if necessary
  try:
    ir_obj = lowered.compiler_ir(dialect="hlo")
    # Handle XLA computation objects and MLIR modules
    if hasattr(ir_obj, "as_hlo_text"):
      hlo_text = ir_obj.as_hlo_text()
    elif hasattr(ir_obj, "operation"):
      try:
        hlo_text = ir_obj.operation.get_asm(enable_debug_info=True)
      except Exception:
        hlo_text = ir_obj.operation.get_asm()
    else:
      try:
        hlo_text = ir_obj.as_text()
      except Exception:
        hlo_text = str(ir_obj)
  except Exception:
    ir_obj = lowered.compiler_ir(dialect="stablehlo")
    if hasattr(ir_obj, "operation"):
      try:
        hlo_text = ir_obj.operation.get_asm(enable_debug_info=True)
      except Exception:
        hlo_text = ir_obj.operation.get_asm()
    else:
      try:
        hlo_text = ir_obj.as_text()
      except Exception:
        hlo_text = str(ir_obj)
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, out_filename)
  with open(out_path, "w") as f:
    f.write(hlo_text)
  return out_path


def dump_llo_from_lowered(lowered: Any, out_dir: str, out_filename: str) -> str:
  """
  Extract a lower-level XLA IR (LMHLO / LLVM if available) from a lowered computation
  and write it to a file. Returns the full output file path.
  """
  ir_text = None
  dialect_candidates = ["llvm", "lmhlo", "mhlo"]
  # Try both precompiled lowered and compiled executable views
  sources = [lowered]
  try:
    compiled = lowered.compile()
    sources.append(compiled)
  except Exception:
    pass
  for source in sources:
    for dialect in dialect_candidates:
      try:
        ir_obj = source.compiler_ir(dialect=dialect)
        if hasattr(ir_obj, "operation"):
          try:
            ir_text = ir_obj.operation.get_asm(enable_debug_info=True)
          except Exception:
            ir_text = ir_obj.operation.get_asm()
        else:
          # Some dialects (e.g., llvm) may expose textual interfaces differently
          if hasattr(ir_obj, "as_text"):
            ir_text = ir_obj.as_text()
          else:
            ir_text = str(ir_obj)
        if ir_text and len(ir_text) > 0:
          break
      except Exception:
        continue
    if ir_text:
      break
  if ir_text is None:
    # As a last resort, try to stringify generic compiler_ir without dialect hints
    try:
      generic = lowered.compiler_ir()
      if hasattr(generic, "operation"):
        ir_text = generic.operation.get_asm(enable_debug_info=True)
      elif hasattr(generic, "as_text"):
        ir_text = generic.as_text()
      else:
        ir_text = str(generic)
    except Exception:
      raise RuntimeError("Unable to extract LLO/LMHLO/LLVM IR from lowered/compiled computation.")
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, out_filename)
  with open(out_path, "w") as f:
    f.write(ir_text)
  return out_path


def profile_jax_functions_xprof(
    tasks: List[Tuple[Callable[..., Any], Tuple[Any, ...]]],
    profile_name: str = "jax_profile",
    kernel_name: str = "kernel_name",
):
  """Profiles a list of JAX functions.

  Args:
    tasks: A list of tuples, where each tuple contains a JAX function and its
      arguments.
    profile_name: The name of the profile, written in log/xprof/plugins/profile.
    kernel_name: The name of the kernel, used to find the latency of the kernel in the trace file.
  Usage:
    tasks = [
        (jit_pdul_barrett_xyzz_pack, (point_a_jax,)),
    ]
    profile_name = "jit_pdul_barrett_xyzz_pack"
    profile_jax_functions(tasks, profile_name, kernel_name="jit_pdul_barrett_xyzz_pack")
  """
  latency = 0
  n = 1 # number of times running the kernel
  final_folder_name = os.path.join(profile_root, profile_name)
  options = jax.profiler.ProfileOptions()
  options.python_tracer_level = 3
  options.host_tracer_level = 3 # https://docs.jax.dev/en/latest/profiling.html#general-options
  options.advanced_configuration = {"tpu_trace_mode" : "TRACE_COMPUTE_AND_SYNC", "tpu_num_chips_to_profile_per_task" : 4}

  repo_root = os.path.dirname(__file__)
  xprof_dir = os.path.join(repo_root, "log/xprof")
  with jax.profiler.trace(xprof_dir):
    # Launch all JAX computations
    results = []
    for func, args_tuple in tasks:
      result = func(*args_tuple)
      results.append(result)

    # Wait for all computations launched in the loop to complete
    if results:
      jax.block_until_ready(results)

  # Rename the newly created timestamped directory to the designated profile_name.
  try:
    if os.path.isdir(profile_root):
      post_dirs = set(os.listdir(profile_root))
      created_dirs = [d for d in (post_dirs - pre_existing_dirs) if os.path.isdir(os.path.join(profile_root, d))]

      target_dir = None
      if created_dirs:
        # Choose the most recently modified among the newly created ones.
        target_dir = max(created_dirs, key=lambda d: os.path.getmtime(os.path.join(profile_root, d)))
      else:
        # Fallback: pick the most recent dir in case set diff failed (e.g., pre list failed).
        all_dirs = [d for d in post_dirs if os.path.isdir(os.path.join(profile_root, d))]
        if all_dirs:
          target_dir = max(all_dirs, key=lambda d: os.path.getmtime(os.path.join(profile_root, d)))

      if target_dir:

        # Avoid overwriting existing destination; add numeric suffix if necessary.
        if os.path.exists(final_folder_name):
          suffix = 1
          while os.path.exists(f"{final_folder_name}_{suffix}"):
            suffix += 1
          final_folder_name = f"{final_folder_name}_{suffix}"
        os.rename(os.path.join(profile_root, target_dir), final_folder_name)
  except Exception as e:
      print(f"Profile rename failed: {e}")

  # Read the trace file and print the latency of the kernel
  # Find the file that ends with 'trace.json.gz' in the destination directory
  trace_file = None
  if os.path.exists(final_folder_name):
    for fname in os.listdir(final_folder_name):
      if fname.endswith("trace.json.gz"):
        trace_file = os.path.join(final_folder_name, fname)
        break
    if trace_file:
      with gzip.open(trace_file, 'rt') as f:
        jtrace = json.loads(f.read())
        if jtrace:
          if "NVIDIA" in  jax.devices()[0].device_kind:
            for e in jtrace["traceEvents"]:
              if 'args' in e and 'tf_op' in e['args']:
                if kernel_name in e['args']["hlo_module"]:
                  latency += e['dur']
          elif "TPU" in  jax.devices()[0].device_kind:
            pid = 999999 # an invalid PID
            for e in jtrace["traceEvents"]:
              if 'args' in e and 'name' in e['args'] and 'TPU:0' in e['args']['name']:
                pid = e['pid']
              if 'args' in e and 'tf_op' in e['args'] and kernel_name in e['args']['tf_op']:
                if e['pid'] == pid:
                  latency += e['dur']
              if 'args' in e and 'hlo_category' in e['args'] and 'copy' in e['args']['hlo_category']:
                if e['pid'] == pid:
                  latency += e['dur']
    else:
      print(f"Trace file not found: {trace_file}")
  else:
    print(f"Final folder name not found: {final_folder_name}")
  return latency


# paper full case evaluation.
original_moduli_51_limbs = [1073753729, 1073738977, 1073753281, 1073739041, 1073753089, 1073747137, 1073752417, 1073739169, 1073745697, 1073739361, 1073752129, 1073746337, 1073748737, 1073746529, 1073748289, 1073747393, 1073749889, 1073748449, 1073751713, 1073749153, 1073750593, 1073749409, 1073751521, 1073750017, 1073751169, 1073750497, 1073751073, 1073750113, 1073750849, 1073739617, 1073746273, 1073745473, 1073745889, 1073742881, 1073745377, 1073739649, 1073745121, 1073741953, 1073744993, 1073739937, 1073744417, 1073742913, 1073744257, 1073742113, 1073743457, 1073742209, 1073743393, 1073740609, 1073742721, 1073741441, 1073741857, 524353]
original_psi_51_limbs = [1093151, 90892563, 108899655, 56634236, 235160291, 12265314, 191995239, 21404433, 40083131, 3916344, 113671079, 34500367, 61894143, 20463380, 13205216, 60050555, 145308815, 87067229, 10533116, 133048918, 13697511, 47895671, 14807533, 10994638, 25005605, 44429319, 77617905, 22756112, 21182116, 46947055, 41148497, 163086225, 60397627, 176334344, 30766686, 77429283, 67466901, 67653750, 4536048, 135444559, 63788661, 110966687, 9716122, 12174708, 49591386, 81862273, 51874541, 12155428, 60746932, 68809976, 28870916, 19017]
extend_moduli_51_limbs = [1152921504606845473, 1152921504606844513, 1152921504606844417, 1152921504606844289, 1152921504606843233, 1152921504606843073, 1152921504606842753, 1152921504606841793, 1152921504606841441, 1152921504606840929]


NTT_PARAMETERS_BY_DEGREE = {
  16: {
    "moduli": [1073759809, 1073759041, 1073759777, 1073758337, 1073759329, 1073758849, 1073759233, 1073738273, 1073754113, 1073738753, 1073753729, 1073738977, 1073753281, 1073739041, 1073753089, 1073747137, 1073752417, 1073739169, 1073745697, 1073739361, 1073752129, 1073746337, 1073748737, 1073746529, 1073748289, 1073747393, 1073749889, 1073748449, 1073751713, 1073749153, 1073750593, 1073749409, 1073751521, 1073750017, 1073751169, 1073750497, 1073751073, 1073750113, 1073750849, 1073739617, 1073746273, 1073745473, 1073745889, 1073742881, 1073745377, 1073739649, 1073745121, 1073741953, 1073744993, 1073739937, 1073744417, 1073742913, 1073744257, 1073742113, 1073743457, 1073742209, 1073743393, 1073740609, 1073742721, 1073741441, 1073741857, 524353],
    "root_of_unity": [149761193, 17168328, 145519847, 68042513, 3491826, 21109149, 48183983, 49547540, 15369996, 12935385, 1093151, 90892563, 108899655, 56634236, 235160291, 12265314, 191995239, 21404433, 40083131, 3916344, 113671079, 34500367, 61894143, 20463380, 13205216, 60050555, 145308815, 87067229, 10533116, 133048918, 13697511, 47895671, 14807533, 10994638, 25005605, 44429319, 77617905, 22756112, 21182116, 46947055, 41148497, 163086225, 60397627, 176334344, 30766686, 77429283, 67466901, 67653750, 4536048, 135444559, 63788661, 110966687, 9716122, 12174708, 49591386, 81862273, 51874541, 12155428, 60746932, 68809976, 28870916, 19017],
  },
  4096: {
    "moduli": [268730369, 268689409, 268361729, 268582913, 268369921, 268460033, 557057, 1152921504606830593, 1152921504606748673],
    "root_of_unity": [8801, 19068, 58939, 11033, 62736, 77090, 474, 116777451583545, 271802498405390],
  },
  8192: {
    "moduli": [269402113, 268091393, 268730369, 268271617, 269221889, 268664833, 268861441, 268369921, 268582913, 557057, 1152921504606830593, 1152921504606748673],
    "root_of_unity": [18987, 2826, 1678, 18925, 2446, 31335, 40892, 65274, 15787, 268, 25959043411404, 100406242475323],
  },
  16384: {
    "moduli": [274726913, 272760833, 274628609, 267059201, 270499841, 267550721, 270237697, 267943937, 268861441, 268042241, 268730369, 268238849, 269844481, 268271617, 269221889, 268369921, 268664833, 557057, 1152921504606748673, 1152921504606683137, 1152921504606584833],
    "root_of_unity": [9358, 15613, 1976, 5381, 15236, 9622, 5177, 2469, 792, 63914, 9742, 12308, 3704, 7216, 7564, 10360, 2023, 19, 62213374832584, 212089012217363, 92166579128688],
  },
  65536: {
    "moduli": [384040961, 376569857, 371458049, 375521281, 371589121, 383778817, 377880577, 379453441, 323092481, 351797249, 349962241, 351404033, 260702209, 308150273, 304742401, 307888129, 302776321, 306708481, 304218113, 347996161, 319291393, 347078657, 323223553, 337248257, 323878913, 336855041, 329515009, 332660737, 329777153, 335413249, 325844993, 330301441, 327548929, 332267521, 328728577, 344850433, 336068609, 340000769, 261488641, 302252033, 297664513, 299499521, 261881857, 295305217, 263323649, 277086209, 263454721, 292159489, 279838721, 291373057, 284950529, 290455553, 281935873, 285474817, 283508737, 288882689, 264634369, 276430849, 270532609, 274726913, 272760833, 276037633, 265420801, 270794753, 268042241, 269221889, 786433],
    "root_of_unity": [1197, 4622, 9335, 5748, 719, 1497, 2281, 3163, 3548, 80, 6577, 4942, 435, 3498, 316, 4503, 1433, 5766, 440, 2739, 1792, 13, 545, 7539, 7418, 7033, 32540, 1301, 4354, 16962, 10301, 289, 4195, 3322, 1005, 1747, 13384, 7659, 2200, 1035, 2142, 6961, 2774, 910, 43, 1949, 4343, 6648, 787, 2879, 4743, 563, 3385, 5648, 5875, 9494, 2122, 852, 6279, 1335, 712, 2017, 929, 142, 5274, 3264, 8],
  },
}

moduli_28_list = {
  degree: params["moduli"]
  for degree, params in NTT_PARAMETERS_BY_DEGREE.items()
}

roof_of_unity = {
  degree: params["root_of_unity"]
  for degree, params in NTT_PARAMETERS_BY_DEGREE.items()
}
