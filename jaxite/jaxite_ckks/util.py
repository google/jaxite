"""Utils for jaxite_ckks.
"""



def is_prime_deterministic(n):
  """Deterministic primality test for n < 2^64.

  Uses Trial Division for speed + Deterministic Miller-Rabin for correctness.

  Args:
    n: The number to test for primality.

  Returns:
    True if n is prime, False otherwise.
  """
  if n < 2:
    return False
  if n == 2 or n == 3:
    return True
  if n % 2 == 0:
    return False

  # 1. SPEED OPTIMIZATION: Trial Division
  # Check divisibility by small primes to fail fast on obvious composites.
  # This filters out ~85% of candidates without expensive modular
  # exponentiation.
  small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
  for p in small_primes:
    if n == p:
      return True
    if n % p == 0:
      return False

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
    if a >= n:
      break

    x = pow(a, d, n)
    if x == 1 or x == n - 1:
      continue

    for _ in range(s - 1):
      x = pow(x, 2, n)
      if x == n - 1:
        break
    else:
      return False  # Composite

  return True  # Prime


def find_moduli_ntt(total_number, precision, ntt_length):
  """Deterministically finds the largest valid NTT moduli.

  Args:
    total_number: Number of moduli to find.
    precision: Bit-width (e.g., 60 for < 2^60).
    ntt_length: The required N-th root of unity (e.g., 1024).

  Returns:
    A list of the largest valid NTT moduli found, up to `total_number`.
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


def to_tuple(a):
  """Create to convert numpy array into tuple."""
  try:
    return tuple(to_tuple(i) for i in a)
  except TypeError:
    return a


def modinv(x: int, q: int) -> int:
  """Returns the inverse of x mod q."""
  return int(pow(x, -1, q))
