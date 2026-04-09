"""Mathematical utilities for CKKS NTT."""

import math
import numpy as np


def prime_factors(n: int) -> set[int]:
  """Returns the set of prime factors of n."""
  factors = set()
  while n % 2 == 0:
    factors.add(2)
    n //= 2
  p = 3
  while p * p <= n:
    while n % p == 0:
      factors.add(p)
      n //= p
    p += 2
  if n > 1:
    factors.add(n)
  return factors


def find_generator(q: int) -> int:
  """Finds a primitive root modulo q."""
  if q == 2:
    return 1
  phi = q - 1
  factors = prime_factors(phi)
  for g in range(2, q):
    if all(pow(g, phi // p, q) != 1 for p in factors):
      return int(g)
  raise ValueError(f"No generator found for q={q}")


def root_of_unity(m: int, q: int) -> int:
  """Returns the canonical primitive m-th root of unity modulo q."""
  if (q - 1) % m != 0:
    raise ValueError(f"m={m} must divide q-1={q-1}")
  g = find_generator(q)
  r = pow(g, (q - 1) // m, q)
  candidates = []

  # For m power of 2, we can optimize the loop
  if (m & (m - 1)) == 0:
    curr = r
    r2 = (r * r) % q
    for _ in range(1, m, 2):
      candidates.append(int(curr))
      curr = (curr * r2) % q
  else:
    for k in range(1, m):
      if math.gcd(k, m) != 1:
        continue
      psi = pow(r, k, q)
      candidates.append(psi)

  if not candidates:
    raise ValueError(f"No primitive {m}-th root found for q={q}")
  return int(min(candidates))


def gen_twiddle_matrix(rows: int, cols: int, q: int, omega: int) -> np.ndarray:
  """Precomputes the twiddle matrix T where T[r, c] = omega^(r*c) mod q."""
  q_int = int(q)
  omega_int = int(omega)

  # Optimization: precompute powers of omega in O(rows * cols)
  # instead of O(rows * cols * log(exponent))
  max_exponent = (rows - 1) * (cols - 1)
  powers = np.empty(max_exponent + 1, dtype=np.uint64)
  curr = 1
  for i in range(max_exponent + 1):
    powers[i] = curr
    curr = (curr * omega_int) % q_int

  # Vectorized creation of the matrix
  r_idx = np.arange(rows, dtype=np.uint64)
  c_idx = np.arange(cols, dtype=np.uint64)
  exponents = np.outer(r_idx, c_idx)

  return powers[exponents]


def gen_twiddle_matrix_inv(
    rows: int, cols: int, q: int, omega: int
) -> np.ndarray:
  """Precomputes the inverse twiddle matrix T_inv where T_inv[r, c] = omega^(-r*c) mod q."""
  inv_omega = pow(int(omega), -1, int(q))
  return gen_twiddle_matrix(rows, cols, q, inv_omega)


def get_bit_reverse_perm(n: int) -> list[int]:
  """Generates a list of indices for bit-reversal permutation of size n."""
  if n <= 0:
    return []
  bits = n.bit_length() - 1
  perm = [0] * n
  for i in range(n):
    r = 0
    temp = i
    for _ in range(bits):
      r = (r << 1) | (temp & 1)
      temp >>= 1
    perm[i] = r
  return perm
