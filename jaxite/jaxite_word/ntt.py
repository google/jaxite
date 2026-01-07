"""JAX implementation of Gentalman Sande NTT."""

import concurrent.futures
import functools

import jax
import jax.numpy as jnp
import numpy as np
from typing import List


########################
# Offline Functions
########################
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


def rechunkify(arr_a, chunkwidth):
  """Rechunkify the input array back to the desired precision.

  This function accumulates partial sums that might exceed the `chunkwidth`
  precision and re-distributes any overflow (carry bits) across chunks
  until all chunks are within the defined `chunkwidth`.

  The process handles potential carries by repeatedly splitting chunks into
  lower and upper halves, shifting, and accumulating. The illustration below
  shows how overlapping partial sums are handled to prevent overflow:

  Data Type Illustration:
             LSB             MSB
            |-----------------> bit
            |   a0
            |  ==--
            |     a1
            |    ==--
            |        a2
            |       ==--
            |          a3
            v         ==--

    whole   a0   a1   a2   a3
    precision ==-- ==-- ==-- ==--

    lower   a0 a1 a2 a3
    half    == == == ==

    upper   a0 a1 a2 a3
    half    -- -- -- --

  Chunk Splitting and Vectorized Accumulation (simplified):
    1.  Split each array element into a lower half (within `chunkwidth`)
        and an upper half (carry bits).
    2.  Pad and align these halves to prepare for accumulation.
    3.  Vectorizedly accumulate the padded halves.
    4.  Repeatedly process any chunks that still exceed `bitmask` by
        splitting them further and redistributing their carry bits until
        all chunks are within `chunkwidth`.
    5.  A final adjustment is made to the top chunk to avoid overflow.

  Args:
      arr_a: The input array of chunked data.
      chunkwidth: The desired precision (e.g., 8, 16, 32 bits) for each chunk.

  Returns:
      The rechunkified array, where each element is within `chunkwidth` and
      carries have been propagated.
  """
  dtype_double_length = jnp.uint16
  if chunkwidth == 16:
    dtype_double_length = jnp.uint32
  elif chunkwidth == 32:
    dtype_double_length = jnp.uint64

  # assume the precision of partial sum is <= 2 * precision of input value.
  bitmask = (1 << chunkwidth) - 1

  # Chunk Splitting -> upper and lower half
  arr_a_lower_half = jnp.bitwise_and(arr_a, bitmask)
  arr_a_upper_half = jnp.right_shift(arr_a, chunkwidth)

  # Padding to align
  arr_a_lower_half_pad = jnp.pad(arr_a_lower_half, (0, 1))
  arr_a_upper_half_pad = jnp.pad(arr_a_upper_half, (1, 0))

  # Vectorized Accumulation
  arr_b = jnp.add(
      arr_a_lower_half_pad.astype(dtype_double_length),
      arr_a_upper_half_pad.astype(dtype_double_length),
  )

  while not jnp.all(arr_b <= bitmask):
    arr_b_lower_half = jnp.bitwise_and(arr_b, bitmask)
    arr_b_carry = jnp.right_shift(arr_b, chunkwidth)
    arr_b = jnp.roll(arr_b_carry, 1, axis=-1)
    arr_b = jnp.add(arr_b_lower_half, arr_b)

  # Vectorized Accumulation
  arr_c = arr_b

  # break top chunk into upper and lower to avoid overflow.
  arr_c = jnp.pad(arr_c, (0, 1))
  arr_c = arr_c.at[-1].set(jnp.right_shift(arr_c[-2], chunkwidth))
  arr_c = arr_c.at[-2].set(jnp.bitwise_and(arr_c[-2], bitmask))

  return arr_c


def smul_as_dense_gemv_bat(
    x, total_in_precision=32, chunkwidth=8, q=4294967291
):
  """This is the implementation of Basis Align Transformation (BAT).

  Major improvement to achieve dense matrix.

  Args:
    x: The input matrix.
    total_in_precision: The total precision of the input matrix.
    chunkwidth: The chunkwidth.
    q: The modulus.

  Returns:
    The dense matrix.

  Steps:
  1. break x into [x0, x1, x2, x3]
  2. reform [x0, x1, x2, x3] into the output
  [
  x0    r00    r00    r00    # 2^0
  x1   x0+r01  r01    r01    # 2^8
  x2   x1+r02 x0+r02  r02    # 2^16
  x3   x2+r03 x1+r03 x0+r03  # 2^24
  ]

  Note: prefilled value are just examples.
    We pick largest 2^32-1 to make sure that intermediate results might
    exceed 32-bit precision range, and expose potential precision overflow.
  """
  dtype_double_length = jnp.uint16
  chunk_upper_bound = (1 << 8) - 1
  if chunkwidth == 16:
    dtype_double_length = jnp.uint32
    chunk_upper_bound = (1 << 16) - 1
  elif chunkwidth == 32:
    dtype_double_length = jnp.uint64
    chunk_upper_bound = (1 << 32) - 1

  total_chunk_num = int(jnp.ceil(total_in_precision / chunkwidth))

  # the number of row in left matrix
  height = total_chunk_num + total_chunk_num - 1
  x_dtype = chunk_decomposition(x, chunkwidth)
  x_dense = jnp.zeros(
      (total_chunk_num + total_chunk_num - 1, total_chunk_num),
      dtype=dtype_double_length,
  )
  for j in range(total_chunk_num):
    upper_idx = min(total_chunk_num, x_dtype.shape[0] + j)
    x_dense = x_dense.at[j:upper_idx, j].set(x_dtype[: upper_idx - j])

  # [
  # x0              # 2^0
  # x1 x0           # 2^8
  # x2 x1 x0        # 2^16
  # x3 x2 x1 x0     # 2^24
  # -----------
  #    x3 x2 x1     # 2^32  iterate all elements in the bottom block
  #       x3 x2     # 2^40
  #          x3     # 2^48
  # ]

  # Perform BAT to the following block of the matrix
  # j    2  1  0
  #     x3 x2 x1   # 2^32  i=0
  #        x3 x2   # 2^40  i=1
  #           x3   # 2^48  i=2

  for i in range(x_dtype.shape[0] - 1):
    for j in range(x_dtype.shape[0] - 1 - i):
      basis = (total_chunk_num + i) * chunkwidth
      projected_data = (int(x_dtype[i + j + 1]) << basis) % q
      r = chunk_decomposition(projected_data, chunkwidth).astype(
          dtype_double_length
      )

      x_dense = x_dense.at[: len(r), total_chunk_num - 1 - j].set(
          jnp.add(r, x_dense[: len(r), total_chunk_num - 1 - j])
      )

  while not jnp.all(x_dense <= chunk_upper_bound) or not jnp.all(
      x_dense[total_chunk_num:, :] == 0
  ):
    for j in range(total_chunk_num - 1):
      # Iterate over different columns
      if not jnp.all(x_dense[:, total_chunk_num - 1 - j] <= chunk_upper_bound):
        arr_new_chunkified = rechunkify(
            x_dense[:, total_chunk_num - 1 - j], chunkwidth
        )
        x_dense = x_dense.at[:, total_chunk_num - 1 - j].set(
            arr_new_chunkified[:height]
        )

    # j    2  1  0
    #     x3 x2 x1   # 2^32  i=0
    #        x3 x2   # 2^40  i=1
    #           x3   # 2^48  i=2
    for i in range(x_dtype.shape[0] - 1):
      for j in range(x_dtype.shape[0] - 1 - i):
        data = x_dense[total_chunk_num + i, total_chunk_num - 1 - j]
        if data > 0:
          basis = (total_chunk_num + i) * chunkwidth
          projected_data = (int(data) << basis) % q
          r = chunk_decomposition(projected_data, chunkwidth).astype(
              dtype_double_length
          )

          x_dense = x_dense.at[: len(r), total_chunk_num - 1 - j].set(
              jnp.add(r, x_dense[: len(r), total_chunk_num - 1 - j])
          )
          x_dense = x_dense.at[
              total_chunk_num + i, total_chunk_num - 1 - j
          ].set(0)
  return x_dense[:total_chunk_num, :].astype(jnp.uint8)


def smul_as_dense_gemv_bat_jax(x, q=4294967291):
  """This is the implementation of bat; Major improvement to achieve dense matrix.

  Args:
    x: The input matrix.
    q: The modulus.

  Returns:
    The dense matrix.

  Steps:
  1. break x into [x0, x1, x2, x3]
  2. reform [x0, x1, x2, x3] into the output
  [
  x0    r00    r00    r00    # 2^0
  x1   x0+r01  r01    r01    # 2^8
  x2   x1+r02 x0+r02  r02    # 2^16
  x3   x2+r03 x1+r03 x0+r03  # 2^24
  ]
  """
  assert x.dtype == jnp.uint32
  chunkwidth = 8
  chunk_upper_bound = (1 << 8) - 1
  total_chunk_num = 4

  # the number of row in left matrix
  height = 7
  x_dtype = jax.lax.bitcast_convert_type(x, new_dtype=jnp.uint8)
  x_dense = jnp.array(
      [
          [x_dtype[0], 0, 0, 0],
          [x_dtype[1], x_dtype[0], 0, 0],
          [x_dtype[2], x_dtype[1], x_dtype[0], 0],
          [x_dtype[3], x_dtype[2], x_dtype[1], x_dtype[0]],
          [0, x_dtype[3], x_dtype[2], x_dtype[1]],
          [0, 0, x_dtype[3], x_dtype[2]],
          [0, 0, 0, x_dtype[3]],
      ],
      dtype=jnp.uint16,
  )

  # [
  # x0              # 2^0
  # x1 x0           # 2^8
  # x2 x1 x0        # 2^16
  # x3 x2 x1 x0     # 2^24
  # -----------
  #    x3 x2 x1     # 2^32  iterate all elements in the bottom block
  #       x3 x2     # 2^40
  #          x3     # 2^48
  # ]

  # Perform BAT to the following block of the matrix
  # j    2  1  0
  #     x3 x2 x1   # 2^32  i=0
  #        x3 x2   # 2^40  i=1
  #           x3   # 2^48  i=2

  for i in range(x_dtype.shape[0] - 1):
    for j in range(x_dtype.shape[0] - 1 - i):
      basis = (total_chunk_num + i) * chunkwidth
      projected_data = ((x_dtype[i + j + 1].astype(jnp.uint64)) << basis) % q
      r = jax.lax.bitcast_convert_type(
          projected_data, new_dtype=jnp.uint8
      ).astype(jnp.uint16)

      x_dense = x_dense.at[:, total_chunk_num - 1 - j].set(
          jnp.add(r[:height], x_dense[:, total_chunk_num - 1 - j])
      )

  while not jnp.all(x_dense <= chunk_upper_bound) or not jnp.all(
      x_dense[total_chunk_num:, :] == 0
  ):
    # for _ in range(2): # rechunkify won't exceed 3 times.
    for j in range(total_chunk_num - 1):
      # Iterate over different columns
      if not jnp.all(x_dense[:, total_chunk_num - 1 - j] <= chunk_upper_bound):
        arr_new_chunkified = rechunkify(
            x_dense[:, total_chunk_num - 1 - j], chunkwidth
        )
        x_dense = x_dense.at[:, total_chunk_num - 1 - j].set(
            arr_new_chunkified[:height]
        )

    # j    2  1  0
    #     x3 x2 x1   # 2^32  i=0
    #        x3 x2   # 2^40  i=1
    #           x3   # 2^48  i=2
    for i in range(x_dtype.shape[0] - 1):
      for j in range(x_dtype.shape[0] - 1 - i):
        data = x_dense[total_chunk_num + i, total_chunk_num - 1 - j]
        if data > 0:
          basis = (total_chunk_num + i) * chunkwidth
          projected_data = (data.astype(jnp.uint64) << basis) % q
          r = jax.lax.bitcast_convert_type(
              projected_data, new_dtype=jnp.uint8
          ).astype(jnp.uint16)
          x_dense = x_dense.at[:, total_chunk_num - 1 - j].set(
              jnp.add(r[:height], x_dense[:, total_chunk_num - 1 - j])
          )
          x_dense = x_dense.at[
              total_chunk_num + i, total_chunk_num - 1 - j
          ].set(0)
  return x_dense[:total_chunk_num, :].astype(jnp.uint8)


def hpmatmul_offline_compile_bat(mat_a, q):
  """Convert the input (m,n) matrix into (m,n,p,q), i.e.

  replace each element in the original matrix by a p*q matrix (p==q).

  Args:
    mat_a: The input matrix.
    q: The modulus.

  Returns:
    The converted matrix.
  """
  assert mat_a.dtype == jnp.uint32  # This version is defined for 32-bit input.
  m, n = mat_a.shape[0], mat_a.shape[1]
  total_in_precision = 32
  chunkwidth = 8
  # Convert left-side matrix
  total_chunk_num = int(jnp.ceil(total_in_precision / chunkwidth))

  left_mat = jnp.zeros(
      (m, n, total_chunk_num, total_chunk_num), dtype=jnp.uint16
  )

  with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for i in range(m):
      futures.extend(
          executor.submit(
              smul_as_dense_gemv_bat,
              mat_a[i, k],
              total_in_precision,
              chunkwidth,
              q,
          )
          for k in range(n)
      )
    index_pairs = []
    for i in range(m):
      for k in range(n):
        index_pairs.append((i, k))
    for future, (i, k) in zip(futures, index_pairs):
      left_mat = left_mat.at[i, k, :, :].set(future.result())

  return left_mat


def find_all_root_of_unity(n: int, q: int) -> List[int]:
  """Find the n-th root of unity in the field of integers modulo q."""
  all_root_of_unity = []
  for v in range(2, q):
    if (v**n) % q == 1:
      find_root = all((v**i) % q != 1 for i in range(1, n))
      if find_root:
        all_root_of_unity.append(v)
        print(v)
  return all_root_of_unity


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


def nth_primitive_root(n, q):
  """Returns a primitive n-th root of unity in GF(q).

  Args:
    n (int): The desired order of the root of unity.
    q (int): The prime modulus of the finite field GF(q).

  Returns:
    int: An element omega in GF(q) such that omega^n = 1 and its order is
    exactly n.

  Precondition: n divides (q-1).
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


def bit_reverse(x, bits):
  """Compute the bit-reversal of integer x with the given number of bits."""
  result = 0
  for i in range(bits):
    if (x >> i) & 1:  # if i-th bit of x is 1
      result |= 1 << (bits - 1 - i)  # set the corresponding reversed bit
  return result


def tonelli_shanks(q, omega):
  """Solve for x in x^2 ≡ omega (mod q) using the Tonelli-Shanks algorithm.

  Args:
    q: The prime modulus.
    omega: The element to find the square root of.

  Returns:
    One square root of omega modulo q, or None if no square root exists.

  Raises:
    ValueError: If the algorithm fails to find the exponent i.
  """
  # Check if a is a quadratic residue mod p
  if pow(omega, (q - 1) // 2, q) != 1:
    return None  # no solution exists

  # Special case: p ≡ 3 (mod 4)
  if q % 4 == 3:
    return pow(omega, (q + 1) // 4, q)

  # Write p - 1 as q_minus_1 * 2^s with q_minus_1 odd.
  s = 0
  q_minus_1 = q - 1
  while q_minus_1 % 2 == 0:
    s += 1
    q_minus_1 //= 2

  # Find a quadratic non-residue z
  z = 2
  while pow(z, (q - 1) // 2, q) != q - 1:
    z += 1

  m = s
  c = pow(z, q_minus_1, q)
  t = pow(omega, q_minus_1, q)
  r = pow(omega, (q_minus_1 + 1) // 2, q)

  while t != 1:
    # Find the least i, 0 < i < m, such that t^(2^i) ≡ 1 (mod p)
    i = 1
    temp = pow(t, 2, q)
    while temp != 1:
      temp = pow(temp, 2, q)
      i += 1
      if i == m:
        raise ValueError("Algorithm failed to find the exponent i")
    # Update values
    b = pow(c, 2 ** (m - i - 1), q)
    m = i
    c = pow(b, 2, q)
    t = (t * c) % q
    r = (r * b) % q
  return r


def compute_psi(omega, n, q):
  """Compute psi such that psi^2 = omega and psi^n = -1 mod q.

  Given an n-th primitive root of unity omega in GF(q), compute psi.

  Args:
    omega: The n-th primitive root of unity.
    n: The order of the root of unity.
    q: The prime modulus.

  Returns:
    An element psi in GF(q) such that psi^2 = omega and psi^n = -1 mod q.
  """
  psi = tonelli_shanks(q, omega)
  if psi is None:
    raise ValueError("No square root exists for omega modulo q.")

  # Check the negacyclic condition: psi^n should equal -1 modulo q.
  if pow(psi, n, q) != q - 1:
    # Try the other square root (q - psi)
    psi = q - psi
  if pow(psi, n, q) != q - 1:
    raise ValueError("Neither square root of omega satisfies psi^n = -1 mod q.")
  return psi


def gen_twiddle_matrix(rows, cols, q, omega):
  """Precompute the twiddle matrix T of shape (rows, cols), where T[r, c] = omega^(r*c) mod q.

  Args:
    rows: The number of rows in the matrix.
    cols: The number of columns in the matrix.
    q: The modulus.
    omega: The primitive root of unity.

  Returns:
    The twiddle matrix.
  """
  twiddle_matrix = np.zeros((rows, cols), dtype=int)
  for r in range(rows):
    for c in range(cols):
      twiddle_matrix[r, c] = pow(omega, r * c, q)
  return twiddle_matrix


def gen_twiddle_matrix_inv(rows, cols, q, omega):
  """Precompute the inverse twiddle matrix T_inv of shape (rows, cols).

  T_inv[r, c] = omega^{- (r*c)} mod q.

  Args:
    rows: The number of rows in the matrix.
    cols: The number of columns in the matrix.
    q: The modulus.
    omega: The primitive root of unity.

  Returns:
    The inverse twiddle matrix.
  """
  twiddle_matrix_inv = np.zeros((rows, cols), dtype=int)
  for r in range(rows):
    for c in range(cols):
      twiddle_matrix_inv[r, c] = pow(omega, -r * c, q)
  return twiddle_matrix_inv


def ntt_original_form(v, q, omega):
  length = len(v)
  coef_mat = gen_twiddle_matrix(length, length, q, omega)
  result = [0] * length
  for k in range(length):
    acc = 0
    for j in range(length):
      acc = (acc + v[j] * coef_mat[j, k]) % q
    result[k] = acc
  return result


def intt_original_form(v, q, omega):
  """Compute the Inverse NTT (naive O(length^2) algorithm) of vector v of length length over GF(q).

  omega_inv is a primitive L-th root of unity for the inverse transform, i.e.

    if the forward NTT uses omega, then we use omega_inv = omega^{-1} mod q.
  The result is normalized by multiplying by the modular inverse of L.

  Args:
    v: The input vector.
    q: The prime modulus.
    omega: The primitive L-th root of unity.

  Returns:
    The inverse NTT of v.
  """

  length = len(v)
  omega_inv = pow(omega, -1, q)  # modular inverse of root
  coef_mat = gen_twiddle_matrix(length, length, q, omega_inv)
  result = [0] * length
  # Compute the modular inverse of L modulo q
  length_inv = pow(length, -1, q)
  for j in range(length):
    acc = 0
    for k in range(length):
      # Using omega_inv^(j*k)
      acc = (acc + v[k] * coef_mat[j, k]) % q
    result[j] = (acc * length_inv) % q

  return result


def ntt_bit_reverse(a, q, omega):
  """Compute the Number Theoretic Transform of array a modulo p using a given primitive omega of unity."""
  n = len(a)
  # Ensure that omega^n ≡ 1 (mod p) and n divides p-1 for validity.
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


def ntt_four_step(x, q, omega, rows, cols):
  """Compute the 4-step NTT of the input vector x (length N = rows * cols) over GF(q).

  Args:
    x: list or 1D numpy array (length N).
    q: prime modulus.
    omega: the primitive N-th root of unity.
    rows: factors of N, so that N = rows * cols.
    cols: factors of N, so that N = rows * cols.

  Returns:
    A list representing the NTT result.

  Process:
    1. Columns:  NTT on each column (length rows) using omega_col = omega^cols.
    2. Twiddle:  Multiply by T[r,c] = omega^(r*c).
    3. Rows:     NTT on each row (length cols) using omega_row = omega^rows.
    4. Reordering: Final output is flatten(transpose(Z)).
  """
  num_elements = rows * cols
  if len(x) != num_elements:
    raise ValueError("Length of x must equal rows * cols")
  omega_row = pow(omega, rows, q)
  omega_col = pow(omega, cols, q)
  matrix_a = np.array(x, dtype=int).reshape((rows, cols))
  y = np.zeros((rows, cols), dtype=int)
  for c in range(cols):
    col = matrix_a[:, c].tolist()
    y[:, c] = ntt_original_form(col, q, omega_col)
  print(f"after Step 1={y}")

  twiddle = gen_twiddle_matrix(rows, cols, q, omega)
  y = (y * twiddle) % q
  print(f"after Step 2={y}")

  matrix_z = np.zeros((rows, cols), dtype=int)
  for r in range(rows):
    row = y[r, :].tolist()
    matrix_z[r, :] = ntt_original_form(row, q, omega_row)
  print(f"after Step 3={matrix_z}")
  matrix_x = np.array(
      matrix_z.T
  ).flatten()  # forward transform reorders via transpose flattening
  print(f"after Step 3 after transpose={matrix_x}")
  return matrix_x.tolist()


def intt_four_step(x, q, omega, rows, cols):
  """Compute the 4-step Inverse NTT of the input vector X (length N = rows * cols) over GF(q).

  Forward transform recap:
    - Columns:  NTT on each column (length rows) using omega_col = omega^cols.
    - Twiddle:  Multiply by T[r,c] = omega^(r*c).
    - Rows:     NTT on each row (length cols) using omega_row = omega^rows.
    - Reordering: Final output is flatten(transpose(Z)).

  To invert, we perform:
    0. Compute the appropriate inverse roots.
    1. Undo the reordering.
    2. Inverse row transform (length cols) on each row.
    3. Multiply by the inverse twiddle matrix T_inv[r,c] = omega^(-r*c).
    4. Inverse column transform (length rows) on each column.
    5. Reassemble the final result.

  Note: The naive inverse NTT (intt_original_form) already divides by the
  transform length.
  Hence, the two stages provide an overall normalization of 1/(rows·cols) = 1/N.

  Args:
    x: list or 1D numpy array (length N) that is the forward NTT result.
    q: prime modulus.
    omega: the primitive N-th root of unity used in the forward transform.
      (Forward transform used: rowNTT with omega_row = omega^R and columnNTT
      with omega_col = omega^C, plus twiddle multiplication T[r,c] =
      omega^(r*c).)
    rows: factors of N, so that N = rows * cols.
    cols: factors of N, so that N = rows * cols.

  Returns:
    A list representing the inverse NTT result (the original vector).
  """
  num_elements = rows * cols
  if len(x) != num_elements:
    raise ValueError("Length of X must equal rows * cols")

  # Step 0: Compute necessary inverse roots and normalization factors.
  # For the inverse column transform (of length rows):
  omega_col = pow(omega, cols, q)
  # For the inverse row transform (of length cols):
  omega_row = pow(omega, rows, q)

  # Step 1: Undo the final reordering of the forward transform.
  # The forward transform did: X = flatten(transpose(Z)) with Z of shape
  # (rows, cols).
  # To recover Z, first reshape X into shape (cols, rows) then transpose.
  matrix_z = np.array(x, dtype=int).reshape((cols, rows)).T
  # Now Z is an rows x cols matrix.

  # Step 2: Inverse row transform.
  # For each row of Y (length cols), compute the inverse NTT using omega_row.
  y = np.zeros((rows, cols), dtype=int)
  for r in range(rows):
    row = matrix_z[r, :].tolist()
    # intt on each row of length cols using inv_omega_row
    # (inverse happens inside intt_original_form)
    y[r, :] = intt_original_form(row, q, omega_row)

  # Step 3: Multiply by the inverse twiddle factor matrix.
  # The forward twiddle matrix was T[r,c] = omega^(r*c). Its inverse is:
  # T_inv[r,c] = omega^{-r*c} mod q.
  twiddle_inv = gen_twiddle_matrix_inv(rows, cols, q, omega)
  y = (y * twiddle_inv) % q

  # Step 4: Inverse column transform.
  # For each column of Z (length rows), compute the inverse NTT using
  # inv_omega_col.
  matrix_a = np.zeros((rows, cols), dtype=int)
  for c in range(cols):
    col = y[:, c].tolist()
    # intt on each column of length rows using inv_omega_col
    # (inverse happens inside intt_original_form).
    matrix_a[:, c] = intt_original_form(col, q, omega_col)

  # Step 5: Reassemble the final result.
  # The forward transform mapped the original vector x to X using a reordering.
  # Here, we flatten A (row-major order) to obtain the original x.
  x_recovered = np.array(matrix_a).flatten()
  return x_recovered.tolist()


def ntt_negacyclic(a, q, psi, rows, cols):
  """Compute the negacyclic NTT of array a (length n) modulo q.

  Args:
    a: list (or 1D array) of integers (length n).
    q: prime modulus.
    psi: an element in GF(q) such that psi^(2*n) = 1 and psi^n = -1 mod q. (That
      is, psi is a primitive 2n-th root of unity; note that then ω = psi^2 is a
      primitive n-th root of unity.)
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

  # a_transformed = ntt_bit_reverse(a_twisted.copy(), q, omega)
  return ntt_four_step(a_twisted.copy(), q, omega, rows, cols)


def intt_negacyclic(a, q, psi, rows, cols):
  """Compute the inverse negacyclic NTT of array a (length n) modulo q.

  Args: a   : list (or 1D array) of integers (length n) in the negacyclic
  evaluation domain. q   : prime modulus. psi : an element in GF(q) such that
  psi^(2*n) = 1 and psi^n = -1 mod q. (That is, psi is a primitive 2n-th root of
  unity; note that then ω = psi^2 is a primitive n-th root of unity.)

  Returns:
    The original input vector (i.e. the inverse transform).

  Process:
    1. Compute the inverse vanilla NTT using ω = psi².
    2. Post-twist: multiply the result by psi^(–i) for coefficient index i.
  """
  n = len(a)
  omega = pow(psi, 2, q)

  # Compute the inverse vanilla NTT.
  # a_inv = intt_bit_reverse(a.copy(), q, omega)
  a_inv = intt_four_step(a.copy(), q, omega, rows, cols)

  # Post-twisting: multiply a_inv[i] by psi^(–i).
  psi_inv = pow(psi, -1, q)
  return [(a_inv[i] * pow(psi_inv, i, q)) % q for i in range(n)]


def ntt_negacyclic_tpu_algorithm(
    a, q, psi, rows, cols, tf_step1, coef_step2, tf_step3
):
  """Compute the negacyclic NTT of array a (length n) modulo q.

  Args:
    a: list (or 1D array) of integers (length n).
    q: prime modulus.
    psi: an element in GF(q) such that psi^(2*n) = 1 and psi^n = -1 mod q. (That
      is, psi is a primitive 2n-th root of unity; note that then ω = psi^2 is a
      primitive n-th root of unity.)
    rows: Number of rows in the matrix.
    cols: Number of columns in the matrix.
    tf_step1: The twiddle factor matrix for step 1.
    coef_step2: The twiddle factor matrix for step 2 (element-wise
    multiplication).
    tf_step3: The twiddle factor matrix for step 3.

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

  num_elements = rows * cols
  if len(a_twisted) != num_elements:
    raise ValueError("Length of a_twisted must equal rows * cols")
  matrix_a = np.array(a_twisted, dtype=int).reshape((rows, cols))
  y = np.matmul(tf_step1, matrix_a)
  y = y % q

  y = y * coef_step2
  y = y % q

  z = np.matmul(y, tf_step3)
  z = z % q
  x = np.array(
      z.T
  ).flatten()  # forward transform reorders via transpose flattening
  return x.tolist()


def intt_negacyclic_tpu_algorithm(
    a, q, psi, rows, cols, inv_tf_step1, inv_coef_step2, inv_tf_step3
):
  """Compute the inverse negacyclic NTT of array a (length n) modulo q using TPU-friendly operations.

  Args: a   : list (or 1D array) of integers (length n) in the negacyclic
  evaluation domain. q   : prime modulus. psi : an element in GF(q) such that
  psi^(2*n) = 1 and psi^n = -1 mod q. (That is, psi is a primitive 2n-th root of
  unity; note that then ω = psi^2 is a primitive n-th root of unity.)
    rows: Number of rows in the matrix.
    cols: Number of columns in the matrix.
    inv_tf_step1: The inverse of the first transform matrix.
    inv_coef_step2: The inverse of the second coefficient matrix.
    inv_tf_step3: The inverse of the third transform matrix.

  Returns:
    The original input vector (i.e. the inverse transform).

  Process:
    1. Compute the inverse vanilla NTT using ω = psi².
    2. Post-twist: multiply the result by psi^(–i) for coefficient index i.
  """
  n = len(a)

  num_elements = rows * cols
  if len(a) != num_elements:
    raise ValueError("Length of a must equal rows * cols")

  # Step 1: Undo the final reordering of the forward transform.
  # The forward transform did: X = flatten(transpose(Z)) with Z of shape
  # (rows, cols).
  # To recover Z, first reshape X into shape (cols, rows) then transpose.
  z = np.array(a, dtype=int).reshape((cols, rows)).T
  # Now z is an rows x cols matrix.

  # Step 2: Inverse row transform.
  # For each row of Y (length cols), compute the inverse NTT using omega_row.
  y = np.matmul(z, inv_tf_step1) % q
  cols_inv = pow(cols, -1, q)
  y = y * cols_inv % q
  # Step 3: Multiply by the inverse twiddle factor matrix.
  # The forward twiddle matrix was T[r,c] = omega^(r*c). Its inverse is:
  # T_inv[r,c] = omega^{-r*c} mod q.
  y = (y * inv_coef_step2) % q

  # Step 4: Inverse column transform.
  # For each column of Z (length rows), compute the inverse NTT using
  # inv_omega_col.
  a = np.matmul(inv_tf_step3, y) % q
  rows_inv = pow(rows, -1, q)
  a = a * rows_inv % q

  # Step 5: Reassemble the final result.
  # The forward transform mapped the original vector x to X using a reordering.
  # Here, we flatten A (row-major order) to obtain the original x.
  x_recovered = np.array(a).flatten()

  # Post-twisting: multiply x_recovered[i] by psi^(–i).
  psi_inv = pow(psi, -1, q)
  return [(x_recovered[i] * pow(psi_inv, i, q)) % q for i in range(n)]


########################
# Online Functions
########################
@functools.partial(
    jax.jit,
    static_argnames=("q", "s", "m"),
)
def barret_reduction(z, q, s, m):
  """Vectorized implementation of the Barrett reduction.

  This implementation sets the internal shift width `w` to `min(s, 32)` so it
  works with small modulus `q < 2^16`.

  Args:
    z: The input value (at most 64 bits).
    q: The modulus.
    s: The bit width of q.
    m: The precomputed value for Barrett reduction.

  Returns:
    The result of the Barrett reduction.
  """
  w = min(s, 32)
  z1 = z & (2**w - 1)
  z2 = z >> w
  t = ((z1 * m) >> w) + (z2 * m)
  t = t >> (s - w)
  z = (z - t * q).astype(jnp.uint32)
  pred = z >= q
  return jnp.where(pred, z - q, z)


@functools.partial(
    jax.jit,
    static_argnames=("q", "s"),
)
def barret_reduction_static_q(z, q, s):
  """Vectorized implementation of the Barrett reduction.

  This implementation specializes on the value of `q`, which allows XLA to
  apply aggressive compile-time optimizations.

  Args:
    z: The input value.
    q: The modulus.
    s: The bit width of q.

  Returns:
      The result of the Barrett reduction.
  """
  # if this implementation fails to pass any test, move this line out
  # and add 'm' into static_argnames
  m = jnp.floor(2**s / q).astype(jnp.uint32)
  w = min(s, 32)
  z1 = z & (2**w - 1)
  z2 = z >> w
  t = ((z1 * m) >> w) + (z2 * m)
  t = t >> (s - w)
  z = (z - t * q).astype(jnp.uint32)
  pred = z >= q
  return jnp.where(pred, z - q, z)


@jax.jit
def hpmatmul_bat_coef_lhs_batch(lhs: jax.Array, y: jax.Array):
  """Input (m, k) Left Matrix -> (m, k, p, q) Left Matrix, where each element in the original (m, k) matrix is replaced by a (p, q) matrix.

  Expect the dtype of `lhs` and `rhs` to be `jnp.uint32`.

  Args:
    lhs: The input left matrix.
    y: The input right matrix.

  Returns:
    The result of the bat coefficient multiplication, with the same batch size
    as the input matrices.
  """
  rhs: jax.Array = jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint8)
  i8_products = jnp.einsum(
      "mkpq,bknq->bmnp",
      lhs,
      rhs,
      preferred_element_type=jnp.int32,
  )
  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
  return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(-1,))


@jax.jit
def hpmatmul_bat_coef_rhs_batch(y: jax.Array, rhs: jax.Array):
  """Input (k, n) right Matrix -> (k, n, p, q) right Matrix, where each element in the original (k, n) matrix is replaced by a (p, q) matrix.

  Expect the dtype of `lhs` and `rhs` to be `jnp.uint32`.

  Args:
    y: The input left matrix.
    rhs: The input right matrix.

  Returns:
    The result of the bat coefficient multiplication, with the same batch size
    as the input matrices.
  """

  lhs: jax.Array = jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint8)
  i8_products = jnp.einsum(
      "bmkq,knpq->bmnp",
      lhs,
      rhs,
      preferred_element_type=jnp.int32,
  )
  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
  return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(-1,))


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=("q", "s", "m"),
)
def ntt_layout_invariant_batch(
    poly_coef_2d,
    tf_step1,
    coef_step2,
    tf_step3,
    q,
    s,
    m,
):
  """Jax implementation of Gentalman Sande NTT, vectorized implementation on VPU."""
  assert poly_coef_2d.dtype == jnp.uint32
  assert tf_step1.dtype == jnp.uint8
  assert coef_step2.dtype == jnp.uint32
  assert tf_step3.dtype == jnp.uint8

  result_step1 = hpmatmul_bat_coef_lhs_batch(tf_step1, poly_coef_2d)

  result_step1_mod_q = barret_reduction(result_step1, q, s, m)

  result_step2 = jax.numpy.multiply(result_step1_mod_q, coef_step2)
  result_step2_mod_q = barret_reduction(result_step2, q, s, m)
  result_step3 = hpmatmul_bat_coef_rhs_batch(result_step2_mod_q, tf_step3)
  return barret_reduction(result_step3, q, s, m)


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=("rows_inv", "q", "s", "m"),
)
def intt_layout_invariant_batch(
    poly_coef_2d,
    tf_step1,
    coef_step2,
    tf_step3,
    rows_inv,
    q,
    s,
    m,
):
  """Jax implementation of Gentalman Sande NTT, vectorized implementation on VPU."""
  assert poly_coef_2d.dtype == jnp.uint32
  assert tf_step1.dtype == jnp.uint8
  assert coef_step2.dtype == jnp.uint32
  assert tf_step3.dtype == jnp.uint8

  result_step1 = hpmatmul_bat_coef_rhs_batch(poly_coef_2d, tf_step1)
  result_step1_mod_q = barret_reduction(result_step1, q, s, m)
  result_step2 = jax.numpy.multiply(result_step1_mod_q, coef_step2)
  result_step2_mod_q = barret_reduction(result_step2, q, s, m)
  result_step3 = hpmatmul_bat_coef_lhs_batch(tf_step3, result_step2_mod_q)
  result_step3_mod_q = barret_reduction(result_step3, q, s, m)
  result_scaled = jax.numpy.multiply(result_step3_mod_q, rows_inv)
  return barret_reduction(result_scaled, q, s, m)
