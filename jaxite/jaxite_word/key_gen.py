import math
import secrets
from typing import Any, Dict, List, Tuple
import jax.numpy as jnp
import jaxite.jaxite_word.rns as rns
import jaxite.jaxite_word.util as util

RnsPolynomial = rns.RnsPolynomial
RnsParams = rns.RnsParams
gen_rns_polynomial = rns.gen_rns_polynomial

MAX_INT_64 = 9223372036854775295
sigma = 3.190000057220458984375


########################
# Key Generation
########################
def gen_ternary_uniform_polynomial(
    degree: int, moduli: list[int]
) -> RnsPolynomial:
  """Generate a uniformly random RNS polynomial in R_Q = Z[X] / (Q, X^N+1)."""
  coeffs_q = [secrets.randbelow(2) for _ in range(degree)]
  return gen_rns_polynomial(degree, coeffs_q, moduli)


def gen_uniform_polynomial(degree: int, moduli: list[int]) -> RnsPolynomial:
  """Generate a uniformly random RNS polynomial in R_Q = Z[X] / (Q, X^N+1)."""
  coeffs_q = []
  for q in moduli:
    coeffs_q.append([secrets.randbelow(q) for _ in range(degree)])
  return RnsPolynomial(degree, moduli, coeffs_q, is_ntt=False)


def gen_gaussian_polynomial(
    degree: int, moduli: list[int], sigma: float
) -> RnsPolynomial:
  """Generate a random Gaussian polynomial in R_Q = Z[X] / (Q, X^N+1).

  Note: Each coefficient is independently sampled from a rounded Gaussian
        distribution with parameter sigma.

  Args:
    degree: The degree N of the ring R_Q.
    moduli: The list of prime moduli q_i's whose product is Q.
    sigma: The standard deviation of the Gaussian distribution.

  Returns:
    An RNS polynomial with coefficients sampled from a Gaussian distribution.
  """
  prng = secrets.SystemRandom()
  coeffs = [round(prng.normalvariate(0, sigma)) for _ in range(degree)]
  return gen_rns_polynomial(degree, coeffs, moduli)


def _validate_private_key(private_key: List[List[int]]) -> Tuple[int, int]:

  if not isinstance(private_key, list) or not private_key:
    raise ValueError("private_key must be a non-empty 2D list")
  num_elements = len(private_key)
  degree = None
  for row in private_key:
    if not isinstance(row, list) or not row:
      raise ValueError(
          "private_key must be a non-empty 2D list with non-empty rows"
      )
    if degree is None:
      degree = len(row)
    elif len(row) != degree:
      raise ValueError(
          "All rows in private_key must have the same length (degree)"
      )
    for coeff in row:
      if not isinstance(coeff, int):
        raise TypeError("All coefficients in private_key must be integers")
  return num_elements, degree  # type: ignore[return-value]


def _mod_q(x: int, q: int) -> int:
  r = x % q
  return r if r >= 0 else r + q


def modulus_switch(
    coefficient: int | List[int],
    cur_moduli: int = 524353,
    target_moduli: int = 1152921504606845473,
) -> int | List[int]:
  """Switch coefficients from cur_moduli to target_moduli using centered representation.

  If c < (cur_moduli + 1) // 2, it is unchanged. Otherwise, it is treated as
  a negative representative (c - cur_moduli) and lifted to Z_{target_moduli}
  by computing target_moduli + (c - cur_moduli).

  Args:
      coefficient: An integer or a list of integers in Z_{cur_moduli}.
      cur_moduli: Current modulus (default: 524353).
      target_moduli: Target modulus (default: 1152921504606845473).

  Returns:
      The switched coefficient with the same container type as input
      (int for int input, List[int] for list input).
  """
  threshold = (cur_moduli + 1) // 2

  def _switch_one(value: int) -> int:
    v = int(value)
    return v if v < threshold else target_moduli + v - cur_moduli

  if isinstance(coefficient, list):
    return [_switch_one(c) for c in coefficient]
  else:
    return _switch_one(int(coefficient))


def gen_evaluation_key(
    private_key: List[List[int]],
    q: int | List[int],
    P: int | List[int] = 1,
    noise_std: float = 3.190000057220458984375,
    noise_scale: int = 1,
    a: List[List[List[int]]] | None = None,
    e: List[List[List[int]]] | None = None,
    dnum: int = 3,
):

  num_elements, degree = _validate_private_key(private_key)

  q_list: List[int] = list(q)
  p_list: List[int] = list(P) if isinstance(P, list) else [int(P)]
  if len(private_key) != len(q_list):
    raise ValueError("private_key must have one row per q modulus (len(q))")
  size_q = len(q_list)
  # size_p = len(p_list)
  # size_qp = size_q + size_p

  sk_q = private_key
  # sOld is s^2 mod q for Q part
  sOld = [
      [_mod_q(sk_q[i][j] * sk_q[i][j], q_list[i]) for j in range(degree)]
      for i in range(size_q)
  ]

  return key_switch_gen(
      sOld=sOld,
      sNew=sk_q,
      q_list=q_list,
      p_list=p_list,
      noise_std=noise_std,
      noise_scale=noise_scale,
      a=a,
      e=e,
      dnum=dnum,
  )


def key_switch_gen(
    sOld: List[List[int]],
    sNew: List[List[int]],
    q_list: List[int],
    p_list: List[int],
    noise_std: float = 3.190000057220458984375,
    noise_scale: int = 1,
    a: List[List[List[int]]] | None = None,
    e: List[List[List[int]]] | None = None,
    dnum: int = 3,
) -> Dict[str, Any]:
  """Construct evaluation key parts given secret forms sOld (Q) and sNew (time domain).

  Args:
      sOld: Secret squared residues modulo each q in Q, shape [|Q|][N].
      sNew: Secret in time domain for Q basis, shape [|Q|][N].
      q_list: List of moduli forming Q.
      p_list: List of moduli forming P.
      noise_std: Standard deviation for error sampling.
      noise_scale: Integer multiplier applied to error samples.
      a: Optional pre-specified a samples per part and limb.
      e: Optional pre-specified e samples per part and limb.
      dnum: Number of partitions over Q for HYBRID scheme.

  Returns:
      Dict with keys: "a", "b", "modulus", "P", and "shape".
  """

  sOut = []
  degree = len(sNew[0])
  for limb, q in zip(sNew, q_list):
    psi = util.root_of_unity(2 * degree, q)
    test_in = util.bit_reverse_array(limb)
    temp = util.intt_negacyclic_bit_reverse(test_in, q, psi)
    sOut.append(temp)

  size_q = len(q_list)
  size_p = len(p_list)
  size_qp = size_q + size_p

  degree = len(sOut[0])
  s_qp = []
  for q_p in p_list:
    temp = [sOut[0][j] for j in range(degree)]
    temp = modulus_switch(temp, q_list[0], q_p)
    s_qp.append(temp)

  s_p_eva = []
  for limb, p in zip(s_qp, p_list):
    psi = util.root_of_unity(2 * degree, p)
    temp = util.ntt_negacyclic_bit_reverse(limb, p, psi)
    s_p_eva.append(util.bit_reverse_array(temp))

  s_qp_eva = sNew + s_p_eva
  P_prod = 1
  for p in p_list:
    P_prod *= p
  P_mod_q = [P_prod % qi for qi in q_list]

  num_per_part_q = (size_q + dnum - 1) // dnum
  num_part_q = math.ceil(size_q / num_per_part_q)

  a_parts: List[List[List[int]]] = []
  b_parts: List[List[List[int]]] = []
  moduli_list = q_list + p_list
  for part in range(num_part_q):
    start_idx = num_per_part_q * part
    end_idx = min(size_q, start_idx + num_per_part_q)
    a_sample = gen_ternary_uniform_polynomial(degree, moduli_list[:size_qp])
    a_rows = [
        util.bit_reverse_array(
            util.ntt_negacyclic_bit_reverse(
                a_sample.coeffs[i],
                modulus_i,
                util.root_of_unity(int(degree << 1), modulus_i),
            )
        )
        for i, modulus_i in enumerate(moduli_list[:size_qp])
    ]
    e_sample = gen_gaussian_polynomial(
        degree, moduli_list[:size_qp], sigma=noise_std
    )
    e_rows = [
        util.bit_reverse_array(
            util.ntt_negacyclic_bit_reverse(
                e_sample.coeffs[i],
                modulus_i,
                util.root_of_unity(int(degree << 1), modulus_i),
            )
        )
        for i, modulus_i in enumerate(moduli_list[:size_qp])
    ]
    b_rows: List[List[int]] = []
    for i in range(size_qp):
      modulus_i = q_list[i] if i < size_q else p_list[i - size_q]
      a_row = a[part][i] if a is not None else a_rows[i]
      e_row = e[part][i] if e is not None else e_rows[i]
      s_row = s_qp_eva[i]
      if i < start_idx or i >= end_idx:
        b_row = [
            _mod_q(
                _mod_q(-a_row[j] * s_row[j], modulus_i)
                + _mod_q(noise_scale * e_row[j], modulus_i),
                modulus_i,
            )
            for j in range(degree)
        ]
      else:
        b_row = [
            _mod_q(
                (
                    _mod_q(-a_row[j] * s_row[j], modulus_i)
                    + _mod_q(P_mod_q[i] * sOld[i][j], modulus_i)
                    + _mod_q(noise_scale * e_row[j], modulus_i)
                ),
                modulus_i,
            )
            for j in range(degree)
        ]
      b_rows.append(b_row)
    a_parts.append(a_rows)
    b_parts.append(b_rows)

  return {
      "a": a_parts if a is None else a,
      "b": b_parts,
      "modulus": {"Q": q_list, "P": p_list},
      "P": p_list,
      "shape": (num_part_q, size_qp, degree),
  }


def find_automorphism_index_2n_complex(i: int, m: int):
  if i == 0:
    return 1
  elif i == m - 1:
    return i
  if not util.is_power_of_two(m):
    raise ValueError("m should be a power of two.")

  g0 = pow(5, -1, m) if i < 0 else 5  # modular inverse of 5 mod m when i < 0
  g = g0
  i_unsigned = abs(i)
  for _ in range(1, i_unsigned):
    g = (g * g0) & (m - 1)  # modulo m since m is a power of two
  return g


def precompute_rotation_key_map(n: int, k: int) -> list[int]:
  m = n << 1
  logm = int(round(math.log2(m)))
  logn = int(round(math.log2(n)))
  precomp = [0] * n
  for j in range(n):
    j_tmp = (j << 1) + 1
    mul = j_tmp * k
    idx = (mul - ((mul >> logm) << logm)) >> 1
    jrev = util.bit_reverse(j, logn)
    idxrev = util.bit_reverse(idx, logn)
    precomp[jrev] = idxrev
  return precomp


def gen_rotation_key(
    sk,
    original_moduli,
    extend_moduli,
    rot_index,
    dnum=3,
    noise_std=3.190000057220458984375,
    noise_scale=1,
    a=None,
    e=None,
):
  n = len(sk[0])
  result = find_automorphism_index_2n_complex(rot_index, 2 * n)
  key_map_idx = util.modinv(result, 2 * n)
  target_order = precompute_rotation_key_map(n, key_map_idx)
  # transform sk based on the order.
  sk_rot = jnp.array(sk)[:, jnp.array(target_order)].tolist()
  ek = key_switch_gen(
      sk,
      sNew=sk_rot,
      q_list=original_moduli,
      p_list=extend_moduli,
      noise_std=noise_std,
      noise_scale=noise_scale,
      a=a,
      e=e,
      dnum=dnum,
  )
  return ek


def gen_pke_pair(
    q_towers: List[int],
    p_towers: List[int],
    degree: int,
    noise_std: float = 3.190000057220458984375,
    noise_scale: int = 1,
    a_ref=None,
    s_ref=None,
    e_ref=None,
) -> Dict[str, Any]:
  """Generate a PKE pair.

  Args:
      q_towers: List of moduli forming Q.
      p_towers: List of moduli forming P.
      degree: The degree N of the ring R_Q.
      noise_std: Standard deviation for error sampling.
      noise_scale: Integer multiplier applied to error samples.

  Returns:
      Dict with keys: "public_key", "secret_key".
  """
  moduli_list = q_towers + p_towers
  s = gen_ternary_uniform_polynomial(degree, moduli_list)
  s = [
      util.bit_reverse_array(
          util.ntt_negacyclic_bit_reverse(
              s.coeffs[i],
              modulus_i,
              util.root_of_unity(int(degree << 1), modulus_i),
          )
      )
      for i, modulus_i in enumerate(moduli_list)
  ]
  s = s_ref if s_ref is not None else s
  a = gen_uniform_polynomial(degree, moduli_list)
  a = [
      util.bit_reverse_array(
          util.ntt_negacyclic_bit_reverse(
              a.coeffs[i],
              modulus_i,
              util.root_of_unity(int(degree << 1), modulus_i),
          )
      )
      for i, modulus_i in enumerate(moduli_list)
  ]
  a = a_ref if a_ref is not None else a
  e = gen_gaussian_polynomial(degree, moduli_list, sigma=noise_std)
  e = [
      util.bit_reverse_array(
          util.ntt_negacyclic_bit_reverse(
              e.coeffs[i],
              modulus_i,
              util.root_of_unity(int(degree << 1), modulus_i),
          )
      )
      for i, modulus_i in enumerate(moduli_list)
  ]
  e = e_ref if e_ref is not None else e

  b = [
      [
          _mod_q(
              _mod_q(e[i][j] * noise_scale, moduli_list[i])
              - _mod_q(a[i][j] * s[i][j], moduli_list[i]),
              moduli_list[i],
          )
          for j in range(degree)
      ]
      for i in range(len(moduli_list))
  ]
  s = s[: len(q_towers)]
  return {
      "public_key": [b, a],
      "secret_key": s,
  }
