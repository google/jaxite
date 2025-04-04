"""Implementation of RNS modular reduction."""

import math
import random

import jax
import jax.numpy as jnp

randint = random.randint


jax.config.update("jax_enable_x64", True)
chunk_dtype = jnp.uint16
mul_res_dtype = jnp.uint32


def to_tuple(a):
  """Create to convert numpy array into tuple."""
  try:
    return tuple(to_tuple(i) for i in a)
  except TypeError:
    return a


def find_moduli(total_modulus, precision):
  """Find moduli for RNS."""
  initial_moduli = 2**precision
  overall_moduli = []
  constant_offset_list = []
  overall_modulus = 1
  for i in range(2 ** (precision >> 1) - 1):
    cur_moduli = initial_moduli - i
    if math.gcd(cur_moduli, overall_modulus) == 1:
      overall_moduli.append(cur_moduli)
      constant_offset_list.append(i)
      overall_modulus *= cur_moduli
      if overall_modulus > total_modulus:
        return overall_moduli, constant_offset_list

  # Find 2**15 - v too
  initial_moduli = 2 ** (precision - 1)
  if overall_modulus < total_modulus:
    for i in range(2 ** (precision >> 1) - 1):
      cur_moduli = initial_moduli - i
      if math.gcd(cur_moduli, overall_modulus) == 1:
        overall_moduli.append(cur_moduli)
        constant_offset_list.append(i << 1)
        overall_modulus *= cur_moduli
        if overall_modulus > total_modulus:
          return overall_moduli, constant_offset_list

  return overall_moduli, constant_offset_list


def hardware_friendly_mod_reduce(x, moduli_t):
  """Convert input value x into the RNS form.

  Args:
    x: Input value as a jnp.uint32 array.
    moduli_t: List of hardware friendly moduli_t.

  Returns:
    The RNS representation of x as a jnp.uint16 array.
  """
  assert x.dtype == jnp.uint32
  x_h = (x >> 16) & 0xFFFF
  x_l = x & 0xFFFF
  x_reduce = x_h * moduli_t + x_l

  x_h_sec = (x_reduce >> 16) & 0xFFFF
  x_l_sec = x_reduce & 0xFFFF
  x_reduce_sec = x_h_sec * moduli_t + x_l_sec

  x_h_third = (x_reduce_sec >> 16) & 0xFFFF
  x_l_third = x_reduce_sec & 0xFFFF
  x_reduce_third = x_h_third * moduli_t + x_l_third
  return x_reduce_third.astype(jnp.uint16)


def to_rns(x, moduli):
  return [x % m for m in moduli]


def rns_reconstruct(x, overall_moduli, icrt_factors):
  x_reconstruct = []
  for value_rns in x:
    big_int = 0
    for idx, tower in enumerate(value_rns):
      big_int += int(tower) * int(icrt_factors[idx])
    x_reconstruct.append(big_int % overall_moduli)
  return x_reconstruct


def rns_icrt_factors_compute(modulus, moduli):
  precomputed = []
  for m in moduli:
    rest = modulus // m  # 0 mod all the other moduli
    inverse = pow(rest % m, -1, m)  # factor to make 1 mod this moduli
    icrt_val = (rest * inverse) % modulus  # combine
    precomputed.append(icrt_val)
  return precomputed


def rns_coefficients_precompute(
    icrt_factors,
    overall_moduli,
    num_bytes,
    moduli_precision,
    overall_modulus,
    q,
):
  """Precompute RNS coefficients.

  Args:
    icrt_factors: Precomputed inverse CRT factors.
    overall_moduli: Array of moduli.
    num_bytes: Number of bytes.
    moduli_precision: Precision of the moduli.
    overall_modulus: Overall modulus.
    q: Target modulus.

  Returns:
    Precomputed RNS coefficients and correction coefficients.
  """
  num_residues = len(overall_moduli)
  # icrt_factors_byteshifted -- (num_residues, num_bytes)
  icrt_factors_byteshifted = [
      [
          (((1 << (8 * pre_id)) * factor) % overall_modulus)
          for pre_id in range(num_bytes)
      ]
      for factor in icrt_factors
  ]
  # icrt_factors_byteshifted_modq -- (num_residues, num_bytes)
  icrt_factors_byteshifted_modq = [
      [(chunk % q) for chunk in factors] for factors in icrt_factors_byteshifted
  ]
  # icrt_factors_byteshifted_modq_rns
  # (num_residues, num_bytes, num_residues) [Convert each byte range into RNS]
  icrt_factors_byteshifted_modq_rns = [
      [to_rns(chunk, overall_moduli) for chunk in factors]
      for factors in icrt_factors_byteshifted_modq
  ]

  rns_mat = jnp.array(
      icrt_factors_byteshifted_modq_rns, dtype=jnp.uint16
  ).reshape(-1, num_residues)

  # calculate quotient estimation
  fix_point = 1 << moduli_precision

  shifted_quotient_estimations = []
  for factors in icrt_factors_byteshifted:
    for chunk in factors:
      shifted_quotient_estimations.append(
          [math.ceil((chunk * fix_point) / overall_modulus)]
      )
  sqe_mat = jnp.array(shifted_quotient_estimations, dtype=jnp.uint16)

  cor_mat = jnp.array(
      [to_rns(-overall_modulus % q, overall_moduli)], dtype=jnp.uint16
  )

  # Convert rns_mat and sqe_mat into various bytes.
  rns_mat_u8 = jax.lax.bitcast_convert_type(rns_mat, jnp.uint8)
  seq_mat_u8 = jax.lax.bitcast_convert_type(sqe_mat, jnp.uint8)
  rns_stack_mat_u8 = jnp.hstack((
      rns_mat_u8[..., 0],
      seq_mat_u8[..., 0],
      rns_mat_u8[..., 1],
      seq_mat_u8[..., 1],
  ))
  return to_tuple(rns_stack_mat_u8.tolist()), to_tuple(cor_mat.tolist())


def rns_mod_reduce(
    data_a_rns,
    data_b_rns,
    moduli,
    moduli_t,
    rns_stack_mat_u8,
    cor_mat,
    icrt_factors,
    overall_modulus,
):
  """Performs RNS modular reduction.

  Args:
    data_a_rns: First input in RNS form.
    data_b_rns: Second input in RNS form.
    moduli: Array of moduli.
    moduli_t: Array of constant offsets.
    rns_stack_mat_u8: Precomputed RNS coefficients.
    cor_mat: Precomputed correction coefficients.
    icrt_factors: Precomputed inverse CRT factors.
    overall_modulus: Overall modulus.

  Returns:
    Result of the modular reduction.
  """
  num_residues = moduli.shape[0]
  mul_res = jnp.multiply(
      data_a_rns.astype(mul_res_dtype), data_b_rns.astype(mul_res_dtype)
  )
  mul_res_tower_red = hardware_friendly_mod_reduce(mul_res, moduli_t)

  # Global Modular reduction
  mul_res_glb_red = jnp.matmul(
      mul_res_tower_red.view(jnp.uint8),
      rns_stack_mat_u8,
      preferred_element_type=mul_res_dtype,
  )

  mul_res_glb_red_u32_l, mul_res_glb_red_u32_h = jnp.split(
      mul_res_glb_red, [num_residues + 1], axis=1
  )
  mul_res_glb_red_u32 = mul_res_glb_red_u32_l + (mul_res_glb_red_u32_h << 8)
  rns_reduce_u32, qe_u32 = jnp.split(
      mul_res_glb_red_u32, [num_residues], axis=1
  )

  # obtain the high 16 bits from the quotient estimation results qe_u32
  c_corrected = rns_reduce_u32 + jnp.matmul(
      qe_u32 >> moduli_precision, cor_mat, preferred_element_type=mul_res_dtype
  )
  c_corrected_reduce = hardware_friendly_mod_reduce(c_corrected, moduli_t)

  result_rns = rns_reconstruct(
      c_corrected_reduce.tolist(), overall_modulus, icrt_factors
  )

  return result_rns


if __name__ == "__main__":
  ###########################
  # User Configured Input
  ###########################
  q = 0x01AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001
  moduli_precision = 16
  extra_bit_to_avoid_addition_overflow = 4
  num_bytes = moduli_precision // 8
  num_residues_for_q = (
      q.bit_length() + moduli_precision - 1
  ) // moduli_precision + 1
  overall_modulus = (
      ((q * 256 * num_bytes * num_residues_for_q) ** 2)
      * extra_bit_to_avoid_addition_overflow
      * 2
  )
  assert overall_modulus == q * q * 256 * 256 * 50 * 50 * 4 * 2
  num_elements = 1024

  ###########################
  # Offline Precompute
  ###########################

  overall_moduli, constant_offset_list = find_moduli(
      overall_modulus, moduli_precision
  )
  assert overall_moduli == [
      65536,
      65535,
      65533,
      65531,
      65527,
      65521,
      65519,
      65509,
      65503,
      65497,
      65491,
      65489,
      65479,
      65477,
      65473,
      65459,
      65449,
      65447,
      65437,
      65431,
      65423,
      65419,
      65413,
      65411,
      65407,
      65393,
      65383,
      65381,
      65371,
      65369,
      65363,
      65357,
      65353,
      65347,
      65339,
      65327,
      65323,
      65321,
      65311,
      65309,
      65293,
      65287,
      32761,
      32749,
      32743,
      32741,
      32719,
      32717,
      32713,
      32707,
  ]
  overall_modulus = 1
  for moduli in overall_moduli:
    overall_modulus *= moduli
  overall_modulus = int(overall_modulus)
  assert len(overall_moduli) == (
      (overall_modulus.bit_length() + moduli_precision - 1) // moduli_precision
  )
  icrt_factors = rns_icrt_factors_compute(overall_modulus, overall_moduli)

  # hardware friendly moduli is 2**precision - t
  # moduli is the jax.array of "2**precision - t"
  # moduli_t is the jax.array of "t"
  rns_stack_mat_u8, cor_mat = rns_coefficients_precompute(
      icrt_factors,
      overall_moduli,
      num_bytes,
      moduli_precision,
      overall_modulus,
      q,
  )
  assert cor_mat == (
      (
          57491,
          26379,
          24673,
          4733,
          47122,
          11996,
          11119,
          12151,
          45048,
          10179,
          3397,
          45514,
          12274,
          62018,
          4316,
          141,
          20271,
          17626,
          20758,
          57875,
          41612,
          44321,
          30081,
          6090,
          16501,
          13984,
          14909,
          14581,
          47918,
          44932,
          34016,
          7605,
          33574,
          30236,
          15843,
          26521,
          52723,
          28347,
          32242,
          11676,
          31854,
          34463,
          30291,
          29806,
          1344,
          25148,
          23069,
          4869,
          6178,
          32502,
      ),
  )
  rns_stack_mat_u8 = jnp.array(rns_stack_mat_u8, dtype=jnp.uint8)
  cor_mat = jnp.array(cor_mat, dtype=jnp.uint16)

  moduli = jnp.array(overall_moduli, dtype=mul_res_dtype)
  moduli_t = jnp.array(constant_offset_list, dtype=chunk_dtype)

  ###########################
  # Generate Random Data
  ###########################

  random_data = [randint(0, q) for _ in range(num_elements)]
  result_ref = [val * val % q for val in random_data]
  data_rns = jnp.array(
      [to_rns(ele, overall_moduli) for ele in random_data], jnp.uint32
  )

  ###########################
  # Compute Modular Reduction in RNS Form
  ###########################
  # Limb-wise modular multiplication
  # (num_elements, num_residue)
  result_rns = rns_mod_reduce(
      data_rns,
      data_rns,
      moduli,
      moduli_t,
      rns_stack_mat_u8,
      cor_mat,
      icrt_factors,
      overall_modulus,
  )
  result_mod_q = [val % q for val in result_rns]
  assert result_ref == result_mod_q
