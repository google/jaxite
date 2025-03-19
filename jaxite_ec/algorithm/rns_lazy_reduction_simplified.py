"""RNS Variant of Lazy Reduction."""

import math
import random

import numpy as np

randint = random.randint
ceil = math.ceil
gcd = math.gcd
log = math.log


def total_modulus(moduli):
  modulus = 1  # Compute the big modulus
  for m in moduli:
    modulus *= m
  return modulus


def rns_precompute(moduli):
  modulus = total_modulus(moduli)
  precomputed = []
  for m in moduli:
    rest = modulus // m  # 0 mod all the other moduli
    inverse = pow(rest % m, -1, m)  # factor to make 1 mod this moduli
    icrt_val = (rest * inverse) % modulus  # combine
    precomputed.append(icrt_val)
  return precomputed


def rns_reconstruct(residues, moduli, precomputed):
  assert len(residues) == len(moduli)
  assert len(moduli) == len(precomputed)
  output = 0
  for i, r in enumerate(residues):
    output += precomputed[i] * int(r)
  return output % total_modulus(moduli)


def to_rns(x, moduli):
  assert x < total_modulus(moduli)
  return [x % m for m in moduli]


def ceil_div(x, y):
  return (x + y - 1) // y


def split_view_32_to_16(a):
  # TODO: compare these versions?
  # return np.right_shift(a, 16, dtype=np.uint16),
  #        np.bitwise_and(a, 2**16-1, dtype=np.uint16)
  v = a.view(dtype=np.uint16)
  return v[:, 1::2], v[:, ::2]


# Reinterpret as matrix of u8
def reinterpret_as_u8(mat):
  return mat.view(dtype=np.uint8)


# for 16 bit moduli
def cond_correct(vals, moduli):
  # outputs 0 or 1
  # Possibility: extract the third byte to get 0 or 1
  comparison = np.greater_equal(vals, moduli).astype(np.int16)
  # when negated: 0 or -1, and then interpret as unsigned: 0 or 0xffff
  mask = np.negative(comparison).view(dtype=np.uint16)
  # bitwise and to select
  correction = np.bitwise_and(moduli, mask)
  # 16 bit elementwise subtract 0 or modulus
  return np.subtract(vals, correction, dtype=vals.dtype)


# Helper function to pack values into matrix row
def to_row(int_val, moduli):
  t = len(moduli)
  row = np.zeros((2, t), dtype=np.uint8)
  rns = to_rns(int_val, moduli)
  for i in range(t):
    v = rns[i]
    u = v >> 8  # high byte
    l = v & (255)  # low byte
    row[0, i] = l
    row[1, i] = u
  return row


def greedy_select(priority_values_lists, target):
  """Greedily selects moduli from a list of priority values until the target is reached."""
  current_modulus = 1
  selected_moduli = []
  for priority_values_sublist in priority_values_lists:
    for value_info in priority_values_sublist:
      val = value_info[0]
      if gcd(val, current_modulus) == 1:
        selected_moduli.append(value_info)
        current_modulus *= val
        if current_modulus >= target:
          return selected_moduli
  print(current_modulus / target)
  raise ValueError("Did not reach target")


def next_odd_smaller(x):
  if x % 2 == 0:
    return x - 1
  return x


def gen_lists(bits_per_word):
  """Generates a list of priority values for greedy selection of moduli."""
  bound = 2 ** (bits_per_word // 2)
  while (bound + 1) ** 2 + bound >= 2**bits_per_word:
    bound -= 1
  # (value, )
  duh = [(2**bits_per_word, bits_per_word, 0)]
  small_t = [
      (2**bits_per_word - i, bits_per_word, i)
      for i in range(1, next_odd_smaller(bound), 2)
  ]
  bits_per_word -= 1
  small_t_extra = [
      (2**bits_per_word - i, bits_per_word, i)
      for i in range(1, next_odd_smaller(bound // 2), 2)
  ]
  small_t_pos = [
      (i, bits_per_word, 2**bits_per_word - i)
      for i in range(
          2**bits_per_word + 1,
          next_odd_smaller((bound // 2) + 2**bits_per_word),
          2,
      )
  ]
  prio_lists = [duh, small_t, small_t_extra, small_t_pos]
  return prio_lists


def gen_rns(bits_per_word, target):
  prio_lists = gen_lists(bits_per_word)
  return greedy_select(prio_lists, target)


# Put the moduli into an array for above algorithm.
def prepare_moduli(moduli_info):
  moduli = np.array(
      [0 if v[0] == 2**16 else v[0] for v in moduli_info], dtype=np.uint16
  )
  # 2**15 - v is being represented as 2**16 - 2 * v, because we wanna all moduli
  # to be 16 bit values. This allows us to use the same compute for both cases.
  # So we need to multiply by 2 when bit_per_word is 15 (not 16 in this case)
  moduli_t = np.array(
      [v[2] if v[1] == 16 else 2 * v[2] for v in moduli_info], dtype=np.uint8
  )
  return (moduli, moduli_t)


# Wide multiply for RNS convolution
def element_mult_16_by_16_to_32(a, b):
  return np.multiply(a, b, dtype=np.uint32)


# Reduce a u32 modulo a 16 bit modulus to u16
def mod_reduce_numpy(vals, moduli_info):
  """Reduces a u32 modulo a 16 bit modulus to u16 using numpy."""
  moduli_v, moduli_t = moduli_info
  u, l = split_view_32_to_16(vals)
  # m = 2**16 - t, where t is small
  # input = l + 2**16 * u
  # 2**16 % (2**16 - t) = t
  # input % m = l + t * u
  # u,l < 2**16 -1, therefore i1 < (t+1)(2**16 - 1)
  i1 = np.add(l, np.multiply(u, moduli_t, dtype=np.uint32), dtype=np.uint32)
  u, l = split_view_32_to_16(i1)
  # redundant form
  # u2 < (t+1), l2 < (2**16 - 1), therefore i2 < (2**16-1) + (t+1)^2.
  # For t < 254, it follows that i2 < 2**16 + modulus and therefore
  # one correction is sufficient.
  # 15 bit moduli are expressed as 2**16 - 2t = 2 * (2**15 - t) and also
  # works; it just has 16 bit representatives
  i2 = np.add(l, np.multiply(u, moduli_t, dtype=np.uint32), dtype=np.uint32)
  out = cond_correct(i2, moduli_v)
  # Just return the lower 16 bits
  return out.astype(np.uint16)


# Multiply as u8 arrays, returning u32 accumulator
def canonical_mult(a, b):
  return np.matmul(reinterpret_as_u8(a), reinterpret_as_u8(b), dtype=np.uint32)


# Modular multiplication using tensor cores and numpy
def mod_multiply_np(a_rns, b_rns, precompute, moduli):
  """Modular multiplication using tensor cores and numpy."""
  print("a", a_rns.shape, a_rns.dtype)
  print("b", b_rns.shape, b_rns.dtype)
  mat, precision, precision_words, correction, moduli_pre = precompute
  print("mat", mat.shape, mat.dtype)
  print("correction", correction.shape, correction.dtype)
  moduli_m, moduli_t = moduli_pre
  print("moduli", moduli_m.shape, moduli_m.dtype)
  print("moduli_t", moduli_t.shape, moduli_t.dtype)
  # Elementwise multiplication
  c_unreduced = element_mult_16_by_16_to_32(a_rns, b_rns)
  c_rns_reduced = mod_reduce_numpy(c_unreduced, moduli_pre)

  # Multiply u8 matrix with u32 accumulator
  c_target = canonical_mult(c_rns_reduced, mat)
  d = 2 * len(moduli)
  t = d // 2
  s_shift = ceil_div(precision_words, 2)
  # Recombine two accumulators to unreduced 16 bit rns
  # Elementwise operation
  # I guess bitshift is actually more efficient on TPU since 8 bit alignment
  # TODO: Interleaving versus block chunk
  # In gen_precompute_matrix, we split each 16-bit residue of
  # ICRT factor (mod moduli) into 2 8-bit chunks.
  # Lower 8-bit chunks of all residue are in the first :t + s_shift columns,
  # higher 8-bit chunks of all residue  are in the last :t + s_shift columns.
  # So here we need to recombine the two 8-bit chunks into 16-bit.
  c_target_combined = c_target[:, : t + s_shift] + (
      c_target[:, t + s_shift :] << 8
  )
  # elementwise
  if precision_words > 2:
    assert precision > 16
    # precision=24 for the parameters in this file if that helps
    quotient = (
        (c_target_combined[:, t + 1] + (c_target_combined[:, t] >> 16))
        >> (precision - 16)
    ).astype(np.uint16)
  else:
    q_row = c_target_combined[:, t]
    quotient = (q_row >> precision).astype(np.uint8)

  # 16 bit * 16 bit < 32 bit product
  # Couldn't figure out how to tell numpy to do wide product here so I just
  # changed types first
  quotient_correction = np.outer(
      quotient.astype(np.uint32), correction.astype(np.uint32)
  )
  c_target = c_target_combined[:, :t]
  # Quotient < 16 + log(2t) bits
  # Accumulator < 16 + 8 + log(2t) bits
  # For t = 51, Quotient < 23 bits and Accumulator < 31 bits, so sum will
  # fit in 32 bits.
  c_target += quotient_correction
  return mod_reduce_numpy(c_target, moduli_pre)


# Demonstrating the simplest version which is 4t^2 (optimal version is 3t^2)
def gen_precompute_matrix(moduli, moduli_info, target_modulus):
  """Generates the precompute matrix for RNS lazy reduction.

  This function precomputes data used later to “correct” and recombine
  RNS‐products so that a final result modulo the original (target) modulus is
  obtained. In other words, given that numbers are represented by their residues
  modulo a set of small moduli, the algorithm builds a matrix and some auxiliary
  parameters that let you later convert a product in the RNS domain back into a
  single number modulo the target modulus.
  """
  modulus = total_modulus(moduli)
  icrt_factors = rns_precompute(moduli)

  # The second place value in each 16 bit residue, since we have a matrix of
  # 8 bit elements
  shifted_icrt_factors = [(256 * p) % modulus for p in icrt_factors]
  # Apply lazy reduction to ICRT factors
  target_icrt_factors = [p % target_modulus for p in icrt_factors]
  target_shifted_icrt_factors = [
      p % target_modulus for p in shifted_icrt_factors
  ]

  # Precision required to correct quotient
  # Estimate how many multiples of the total modulus to cancel.
  # Why 2? There are four consecutive additions, which requires 2 more bits.
  precision_requirement = sum(moduli).bit_length() + 2
  fixed_point = 2**precision_requirement
  quotient_estimations = [
      ceil_div(p * fixed_point, modulus) for p in icrt_factors
  ]
  shifted_quotient_estimations = [
      ceil_div(p * fixed_point, modulus) for p in shifted_icrt_factors
  ]

  precision_words = 0
  for q in quotient_estimations:
    w = ceil(log(q, 2**8))
    if w > precision_words:
      precision_words = w
  print(precision_words)
  if precision_words == 3:
    precision_words = 4  # easier to just add a zero column.
  # 16 bit moduli
  d = 2 * len(moduli)
  t = d // 2
  mat = np.zeros((d, d + precision_words), dtype=np.uint8)
  s_shift = ceil_div(precision_words, 2)

  quotient_estimations_mat = np.matrix(
      [quotient_estimations], dtype=np.uint32
  ).T.view(dtype=np.uint8)
  shifted_quotient_estimations_mat = np.matrix(
      [shifted_quotient_estimations], dtype=np.uint32
  ).T.view(dtype=np.uint8)

  # Arrange values in matrix following base system
  for i in range(t):
    r = to_row(target_icrt_factors[i], moduli)
    mat[2 * i, :t] = r[0, :]
    mat[2 * i, t : t + s_shift] = quotient_estimations_mat[i, ::2]
    mat[2 * i, t + s_shift : 2 * t + s_shift] = r[1, :]
    mat[2 * i, 2 * t + s_shift :] = quotient_estimations_mat[i, 1::2]
    r = to_row(target_shifted_icrt_factors[i], moduli)
    mat[2 * i + 1, :t] = r[0, :]
    mat[2 * i + 1, t : t + s_shift] = shifted_quotient_estimations_mat[i, ::2]
    mat[2 * i + 1, t + s_shift : 2 * t + s_shift] = r[1, :]
    mat[2 * i + 1, 2 * t + s_shift :] = shifted_quotient_estimations_mat[
        i, 1::2
    ]

  # other things needed
  correction_factor = np.array(
      to_rns(-modulus % target_modulus, moduli), dtype=np.uint16
  )
  # correction_matrix
  moduli_pre = prepare_moduli(moduli_info)
  return (
      mat,
      precision_requirement,
      precision_words,
      correction_factor,
      moduli_pre,
  )


# Test modular multiplication
def test_mod_multiply(q, moduli_info_in_test):
  """Test modular multiplication."""
  moduli_info = prepare_moduli(moduli_info_in_test)
  moduli_np, _ = moduli_info
  moduli_list = [2**16 if int(m) == 0 else int(m) for m in moduli_np]
  assert total_modulus(moduli_list) > q * q
  precompute = gen_precompute_matrix(moduli_list, moduli_info_in_test, q)
  rns_p = rns_precompute(moduli_list)
  batch_size = 16
  test_values = [randint(0, q) for _ in range(3 * batch_size)]
  values_rns = [to_rns(v, moduli_list) for v in test_values]
  for i, v_rns in enumerate(values_rns):
    assert rns_reconstruct(v_rns, moduli_list, rns_p) == test_values[i]
  a = test_values[:batch_size]
  b = test_values[batch_size : 2 * batch_size]
  b2 = test_values[2 * batch_size :]
  a_rns = np.matrix(values_rns[:batch_size], dtype=np.uint16)
  b_rns = np.matrix(values_rns[batch_size : 2 * batch_size], dtype=np.uint16)
  b2_rns = np.matrix(values_rns[2 * batch_size :], dtype=np.uint16)
  c_rns = mod_multiply_np(a_rns, b_rns, precompute, moduli_list)
  for i in range(batch_size):
    c_int = rns_reconstruct(
        [int(c_rns[i, j]) for j in range(len(moduli_list))], moduli_list, rns_p
    )
    assert c_int % q == (a[i] * b[i]) % q
  # Test can apply again
  c2_rns = mod_multiply_np(c_rns, b2_rns, precompute, moduli_list)
  for i in range(batch_size):
    c_int = rns_reconstruct(
        [int(c2_rns[i, j]) for j in range(len(moduli_list))], moduli_list, rns_p
    )
    assert c_int % q == (a[i] * b[i] * b2[i]) % q
  print("Multiply pass")


def test_case():
  q = 0x01AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001
  # Generate 50 moduli using 16-bit words such that the total modulus exceeds the target.
  moduli_q = gen_rns(16, q * q * 256 * 256 * 50 * 50 * 4 * 2)
  print(len(moduli_q), moduli_q)
  test_mod_multiply(q, moduli_q)
