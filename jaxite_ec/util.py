"""Utility functions for jaxite_ec.

Note that: All functions that directly take Python int as input cannot be
jitted.
"""

import csv
import json
import math
from typing import Any, Callable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# copybara: from google3.perftools.accelerators.xprof.api.python import xprof_analysis_client
# copybara: from google3.perftools.accelerators.xprof.api.python import xprof_session

gcd = math.gcd


####################################
# BLS12-377 Curve Configurations
####################################

MODULUS_377_INT = 0x01AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001
MU_377_INT = 0x98542343310183A5DB0F28160BBD3DCEEEB43799DDAC681ABCB52236169B40B43B5A1DE2710A9647E7F56317936BFF32
TWIST_D_INT = 122268283598675559488486339158635529096981886914877139579534153582033676785385790730042363341236035746924960903179


####################################
# Global Configurations
####################################

BASE = 16
BASE_TYPE = jnp.uint16  # this type must match the BASE, i.e. jnp.uint<BASE>
U16_MASK = 0xFFFF
U32_MASK = 0xFFFFFFFF

U8_CHUNK_NUM = 48
U16_CHUNK_NUM = 24
U32_CHUNK_NUM = 12
U16_CHUNK_SHIFT_BITS = 16
U32_CHUNK_SHIFT_BITS = 32

U16_EXT_CHUNK_NUM = 25


BARRETT_SHIFT_U8 = 95  # BARRETT Params for k = 380
CHUNK_PRECISION = 8

# Lazy Reduction Logics
MODULUS_377_S16_INT = MODULUS_377_INT << 16

# Pippenger Logics
COORDINATE_NUM = 4

# RNS Reduction Logics
# Hardware friendly moduli factors are 2**16 - v for v in the following list
RNS_MODULI_T = (
    0,
    1,
    3,
    5,
    9,
    15,
    17,
    27,
    33,
    39,
    45,
    47,
    57,
    59,
    63,
    77,
    87,
    89,
    99,
    105,
    113,
    117,
    123,
    125,
    129,
    143,
    153,
    155,
    165,
    167,
    173,
    179,
    183,
    189,
    197,
    209,
    213,
    215,
    225,
    227,
    243,
    249,
    14,
    38,
    50,
    54,
    98,
    102,
    110,
    122,
)

MODULI = tuple([
    2**16 if i == 0 else 2**16 - int(i) if i % 2 == 1 else 2**15 - (int(i) // 2)
    for i in RNS_MODULI_T
])


RNS_PRECISION = 16
NUM_MODULI = len(RNS_MODULI_T)
# Maximum number of consecutive additions/subtractions
ADDITION_BOUND = 4

# Warning: specific to target modulus and addition bound
MODULI_SUB = tuple([
    ((512 * NUM_MODULI * MODULUS_377_INT * ADDITION_BOUND) - 2**16) % m
    for m in MODULI
])
TWIST_D_RNS = tuple([TWIST_D_INT % MODULI[i] for i in range(len(MODULI))])


####################################
# Utility Functions
####################################


def print_hex_values(int_list):
  hex_values = " ".join((hex(value)) for value in int_list)
  print(hex_values)


def array_to_int(jax_array: jax.Array, base) -> int:
  """Converts a JAX array to a single Python integer."""
  result = 0

  for i, elem in enumerate(jax_array):
    result |= int(elem) << (i * base)

  return result


def int_to_array(
    python_int, base=BASE, dtype=jnp.uint16, array_size=U16_CHUNK_NUM
):
  """Converts a Python integer to a JAX array."""
  mask = (1 << base) - 1

  elements = []
  while python_int > 0:
    elements.append(python_int & mask)  # Extract the lower bits
    python_int >>= base  # Shift to remove the extracted bits

  # we pad or trim the result to match the desired size
  if array_size is not None:
    assert array_size >= len(elements)
    elements = elements[:array_size] + [0] * (array_size - len(elements))

  return jnp.array(elements, dtype=dtype)


def array_to_int_list(jax_array, base):
  """Converts JAX array to single integer."""
  result_list = []
  for i in range(jax_array.shape[0]):
    value_vector = jax_array[i]
    value_int = array_to_int(value_vector, base)
    result_list.append(value_int)
  return result_list


def int_list_to_array(int_list, base=BASE, array_size=U16_CHUNK_NUM):
  """Converts a list of integers to a JAX array."""
  chunked_arrays = []
  for int_value in int_list:
    chunked_arrays.append(int_to_array(int_value, base, array_size=array_size))
  return jnp.array(chunked_arrays)


def int_point_to_jax_point_pack(
    coordinates: List[int], base=BASE, chunk_num=U16_CHUNK_NUM
):
  result = []
  for i in range(len(coordinates)):
    result.append(int_to_array(coordinates[i], base, array_size=chunk_num))
  return jnp.array(result)


def jax_point_pack_to_int_point(point: jax.Array):
  coordinate_num = point.shape[0]
  coordinates = []
  for i in range(coordinate_num):
    c = array_to_int(point[i], BASE)
    coordinates.append(c)
  return coordinates


# RNS related data formal conversion
def int_to_array_rns(x):
  return [x % m for m in MODULI]


def array_rns_to_int(residues):
  rns_precompute_values = rns_precompute(MODULI)
  return rns_reconstruct(residues, MODULI, rns_precompute_values)


def int_list_to_array_rns(int_list) -> jnp.ndarray:
  """Converts a list of integers to a JAX array."""
  limbs = []
  for int_value in int_list:
    limbs.append(int_to_array_rns(int_value))
  return jnp.array(limbs)


def array_rns_to_int_list(jax_array):
  """Converts JAX array to single integer."""
  result_list = []
  for i in range(jax_array.shape[0]):
    value_vector = jax_array[i]
    value_int = array_rns_to_int(value_vector)
    result_list.append(value_int)
  return result_list


def int_point_to_jax_rns_point_pack(coordinates: List[int]):
  result = []
  for i in range(len(coordinates)):
    result.append(int_to_array_rns(coordinates[i]))
  return jnp.array(result)


def jax_rns_point_pack_to_int_point(point: jax.Array):
  coordinate_num = point.shape[0]
  coordinates = []
  for i in range(coordinate_num):
    c = array_rns_to_int(point[i])
    coordinates.append(c)
  return coordinates


def int_point_batch_to_jax_point_pack(
    points: List[List[int]], base=BASE, chunk_num=U16_CHUNK_NUM
):
  result = []
  for i in range(len(points)):
    result.append(int_point_to_jax_point_pack(points[i], base, chunk_num))
  return jnp.transpose(jnp.array(result), (1, 0, 2))


def jax_point_pack_to_int_point_batch(point_pack: jnp.ndarray, base=BASE):
  points = jnp.transpose(point_pack, (1, 0, 2))
  results = []
  for i in range(len(points)):
    results.append(array_to_int_list(points[i], base))
  return results


def int_point_batch_to_jax_rns_point_pack(points: List[List[int]]):
  result = []
  for i in range(len(points)):
    result.append(int_point_to_jax_rns_point_pack(points[i]))
  return jnp.transpose(jnp.array(result), (1, 0, 2))


def jax_rns_point_pack_to_int_point_batch(point_pack: jnp.ndarray):
  points = jnp.transpose(point_pack, (1, 0, 2))
  results = []
  for i in range(len(points)):
    results.append(array_rns_to_int_list(points[i]))
  return results


# RNS helpers
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


def to_tuple(a):
  """Create to convert numpy array into tuple."""
  try:
    return tuple(to_tuple(i) for i in a)
  except TypeError:
    return a


# The following function achieves the same function as int_to_array, but it
# can be pre-run (Google restriction), and returns a tuple.
def int_to_precomputed_array(
    python_int, base=BASE, dtype=jnp.uint16, array_size=U16_CHUNK_NUM
):
  """Converts a Python integer to a JAX array."""
  mask = (1 << base) - 1

  elements = []
  while python_int > 0:
    elements.append(python_int & mask)  # Extract the lower bits
    python_int >>= base  # Shift to remove the extracted bits

  # we pad or trim the result to match the desired size
  if array_size is not None:
    assert array_size >= len(elements)
    elements = elements[:array_size] + [0] * (array_size - len(elements))

  return to_tuple(np.array(elements, dtype=dtype).tolist())


####################################
# Performance Profiler Functions (Google Internal)
####################################


def profile_jax_functions(
    tasks: List[Tuple[Callable[..., Any], Tuple[Any, ...]]],
    profile_name: str = "jax_profile",
):
  """Profiles a list of JAX functions.

  Args:
    tasks: A list of tuples, where each tuple contains a JAX function and its
      arguments.
    profile_name: The name of the profile.

  Usage:
    tasks = [
        (jit_pdul_barrett_xyzz_pack, (point_a_jax,)),
    ]
    profile_name = "jit_pdul_barrett_xyzz_pack"
    profile_jax_functions(tasks, profile_name)
  """
  session_id = None

  # copybara: session = xprof_session.XprofSession()
  # copybara: session.start_session()
  try:
    # Launch all JAX computations
    results = []
    for func, args_tuple in tasks:
      result = func(*args_tuple)
      results.append(result)

    # Wait for all computations launched in the loop to complete
    if results:
      jax.block_until_ready(results)

  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Error type: {type(e).__name__}")
    print(f"Error details: {e}")
    # Attempt to end the session even if there was an error
    # copybara: session_id = session.end_session_and_get_session_id()
    print("Xprof session ended due to error.")
    if session_id:
      print(f"{profile_name}:  http://xprof/?session_id={session_id}")
  finally:
    if session_id is None:
      # copybara: session_id = session.end_session_and_get_session_id()
      print(f"{profile_name}: http://xprof/?session_id={session_id}")
      # copybara: client = xprof_analysis_client.XprofAnalysisClient()
      trace = (
          client.get_profile_data("trace_viewer.json", session_id)
          if client
          else None
      )
      jtrace = json.loads(trace[1]) if trace else None
      if jtrace:
        for e in jtrace["traceEvents"]:
          if profile_name in e["name"]:
            print(f"{profile_name} latency: {e['dur']}\n")


####################################
# Lazy Reduction -- Offline Precompute
####################################


def construct_lazy_matrix(p, chunk_precision=8, chunk_num_u8=U8_CHUNK_NUM):
  """Construct the lazy matrix.

  Args:
    p: The modulus.
    chunk_precision: The chunk precision.
    chunk_num_u8: The number of chunks in the u8 value.

  Returns:
    lazy_mat: The lazy matrix.

  Note that: this function runs on CPU of the TPU-VM, which cannot be jitted.
  """
  lazy_mat_list = []
  for i in range(chunk_num_u8 + 4):
    val = int(int(256) ** (chunk_num_u8 + i)) % p
    lazy_mat_list.append(
        int_to_precomputed_array(val, chunk_precision, array_size=chunk_num_u8)
    )
  return to_tuple(lazy_mat_list)


MODULUS_377_LAZY_MAT = construct_lazy_matrix(MODULUS_377_INT)


####################################
# RNS Reduction -- Offline Precompute
####################################


def find_moduli(total_modulus, precision):
  """Finds a list of moduli close to the given precision.

  Args:
    total_modulus: The target modulus.
    precision: The desired precision of the moduli.

  Returns:
    A tuple containing two lists:
      - overall_moduli: A list of moduli close to the given precision.
      - overall_constant_offset: A list of constant offsets for the moduli.
  """
  initial_moduli = 2**precision
  overall_moduli = []
  overall_constant_offset = []
  overall_modulus = 1
  for i in range(2 ** (precision >> 1) - 1):
    cur_moduli = initial_moduli - i
    if math.gcd(cur_moduli, overall_modulus) == 1:
      overall_moduli.append(cur_moduli)
      overall_constant_offset.append(i)
      overall_modulus *= cur_moduli
      if overall_modulus > total_modulus:
        return to_tuple(overall_moduli), to_tuple(overall_constant_offset)

  # Find 2**15 - v too
  initial_moduli = 2 ** (precision - 1)
  if overall_modulus < total_modulus:
    for i in range(2 ** (precision >> 1) - 1):
      cur_moduli = initial_moduli - i
      if math.gcd(cur_moduli, overall_modulus) == 1:
        overall_moduli.append(cur_moduli)
        overall_constant_offset.append(i << 1)
        overall_modulus *= cur_moduli
        if overall_modulus > total_modulus:
          return to_tuple(overall_moduli), to_tuple(overall_constant_offset)

  return to_tuple(overall_moduli), to_tuple(overall_constant_offset)


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

  rns_mat = np.array(
      icrt_factors_byteshifted_modq_rns, dtype=np.uint16
  ).reshape(-1, num_residues)

  # calculate quotient estimation
  fix_point = 1 << moduli_precision

  shifted_quotient_estimations = []
  for factors in icrt_factors_byteshifted:
    for chunk in factors:
      shifted_quotient_estimations.append(
          [math.ceil((chunk * fix_point) / overall_modulus)]
      )
  sqe_mat = np.array(shifted_quotient_estimations, dtype=np.uint16)

  cor_mat = np.array(
      [to_rns(-overall_modulus % q, overall_moduli)], dtype=np.uint16
  )

  # Convert rns_mat and sqe_mat into various bytes.
  # Version 1: split precision into different chunks.
  # rns_mat_u8 = rns_mat.view(np.uint8).reshape(*rns_mat.shape, num_bytes)
  # seq_mat_u8 = sqe_mat.view(np.uint8).reshape(*sqe_mat.shape, num_bytes)
  # rns_stack_mat_u8 = np.hstack((
  #     rns_mat_u8[..., 0],
  #     seq_mat_u8[..., 0],
  #     rns_mat_u8[..., 1],
  #     seq_mat_u8[..., 1],
  # ))
  # Version 2: interleave precision -- tested to be faster.
  rns_stack_mat_u8 = np.hstack(
      (rns_mat.view(jnp.uint8), sqe_mat.view(jnp.uint8))
  )
  return to_tuple(rns_stack_mat_u8.tolist()), to_tuple(cor_mat.tolist())


def get_parts(u16mat):
  assert u16mat.dtype == np.uint16
  u16bytes = u16mat.view(np.uint8)
  return [u16bytes[:, ::2], u16bytes[:, 1::2]]


M = MODULUS_377_INT * MODULUS_377_INT * 256 * 256 * 50 * 50 * 4 * 2
moduli_precision = 16
num_bytes = moduli_precision // 8  # 2
# hardware friendly moduli is 2**precision - t
# overall_moduli is the jax.array of "2**precision - t"
# overall_constant_offset is the jax.array of "t"
overall_moduli, overall_constant_offset = find_moduli(M, moduli_precision)
M = 1
for moduli in overall_moduli:
  M *= moduli
M = int(M)
assert len(overall_moduli) == (
    (M.bit_length() + moduli_precision - 1) // moduli_precision
)

icrt_factors = rns_icrt_factors_compute(M, overall_moduli)

RNS_STACK_MAT_NEW, COR_MAT_NEW = rns_coefficients_precompute(
    icrt_factors,
    overall_moduli,
    num_bytes,
    moduli_precision,
    M,
    MODULUS_377_INT,
)


def construct_rns_matrix(q):
  return rns_coefficients_precompute(
      icrt_factors, overall_moduli, num_bytes, moduli_precision, M, q
  )


###############################
# Break High-precision Integer into Chunkcs
###############################


MODULI = overall_moduli
RNS_MODULI_T = overall_constant_offset
RNS_MAT = (RNS_STACK_MAT_NEW, COR_MAT_NEW)
MODULUS_377_INT_CHUNK = int_to_precomputed_array(
    MODULUS_377_INT, base=BASE, array_size=U16_CHUNK_NUM
)
MU_377_INT_CHUNK = int_to_precomputed_array(
    MU_377_INT, base=BASE, array_size=U16_CHUNK_NUM
)
TWIST_D_INT_CHUNK = int_to_precomputed_array(
    TWIST_D_INT, base=BASE, array_size=U16_EXT_CHUNK_NUM
)
MODULUS_377_S16_INT_CHUNK = int_to_precomputed_array(
    MODULUS_377_S16_INT, base=BASE, array_size=U16_EXT_CHUNK_NUM
)
