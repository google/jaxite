import cmath
import math
import random
from typing import List
import jaxite.jaxite_word.ciphertext as ct
import jax.numpy as jnp
import jaxite.jaxite_word.key_gen as kg
import numpy as np
import jaxite.jaxite_word.util as util


MAX_INT_64 = 9223372036854775295
sigma = 3.190000057220458984375


########################
# Common Functions
########################
def _roots(m: int) -> List[complex]:
  return [cmath.exp(1j * 2 * math.pi * k / m) for k in range(m)]


def _rot_group(m: int, nh: int, g: int = 5) -> List[int]:
  # CKKS rotation subgroup (powers of 5 mod m)
  r = [1]
  for _ in range(1, nh):
    r.append((r[-1] * g) % m)
  return r


def _bitrev_inplace(a: List[complex]) -> None:
  n = len(a)
  j = 0
  for i in range(1, n):
    bit = n >> 1
    while j & bit:
      j ^= bit
      bit >>= 1
    j ^= bit
    if i < j:
      a[i], a[j] = a[j], a[i]


def FFTSpecialInv(vals: List[complex], cycl_order: int) -> None:
  """CKKS 'special' inverse FFT (DIF-style), in-place:

  - twiddles via rotation group,
  - per-stage idx = ((lenq - (rg[j] % lenq)) % lenq) * (m/lenq),
  - scale by 1/Nh,
  - *post* bit-reversal (to match OpenFHE ordering in your test).
  """
  m = cycl_order
  nh = len(vals)
  roots = _roots(m)
  rg = _rot_group(m, nh, g=5)

  length = nh
  while length >= 1:
    half = length >> 1
    lenq = length << 2
    step = m // lenq
    for i in range(0, nh, length):
      for j in range(half):
        mod = rg[j] % lenq
        idx = ((lenq - mod) % lenq) * step
        u = vals[i + j]
        t = vals[i + j + half]
        vals[i + j] = u + t
        vals[i + j + half] = (u - t) * roots[idx]
    length >>= 1

  inv = 1.0 / nh
  for k in range(nh):
    vals[k] *= inv

  _bitrev_inplace(vals)


def FFTSpecial(vals: List[complex], cycl_order: int) -> None:
  """CKKS 'special' forward FFT (DIT-style), in-place:

  - *pre* bit-reversal,
  - per-stage idx = (rg[j] % lenq) * (m/lenq),
  - no final scale.
  """
  m = cycl_order
  nh = len(vals)
  roots = _roots(m)
  rg = _rot_group(m, nh, g=5)

  _bitrev_inplace(vals)

  length = 2
  while length <= nh:
    half = length >> 1
    lenq = length << 2
    step = m // lenq
    for i in range(0, nh, length):
      for j in range(half):
        mod = rg[j] % lenq
        idx = mod * step
        u = vals[i + j]
        v = vals[i + j + half] * roots[idx]
        vals[i + j] = u + v
        vals[i + j + half] = u - v
    length <<= 1


def nearest_int(x: float) -> int:
  # matches OpenFHE's usual "nearest" behavior used for CKKS encode
  return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def slot_to_coeffs(y: List[complex]) -> List[float]:
  """Slot -> real polynomial coefficients (length N=2*Nh).

  Adjust here if your build packs coefficients differently. Current:
  [Re(y_0..y_{Nh-1}), Im(y_0..y_{Nh-1})]
  """
  nh = len(y)
  re = [y[i].real for i in range(nh)]
  im = [y[i].imag for i in range(nh)]
  return re + im


def fit_to_native_vector(
    vec: List[int], big_bound: int, modulus: int, ring_dim: int
) -> List[int]:
  """Python equivalent of CKKSPackedEncoding::FitToNativeVector.

  - Places dslots values into a length-ring_dim vector at indices i*gap

    where gap = ring_dim // dslots.
  - Maps each input value n using bigBound and modulus:
      if n > bigBound/2: (n - (bigBound - modulus)) mod modulus
      else:               n mod modulus
  """
  dslots = len(vec)
  if dslots == 0:
    return [0] * ring_dim

  big_value_half = big_bound >> 1
  diff = big_bound - modulus
  gap = ring_dim // dslots

  native = [0] * ring_dim
  for i, val in enumerate(vec):
    n = int(val)
    if n > big_value_half:
      mapped = (n - diff) % modulus
    else:
      mapped = n % modulus
    native[gap * i] = mapped
  return native


def ckks_encrypt(
    plaintext: List[List[int]],
    public_key: List[List[List[int]]],
    q_towers: List[int],
    noise_scale_degree: int = 1,
    sigma=3.190000057220458984375,
    v=None,
    e=None,
    ba=None,
    e_ref=None,
    v_ref=None,
):
  # plaintext is now (degree, moduli)
  degree = len(plaintext)
  num_towers = len(q_towers)
  psi = None
  if v is None:
    psi = [
        util.root_of_unity(int(2 * degree), q_towers[t_id])
        for t_id in range(num_towers)
    ]
    v = kg.gen_ternary_uniform_polynomial(degree, q_towers).coeffs
    v = [
        util.bit_reverse_array(
            util.ntt_negacyclic_bit_reverse(v[t_id], q_towers[t_id], psi[t_id])
        )
        for t_id in range(num_towers)
    ]
    if v_ref is not None:
      np.testing.assert_array_equal(v, v_ref)
  if e is None:
    # This noise is too large! I need to copy the OpenFHE's implementation to fix it
    psi = [
        util.root_of_unity(int(2 * degree), q_towers[t_id])
        for t_id in range(num_towers)
    ]
    e0 = kg.gen_gaussian_polynomial(degree, q_towers, sigma=sigma).coeffs
    e0 = [
        util.bit_reverse_array(
            util.ntt_negacyclic_bit_reverse(
                (e0[t_id]), q_towers[t_id], psi[t_id]
            )
        )
        for t_id in range(num_towers)
    ]
    e1 = kg.gen_gaussian_polynomial(degree, q_towers, sigma=sigma).coeffs
    e1 = [
        util.bit_reverse_array(
            util.ntt_negacyclic_bit_reverse(
                (e1[t_id]), q_towers[t_id], psi[t_id]
            )
        )
        for t_id in range(num_towers)
    ]
    if e_ref is not None:
      np.testing.assert_array_equal(e0, e_ref[0])
      np.testing.assert_array_equal(e1, e_ref[1])
  else:
    e0, e1 = e[0], e[1]
  ns = noise_scale_degree
  if len(plaintext[0]) < len(public_key[0]):
    diff_length = len(public_key[0]) - len(plaintext[0])
    public_key[0] = public_key[0][:-diff_length]
    public_key[1] = public_key[1][:-diff_length]

  # Prepare c0, c1 accumulators with shape (degree, moduli)
  c0 = [[0] * num_towers for _ in range(degree)]
  c1 = [[0] * num_towers for _ in range(degree)]

  # We need to iterate carefully.
  # v is (moduli, degree)
  # public_key is (2, moduli, degree) -> public_key[0], public_key[1] are (moduli, degree)
  # e0, e1 are (moduli, degree)
  # But we want output (degree, moduli)

  for i in range(degree):
    # plaintext[i] is [m0, m1, ..., mk] corresponding to degree i
    p_i_moduli = plaintext[i]

    for j in range(num_towers):
      q_j = q_towers[j]
      v_ji = v[j][i]
      pk0_ji = public_key[0][j][i]
      e0_ji = e0[j][i]

      val0 = (v_ji * pk0_ji + ns * e0_ji) % q_j

      # Add plaintext
      val0 = (val0 + p_i_moduli[j]) % q_j
      c0[i][j] = val0

      pk1_ji = public_key[1][j][i]
      e1_ji = e1[j][i]

      val1 = (v_ji * pk1_ji + ns * e1_ji) % q_j
      c1[i][j] = val1

  if ba is not None:
    # ba corresponds to the value BEFORE adding plaintext (c0_pre, c1)
    # c0 currently holds val0 + plaintext. We need to subtract plaintext for verification.
    # c1 holds val1 (no plaintext added), so it's fine.
    c0_T = [
        [(c0[d][m] - plaintext[d][m]) % q_towers[m] for d in range(degree)]
        for m in range(num_towers)
    ]
    c1_T = [[c1[d][m] for d in range(degree)] for m in range(num_towers)]
    np.testing.assert_array_equal(c0_T, ba[0])
    np.testing.assert_array_equal(c1_T, ba[1])

  return [c0, c1]


def ckks_decrypt(
    ciphertext: List[List[List[int]]],
    private_key: List[List[int]],
    q_towers: List[int],
):
  # ciphertext is list of elements. element 0 is c0, etc.
  # each element is (degree, moduli)
  num_elements = len(ciphertext)
  degree = len(ciphertext[0])
  num_towers = len(ciphertext[0][0])

  s = private_key  # (moduli, degree)

  if num_towers < len(s):
    diff_length = len(s) - num_towers
    s = s[:-diff_length]

  # Pre-transpose s for easier access or just index carefully
  # s is (moduli, degree)

  # We want to compute: M(X) = c0 + c1*s + ...
  # Result should be (degree, moduli) initially before NTT/CRT?
  # Actually decrypt returns coefficients.

  # Let's accumulate in (moduli, degree) for the final NTT part which expects that layout usually,
  # OR we adapt the rest of the function.
  # The original returned `first_element_coef` which was (moduli, degree).
  # But we want "ciphertext/plaintext in the layout of (degree, moduli)".
  # So we should probably return (degree, moduli).

  # Let's accumulate in (degree, moduli).

  res_poly = [[0] * num_towers for _ in range(degree)]

  # s_power starts as s^1. s is (moduli, degree).
  # We need s^k in (moduli, degree).

  s_powers = [s]  # s^1
  # Generate powers if needed (for num_elements > 2)
  # original code: s_power updated iteratively.

  cur_s_power = [list(row) for row in s]  # Copy s

  # c0
  for d in range(degree):
    for m in range(num_towers):
      res_poly[d][m] = ciphertext[0][d][m]

  for i in range(1, num_elements):
    ci = ciphertext[i]  # (degree, moduli)

    for d in range(degree):
      for m in range(num_towers):
        # + ci * s^i
        term = (ci[d][m] * cur_s_power[m][d]) % q_towers[m]
        res_poly[d][m] = (res_poly[d][m] + term) % q_towers[m]

    if i < num_elements - 1:
      # Update s_power to s^(i+1)
      # s^(i+1) = s^i * s
      new_s_power = [[0] * degree for _ in range(num_towers)]
      for m in range(num_towers):
        qi = q_towers[m]
        for d in range(degree):
          new_s_power[m][d] = (cur_s_power[m][d] * s[m][d]) % qi
      cur_s_power = new_s_power

  # Now res_poly is (degree, moduli)
  # We need to do inverse NTT.
  # Existing utils utilize (moduli, degree) usually?
  # util.bit_reverse_array takes 1D list.
  # util.intt_negacyclic_bit_reverse takes 1D list.

  # So we can process row by row if we transpose or col by col.
  # The original returned `first_element_coef` as list of lists (moduli, degree).
  # We want to return (degree, moduli).

  final_res = [[0] * num_towers for _ in range(degree)]

  for m in range(num_towers):
    # Extract column m
    col = [res_poly[d][m] for d in range(degree)]

    # bit reverse
    col_rev = util.bit_reverse_array(col)

    # intt
    coef = util.intt_negacyclic_bit_reverse(
        col_rev, q_towers[m], util.root_of_unity(2 * degree, q_towers[m])
    )

    for d in range(degree):
      final_res[d][m] = coef[d]

  return final_res


def ckks_encode(
    slots: List[complex],
    cycl_order: int,
    q_towers: List[int],
    p_towers: List[int],
    scale: float,
    noise_scale_degree: int = 1,
    max_bits_in_word: int = 61,
):
  """Encode slots to DCRTPoly EVAL form with given (Q,P) towers, NATIVE_INT=64.

  Returns dict with residues for Q and P towers and the scaled integer coeffs.
  """
  nh = len(slots)
  N = 2 * nh
  m = cycl_order
  assert m == 4 * nh, "cycl_order must be 4*Nh for CKKS special FFT size"

  # 1) inverse special FFT
  y = list(slots)
  FFTSpecialInv(y, m)

  # 2) slot->coeff packing
  coeffs = slot_to_coeffs(y)  # length N

  # 3) scale and determine bit length like OpenFHE (NATIVEINT==64)
  #    Find logc = ceil(log2(max(|scaled_real|, |scaled_imag|))) across slots
  scaled_vals = [scale * v for v in coeffs]
  logc = -(10**9)
  for v in scaled_vals:
    absv = abs(v)
    if absv != 0.0:
      logci = int(math.ceil(math.log2(absv)))
      if logc < logci:
        logc = logci
  if logc == -(10**9):
    logc = 0
  if logc < 0:
    raise ValueError("Scaling factor too small")
  # 4) approxFactor to keep values within 60-bit word, then quantize
  log_valid = logc if logc <= max_bits_in_word else max_bits_in_word
  log_approx = logc - log_valid
  approx_factor = 2.0**log_approx
  # Quantize with round-to-nearest after dividing by approx_factor
  ints_base = [nearest_int(v / approx_factor) for v in scaled_vals]
  ints_base = [x + MAX_INT_64 if x < 0 else x for x in ints_base]

  elements = [
      fit_to_native_vector(ints_base, MAX_INT_64, q_tower, N)
      for q_tower in q_towers
  ]

  # 5) Scale back up by approx_factor (power of two) in the ring
  if log_approx > 0:
    step = 1 << log_approx
    ints = [
        [x * step % q_towers[mod_id] for x in elements[mod_id]]
        for mod_id in range(len(q_towers))
    ]
  else:
    ints = elements

  # 6) If noise scale degree > 1, multiply by round(scale)^(d-1)
  if noise_scale_degree > 1:
    int_pow_p = int(round(scale))
    if int_pow_p != 1:
      power = pow(int_pow_p, noise_scale_degree - 1)
      for mod_id in range(len(q_towers)):
        ints[mod_id] = [x * power % q_towers[mod_id] for x in ints[mod_id]]

  # 4) residues per tower
  Q_res = [
      util.ntt_negacyclic_bit_reverse(
          ints[mod_id],
          q_towers[mod_id],
          util.root_of_unity(m, q_towers[mod_id]),
      )
      for mod_id in range(len(q_towers))
  ]
  # Current Q_res is (moduli, degree)
  # We want to return (degree, moduli)

  Q_res_T = [[0] * len(q_towers) for _ in range(N)]
  for mod_id in range(len(q_towers)):
    rev = util.bit_reverse_array(Q_res[mod_id])
    for deg in range(N):
      Q_res_T[deg][mod_id] = rev[deg]

  return Q_res_T


def ckks_decode(
    plaintext: List[int],
    scalingFactor: float,
    slots: int,
    q: int,
    p: int,
    CKKS_M_FACTOR: int = 1,
    ADD_NOISE: bool = False,
):
  # Ported from notebook implementation
  degree = len(plaintext)
  q_half = q >> 1
  Nh = degree // 2
  gap = Nh // slots
  powP_positive = pow(2, p)
  powP = pow(2, -p)

  # Step 1: scale back to intermediate complex vector m(X)
  sf_pre = (1.0 / scalingFactor) * powP_positive

  real_part_list = []
  imag_part_list = []
  for i in range(slots):
    # real part from first half
    r_val = plaintext[i]
    if r_val > q_half:
      real_part = -((q - r_val) * sf_pre)
    else:
      real_part = r_val * sf_pre
    real_part_list.append(int(real_part))

    # imag part from second half
    im_val = plaintext[i + Nh]
    if im_val > q_half:
      imag_part = -((q - im_val) * sf_pre)
    else:
      imag_part = im_val * sf_pre
    imag_part_list.append(int(imag_part))

  curValues = [
      complex(real_part_list[i], imag_part_list[i]) for i in range(slots)
  ]

  # Step 2: compute conjugate vector and estimated stddev (per OpenFHE logic)
  def _conjugate(vec: List[complex]) -> List[complex]:
    n = len(vec)
    result = [0j] * n
    for idx in range(1, n):
      z = vec[n - idx]
      result[idx] = complex(-z.imag, -z.real)
    z0 = vec[0]
    result[0] = complex(z0.real, -z0.imag)
    return result

  def _stddev(vec: List[complex], conjugate: List[complex]) -> float:
    import math as math

    s = len(vec)
    if s == 1:
      return vec[0].imag
    dslots = s * 2
    complex_values = [vec[i] - conjugate[i] for i in range(s // 2 + 1)]
    mean = 2 * sum((cv.real + cv.imag) for cv in complex_values[1 : (s // 2)])
    mean += complex_values[0].imag
    mean += 2 * complex_values[s // 2].real
    mean /= dslots - 1.0
    variance = 2 * sum(
        ((cv.real - mean) ** 2 + (cv.imag - mean) ** 2)
        for cv in complex_values[1 : (s // 2)]
    )
    variance += (complex_values[0].imag - mean) ** 2
    variance += 2 * (complex_values[s // 2].real - mean) ** 2
    variance /= dslots - 2.0
    return 0.5 * math.sqrt(variance)

  conjugate = _conjugate(curValues)

  stddev_dbl = _stddev(curValues, conjugate)
  logstd = math.log2(stddev_dbl) if stddev_dbl > 0 else float("-inf")
  if stddev_dbl < 0.125 * math.sqrt(degree):
    stddev_dbl = 0.125 * math.sqrt(degree)
  if logstd > p - 5.0:
    raise Exception(
        "The decryption failed because the approximation error is too high."
        " Check the parameters. "
    )

  stddev = math.sqrt(CKKS_M_FACTOR + 1) * stddev_dbl
  scale = 0.5 * powP

  # For security, add tiny Gaussian noise scaled by 2^{-p}; it doesn't affect ~1e-3 accuracy
  rng = random.Random()

  def _gauss():
    return rng.gauss(0.0, stddev)

  if ADD_NOISE:
    curValues = [
        complex(
            real_part_list[i] * scale
            + conjugate[i].real * scale
            + powP * _gauss(),
            imag_part_list[i] * scale
            + conjugate[i].imag * scale
            + powP * _gauss(),
        )
        for i in range(slots)
    ]
  else:
    curValues = [
        complex(
            real_part_list[i] * scale + conjugate[i].real * scale,
            imag_part_list[i] * scale + conjugate[i].imag * scale,
        )
        for i in range(slots)
    ]

  # Step 3: Special forward FFT to slot values
  FFTSpecial(curValues, degree * 2)
  curValues = [complex(curValues[i].real, 0.0) for i in range(slots)]
  # Return real parts only
  return curValues


def _crt_combine_rns_plaintext(
    rns_plaintext: List[List[int]], moduli: List[int]
) -> List[int]:
  """Combine residues modulo pairwise-coprime moduli using the standard CRT formula.

  rns_plaintext is (degree, moduli).
  """
  M = 1
  for q in moduli:
    M *= q
  Mi_list = [M // qi for qi in moduli]
  inv_list = [pow(Mi, -1, qi) for Mi, qi in zip(Mi_list, moduli)]

  degree = len(rns_plaintext)
  num_moduli = len(moduli)

  result = []
  for d in range(degree):
    X = 0
    # rns_plaintext[d] is [r0, r1, ...] for degree d
    residues = rns_plaintext[d]

    for i in range(num_moduli):
      ri = residues[i]
      qi = moduli[i]
      Mi = Mi_list[i]
      inv = inv_list[i]
      X += (int(ri) % int(qi)) * int(Mi) * int(inv)

    result.append(X % M)
  return result


########################
# CKKS Context Class
########################
class CKKSContext:

  def __init__(self, parameters: dict):
    self.parameters = parameters
    self.degree = parameters["degree"]
    self.num_slots = parameters.get("num_slots", self.degree // 2)
    self.scalingFactor = parameters.get("scalingFactor", 0.0)
    self.output_scale = parameters.get("output_scale", 0.0)
    self.q_towers = parameters["q_towers"]
    self.p_towers = parameters.get("p_towers", [])
    self.p = parameters.get("p", 0)
    self.CKKS_M_FACTOR = parameters.get("CKKS_M_FACTOR", 1)
    self.moduli = self.q_towers

    self.public_key = parameters.get("public_key", None)
    self.secret_key = parameters.get("secret_key", None)
    self.rotation_key = parameters.get("rotation_key", None)
    self.evaluation_key = parameters.get("evaluation_key", None)

  def encrypt(
      self,
      plaintext: ct.Ciphertext,
      v=None,
      e=None,
      ba=None,
      e_ref=None,
      v_ref=None,
  ) -> ct.Ciphertext:
    if self.public_key is None:
      raise ValueError("Public key is not set in the context.")

    element = plaintext.get_element(0)[0]  # Shape: (degree, num_moduli)
    # element is already (degree, moduli), no transpose needed
    encoded_values = element.tolist()

    c_poly = ckks_encrypt(
        plaintext=encoded_values,
        public_key=self.public_key,
        q_towers=self.q_towers,
        noise_scale_degree=self.parameters.get("noise_scale_degree", 1),
        sigma=self.parameters.get("sigma", 3.190000057220458984375),
        v=v,
        e=e,
        ba=ba,
        e_ref=e_ref,
        v_ref=v_ref,
    )

    shapes = {
        "batch": 1,
        "num_elements": 2,
        "num_moduli": len(self.q_towers),
        "degree": self.degree,
        "precision": 32,
    }

    res_ct = ct.Ciphertext(shapes, parameters={"moduli": self.q_towers})

    # c0, c1 are (degree, moduli) naturally now
    c0 = jnp.expand_dims(
        jnp.array(c_poly[0], dtype=jnp.uint64), axis=0
    )  # (1, degree, moduli)
    c1 = jnp.expand_dims(jnp.array(c_poly[1], dtype=jnp.uint64), axis=0)

    res_ct.set_element(0, c0)
    res_ct.set_element(1, c1)

    return res_ct

  def decrypt(self, ciphertext: ct.Ciphertext) -> ct.Ciphertext:
    if self.secret_key is None:
      raise ValueError("Secret key is not set in the context.")
    c_list = [
        ciphertext.ciphertext[0, 0].tolist(),  # c0
        ciphertext.ciphertext[1, 0].tolist(),  # c1
    ]
    num_elems = ciphertext.num_elements
    c_list = []
    for i in range(num_elems):
      c_list.append(ciphertext.ciphertext[0, i].tolist())  # (degree, moduli)
    num_moduli_ct = ciphertext.num_moduli
    current_q_towers = self.q_towers[:num_moduli_ct]
    decrypted_poly_rns = ckks_decrypt(
        ciphertext=c_list,
        private_key=self.secret_key,
        q_towers=current_q_towers,
    )
    shapes = {
        "batch": 1,
        "num_elements": 1,
        "num_moduli": len(current_q_towers),
        "degree": self.degree,
        "precision": 32,
    }

    res_ct = ct.Ciphertext(shapes, parameters={"moduli": current_q_towers})
    # decrypted_poly_rns is (degree, moduli)
    elem = jnp.expand_dims(
        jnp.array(decrypted_poly_rns, dtype=jnp.uint32), axis=0
    )
    res_ct.set_element(0, elem)

    return res_ct

  def encode(self, slots: List[complex], shift: int = 0) -> ct.Ciphertext:
    m = self.degree * 2
    encoded_rns = ckks_encode(
        slots=slots,
        cycl_order=m,
        q_towers=self.q_towers,
        p_towers=self.p_towers,
        scale=self.scalingFactor,
        max_bits_in_word=self.parameters.get("max_bits_in_word", 61),
    )

    shapes = {
        "batch": 1,
        "num_elements": 1,
        "num_moduli": len(self.q_towers),
        "degree": self.degree,
        "precision": 32,
    }

    res_ct = ct.Ciphertext(shapes, parameters={"moduli": self.q_towers})
    # encoded_rns is (degree, moduli)
    elem = jnp.expand_dims(jnp.array(encoded_rns, dtype=jnp.uint64), axis=0)
    res_ct.set_element(0, elem)

    return res_ct

  def decode(
      self, encoded_plaintext: ct.Ciphertext, is_ntt: bool = False
  ) -> jnp.ndarray:
    rns_poly = encoded_plaintext.ciphertext[0, 0].tolist()  # (degree, moduli)
    num_towers = len(encoded_plaintext.moduli)

    if is_ntt:
      # rns_poly is (degree, moduli).
      new_poly = [[0] * num_towers for _ in range(self.degree)]
      for t_id in range(num_towers):
        # get column
        col = [rns_poly[d][t_id] for d in range(self.degree)]

        rev = util.bit_reverse_array(col)
        intt_vals = util.intt_negacyclic_bit_reverse(
            rev,
            self.q_towers[t_id],
            util.root_of_unity(2 * self.degree, self.q_towers[t_id]),
        )

        for d in range(self.degree):
          new_poly[d][t_id] = intt_vals[d]
      rns_poly = new_poly

    plain_combined = _crt_combine_rns_plaintext(
        rns_poly, self.q_towers[:num_towers]
    )

    big_q = 1
    for qi in self.q_towers[:num_towers]:
      big_q *= qi

    res = ckks_decode(
        plaintext=plain_combined,
        scalingFactor=self.output_scale,
        slots=self.num_slots,
        q=big_q,
        p=self.p,
        CKKS_M_FACTOR=self.CKKS_M_FACTOR,
    )

    return jnp.array(res)

  def multiply(
      self, ciphertext1: ct.Ciphertext, ciphertext2: ct.Ciphertext
  ) -> ct.Ciphertext:
    raise NotImplementedError("Mul function not implemented")

  def rotate(self, ciphertext: ct.Ciphertext, shift: int) -> ct.Ciphertext:
    raise NotImplementedError("Rotate function not implemented")

  def rescale(self, ciphertext: ct.Ciphertext) -> ct.Ciphertext:
    raise NotImplementedError("Rescale function not implemented")

  def add(
      self, ciphertext1: ct.Ciphertext, ciphertext2: ct.Ciphertext
  ) -> ct.Ciphertext:
    raise NotImplementedError("Add function not implemented")

  def sub(
      self, ciphertext1: ct.Ciphertext, ciphertext2: ct.Ciphertext
  ) -> ct.Ciphertext:
    raise NotImplementedError("Sub function not implemented")
