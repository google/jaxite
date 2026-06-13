"""A module for operations on test CKKS evaluation kernels including.

- NTT
"""

import functools
import json
import math

import jax
import jax.numpy as jnp
from jaxite.jaxite_word import ntt
import numpy as np

# copybara: from google3.perftools.accelerators.xprof.api.python import xprof_analysis_client
# copybara: from google3.perftools.accelerators.xprof.api.python import xprof_session
from absl.testing import absltest
from absl.testing import parameterized


jax.config.update("jax_enable_x64", True)


TEST_PARAMS = [
    (
        "test_degree_4",
        113,
        1,
        2,
        2,
        [1, 2, 4, 1],
    ),
    (
        "test_degree_8",
        113,
        1,
        2,
        4,
        [1, 2, 4, 1, 3, 5, 6, 8],
    ),
]


TEST_PARAMS_FULL = [
    (
        "test_degree_16_case1",
        1152921504606844513,
        7645792537133126,
        1,
        4,
        4,
        [
            83963236163890886,
            351982916215618729,
            19109541039909111,
            209897461014368538,
            310692812327896771,
            1059668300217304028,
            1050480625312143092,
            318185408722083429,
            691103099785523933,
            925312603743652858,
            943007261004624982,
            1151476381807175556,
            70951954169609321,
            259753300466854865,
            717282208767975565,
            714233289979413983,
        ],
    ),
    (
        "test_degree_16_case2",
        1152921504606844417,
        97466480447807994,
        1,
        4,
        4,
        [
            83963265707149094,
            351982931975631049,
            19109585857500759,
            209897508168723930,
            310692857143987363,
            1059668342128390172,
            1050480648209416628,
            318185432882906469,
            691103122593273821,
            925312641738174490,
            943007287323534262,
            1151476401561067428,
            70951966266556073,
            259753315402657457,
            717282250621563469,
            714233320002290303,
        ],
    ),
]


class CKKSEvalNTTTest(parameterized.TestCase):
  """A base class for running bootstrap tests.

  Example Test Case:
    If use GF(17) and N = 8 (so q=17 and N=8).
    In GF(17), the multiplicative group has order 16.
    Suppose the forward transform used a primitive 8th root of unity.
    For example, we can use omega = 2, since 2^8 mod 17 == 1 and its order is 8.
  """

  def __init__(self, *args, **kwargs):
    super(CKKSEvalNTTTest, self).__init__(*args, **kwargs)
    self.random_key = jax.random.key(0)

  @parameterized.named_parameters(*TEST_PARAMS)
  def test_vanilla_ntt_original_form(
      self,
      q,
      batch,
      r,
      c,
      test_in,
  ):
    print("Test test_vanilla_ntt_original_form")
    n = r * c
    row_count, col_count = r, c  # for example, n = row_count * col_count
    assert row_count * col_count == n
    omega = ntt.nth_primitive_root(n, q)

    ntt_result = ntt.ntt_original_form(test_in, q, omega)
    print("Forward original form NTT of x:", ntt_result)

    x_recovered = ntt.intt_original_form(ntt_result, q, omega)
    print("Recovered x from inverse original form NTT:", x_recovered)

    self.assertEqual(test_in, x_recovered)

  @parameterized.named_parameters(*TEST_PARAMS)
  def test_vanilla_ntt_cooley_tukey(
      self,
      q,
      batch,
      r,
      c,
      test_in,
  ):
    print("Test test_vanilla_ntt_cooley_tukey")

    n = r * c
    row_count, col_count = r, c  # for example, n = row_count * col_count
    assert row_count * col_count == n
    omega = ntt.nth_primitive_root(n, q)

    ntt_result = ntt.ntt_bit_reverse(test_in, q, omega)
    print("Forward bit-reverse NTT of x:", ntt_result)

    x_recovered = ntt.intt_bit_reverse(ntt_result, q, omega)
    print("Recovered x from inverse bit-reverse NTT:", x_recovered)

    self.assertEqual(test_in, x_recovered)

  @parameterized.named_parameters(*TEST_PARAMS)
  def test_vanilla_ntt_4_step(
      self,
      q,
      batch,
      r,
      c,
      test_in,
  ):
    print("Test test_vanilla_ntt_4_step")

    n = r * c
    row_count, col_count = r, c  # for example, n = row_count * col_count
    assert row_count * col_count == n
    omega = ntt.nth_primitive_root(n, q)
    print("omega=", omega)
    ntt_result = ntt.ntt_four_step(test_in, q, omega, row_count, col_count)
    print("Forward 4-step NTT of x:", ntt_result)

    x_recovered = ntt.intt_four_step(ntt_result, q, omega, row_count, col_count)
    # x_recovered = ntt.intt_bit_reverse(ntt_result, q, omega)
    print("Recovered x from inverse 4-step NTT:", x_recovered)

    self.assertEqual(test_in, x_recovered)

  @parameterized.named_parameters(*TEST_PARAMS)
  def test_negacyclic_ntt_4_step(
      self,
      q,
      batch,
      r,
      c,
      test_in,
  ):
    print("Test test_negacyclic_ntt_4_step_degree_4")

    n = r * c
    row_count, col_count = r, c  # for example, n = row_count * col_count
    assert row_count * col_count == n
    omega = ntt.nth_primitive_root(n, q)
    psi = ntt.compute_psi(omega, n, q)

    ntt_result = ntt.ntt_negacyclic(test_in, q, psi, row_count, col_count)
    print("Forward negacyclic NTT of x:", ntt_result)

    x_recovered = ntt.intt_negacyclic(ntt_result, q, psi, row_count, col_count)
    print("Recovered x from inverse negacyclic NTT:", x_recovered)

    self.assertEqual(test_in, x_recovered)

  @parameterized.named_parameters(*TEST_PARAMS)
  def test_barrett_reduction(
      self,
      q,
      batch,
      r,
      c,
      test_in,
  ):
    print("Test test_negacyclic_ntt_4_step_degree_4")
    test_in = [[[69, 95, 147, 139], [7617, 6977, 8472, 7687]]]
    result_ref = [[[69, 95, 34, 26], [46, 84, 110, 3]]]
    s = 2 * math.ceil(math.log2(q))
    m = math.floor(2**s / q)
    print(f"s={s}, m={m}")
    ntt_result = ntt.barret_reduction(
        jnp.array(test_in, dtype=jnp.uint64), q, s, m
    )
    print("Forward negacyclic NTT of x:", ntt_result.tolist())

    self.assertEqual(result_ref, ntt_result.tolist())

  @parameterized.named_parameters(*TEST_PARAMS)
  def test_negacyclic_ntt_tpu_algorithm(
      self,
      q,
      batch,
      r,
      c,
      test_in,
  ):
    """This is testing the 4-step implementation of negacyclic NTT O(N sqrt(N)) complexity.

    The coefficients generation parts have been moved to offline, while online
    performs the
    computation.
    """
    print("Test test_negacyclic_ntt_tpu_algorithm")
    # We use GF(q) and N = r*c (so q=q and N=r*c).
    # In GF(q), the multiplicative group has order q-1.

    n = r * c
    row_count, col_count = r, c  # for example, n = row_count * col_count
    assert row_count * col_count == n

    omega = ntt.nth_primitive_root(n, q)
    psi = ntt.compute_psi(omega, n, q)

    omega_col = pow(omega, c, q)
    omega_row = pow(omega, r, q)
    tf_mat_step1 = jnp.array(
        ntt.gen_twiddle_matrix(r, r, q, omega_col), dtype=int
    )
    coef_step2 = jnp.array(ntt.gen_twiddle_matrix(r, c, q, omega), dtype=int)
    tf_mat_step3 = jnp.array(
        ntt.gen_twiddle_matrix(c, c, q, omega_row), dtype=int
    )

    inv_omega_col = pow(
        omega_col, -1, q
    )  # inverse primitive R-th root for columns
    inv_omega_row = pow(
        omega_row, -1, q
    )  # inverse primitive C-th root for rows
    inv_tf_mat_step3 = jnp.array(
        ntt.gen_twiddle_matrix(r, r, q, inv_omega_col), dtype=int
    )
    inv_coef_step2 = jnp.array(
        ntt.gen_twiddle_matrix_inv(r, c, q, omega), dtype=int
    )
    inv_tf_mat_step1 = jnp.array(
        ntt.gen_twiddle_matrix(c, c, q, inv_omega_row), dtype=int
    )

    np.testing.assert_array_equal(tf_mat_step1.T, tf_mat_step1)
    np.testing.assert_array_equal(tf_mat_step3.T, tf_mat_step3)
    np.testing.assert_array_equal(inv_tf_mat_step1.T, inv_tf_mat_step1)
    np.testing.assert_array_equal(inv_tf_mat_step3.T, inv_tf_mat_step3)

    ntt_result = ntt.ntt_negacyclic_tpu_algorithm(
        test_in, q, psi, r, c, tf_mat_step1, coef_step2, tf_mat_step3
    )
    print("Forward negacyclic NTT of x:", ntt_result)

    x_recovered = ntt.intt_negacyclic_tpu_algorithm(
        ntt_result,
        q,
        psi,
        r,
        c,
        inv_tf_mat_step1,
        inv_coef_step2,
        inv_tf_mat_step3,
    )
    print("Recovered x from inverse negacyclic NTT:", x_recovered)

    self.assertEqual(test_in, x_recovered)

  @parameterized.named_parameters(*TEST_PARAMS)
  def test_ntt_layout_invariant_batch(
      self,
      q,
      batch,
      r,
      c,
      test_in,
  ):
    print("Test ntt_layout_invariant_batch")
    s = 2 * math.ceil(math.log2(q))
    m = math.floor(2**s / q)
    n = r * c
    omega = ntt.nth_primitive_root(n, q)
    psi = ntt.compute_psi(omega, n, q)
    # psi_inv = pow(psi, -1, q)

    omega_col = pow(omega, c, q)
    omega_row = pow(omega, r, q)
    tf_mat_step1 = jnp.array(
        ntt.gen_twiddle_matrix(r, r, q, omega_col), dtype=int
    )
    coef_step2 = jnp.array(ntt.gen_twiddle_matrix(r, c, q, omega), dtype=int)
    tf_mat_step3 = jnp.array(
        ntt.gen_twiddle_matrix(c, c, q, omega_row), dtype=int
    )

    np.testing.assert_array_equal(tf_mat_step1.T, tf_mat_step1)
    np.testing.assert_array_equal(tf_mat_step3.T, tf_mat_step3)

    tf_mat_bat_step1 = ntt.hpmatmul_offline_compile_bat(
        tf_mat_step1.astype(jnp.uint32), q
    )
    coef_step2 = coef_step2.astype(jnp.uint32)
    tf_mat_bat_step3 = ntt.hpmatmul_offline_compile_bat(
        tf_mat_step3.astype(jnp.uint32), q
    )

    tf_step1 = tf_mat_bat_step1.astype(jnp.uint8)
    tf_step3 = tf_mat_bat_step3.astype(jnp.uint8)
    assert tf_step1.shape == (r, r, 4, 4)
    assert coef_step2.shape == (r, c)
    assert tf_step3.shape == (c, c, 4, 4)

    if c == r:
      np.testing.assert_array_equal(tf_mat_step1, tf_mat_step3)
    np.testing.assert_array_equal(tf_mat_step1.T, tf_mat_step1)
    np.testing.assert_array_equal(tf_mat_step3.T, tf_mat_step3)

    ntt_result = ntt.ntt_negacyclic_tpu_algorithm(
        test_in, q, psi, r, c, tf_mat_step1, coef_step2, tf_mat_step3
    )

    dut = functools.partial(ntt.ntt_layout_invariant_batch, q=q, s=s, m=m)
    test_in_twisted = jnp.array(
        [(test_in[i] * pow(psi, i, q)) % q for i in range(n)], jnp.uint32
    )
    test_in_twisted = test_in_twisted.reshape(batch, r, c)
    result = dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    result = np.array(result.T).flatten().tolist()
    print([result[i] for i in range(n)])

    print(f"input={test_in_twisted}, after NTT = {result}")

    self.assertEqual(ntt_result, result)

    jax.block_until_ready(
        dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    )

    # copybara: session = xprof_session.XprofSession()
    # copybara: session.start_session()
    jax.block_until_ready(
        dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    )
    jax.block_until_ready(
        dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    )
    jax.block_until_ready(
        dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    )
    jax.block_until_ready(
        dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    )
    jax.block_until_ready(
        dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    )
    jax.block_until_ready(
        dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    )
    jax.block_until_ready(
        dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    )
    jax.block_until_ready(
        dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    )
    # copybara: session_id = session.end_session_and_get_session_id()
    print(f"session_id: http://xprof/?session_id={session_id}")
    # copybara: client = xprof_analysis_client.XprofAnalysisClient()
    trace = client.get_profile_data("trace_viewer.json", session_id)
    jtrace = json.loads(trace[1])
    results = []
    for e in jtrace["traceEvents"]:
      if "ntt_layout_invariant_batch" in e["name"]:
        results.append(e["dur"])
    print(jnp.mean(jnp.array(results[:8])), "us")

  @parameterized.named_parameters(*TEST_PARAMS_FULL)
  def test_ntt_intt_layout_invariant_batch(
      self,
      q,
      psi,
      batch,
      r,
      c,
      test_in,
  ):
    if math.log2(q) > 32:
      print(
          "Skip this test as we don't support modulus > 32, because numpy as"
          " max precision as 64"
      )
      return
    s = 2 * math.ceil(math.log2(q))
    m = math.floor(2**s / q)
    n = r * c
    if psi is not None:
      omega = (psi**2) % q
    else:
      omega = ntt.nth_primitive_root(n, q)

    omega_col = pow(omega, c, q)
    omega_row = pow(omega, r, q)
    tf_mat_step1 = jnp.array(
        ntt.gen_twiddle_matrix(r, r, q, omega_col), dtype=int
    )
    coef_step2 = jnp.array(ntt.gen_twiddle_matrix(r, c, q, omega), dtype=int)
    tf_mat_step3 = jnp.array(
        ntt.gen_twiddle_matrix(c, c, q, omega_row), dtype=int
    )

    inv_omega_col = pow(omega_col, -1, q)
    # inverse primitive R-th root for columns
    inv_omega_row = pow(omega_row, -1, q)
    # inverse primitive C-th root for rows
    inv_tf_mat_step3 = jnp.array(
        ntt.gen_twiddle_matrix(r, r, q, inv_omega_col), dtype=int
    )
    # intt needs to scale the corresponding coefficients.
    inv_c = pow(c, -1, q)
    inv_r = pow(r, -1, q)
    inv_coef_step2_ori = jnp.array(
        ntt.gen_twiddle_matrix_inv(r, c, q, omega), dtype=int
    )
    inv_coef_step2 = inv_c * inv_coef_step2_ori
    inv_tf_mat_step1 = jnp.array(
        ntt.gen_twiddle_matrix(c, c, q, inv_omega_row), dtype=int
    )

    if c == r:
      np.testing.assert_array_equal(tf_mat_step1, tf_mat_step3)
    np.testing.assert_array_equal(tf_mat_step1.T, tf_mat_step1)
    np.testing.assert_array_equal(tf_mat_step3.T, tf_mat_step3)
    np.testing.assert_array_equal(inv_tf_mat_step1.T, inv_tf_mat_step1)
    np.testing.assert_array_equal(inv_tf_mat_step3.T, inv_tf_mat_step3)

    tf_step1 = ntt.hpmatmul_offline_compile_bat(
        tf_mat_step1.astype(jnp.uint32), q
    ).astype(jnp.uint8)
    coef_step2 = coef_step2.astype(jnp.uint32)
    tf_step3 = ntt.hpmatmul_offline_compile_bat(
        tf_mat_step3.astype(jnp.uint32), q
    ).astype(jnp.uint8)
    inv_tf_step1 = ntt.hpmatmul_offline_compile_bat(
        inv_tf_mat_step1.astype(jnp.uint32), q
    ).astype(jnp.uint8)
    inv_coef_step2 = inv_coef_step2.astype(jnp.uint32)
    inv_tf_step3 = ntt.hpmatmul_offline_compile_bat(
        inv_tf_mat_step3.astype(jnp.uint32), q
    ).astype(jnp.uint8)

    assert tf_step1.shape == (r, r, 4, 4)
    assert coef_step2.shape == (r, c)
    assert tf_step3.shape == (c, c, 4, 4)
    assert inv_tf_step1.shape == (c, c, 4, 4)
    assert inv_coef_step2.shape == (r, c)
    assert inv_tf_step3.shape == (r, r, 4, 4)

    ntt_result = ntt.ntt_negacyclic_tpu_algorithm(
        test_in, q, psi, r, c, tf_mat_step1, coef_step2, tf_mat_step3
    )

    dut = functools.partial(ntt.ntt_layout_invariant_batch, q=q, s=s, m=m)
    test_in_twisted = jnp.array(
        [(test_in[i] * pow(psi, i, q)) % q for i in range(n)], jnp.uint32
    )
    test_in_twisted = test_in_twisted.reshape(batch, r, c)
    result_jax = dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    result = np.array(result_jax.T).flatten().tolist()
    self.assertEqual(ntt_result, result)

    # INTT 
    x_ref = ntt.intt_negacyclic_tpu_algorithm(
        ntt_result,
        q,
        psi,
        r,
        c,
        inv_tf_mat_step1,
        inv_coef_step2_ori,
        inv_tf_mat_step3,
    )
    self.assertEqual(x_ref, test_in)

    x_untwisted = ntt.intt_layout_invariant_batch(
        result_jax,
        inv_tf_step1,
        inv_coef_step2,
        inv_tf_step3,
        inv_r,
        q,
        s,
        m,
    )
    psi_inv = pow(psi, -1, q)
    x_untwisted = x_untwisted.flatten().tolist()
    x_recovered = [
        (x_untwisted[i] * pow(psi_inv, i, q)) % q
        for i in range(len(x_untwisted))
    ]
    self.assertEqual(x_recovered, test_in)

    jax.block_until_ready(
        dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    )

    # copybara: session = xprof_session.XprofSession()
    # copybara: session.start_session()
    jax.block_until_ready(
        dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    )
    jax.block_until_ready(
        dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    )
    jax.block_until_ready(
        dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    )
    jax.block_until_ready(
        dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    )
    jax.block_until_ready(
        dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    )
    jax.block_until_ready(
        dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    )
    jax.block_until_ready(
        dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    )
    jax.block_until_ready(
        dut(test_in_twisted, tf_step1, coef_step2, tf_step3)
    )
    # copybara: session_id = session.end_session_and_get_session_id()
    print(f"session_id: http://xprof/?session_id={session_id}")
    # copybara: client = xprof_analysis_client.XprofAnalysisClient()
    trace = client.get_profile_data("trace_viewer.json", session_id)
    jtrace = json.loads(trace[1])
    results = []
    for e in jtrace["traceEvents"]:
      if "ntt_layout_invariant_batch" in e["name"]:
        results.append(e["dur"])
    print(jnp.mean(jnp.array(results[:8])), "us")


if __name__ == "__main__":
  absltest.main()
