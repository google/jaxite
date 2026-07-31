# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for SHIP half-bootstrapping implementation."""

import math

import hypothesis
from hypothesis import strategies as st
import jax
from jaxite.jaxite_ckks import bootstrapping
from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import encrypt
from jaxite.jaxite_ckks import key_gen
from jaxite.jaxite_ckks import math as ckks_math
from jaxite.jaxite_ckks import ntt_cpu
from jaxite.jaxite_ckks import random
from jaxite.jaxite_ckks import types
import numpy as np

from absl.testing import absltest

jax.config.update("jax_enable_x64", True)

# Test primes matching standard CKKS parameters
TEST_PRIMES = [
    1073742881,
    1073742721,
    1073742113,
    1073741953,
    1073741857,
    1073741441,
]


def encode_coeffs(coeffs, degree, scale, moduli):
  """Encodes coefficients directly to Plaintext (no inverse FFT)."""
  coeffs_arr = np.array(coeffs)
  if len(coeffs_arr) < degree:
    coeffs_arr = np.pad(coeffs_arr, (0, degree - len(coeffs_arr)))
  scaled_coeffs = np.round(coeffs_arr * scale)
  moduli_arr = np.array(moduli, dtype=np.uint64)
  poly = (scaled_coeffs[:, None] % moduli_arr[None, :]).astype(np.uint64)
  poly_ntt = ntt_cpu.ntt_negacyclic_poly(poly, moduli)
  return types.Plaintext(
      data=jax.numpy.array(poly_ntt, dtype=jax.numpy.uint32),
      moduli=jax.numpy.array(moduli, dtype=jax.numpy.uint32),
  )


def get_phase_shifts(degree, j):
  """Helper to calculate phase shift for non-zero key index j."""
  slots = degree // 2
  indices = np.arange(slots)
  return np.exp(2j * np.pi * indices * j / degree)


class BootstrappingTest(absltest.TestCase):

  def test_half_bootstrap_hw1(self):
    degree = 16
    q_limbs = TEST_PRIMES[:-1]
    p_limbs = TEST_PRIMES[-1:]
    scale = 2**30
    theta = 2
    dnum = 1
    gamma = 1.0
    num_slots = degree // 2

    test_random_source = random.ZeroNoiseRandomSource()

    # 1. Generate level 0 secret key and public key
    s_coeffs = np.zeros((degree, 1), dtype=np.int64)
    s_coeffs[0, 0] = 1
    sk_level0_data = ntt_cpu.ntt_negacyclic_poly(s_coeffs, [q_limbs[0]])
    sk = types.SecretKey(
        data=np.array(sk_level0_data, dtype=np.uint32),
        moduli=np.array([q_limbs[0]], dtype=np.uint32),
    )

    pk, _ = key_gen.keygen(
        degree,
        [q_limbs[0]],
        random_source=test_random_source,
        sk=sk,
    )

    # 2. Extend secret key and precompute constants
    sk = key_gen.extend_secret_key(sk, q_limbs)
    ship_kernel = bootstrapping.SHIP()
    ship_kernel.precompute_constants(
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        degree=degree,
        dnum=dnum,
        sk=sk,
        theta=theta,
        random_source=test_random_source,
    )

    # The input must be a coefficient-domain ciphertext at level 0 with scale =
    # q0, encoding floor(q0·μᵢ/γ) in polynomial coeff i.
    mu = [0.1] * num_slots  # this is the coeffs of the message
    plain_mu = encode_coeffs(mu, degree, q_limbs[0] / gamma, [q_limbs[0]])

    # 3. Encrypt input message
    encryptor = encrypt.Encrypt(pk)
    ct_in = encryptor.encrypt(plain_mu, random_source=test_random_source)

    # 4. Execute half_bootstrap
    ct_out = ship_kernel.half_bootstrap(
        ct_in=ct_in,
        theta=theta,
        scale=scale,
        gamma=gamma,
    )

    # 5. Decrypt and decode using input scale
    decryptor = encrypt.Decrypt(sk)
    plain_out = decryptor.decrypt(ct_out)

    # by encoding .1, we are relatively close to the approx sin(x) = x
    decoder_out = encode.Decode(scale, num_slots)
    actual_slots = decoder_out.decode(plain_out)

    expected = [
        gamma / (2 * math.pi) * math.sin(2 * math.pi * m / gamma) for m in mu
    ]
    np.testing.assert_allclose(
        actual_slots, expected, rtol=1e-3, atol=max(1e-3, 1000.0 / scale)
    )

  def test_half_bootstrap_hw1_nonzero(self):
    degree = 16
    q_limbs = TEST_PRIMES[:-1]
    p_limbs = TEST_PRIMES[-1:]
    scale = 2**30
    theta = 2
    dnum = 1
    gamma = 1.0
    num_slots = degree // 2

    test_random_source = random.ZeroNoiseRandomSource()

    # 1. Generate level 0 secret key and public key
    s_coeffs = np.zeros((degree, 1), dtype=np.int64)
    s_coeffs[2, 0] = 1
    sk_level0_data = ntt_cpu.ntt_negacyclic_poly(s_coeffs, [q_limbs[0]])
    sk = types.SecretKey(
        data=np.array(sk_level0_data, dtype=np.uint32),
        moduli=np.array([q_limbs[0]], dtype=np.uint32),
    )

    pk, _ = key_gen.keygen(
        degree,
        [q_limbs[0]],
        random_source=test_random_source,
        sk=sk,
    )

    # 2. Extend secret key and precompute constants
    sk = key_gen.extend_secret_key(sk, q_limbs)
    ship_kernel = bootstrapping.SHIP()
    ship_kernel.precompute_constants(
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        degree=degree,
        dnum=dnum,
        sk=sk,
        theta=theta,
        random_source=test_random_source,
    )

    # The input must be a coefficient-domain ciphertext at level 0 with scale =
    # q0, encoding floor(q0·μᵢ/γ) in polynomial coeff i.
    mu = [0.1 * i for i in range(num_slots)]  # .1, .2, .3, .4, .5, .6, .7, .8
    plain_mu = encode_coeffs(mu, degree, q_limbs[0] / gamma, [q_limbs[0]])

    # 3. Encrypt input message
    encryptor = encrypt.Encrypt(pk)
    ct_in = encryptor.encrypt(plain_mu, random_source=test_random_source)

    # 4. Execute half_bootstrap
    ct_out = ship_kernel.half_bootstrap(
        ct_in=ct_in,
        theta=theta,
        scale=scale,
        gamma=gamma,
    )

    # 5. Decrypt and decode using input scale
    decryptor = encrypt.Decrypt(sk)
    plain_out = decryptor.decrypt(ct_out)

    decoder_out = encode.Decode(scale, num_slots)
    actual_slots = decoder_out.decode(plain_out)

    expected = [
        gamma / (2 * math.pi) * math.sin(2 * math.pi * m / gamma) for m in mu
    ]
    np.testing.assert_allclose(
        actual_slots, expected, rtol=1e-3, atol=max(1e-3, 1000.0 / scale)
    )

  def test_half_bootstrap_hw1_second_half(self):
    degree = 16
    q_limbs = TEST_PRIMES[:-1]
    p_limbs = TEST_PRIMES[-1:]
    scale = 2**30
    theta = 2
    dnum = 1
    gamma = 1.0
    num_slots = degree // 2

    test_random_source = random.ZeroNoiseRandomSource()

    # index 9 is in second half (9 >= 8)
    s_coeffs = np.zeros((degree, 1), dtype=np.int64)
    s_coeffs[9, 0] = 1
    sk_level0_data = ntt_cpu.ntt_negacyclic_poly(s_coeffs, [q_limbs[0]])
    sk = types.SecretKey(
        data=np.array(sk_level0_data, dtype=np.uint32),
        moduli=np.array([q_limbs[0]], dtype=np.uint32),
    )

    pk, _ = key_gen.keygen(
        degree,
        [q_limbs[0]],
        random_source=test_random_source,
        sk=sk,
    )

    sk = key_gen.extend_secret_key(sk, q_limbs)
    ship_kernel = bootstrapping.SHIP()
    ship_kernel.precompute_constants(
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        degree=degree,
        dnum=dnum,
        sk=sk,
        theta=theta,
        random_source=test_random_source,
    )

    mu = [0.1 * i for i in range(num_slots)]
    plain_mu = encode_coeffs(mu, degree, q_limbs[0] / gamma, [q_limbs[0]])

    encryptor = encrypt.Encrypt(pk)
    ct_in = encryptor.encrypt(plain_mu, random_source=test_random_source)

    ct_out = ship_kernel.half_bootstrap(
        ct_in=ct_in,
        theta=theta,
        scale=scale,
        gamma=gamma,
    )

    decryptor = encrypt.Decrypt(sk)
    plain_out = decryptor.decrypt(ct_out)

    decoder_out = encode.Decode(scale, num_slots)
    actual_slots = decoder_out.decode(plain_out)

    expected = [
        gamma / (2 * math.pi) * math.sin(2 * math.pi * m / gamma) for m in mu
    ]
    np.testing.assert_allclose(
        actual_slots, expected, rtol=1e-3, atol=max(1e-3, 1000.0 / scale)
    )

  def test_half_bootstrap_hw2(self):
    degree = 16
    q_limbs = TEST_PRIMES[:-1]
    p_limbs = TEST_PRIMES[-1:]
    scale = 2**30
    theta = 2
    dnum = 1
    gamma = 1.0
    num_slots = degree // 2

    test_random_source = random.ZeroNoiseRandomSource()

    # 1. Generate level 0 secret key and public key
    s_coeffs = np.zeros((degree, 1), dtype=np.int64)
    s_coeffs[2, 0] = 1
    s_coeffs[4, 0] = 1
    sk_level0_data = ntt_cpu.ntt_negacyclic_poly(s_coeffs, [q_limbs[0]])
    sk = types.SecretKey(
        data=np.array(sk_level0_data, dtype=np.uint32),
        moduli=np.array([q_limbs[0]], dtype=np.uint32),
    )

    pk, _ = key_gen.keygen(
        degree,
        [q_limbs[0]],
        random_source=test_random_source,
        sk=sk,
    )

    # 2. Extend secret key and precompute constants
    sk = key_gen.extend_secret_key(sk, q_limbs)
    ship_kernel = bootstrapping.SHIP()
    ship_kernel.precompute_constants(
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        degree=degree,
        dnum=dnum,
        sk=sk,
        theta=theta,
        random_source=test_random_source,
    )

    # The input must be a coefficient-domain ciphertext at level 0 with scale =
    # q0, encoding floor(q0·μᵢ/γ) in polynomial coeff i.
    mu = [0.1 * i for i in range(num_slots)]  # .1, .2, .3, .4, .5, .6, .7, .8
    plain_mu = encode_coeffs(mu, degree, q_limbs[0] / gamma, [q_limbs[0]])

    # 3. Encrypt input message
    encryptor = encrypt.Encrypt(pk)
    ct_in = encryptor.encrypt(plain_mu, random_source=test_random_source)

    # 4. Execute half_bootstrap
    ct_out = ship_kernel.half_bootstrap(
        ct_in=ct_in,
        theta=theta,
        scale=scale,
        gamma=gamma,
    )

    # 5. Decrypt and decode using input scale
    decryptor = encrypt.Decrypt(sk)
    plain_out = decryptor.decrypt(ct_out)

    decoder_out = encode.Decode(scale, num_slots)
    actual_slots = decoder_out.decode(plain_out)

    expected = [
        gamma / (2 * math.pi) * math.sin(2 * math.pi * m / gamma) for m in mu
    ]
    np.testing.assert_allclose(
        actual_slots, expected, rtol=1e-3, atol=max(1e-3, 1000.0 / scale)
    )

  def test_half_bootstrap_hw2_second_half(self):
    degree = 16
    q_limbs = TEST_PRIMES[:-1]
    p_limbs = TEST_PRIMES[-1:]
    scale = 2**30
    theta = 2
    dnum = 5
    gamma = 1.0
    num_slots = degree // 2

    test_random_source = random.ZeroNoiseRandomSource()

    # 1. Generate level 0 secret key and public key
    s_coeffs = np.zeros((degree, 1), dtype=np.int64)
    s_coeffs[2, 0] = 1
    s_coeffs[9, 0] = 1
    sk_level0_data = ntt_cpu.ntt_negacyclic_poly(s_coeffs, [q_limbs[0]])
    sk = types.SecretKey(
        data=np.array(sk_level0_data, dtype=np.uint32),
        moduli=np.array([q_limbs[0]], dtype=np.uint32),
    )

    pk, _ = key_gen.keygen(
        degree,
        [q_limbs[0]],
        random_source=test_random_source,
        sk=sk,
    )

    # 2. Extend secret key and precompute constants
    sk = key_gen.extend_secret_key(sk, q_limbs)
    ship_kernel = bootstrapping.SHIP()
    ship_kernel.precompute_constants(
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        degree=degree,
        dnum=dnum,
        sk=sk,
        theta=theta,
        random_source=test_random_source,
    )

    mu = [0.1 * i for i in range(num_slots)]
    plain_mu = encode_coeffs(mu, degree, q_limbs[0] / gamma, [q_limbs[0]])

    encryptor = encrypt.Encrypt(pk)
    ct_in = encryptor.encrypt(plain_mu, random_source=test_random_source)

    ct_out = ship_kernel.half_bootstrap(
        ct_in=ct_in,
        theta=theta,
        scale=scale,
        gamma=gamma,
    )

    decryptor = encrypt.Decrypt(sk)
    plain_out = decryptor.decrypt(ct_out)

    decoder_out = encode.Decode(scale, num_slots)
    actual_slots = decoder_out.decode(plain_out)

    expected = [
        gamma / (2 * math.pi) * math.sin(2 * math.pi * m / gamma) for m in mu
    ]
    np.testing.assert_allclose(
        actual_slots, expected, rtol=1e-3, atol=max(1e-3, 1000.0 / scale)
    )

  def test_half_bootstrap_hw3(self):
    degree = 16
    q_limbs = TEST_PRIMES[:-1]
    p_limbs = TEST_PRIMES[-1:]
    scale = 2**30
    theta = 2
    dnum = 1
    gamma = 1.0
    num_slots = degree // 2

    test_random_source = random.ZeroNoiseRandomSource()

    # 1. Generate level 0 secret key and public key
    s_coeffs = np.zeros((degree, 1), dtype=np.int64)
    s_coeffs[2, 0] = 1
    s_coeffs[4, 0] = 1
    s_coeffs[6, 0] = 1
    sk_level0_data = ntt_cpu.ntt_negacyclic_poly(s_coeffs, [q_limbs[0]])
    sk = types.SecretKey(
        data=np.array(sk_level0_data, dtype=np.uint32),
        moduli=np.array([q_limbs[0]], dtype=np.uint32),
    )

    pk, _ = key_gen.keygen(
        degree,
        [q_limbs[0]],
        random_source=test_random_source,
        sk=sk,
    )

    # 2. Extend secret key and precompute constants
    sk = key_gen.extend_secret_key(sk, q_limbs)
    ship_kernel = bootstrapping.SHIP()
    ship_kernel.precompute_constants(
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        degree=degree,
        dnum=dnum,
        sk=sk,
        theta=theta,
        random_source=test_random_source,
    )

    # The input must be a coefficient-domain ciphertext at level 0 with scale =
    # q0, encoding floor(q0·μᵢ/γ) in polynomial coeff i.
    mu = [0.1 * i for i in range(num_slots)]  # .1, .2, .3, .4, .5, .6, .7, .8
    plain_mu = encode_coeffs(mu, degree, q_limbs[0] / gamma, [q_limbs[0]])

    # 3. Encrypt input message
    encryptor = encrypt.Encrypt(pk)
    ct_in = encryptor.encrypt(plain_mu, random_source=test_random_source)

    # 4. Execute half_bootstrap
    ct_out = ship_kernel.half_bootstrap(
        ct_in=ct_in,
        theta=theta,
        scale=scale,
        gamma=gamma,
    )

    # 5. Decrypt and decode using input scale
    decryptor = encrypt.Decrypt(sk)
    plain_out = decryptor.decrypt(ct_out)

    decoder_out = encode.Decode(scale, num_slots)
    actual_slots = decoder_out.decode(plain_out)

    expected = [
        gamma / (2 * math.pi) * math.sin(2 * math.pi * m / gamma) for m in mu
    ]
    np.testing.assert_allclose(
        actual_slots, expected, rtol=1e-3, atol=max(1e-3, 1000.0 / scale)
    )

  def test_half_bootstrap_ll13(self):
    degree = 1024
    q_limbs = [16957441, 17006593, 17252353, 17367041, 17416193, 17448961]
    p_limbs = [134250497]
    scale = 2**24
    theta = 6
    dnum = 8
    gamma = 1.0
    num_slots = degree // 2

    test_random_source = random.ZeroNoiseRandomSource()

    # 1. Generate level 0 secret key (Hamming weight 5) and public key
    s_coeffs = np.zeros((degree, 1), dtype=np.int64)
    s_coeffs[2, 0] = 1
    s_coeffs[8, 0] = 1
    s_coeffs[149, 0] = 1
    s_coeffs[646, 0] = 1
    s_coeffs[1023, 0] = 1
    sk_level0_data = ntt_cpu.ntt_negacyclic_poly(s_coeffs, [q_limbs[0]])
    sk = types.SecretKey(
        data=np.array(sk_level0_data, dtype=np.uint32),
        moduli=np.array([q_limbs[0]], dtype=np.uint32),
    )

    pk, _ = key_gen.keygen(
        degree,
        [q_limbs[0]],
        random_source=test_random_source,
        sk=sk,
    )

    # 2. Extend secret key and precompute constants
    sk = key_gen.extend_secret_key(sk, q_limbs)
    ship_kernel = bootstrapping.SHIP()
    ship_kernel.precompute_constants(
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        degree=degree,
        dnum=dnum,
        sk=sk,
        theta=theta,
        random_source=test_random_source,
    )

    # Define input message coefficients mu
    mu = [0.1 * (i % 8) for i in range(num_slots)]
    plain_mu = encode_coeffs(mu, degree, q_limbs[0] / gamma, [q_limbs[0]])

    # 3. Encrypt input message
    encryptor = encrypt.Encrypt(pk)
    ct_in = encryptor.encrypt(plain_mu, random_source=test_random_source)

    # 4. Execute half_bootstrap
    ct_out = ship_kernel.half_bootstrap(
        ct_in=ct_in,
        theta=theta,
        scale=scale,
        gamma=gamma,
    )

    # 5. Decrypt and decode using input scale
    decryptor = encrypt.Decrypt(sk)
    plain_out = decryptor.decrypt(ct_out)

    decoder_out = encode.Decode(scale, num_slots)
    actual_slots = decoder_out.decode(plain_out)

    expected = [
        gamma / (2 * math.pi) * math.sin(2 * math.pi * m / gamma) for m in mu
    ]
    # LL13 has relatively small primes, so using appropriate error tolerance
    np.testing.assert_allclose(actual_slots, expected, rtol=1e-3, atol=1e-3)


# Strategy to generate (degree, scale, scale_bits, dnum, h, indices) for
# hypothesis tests.
@st.composite
def bootstrapping_params_strategy(draw):
  degree = draw(st.sampled_from([16, 32, 64]))
  # scale_bits determines the size of scale and primes
  scale_bits = draw(st.integers(min_value=15, max_value=29))
  scale = 2**scale_bits
  # dnum must divide len(q_limbs) = 5. So dnum can be 1 or 5.
  dnum = draw(st.sampled_from([1, 5]))
  # Hamming weight h. Must be <= degree. Since min(degree)=16, h in [1, 5] is
  # safe.
  h = draw(st.integers(min_value=1, max_value=5))
  # Generate h unique indices in [0, degree-1]
  indices = draw(
      st.lists(
          st.integers(min_value=0, max_value=degree - 1),
          min_size=h,
          max_size=h,
          unique=True,
      )
  )
  return degree, scale, scale_bits, dnum, h, indices


class BootstrappingHypothesisTest(absltest.TestCase):

  @hypothesis.settings(max_examples=10, deadline=None)
  @hypothesis.given(bootstrapping_params_strategy())
  def test_half_bootstrap_hypothesis(self, params):
    degree, scale, scale_bits, dnum, _, indices = params

    # Dynamically generate primes matching the scale
    # We need 5 q limbs of size scale_bits + 1, and 1 p limb of size
    # scale_bits + 2
    spec = [
        ("q", scale_bits + 1, 5),
        ("p", scale_bits + 2, 1),
    ]
    primes_dict = ckks_math.find_distinct_primes(degree, spec)
    q_limbs = primes_dict["q"]
    p_limbs = primes_dict["p"]

    theta = 2
    gamma = 1.0
    num_slots = degree // 2

    test_random_source = random.ZeroNoiseRandomSource()

    # 1. Generate level 0 secret key with Hamming weight h
    s_coeffs = np.zeros((degree, 1), dtype=np.int64)
    for idx in indices:
      s_coeffs[idx, 0] = 1

    sk_level0_data = ntt_cpu.ntt_negacyclic_poly(s_coeffs, [q_limbs[0]])
    sk = types.SecretKey(
        data=np.array(sk_level0_data, dtype=np.uint32),
        moduli=np.array([q_limbs[0]], dtype=np.uint32),
    )

    pk, _ = key_gen.keygen(
        degree,
        [q_limbs[0]],
        random_source=test_random_source,
        sk=sk,
    )

    # 2. Extend secret key and precompute constants
    sk = key_gen.extend_secret_key(sk, q_limbs)
    ship_kernel = bootstrapping.SHIP()
    ship_kernel.precompute_constants(
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        degree=degree,
        dnum=dnum,
        sk=sk,
        theta=theta,
        random_source=test_random_source,
    )

    # Generate a message with small values to avoid large approximation errors
    mu = [0.05 * (i % 8) for i in range(num_slots)]
    plain_mu = encode_coeffs(mu, degree, q_limbs[0] / gamma, [q_limbs[0]])

    # 3. Encrypt input message
    encryptor = encrypt.Encrypt(pk)
    ct_in = encryptor.encrypt(plain_mu, random_source=test_random_source)

    # 4. Execute half_bootstrap
    ct_out = ship_kernel.half_bootstrap(
        ct_in=ct_in,
        theta=theta,
        scale=scale,
        gamma=gamma,
    )

    # 5. Decrypt and decode
    decryptor = encrypt.Decrypt(sk)
    plain_out = decryptor.decrypt(ct_out)

    decoder_out = encode.Decode(scale, num_slots)
    actual_slots = decoder_out.decode(plain_out)

    expected = [
        gamma / (2 * math.pi) * math.sin(2 * math.pi * m / gamma) for m in mu
    ]

    # Dynamic tolerance based on scale to allow more error for smaller scales
    atol = max(1e-3, 1000.0 / scale)
    rtol = 1e-3
    np.testing.assert_allclose(actual_slots, expected, rtol=rtol, atol=atol)


if __name__ == "__main__":
  absltest.main()
