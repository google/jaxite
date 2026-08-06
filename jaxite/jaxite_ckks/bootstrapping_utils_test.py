"""Tests for SHIP CKKS bootstrapping utilities."""

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import bootstrapping_utils as boot_utils
from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import ntt
from jaxite.jaxite_ckks import types
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

jax.config.update('jax_enable_x64', True)

TEST_PRIMES = [
    1073184769,
    1073479681,
    1073577985,
    1073676289,
]


class BootstrappingUtilsTest(parameterized.TestCase):

  @parameterized.parameters(4, 8, 16, 64, 256)
  def test_rot_group_properties(self, degree):
    m = degree * 2
    nh = degree // 2
    r = boot_utils.rot_group(m, nh, g=5)

    self.assertLen(r, nh)
    self.assertEqual(r[0], 1)
    # All elements must be odd and coprime to m (which is power of 2)
    for val in r:
      self.assertEqual(val % 2, 1)

    # All elements must be unique
    self.assertLen(set(r), nh)

  @parameterized.parameters(2, 4, 8, 16, 128, 512)
  def test_bit_reverse_permutation(self, size):
    indices = boot_utils.get_bit_reverse_perm_indices(size)

    # Must be a valid permutation of range(size)
    self.assertLen(indices, size)
    self.assertEqual(set(indices.tolist()), set(range(size)))

    # Verify bit reversal manually for all elements
    n = int(np.log2(size))
    for i in range(size):
      expected_rev = int(format(i, f'0{n}b')[::-1], 2)
      self.assertEqual(indices[i], expected_rev)

  @parameterized.parameters(8, 16, 64, 256, 1024)
  def test_fft_special_inv_mathematical_correctness(self, degree):
    # The specialized inverse FFT maps slots s to y such that the polynomial
    # coefficients generated from y, when evaluated at the roots 5^j,
    # yield the original slots.
    num_slots = degree // 2

    # Generate random complex slots
    np.random.seed(degree)
    slots = np.random.randn(num_slots) + 1j * np.random.randn(num_slots)
    slots_jax = jnp.array(slots, dtype=jnp.complex128)

    # Run inverse FFT
    fft_kernel = boot_utils.SpecialInverseFFT()
    fft_kernel.precompute_constants(degree)
    y = fft_kernel(slots_jax)
    self.assertEqual(y.shape, (num_slots,))

    # Apply reference forward FFT in-place
    y_np = np.array(y)
    encode.fft_special(y_np, degree * 2)

    # Verify that the forward FFT restores the original slots
    np.testing.assert_allclose(y_np, slots, rtol=1e-4, atol=1e-4)

  @parameterized.parameters(8, 16, 64, 128)
  def test_fft_special_inv_jax_jit_compilation(self, degree):
    num_slots = degree // 2
    slots = jnp.ones(num_slots, dtype=jnp.complex128)

    # Verify that it compiles under JIT and runs on device/TPU without warnings
    fft_kernel = boot_utils.SpecialInverseFFT()
    fft_kernel.precompute_constants(degree)

    @jax.jit
    def run_fft(slots, kernel):
      return kernel(slots)

    res = run_fft(slots, fft_kernel)
    self.assertEqual(res.shape, (num_slots,))

  @parameterized.parameters(16, 32, 64)
  def test_encode_jax_vs_cpu_reference(self, degree):
    num_slots = degree // 2
    scale = 2**20
    moduli = np.array(TEST_PRIMES[:2], dtype=np.uint32)

    # Generate random slots
    np.random.seed(degree)
    slots = np.random.randn(num_slots) + 1j * np.random.randn(num_slots)
    slots_jax = jnp.array(slots, dtype=jnp.complex128)

    # NTT kernel
    ntt_kernel = ntt.NTTBarrett()
    ntt_kernel.precompute_constants(moduli.tolist(), 2, degree // 2)

    # FFT kernel
    fft_kernel = boot_utils.SpecialInverseFFT()
    fft_kernel.precompute_constants(degree)

    # Trace CPU intermediate steps
    nh = degree // 2
    y_ref = np.array(slots.tolist(), dtype=complex)
    if len(y_ref) < nh:
      y_ref = np.pad(y_ref, (0, nh - len(y_ref)))
    encode.fft_special_inv(y_ref, degree * 2)
    coeffs_ref = np.concatenate([y_ref.real, y_ref.imag])
    scaled_coeffs_ref = np.round(coeffs_ref * scale)
    # Trace JAX intermediate steps
    y_jax = fft_kernel(slots_jax)
    coeffs_jax = jnp.concatenate([jnp.real(y_jax), jnp.imag(y_jax)])
    scaled_coeffs_jax = jnp.round(coeffs_jax * scale)

    # Compare intermediate steps
    np.testing.assert_allclose(np.array(y_jax), y_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        np.array(coeffs_jax), coeffs_ref, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        np.array(scaled_coeffs_jax), scaled_coeffs_ref, rtol=1e-4, atol=1e-4
    )
    # Compare final plaintext by decoding back to slots
    encoder_jax = boot_utils.OnlineEncoder(degree=degree, fft_kernel=fft_kernel)
    pt_jax = encoder_jax.encode(slots_jax, scale, jnp.array(moduli), ntt_kernel)
    encoder = encode.Encode(degree, moduli.tolist(), scale)
    pt_ref = encoder.encode(slots.tolist())

    decoder = encode.Decode(scale, num_slots)
    decoded_jax = decoder.decode(pt_jax, is_slot_form=True)
    decoded_ref = decoder.decode(pt_ref, is_slot_form=True)

    np.testing.assert_allclose(decoded_jax, slots, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(decoded_ref, slots, rtol=1e-3, atol=1e-3)
    np.testing.assert_array_equal(pt_jax.moduli, pt_ref.moduli)

  @parameterized.parameters(8, 16, 32, 64)
  def test_encode_jax_jit(self, degree):
    num_slots = degree // 2
    scale = 2**20
    moduli = jnp.array(TEST_PRIMES[:2], dtype=jnp.uint32)

    ntt_kernel = ntt.NTTBarrett()
    ntt_kernel.precompute_constants(moduli.tolist(), 2, degree // 2)

    fft_kernel = boot_utils.SpecialInverseFFT()
    fft_kernel.precompute_constants(degree)

    slots = jnp.ones(num_slots, dtype=jnp.complex128)

    encoder_jax = boot_utils.OnlineEncoder()
    encoder_jax.precompute_constants(degree, fft_kernel)

    def run_encode(slots, scale, moduli, ntt_kernel, encoder):
      return encoder.encode(slots, scale, moduli, ntt_kernel)

    jitted_encode = jax.jit(run_encode, static_argnums=(1,))
    pt = jitted_encode(slots, scale, moduli, ntt_kernel, encoder_jax)

    self.assertIsInstance(pt, types.Plaintext)
    self.assertEqual(pt.data.shape, (degree, 2))

  @parameterized.parameters(
      (1073692673, 8, 1.0),
      (1073643521, 16, 2.0),
  )
  def test_compute_v0_slots(self, q0, num_slots, gamma):
    np.random.seed(q0)
    b_coeffs_flat = jnp.array(
        np.random.randint(0, q0, size=num_slots), dtype=jnp.uint32
    )

    v0_slots = boot_utils.compute_v0_slots(b_coeffs_flat, q0, num_slots, gamma)
    self.assertEqual(v0_slots.shape, (num_slots,))
    self.assertEqual(v0_slots.dtype, jnp.complex128)

    w = np.exp(2j * np.pi / q0)
    coeff = gamma / (4.0j * np.pi)
    expected = coeff * (w ** np.array(b_coeffs_flat))
    np.testing.assert_allclose(v0_slots, expected, rtol=1e-4, atol=1e-4)

  @parameterized.parameters(
      (1073692673, 8),
      (1073643521, 16),
  )
  def test_compute_a_slots(self, q0, num_slots):
    np.random.seed(q0)
    a_coeffs_flat = jnp.array(
        np.random.randint(0, q0, size=2 * num_slots), dtype=jnp.uint32
    )

    w1_a_slots, w2_a_slots = boot_utils.compute_a_slots(
        a_coeffs_flat[:num_slots], a_coeffs_flat[num_slots:], q0
    )
    self.assertEqual(w1_a_slots.shape, (num_slots,))
    self.assertEqual(w2_a_slots.shape, (num_slots,))
    self.assertEqual(w1_a_slots.dtype, jnp.complex128)
    self.assertEqual(w2_a_slots.dtype, jnp.complex128)

    w = np.exp(2j * np.pi / q0)
    expected_w1 = w ** np.array(a_coeffs_flat[:num_slots])
    expected_w2 = w ** np.array(a_coeffs_flat[num_slots:])
    np.testing.assert_allclose(w1_a_slots, expected_w1, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(w2_a_slots, expected_w2, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
