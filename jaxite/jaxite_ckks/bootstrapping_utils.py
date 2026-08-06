"""Utilities for SHIP CKKS bootstrapping."""

import dataclasses
import math
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import math as ckks_math
from jaxite.jaxite_ckks import ntt
from jaxite.jaxite_ckks import types
import numpy as np


def rot_group(m: int, nh: int, g: int = 5) -> list[int]:
  """Generates the rotation group."""
  r = [1]
  for _ in range(1, nh):
    r.append((r[-1] * g) % m)
  return r


def get_bit_reverse_perm_indices(size: int) -> jax.Array:
  """Returns the bit-reversal permutation indices of a given size."""
  indices = jnp.arange(size, dtype=jnp.int32)
  bits = int(math.log2(size))
  return ckks_math.bit_reverse(indices, bits)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class SpecialInverseFFT:
  """Kernel for specialized inverse FFT for CKKS slots encoding."""

  degree: int = 0
  br_indices: jax.Array = dataclasses.field(  # pytype: disable=annotation-type-mismatch
      default_factory=lambda: np.empty((0,), dtype=np.int32)
  )
  level_roots_real: tuple[jax.Array, ...] = ()
  level_roots_imag: tuple[jax.Array, ...] = ()

  def tree_flatten(self):
    children = (self.br_indices, self.level_roots_real, self.level_roots_imag)
    aux_data = (self.degree,)
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    obj = cls()
    obj.degree = aux_data[0]
    obj.br_indices = children[0]
    obj.level_roots_real = children[1]
    obj.level_roots_imag = children[2]
    return obj

  def precompute_constants(self, degree: int):
    """Precomputes roots of unity and bit-reversal indices."""
    self.degree = degree
    m = self.degree * 2
    nh = self.degree // 2
    roots = np.exp(2j * np.pi * np.arange(m) / m)
    roots_real = np.real(roots)
    roots_imag = np.imag(roots)
    rg = np.array(rot_group(m, nh, g=5))

    level_roots_real = []
    level_roots_imag = []
    length = nh
    while length >= 2:
      half = length >> 1
      lenq = length << 2
      step = m // lenq
      mod = rg[:half] % lenq
      idx = ((lenq - mod) % lenq) * step
      level_roots_real.append(jnp.array(roots_real[idx], dtype=jnp.float64))
      level_roots_imag.append(jnp.array(roots_imag[idx], dtype=jnp.float64))
      length >>= 1

    indices = jnp.arange(nh, dtype=jnp.int32)
    bits = int(math.log2(nh))
    self.br_indices = ckks_math.bit_reverse(indices, bits)
    self.level_roots_real = tuple(level_roots_real)
    self.level_roots_imag = tuple(level_roots_imag)

  def __call__(self, vals: jax.Array) -> jax.Array:
    nh = self.degree // 2

    curr_real = jax.lax.real(vals)
    curr_imag = jax.lax.imag(vals)

    num_levels = len(self.level_roots_real)
    length = nh
    for i in range(num_levels):
      half = length >> 1
      r_real = self.level_roots_real[i]
      r_imag = self.level_roots_imag[i]

      shape = curr_real.shape
      curr_real_reshaped = curr_real.reshape(-1, length)
      curr_imag_reshaped = curr_imag.reshape(-1, length)
      u_real = curr_real_reshaped[:, :half]
      u_imag = curr_imag_reshaped[:, :half]
      t_real = curr_real_reshaped[:, half:]
      t_imag = curr_imag_reshaped[:, half:]

      new_u_real = u_real + t_real
      new_u_imag = u_imag + t_imag

      diff_real = u_real - t_real
      diff_imag = u_imag - t_imag

      new_v_real = diff_real * r_real[None, :] - diff_imag * r_imag[None, :]
      new_v_imag = diff_real * r_imag[None, :] + diff_imag * r_real[None, :]

      curr_real = jnp.concatenate([new_u_real, new_v_real], axis=1).reshape(
          shape
      )
      curr_imag = jnp.concatenate([new_u_imag, new_v_imag], axis=1).reshape(
          shape
      )
      length >>= 1

    curr_real = curr_real * jnp.array(1.0 / nh, dtype=jnp.float64)
    curr_imag = curr_imag * jnp.array(1.0 / nh, dtype=jnp.float64)
    curr_real = curr_real[..., self.br_indices]
    curr_imag = curr_imag[..., self.br_indices]
    return jax.lax.complex(curr_real, curr_imag)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class OnlineEncoder:
  """Kernel for online CKKS slot encoding."""

  degree: int = 0
  fft_kernel: SpecialInverseFFT = dataclasses.field(
      default_factory=SpecialInverseFFT
  )

  def precompute_constants(self, degree, fft_kernel):
    self.degree = degree
    self.fft_kernel = fft_kernel

  def tree_flatten(self):
    children = (self.fft_kernel,)
    aux_data = (self.degree,)
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    obj = cls()
    obj.degree = aux_data[0]
    obj.fft_kernel = children[0]
    return obj

  def encode(
      self,
      slots: jax.Array,
      scale: float | jax.Array,
      moduli: jax.Array,
      ntt_kernel: ntt.NTTBarrett,
  ) -> types.Plaintext:
    """JAX-compatible CKKS slot encoding."""
    y = self.fft_kernel(slots)

    # Concat real and imag parts to get coefficients
    coeffs = jnp.concatenate([jnp.real(y), jnp.imag(y)])

    # Scale and round
    scaled_coeffs = jnp.round(coeffs * scale)

    # Modulo reduction for each modulus
    poly = (scaled_coeffs[:, None] % moduli[None, :]).astype(jnp.uint32)

    # Transform to NTT domain
    r = ntt_kernel.constants.r
    c = ntt_kernel.constants.c
    num_moduli = moduli.shape[0]
    poly_reshaped = poly.reshape(1, r, c, num_moduli)
    poly_ntt = ntt_kernel.ntt(poly_reshaped)
    poly_ntt_flat = poly_ntt.reshape(self.degree, num_moduli)

    return types.Plaintext(poly_ntt_flat, moduli.astype(jnp.uint32))


def compute_v0_slots(
    b_coeffs_flat: jax.Array,
    q0: int | jax.Array,
    num_slots: int,
    gamma: float = 1.0,
) -> jax.Array:
  """Computes v0 slot values (Algorithm 1, Step 1)."""
  q0_jax = jnp.asarray(q0)
  phase = (2.0 * np.pi) / q0_jax.astype(jnp.float64)
  exponent_real = b_coeffs_flat[:num_slots].astype(jnp.float64) * phase
  exponent_complex = jax.lax.complex(
      jnp.zeros_like(exponent_real), exponent_real
  )
  w_b = jnp.exp(exponent_complex)
  coeff = gamma * jnp.array(-0.25j / np.pi, dtype=jnp.complex128)
  return coeff * w_b


def compute_a_slots(
    a_lower: jax.Array,
    a_upper: jax.Array,
    q0: int | jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Computes w1_a and w2_a slot values (Algorithm 1, Steps 2-3)."""
  q0_jax = jnp.asarray(q0)
  phase = (2.0 * np.pi) / q0_jax.astype(jnp.float64)
  exponent_real_1 = a_lower.astype(jnp.float64) * phase
  exponent_complex_1 = jax.lax.complex(
      jnp.zeros_like(exponent_real_1), exponent_real_1
  )
  w1_a_slots = jnp.exp(exponent_complex_1)
  exponent_real_2 = a_upper.astype(jnp.float64) * phase
  exponent_complex_2 = jax.lax.complex(
      jnp.zeros_like(exponent_real_2), exponent_real_2
  )
  w2_a_slots = jnp.exp(exponent_complex_2)
  return w1_a_slots, w2_a_slots
