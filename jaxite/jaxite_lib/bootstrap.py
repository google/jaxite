"""The API for bootstrapping in CGGI."""

import dataclasses
import functools
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jaxite.jaxite_lib import decomposition
from jaxite.jaxite_lib import key_switch
from jaxite.jaxite_lib import lwe
from jaxite.jaxite_lib import matrix_utils
from jaxite.jaxite_lib import parameters
from jaxite.jaxite_lib import polymul_kernel
from jaxite.jaxite_lib import random_source
from jaxite.jaxite_lib import rgsw
from jaxite.jaxite_lib import rlwe
from jaxite.jaxite_lib import types

GEN_BSK_NUM_BATCHES = 15
NON_DIVISIBLE_BATCH_SIZE_WARNING = (
    "Expected lwe_sk_dim to be a multiple of %s, but was %s. "
    "This is OK for tests with small LWE secret key sizes, but may cause "
    "memory exhaustion errors with prod security params. "
    "Proceeding with unbatched BSK keygen."
)


@dataclasses.dataclass
class BootstrappingKey:
  """An array with row j an RGSW encryption of bit j of an LWE secret key."""

  encrypted_lwe_sk_bits: jnp.ndarray


def gen_bootstrapping_key(
    lwe_sk: lwe.LweSecretKey,
    rgsw_sk: rgsw.RgswSecretKey,
    decomposition_params: decomposition.DecompositionParameters,
    prg: random_source.RandomSource,
) -> BootstrappingKey:
  """Generate a bootstrapping key for the given LWE secret key.

  A bootstrapping key is a list of N RGSW ciphertexts, each encrypting a bit
  of a given LWE secret key sk_in.

  Args:
    lwe_sk: the input LWE secret key, to be encrypted by the RGSW key
    rgsw_sk: the RGSW secret key, to use to encrypt the LWE key
    decomposition_params: bit-decomposition parameters needed for RGSW
      encryption
    prg: the random source

  Returns:
    A bootstrapping key.
  """
  k = num_blocks = rgsw_sk.key.rlwe_dimension
  rlwe_sk = rgsw_sk.to_rlwe_secret_key()
  lwe_sk_dim = lwe_sk.key_data.shape[0]
  levels = decomposition_params.level_count
  if lwe_sk_dim % 2 != 0:
    raise ValueError(
        "BMMP bootstrap loop-unrolling technique "
        "requires an even lwe_sk dimension, but got "
        f"{lwe_sk_dim}"
    )
  num_bsk_encryptions = lwe_sk_dim + lwe_sk_dim // 2

  ai_samples = prg.uniform(
      shape=(
          num_bsk_encryptions,
          num_blocks + 1,
          levels,
          k,
          rlwe_sk.modulus_degree,
      ),
      dtype=jnp.uint32,
  )
  error_samples = prg.rounded_normal(
      shape=(num_bsk_encryptions, num_blocks + 1, levels),
      dtype=jnp.uint32,
  )

  # Using the improved blind rotate from Bourse-Minelli-Minihold-Paillier
  # (BMMP17: https://eprint.iacr.org/2017/1114), a trick uses a larger
  # bootstrapping key to reduce the number of external products required by 1/2.
  # Rather than encrypt the secret key bits of the LWE key separately, we
  # encrypt:
  #
  #  BSK_{3i}   = s_{2i} * s_{2i+1},
  #  BSK_{3i+1} = s_{2i} * (1 − s_{2i+1}),
  #  BSK_{3i+2} = (1 − s_{2i}) * s_{2i+1}
  #
  # which enables a bootstrap operation that involves 1/2 as many external
  # products, though this causes the bootstrapping key to be 50% larger.
  lwe_sk_data = lwe_sk.key_data.astype(jnp.uint32)
  bsk_input_pairs = jnp.zeros(num_bsk_encryptions, dtype=jnp.uint32)

  bsk_input_pairs = bsk_input_pairs.at[::3].set(
      jnp.multiply(lwe_sk_data[::2], lwe_sk_data[1::2])
  )
  bsk_input_pairs = bsk_input_pairs.at[1::3].set(
      jnp.multiply(lwe_sk_data[::2], 1 - lwe_sk_data[1::2])
  )
  bsk_input_pairs = bsk_input_pairs.at[2::3].set(
      jnp.multiply(1 - lwe_sk_data[::2], lwe_sk_data[1::2])
  )

  # Applying vmap to the entire jit_encrypt over all sk bits will exhaust the
  # tensor core's memory with prod security parameters. It will eagerly allocate
  # a u32[945,2,8,1,1024,1024] which has size ~60 GiB, but a Dragonfish chip has
  # only 16 GiB. To reduce memory usage, we can use lax.map to batch the
  # computation in 15 batches and then reshape the result at the end. Doing this
  # results in intermediate allocations of size ~4 GiB for prod security
  # parameters, which provides enough space for the remaining allocations of
  # random numbers.
  num_batches = GEN_BSK_NUM_BATCHES
  batch_size = num_bsk_encryptions // num_batches
  if num_bsk_encryptions % num_batches != 0:
    print(NON_DIVISIBLE_BATCH_SIZE_WARNING % (batch_size, num_bsk_encryptions))
    num_batches = 1
    batch_size = num_bsk_encryptions

  ai_samples = ai_samples.reshape((
      num_batches,
      batch_size,
      num_blocks + 1,
      levels,
      k,
      rlwe_sk.modulus_degree,
  ))
  error_samples = error_samples.reshape((
      num_batches,
      batch_size,
      num_blocks + 1,
      levels,
  ))

  bsk_input_pairs = bsk_input_pairs.reshape((num_batches, batch_size))

  vmapped_encrypt = jax.vmap(
      rgsw.jit_encrypt,
      in_axes=(0, None, 0, 0, None, None, None, None),
      out_axes=0,
  )

  def process_one_batch(i):
    ai_samples_batch = ai_samples[i]
    error_samples_batch = error_samples[i]
    input_batch = bsk_input_pairs[i]
    return vmapped_encrypt(
        input_batch,
        rlwe_sk.data,
        ai_samples_batch,
        error_samples_batch,
        decomposition_params,
        num_blocks,
        rlwe_sk.log_coefficient_modulus,
        rlwe_sk.modulus_degree,
    )

  encrypted_lwe_sk_bits = jax.lax.map(
      process_one_batch, jnp.arange(num_batches)
  )
  # The output is stacked by batches, and this reshape combines the first two
  # axes into a single axis of size num_bsk_encryptions.
  encrypted_lwe_sk_bits = encrypted_lwe_sk_bits.reshape((
      num_bsk_encryptions,
      (num_blocks + 1) * levels,
      num_blocks + 1,
      rlwe_sk.modulus_degree,
  ))

  return BootstrappingKey(encrypted_lwe_sk_bits=encrypted_lwe_sk_bits)


@jax.named_call
@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def jit_bootstrap(
    ciphertext: types.LweCiphertext,
    test_poly_ciphertext_message: jnp.ndarray,
    bsk_encrypted_lwe_sk_bits: jnp.ndarray,
    ksk_key_data: jnp.ndarray,
    ks_decomposition_params: decomposition.DecompositionParameters,
    bs_decomposition_params: decomposition.DecompositionParameters,
    scheme_params: parameters.SchemeParameters,
) -> types.LweCiphertext:
  """Apply functional bootstrap to reduce noise in the input ciphertext.

  Args:
    ciphertext: the input LWE ciphertext.
    test_poly_ciphertext_message: The RLWE encrypted polynomial of the cleartext
      message values used to retrieve the target sample during bootstrapping.
    bsk_encrypted_lwe_sk_bits: a bootstrapping key, encrypting the bits of the
      secret key used to encrypt `ciphertext`
    ksk_key_data: a key switching key, required to switch back to the original
      LWE key after bootstrapping switches to an LWE key determined by `bsk`.
    ks_decomposition_params: the decomposition parameters used in the key
      switching key.
    bs_decomposition_params: the decomposition parameters used in the
      bootstrapping key.
    scheme_params: the scheme parameters, used to discretize the ciphertext.

  Returns:
    An encryption of the same underlying message as `ciphertext`, but with
    reduced noise.
  """
  mod_degree = scheme_params.polynomial_modulus_degree
  test_poly_log_coefficient_modulus = scheme_params.log_plaintext_modulus

  approx_ciphertext = lwe.switch_modulus(
      ciphertext,
      log_input_modulus=scheme_params.log_plaintext_modulus,
      log_output_modulus=scheme_params.log_mod_degree + 1,
  )

  # The target LWE sample is contained in the constant term of the output
  # polynomial, which sample_extract extracts as an LWE ciphertext.
  rotated = jit_blind_rotate(
      test_poly_ciphertext_message,
      approx_ciphertext,
      bsk_encrypted_lwe_sk_bits,
      test_poly_log_coefficient_modulus,
      bs_decomposition_params,
  )

  extracted = jit_sample_extract(rotated, mod_degree)

  key_switched = key_switch.jit_switch_key(
      ksk_key_data,
      extracted,
      ks_decomposition_params.level_count,
      ks_decomposition_params.log_base,
  )

  return key_switched


@jax.named_call
def bootstrap(
    ciphertext: types.LweCiphertext,
    test_poly_ciphertext: rlwe.RlweCiphertext,
    bsk: BootstrappingKey,
    ksk: key_switch.LweKeySwitchingKey,
    ks_decomposition_params: decomposition.DecompositionParameters,
    bs_decomposition_params: decomposition.DecompositionParameters,
    scheme_params: parameters.SchemeParameters,
    callback: Optional[Callable[[str, Any], None]] = None,
    **kwargs: Any,
) -> types.LweCiphertext:
  """Apply functional bootstrap to reduce noise in the input ciphertext.

  Args:
    ciphertext: the input LWE ciphertext.
    test_poly_ciphertext: The RLWE encrypted polynomial of the cleartext message
      values used to retrieve the target sample during bootstrapping.
    bsk: a bootstrapping key, encrypting the bits of the secret key used to
      encrypt `ciphertext`.
    ksk: a key switching key, required to switch back to the original LWE key
      after bootstrapping switches to an LWE key determined by `bsk`.
    ks_decomposition_params: the decomposition parameters used in the key
      switching key.
    bs_decomposition_params: the decomposition parameters used in the
      bootstrapping key.
    scheme_params: the scheme parameters, used to discretize the ciphertext.
    callback: an optional callback for tests.
    **kwargs: kwargs forwarded to the callback, if present.

  Returns:
    An encryption of the same underlying message as `ciphertext`, but with
    reduced noise.
  """
  if callback:
    callback("initial", ciphertext, **kwargs)

  mod_degree = scheme_params.polynomial_modulus_degree
  test_poly_log_coefficient_modulus = scheme_params.log_plaintext_modulus

  approx_ciphertext = lwe.switch_modulus(
      ciphertext,
      log_input_modulus=scheme_params.log_plaintext_modulus,
      log_output_modulus=scheme_params.log_mod_degree + 1,
  )
  if callback:
    callback("approx_ciphertext", approx_ciphertext, **kwargs)

  # The target LWE sample is contained in the constant term of the output
  # polynomial, which sample_extract extracts as an LWE ciphertext.
  rotated = jit_blind_rotate(
      test_poly_ciphertext.message,
      approx_ciphertext,
      bsk.encrypted_lwe_sk_bits,
      test_poly_log_coefficient_modulus,
      bs_decomposition_params,
  )
  if callback:
    callback("rotated", rotated, **kwargs)

  extracted = jit_sample_extract(rotated, mod_degree)
  if callback:
    callback("extracted", extracted, **kwargs)

  key_switched = key_switch.jit_switch_key(
      ksk.key_data,
      extracted,
      ks_decomposition_params.level_count,
      ks_decomposition_params.log_base,
  )
  if callback:
    callback("key_switched", key_switched, **kwargs)

  return key_switched


def external_product(
    rgsw_ct: rgsw.RgswCiphertext,
    rlwe_ct: rlwe.RlweCiphertext,
    decomposition_params: decomposition.DecompositionParameters,
) -> rlwe.RlweCiphertext:
  """Compute the external product of an RSGW and RLWE ciphertext."""
  output = jit_external_product(
      rgsw_ct.message, rlwe_ct.message, decomposition_params
  )
  return rlwe.RlweCiphertext(
      log_coefficient_modulus=rlwe_ct.log_coefficient_modulus,
      modulus_degree=rlwe_ct.modulus_degree,
      message=output,
  )


@functools.partial(jax.jit, static_argnames="decomposition_params")
def jit_external_product(
    rgsw_ct: jnp.ndarray,
    rlwe_ct: jnp.ndarray,
    decomposition_params: decomposition.DecompositionParameters,
) -> rlwe.RlweCiphertext:
  """Compute the external product of an RSGW and RLWE ciphertext."""
  decomposed_rlwe = decomposition.decompose_rlwe_ciphertext(
      rlwe_ct, decomposition_params
  )
  return polymul_kernel.negacyclic_vector_matrix_polymul(
      decomposed_rlwe, rgsw_ct
  )


def cmux(
    control: rgsw.RgswCiphertext,
    eq_zero: rlwe.RlweCiphertext,
    neq_zero: rlwe.RlweCiphertext,
    decomposition_params: decomposition.DecompositionParameters,
) -> rlwe.RlweCiphertext:
  """Compute CMUX: controlled multiplexer.

  Args:
    control: RGSW ciphertext acting as a selector between eq_zero and neq_zero
    eq_zero: RLWE ciphertext selected if control=0
    neq_zero: RLWE ciphertext selected if control=1
    decomposition_params: decomposition parameters for the external product

  Returns:
    RlwePlaintext: selected RLWE ciphertext
  """
  if eq_zero.coefficient_modulus != neq_zero.coefficient_modulus:
    raise ValueError(
        (
            "Bad params, rlwe ciphertexts must have same coeff modulus"
            " Instead found:\n"
        ),
        f"eq_zero.coeff_modulus={eq_zero.coefficient_modulus}, ",
        f"neq_zero.coeff_modulus={neq_zero.coefficient_modulus}.",
    )
  if eq_zero.modulus_degree != neq_zero.modulus_degree:
    raise ValueError(
        (
            "Bad params, rlwe ciphertexts must have same mod degree."
            " Instead found:\n"
        ),
        f"eq_zero.modulus_degree={eq_zero.modulus_degree}, ",
        f"neq_zero.modulus_degree={neq_zero.modulus_degree}.",
    )
  modulus_degree = eq_zero.modulus_degree
  output = jit_cmux(
      control.message, eq_zero.message, neq_zero.message, decomposition_params
  )
  return rlwe.RlweCiphertext(
      log_coefficient_modulus=eq_zero.log_coefficient_modulus,
      modulus_degree=modulus_degree,
      message=output,
  )


@jax.named_call
@functools.partial(jax.jit, static_argnames="decomposition_params")
def jit_cmux(
    control: jnp.ndarray,
    eq_zero: jnp.ndarray,
    neq_zero: jnp.ndarray,
    decomposition_params: decomposition.DecompositionParameters,
) -> rlwe.RlweCiphertext:
  """A jitted cmux."""
  return (
      eq_zero
      + jit_external_product(  # pytype: disable=bad-return-type  # jax-ndarray
          rgsw_ct=control,
          rlwe_ct=neq_zero - eq_zero,
          decomposition_params=decomposition_params,
      )
  )


def blind_rotate(
    rot_polynomial: rlwe.RlweCiphertext,
    coefficient_index: types.LweCiphertext,
    bsk: BootstrappingKey,
    decomposition_params: decomposition.DecompositionParameters,
) -> rlwe.RlweCiphertext:
  """Rotate an encrypted polynomial `coefficient_index` times.

  Args:
    rot_polynomial: an RLWE encryption of the polynomial P to rotate. This is
      referred to as TGLWE_s'(v) in the TFHE explainer paper
      (https://eprint.iacr.org/2021/1402.pdf). Note that (0, 0, ... v) is a
      valid RLWE encryption of v.
    coefficient_index: the encrypted index J to rotate to the constant term.
      This is referred to as `c^tilde = (a_1^tilde, ... a_n^tilde, b^tilde)`.
    bsk: bootstrapping key, an RGSW encryption of the bits of the secret key `s`
      used to encrypt coefficient_index
    decomposition_params: decomposition parameters for the external product
      operation used in CMUX

  Returns:
    An encryption of X^{-J} * P, which has the J-th coefficient of P as its
    constant term.
  """
  output = jit_blind_rotate(
      rot_polynomial.message,
      coefficient_index,
      bsk.encrypted_lwe_sk_bits,
      rot_polynomial.log_coefficient_modulus,
      decomposition_params,
  )
  return rlwe.RlweCiphertext(
      log_coefficient_modulus=rot_polynomial.log_coefficient_modulus,
      modulus_degree=rot_polynomial.modulus_degree,
      message=output,
  )


@jax.named_call
@functools.partial(jax.jit, static_argnums=(3, 4))
def jit_blind_rotate(
    rot_polynomial: jnp.ndarray,
    coefficient_index: types.LweCiphertext,
    bsk: jnp.ndarray,
    log_coefficient_modulus: int,
    decomposition_params: decomposition.DecompositionParameters,
) -> rlwe.RlweCiphertext:
  """Rotate an encrypted polynomial `coefficient_index` times."""
  # Calculate c' = X^{-b^tilde} * RLWE_s'(v) (for each entry in RLWE_s'(v))
  b_tilde = coefficient_index[coefficient_index.shape[0] - 1]
  c_prime = matrix_utils.monomial_mul_list(
      rot_polynomial,
      -b_tilde,
      log_coefficient_modulus,
  ).astype(jnp.uint32)

  # Using the improved blind rotate from Bourse-Minelli-Minihold-Paillier
  # (BMMP17: https://eprint.iacr.org/2017/1114), a trick uses a larger
  # bootstrapping key to reduce the number of external products required by 1/2.
  num_loop_terms = (coefficient_index.shape[0] - 1) // 2

  def one_external_product(j, c_prime_accum):
    # Doing this computation inside the external product loop improves cache
    # locality, resulting in reduced data copying.
    power1 = coefficient_index[2 * j] + coefficient_index[2 * j + 1]
    power2 = coefficient_index[2 * j]
    power3 = coefficient_index[2 * j + 1]
    bmmp_factor = (
        matrix_utils.scale_by_x_power_n_minus_1(
            power1, bsk[3 * j], log_modulus=log_coefficient_modulus
        )
        + matrix_utils.scale_by_x_power_n_minus_1(
            power2, bsk[3 * j + 1], log_modulus=log_coefficient_modulus
        )
        + matrix_utils.scale_by_x_power_n_minus_1(
            power3, bsk[3 * j + 2], log_modulus=log_coefficient_modulus
        )
    ).astype(jnp.uint32)
    return c_prime_accum + jit_external_product(
        rgsw_ct=bmmp_factor,
        rlwe_ct=c_prime_accum,
        decomposition_params=decomposition_params,
    )

  return jax.lax.fori_loop(0, num_loop_terms, one_external_product, c_prime)


def sample_extract(ciphertext: rlwe.RlweCiphertext) -> types.LweCiphertext:
  """Extracts an LWE encryption of the constant term encrypted by the input."""
  return jit_sample_extract(ciphertext.message, ciphertext.modulus_degree)


@jax.named_call
@functools.partial(jax.jit, static_argnames="poly_deg")
def jit_sample_extract(
    rlwe_ciphertext: jnp.ndarray, poly_deg: jnp.uint32
) -> types.LweCiphertext:
  """Extracts an LWE encryption of the constant term encrypted by the input.

  Given an RLWE ciphertext with modulus_degree polyN and rlwe_dimension k,
  outputs an LWE ciphertext with lwe_dimension k*polyN.

  Args:
    rlwe_ciphertext: an RLWE encryption of the polynomial to sample extract.
    poly_deg: the polynomial degree of the rlwe_ciphertext.

  Returns:
    An LWE encryption of the constant term of the input polynomial.
  """
  k = jnp.shape(rlwe_ciphertext)[0] - 1  # rlwe_dimension
  ones = jnp.ones(poly_deg, dtype=jnp.int32)
  indices = jax.lax.broadcasted_iota(
      dtype=jnp.int32, shape=ones.shape, dimension=0
  )
  signed_ones = jnp.where(indices > 0, -ones, ones)
  extraction_coefficients = jnp.broadcast_to(signed_ones, (k, poly_deg))

  # extracts the rlwe_ciphertext into a matrix, accessing the last axis by
  # indices via [0, poly_deg-1, poly_deg-2, ..., 1]
  extracted_values = jnp.flip(
      jnp.roll(rlwe_ciphertext[:-1, :], -1, axis=-1), axis=-1
  )
  extracted_sample = extraction_coefficients * extracted_values
  b_term_constant_coeff = rlwe_ciphertext[-1][0]

  return jnp.append(
      extracted_sample.flatten(),
      jnp.array([b_term_constant_coeff], dtype=jnp.uint32),
  )
