"""LWE encryption scheme."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jaxite.jaxite_lib import encoding
from jaxite.jaxite_lib import matrix_utils
from jaxite.jaxite_lib import parameters
from jaxite.jaxite_lib import random_source
from jaxite.jaxite_lib import types


@dataclasses.dataclass
class LweSecretKey:
  """A secret key for the LWE encryption scheme."""

  # the q in Z/2^qZ
  log_modulus: int

  # the length of the sampled key_data vector s,
  # equal to len(LweCiphertext) - 1.
  # Encryption involves sampling a vector a = (a_1, ..., a_{lwe_dimension})
  # and computing a dot product <a, s> + error.
  lwe_dimension: int

  # the binary values (s_1, ..., s_{lwe_dimension})
  # used as a dot product multiplicand when encrypting.
  key_data: jnp.ndarray

  @property
  def modulus(self) -> jnp.uint32:
    return 2 ** jnp.uint32(self.log_modulus)


def gen_key(
    params: parameters.SchemeParameters, prg: random_source.RandomSource
) -> LweSecretKey:
  """Generate an LWE secret key."""
  return LweSecretKey(
      log_modulus=params.log_plaintext_modulus,
      lwe_dimension=params.lwe_dimension,
      key_data=prg.sk_uniform(shape=(params.lwe_dimension,), dtype=jnp.uint32),
  )


def encrypt(
    plaintext: types.LwePlaintext,
    sk: LweSecretKey,
    prg: random_source.RandomSource,
) -> types.LweCiphertext:
  """Encrypt an LWE plaintext."""
  ai_samples = prg.uniform(shape=(sk.lwe_dimension,), dtype=jnp.uint32)
  error_sample = prg.rounded_normal()
  return jit_encrypt(
      plaintext,
      sk.key_data,
      ai_samples,
      error_sample,
      sk.log_modulus,
  )


@functools.partial(jax.jit, static_argnames="log_modulus")
def jit_encrypt(
    plaintext: types.LwePlaintext,
    key_data: jnp.ndarray,
    ai_samples: jnp.ndarray,
    error_sample: jnp.uint32,
    log_modulus: int,
) -> types.LweCiphertext:
  """Encrypt an LWE plaintext with pre-computed randomness."""
  clean_product = jnp.dot(ai_samples, key_data) + plaintext
  obfuscated_product = clean_product + error_sample
  if log_modulus < 32:
    obfuscated_product = obfuscated_product % (jnp.uint32(2) ** log_modulus)
  return jnp.append(ai_samples, obfuscated_product)


def decrypt_without_denoising(
    ciphertext: types.LweCiphertext,
    sk: LweSecretKey,
) -> types.LwePlaintext:
  """Decrypt an LWE ciphertext without removing noise."""
  obfuscated_plaintext = ciphertext[-1] - jnp.dot(ciphertext[:-1], sk.key_data)
  if sk.log_modulus < 32:
    obfuscated_plaintext = jnp.mod(obfuscated_plaintext, sk.modulus)

  return obfuscated_plaintext


def decrypt(
    ciphertext: types.LweCiphertext,
    sk: LweSecretKey,
    encoding_params: encoding.EncodingParameters,
) -> types.LweCleartext:
  """Decrypt and remove the noise from an LWE ciphertext."""
  return encoding.remove_noise(  # pytype: disable=bad-return-type  # jax-ndarray
      decrypt_without_denoising(ciphertext, sk), encoding_params
  )


def noiseless_embedding(
    plaintext: types.LwePlaintext, lwe_dimension: int
) -> types.LweCiphertext:
  """Returns a noiseless LweCiphertext embedding of `plaintext`."""
  samples = jnp.zeros((lwe_dimension,), dtype=jnp.uint32)
  return jnp.append(samples, plaintext)


@jax.jit
def switch_modulus(
    ciphertext: types.LweCiphertext,
    log_input_modulus: jnp.uint32,
    log_output_modulus: jnp.uint32,
) -> types.LweCiphertext:
  """Perform a modulus switch on the input ciphertext.

  The input and output moduli must be powers of two, and they're passed in as
  their logarithms.

  The extra error introduced by modulus switching is at most 0.5 * (lwe_dim+1),
  (note that error is interpreted in the new modulus), and the total error after
  modulus switching is at most

    e_input + 0.5 (lwe_dim + 1),

  where e_input is the error present in `ciphertext`.

  Args:
    ciphertext: the LWE ciphertext to modulus switch
    log_input_modulus: the log2 of the modulus to switch from
    log_output_modulus: the log2 of the modulus to switch to

  Returns:
    An LWE cipherext of the same underlying message, but with values expressed
    with respect to 2**log_output_modulus.
  """
  divisor = jnp.uint32(2) ** (log_input_modulus - log_output_modulus)
  scaled_ciphertext = matrix_utils.integer_div(ciphertext, divisor)
  approx_ciphertext = jnp.mod(
      scaled_ciphertext, jnp.uint32(2) ** log_output_modulus
  )
  return approx_ciphertext
