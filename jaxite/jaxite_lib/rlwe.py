"""RLWE encryption scheme."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jaxite.jaxite_lib import encoding
from jaxite.jaxite_lib import lwe
from jaxite.jaxite_lib import matrix_utils
from jaxite.jaxite_lib import parameters
from jaxite.jaxite_lib import random_source


@dataclasses.dataclass
class RlwePlaintext:
  """An RLWE plaintext is a polynomial in (Z/qZ)[X] / (X^N + 1)."""

  # the log of the modulus q of the polynomial coefficients
  log_coefficient_modulus: int

  # the degree N of the ring modulus polynomial.
  modulus_degree: int

  # the coefficients of the polynomial,
  # starting from lowest degree to highest.
  message: jnp.ndarray

  @property
  def coefficient_modulus(self) -> jnp.uint32:
    return 2 ** jnp.uint32(self.log_coefficient_modulus)

  def __str__(self) -> str:
    # this does not need to be fast because it will only be used in development.
    s = ' + '.join(
        f'{coeff} x^{power}'
        for (power, coeff) in enumerate(self.message)
        if coeff != 0
    )
    return s if s else '0'


@dataclasses.dataclass
class RlweCiphertext:
  """An RLWE ciphertext.

  An RLWE ciphertext is a list of k+1 polynomials in (Z/qZ)[X] / (X^N + 1),
  where the last polynomial is a dot product with the secret key, plus noise.
  """

  # the log of the modulus q of the polynomial coefficients
  log_coefficient_modulus: int

  # the degree N of the ring modulus polynomial.
  modulus_degree: int

  # the polynomials, packed so each row corresponds to one polynomial, and each
  # column corresponds to a coefficient of the same degree.  The first colum is
  # the lowest degree.
  message: jnp.ndarray

  @property
  def coefficient_modulus(self) -> jnp.uint32:
    return 2 ** jnp.uint32(self.log_coefficient_modulus)

  def _single_row_str(self, row):
    return str(
        RlwePlaintext(
            log_coefficient_modulus=self.log_coefficient_modulus,
            modulus_degree=self.modulus_degree,
            message=row,
        )
    )

  def __str__(self) -> str:
    # this does not need to be fast because it will only be used in development.
    inner = '\n '.join(self._single_row_str(row) for row in self.message)
    return f'[\n {inner}\n]'

  def __repr__(self) -> str:
    return str(self)


@dataclasses.dataclass
class RlweSecretKey:
  """A secret key for the RLWE encryption scheme."""

  # the log of q in Z/qZ
  log_coefficient_modulus: int

  # the power of two N in the polynomial modulus x^N + 1
  modulus_degree: int

  # the number of samples chosen, equal to len(RlweCiphertext) - 1
  rlwe_dimension: int

  # the binary polynomial values (s_1, ..., s_{rlwe_dimension})
  # used as a dot product multiplicand when encrypting.
  # each entry is a polynomial in {0,1}[X]/ (X^N + 1)
  data: jnp.ndarray

  @property
  def coefficient_modulus(self) -> jnp.uint32:
    return jnp.uint32(2) ** self.log_coefficient_modulus


def gen_key(
    params: parameters.SchemeParameters, prg: random_source.RandomSource
) -> RlweSecretKey:
  """Generate an RLWE secret key."""
  return RlweSecretKey(
      log_coefficient_modulus=params.log_plaintext_modulus,
      modulus_degree=params.polynomial_modulus_degree,
      rlwe_dimension=params.rlwe_dimension,
      data=prg.sk_uniform(
          shape=(params.rlwe_dimension, params.polynomial_modulus_degree),
          dtype=jnp.uint32,
      ),
  )


def encrypt(
    plaintext: RlwePlaintext, sk: RlweSecretKey, prg: random_source.RandomSource
) -> RlweCiphertext:
  """Encrypt an RLWE plaintext."""
  ai_samples = prg.uniform(
      shape=(sk.rlwe_dimension, sk.modulus_degree),
      dtype=jnp.uint32,
  )
  error_sample = prg.rounded_normal(shape=(sk.modulus_degree,)).astype(
      jnp.uint32
  )
  ciphertext_data = jit_encrypt(
      plaintext.message,
      sk.data,
      ai_samples,
      error_sample,
      log_coefficient_modulus=sk.log_coefficient_modulus,
  )

  return RlweCiphertext(
      log_coefficient_modulus=sk.log_coefficient_modulus,
      modulus_degree=sk.modulus_degree,
      message=ciphertext_data,
  )


@functools.partial(jax.jit, static_argnames='log_coefficient_modulus')
def jit_encrypt(
    plaintext: jnp.ndarray,
    key_data: jnp.ndarray,
    ai_samples: jnp.ndarray,
    error_sample: jnp.ndarray,
    log_coefficient_modulus: int,
) -> jnp.ndarray:
  """Encrypt an RLWE plaintext with pre-computed randomness."""
  clean_product = (
      matrix_utils.poly_dot_product(
          ai_samples,
          key_data,
      )
      + plaintext
  )
  obfuscated_product = clean_product + error_sample

  if log_coefficient_modulus < 32:
    modulus = jnp.uint32(2) ** log_coefficient_modulus
    obfuscated_product = obfuscated_product % modulus

  return jnp.append(ai_samples, jnp.array([obfuscated_product]), axis=0)


def decrypt(
    ciphertext: RlweCiphertext,
    sk: RlweSecretKey,
    encoding_params: encoding.EncodingParameters,
) -> RlwePlaintext:
  """Decrypt an RLWE ciphertext."""
  dot_prod = matrix_utils.poly_dot_product(ciphertext.message[:-1], sk.data)
  obfuscated_plaintext = ciphertext.message[-1] - dot_prod
  if sk.log_coefficient_modulus < 32:
    obfuscated_plaintext = jnp.mod(obfuscated_plaintext, sk.coefficient_modulus)
  msg = encoding.remove_noise(obfuscated_plaintext, encoding_params)
  return RlwePlaintext(
      log_coefficient_modulus=sk.log_coefficient_modulus,
      modulus_degree=sk.modulus_degree,
      message=msg,
  )


@jax.named_call
def flatten_key(sk: RlweSecretKey) -> lwe.LweSecretKey:
  """Flattens rlwe secret key to use for decrypting after sample extraction.

  Args:
    sk: an RLWE secret key that was used to encrypt the input to a sample
      extract.

  Returns:
    An LWE secret key that can be used to decrypt the LWE ciphertext output of
    a sample extraction.
  """
  poly_deg, k = sk.modulus_degree, sk.rlwe_dimension
  return lwe.LweSecretKey(
      log_modulus=sk.log_coefficient_modulus,
      lwe_dimension=poly_deg * k,
      key_data=sk.data.reshape(poly_deg * k),
  )
