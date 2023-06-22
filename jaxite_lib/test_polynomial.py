"""A module for operations on test polynomials.

In general, a test polynomial is a degree-N polynomial p(x) in the ring Z/qZ[x]
mod (x^N+1) which represents the lookup table of a single-input function f(x) on
M-bit integers (2^M <= N). The simplest example, and the function used for
non-functional bootstrapping, is the identity function on the message space
Z/2^M Z.

This module constructs test polynomials in such a way as to make them usable for
FHE, in the sense that blind_rotate is used to index a specific coefficient a_j
of the test polynomial. In particular, j = b - <a,s> is the underlying plaintext
of an LWE encrypted message, plus some error term.

The complexity arises because the index j is approximate, and may be negative. A
negative index is interpreted modulo N, but because the ring Z/qZ[x] / (x^N+1)
is negacyclic (i.e., x^N = -1, not 1), the test polynomial p(x) must be
constructed with special care to work with blind rotate. The steps are as
follows:

  1. Start with a raw lookup table as a list L with L[i] = f(i) mod 2^M. This
     table must be length 2^M to account for all possible inputs to the
     function.

  2. Repeat each coefficient some number of times, which corresponds to adding
     protection against approximation error in the input index. The amount of
     repetition per block is N / 2^M.

  3. Rotate the coefficients backwards by half the width of one repetition
     block, to account for the possibility that the input index is negative
     due to error in the blind_rotate input j. I.e., an error-free index input
     to blind_rotate lands in the center of its block.

  4. Encode the coefficients so as to be compatible with the encoding of LWE
     plaintexts, i.e., in such a way that a coefficient that is < 2^M can be
     decoded as if it were an LWE plaintext. This boils down to using the
     padding bits to store test polynomial coefficients that need more than M
     bits. This is required because the output of blind_rotate is interpreted
     as an LWE ciphertext.
"""

import jax.numpy as jnp
from jaxite.jaxite_lib import encoding
from jaxite.jaxite_lib import parameters
from jaxite.jaxite_lib import random_source
from jaxite.jaxite_lib import rlwe


def _pad_for_error(
    polynomial: jnp.ndarray, mod_degree: int, coeff_mod: int
) -> jnp.ndarray:
  """Pads the coefficients of the input polynomial into blocks.

  The curious behavior of this method is that the first block is split across
  the lowest and highest degrees. The test polynomials output by the methods in
  this module are used to "lookup" coefficients by index, but the index input to
  the lookup is scaled and approximate, with symmetric error.

  The width of the block is related to how much error is allowed in the index
  before the wrong value is looked up, and the rotation ensures that symmetric
  error for index 0 (mod polynomial.modulus_degree) will wrap around to the
  highest coefficient and provide the right answer.

  Because the coefficients in the part of the first block wrap around in the
  negative direction (due to the negacyclic property), we must negate those
  terms.

  Args:
    polynomial: the input polynomial coefficients, as a jax array.
    mod_degree: the degree of the resulting polynomial
    coeff_mod: the modulus of the coefficients

  Returns:
    The coefficients padded into blocks of width mod_degree / len(polynomial)
  """
  # We want to fill up the entire output polynomial, which has `mod_degree` many
  # coefficients, with no space left over. Since mod_degree and the message
  # space size are both powers of two, this division is exact and each block has
  # the same size, which is also a power of two.
  block_width = mod_degree // polynomial.shape[0]
  blocks = jnp.repeat(polynomial, block_width)

  split_index = block_width // 2
  first_half_block = blocks[:split_index]
  # Because the coefficient wraps around, we must negate those terms to adhere
  # to the negacyclic property.
  negated_half_block = jnp.mod(-first_half_block, coeff_mod)
  return jnp.concatenate((blocks[split_index:], negated_half_block)).astype(
      jnp.uint32
  )


def gen_test_polynomial(
    cleartext_coefficients: jnp.ndarray,
    encoding_params: encoding.EncodingParameters,
    scheme_params: parameters.SchemeParameters,
) -> rlwe.RlwePlaintext:
  """Generates a test polynomial with padding and encoding.

  The test polynomial generated encodes cleartext_coefficients, and then the
  result is padded so that the total polynomial degree is
  polynomial_modulus_degree - 1.

  Args:
    cleartext_coefficients: the unpadded coefficients of the test polynomial.
    encoding_params: the encoding parameters that define the message space size.
    scheme_params: the scheme parameters that define the polynomial modulus.

  Returns:
    An RlwePlaintext with the appropriate range and padding.

  Raises:
    ValueError: when the `cleartext_coefficients` provided exceeds the desired
    degree of the test polynomial, `mod_degree`.
  """
  mod_degree = scheme_params.polynomial_modulus_degree
  message_space_size = 2**encoding_params.message_bit_length
  if mod_degree < len(cleartext_coefficients):
    raise ValueError(
        f"Degree ({mod_degree}) must exceed the number of "
        f"coefficients ({len(cleartext_coefficients)})."
    )

  if len(cleartext_coefficients) != message_space_size:
    raise ValueError(
        "Coefficients must have length equal to the size of the "
        f"message space, but length was {len(cleartext_coefficients)}"
    )

  padded = _pad_for_error(
      cleartext_coefficients, mod_degree, message_space_size
  )

  # Encode the coefficients, because blind_rotate will produce as output one of
  # the coefficients, which is then interpreted as an LWE ciphertext.
  test_polynomial = encoding.encode(padded, encoding_params)

  return rlwe.RlwePlaintext(
      message=test_polynomial,
      log_coefficient_modulus=scheme_params.log_plaintext_modulus,
      modulus_degree=mod_degree,
  )


def identity_test_polynomial(
    encoding_params: encoding.EncodingParameters,
    scheme_params: parameters.SchemeParameters,
) -> rlwe.RlwePlaintext:
  """Generates a test polynomial for the identity function.

  A helper for gen_test_polynomial which generates the polynomial

    0 + 1x + 2x^2 + ... + (p-1)x^{p-1},

  where p is the size of the message space. This polynomial is then padded and
  adjusted for error appropriately.

  Args:
    encoding_params: the encoding parameters that define the message space size.
    scheme_params: the scheme parameters that define the polynomial modulus.

  Returns:
    An RlwePlaintext representing a padded test polynomial for the identity
    function.
  """
  coeffs = jnp.arange(2**encoding_params.message_bit_length, dtype=jnp.uint32)
  return gen_test_polynomial(
      cleartext_coefficients=coeffs,
      encoding_params=encoding_params,
      scheme_params=scheme_params,
  )


def trivial_encryption(
    test_polynomial: rlwe.RlwePlaintext,
    scheme_params: parameters.SchemeParameters,
) -> rlwe.RlweCiphertext:
  """Trivially encrypts the test polynomial.

  The encryption of the test polynomial is simple for a nuanced reason.  We
  need a valid RLWE encryption with respect to the same secret key used to
  generate the bootstrapping key (the RGSW SK that is used for cmux/external
  product), but that SK is unknown to this function because it is not public.

  However, the "encryption" (0, 0, ..., msg) is a valid RLWE encryption of msg
  with repect to any secret key. This is equivalent to RLWE encryption with a
  PRG that generates zero for every call.

  Args:
    test_polynomial: the test polynomial to encrypt.
    scheme_params: the scheme parameters used for the dimension of the
      encryption.

  Returns:
    An RlweCiphertext with identically zero samples, which encrypts the test
    polynomial with respect to any secret key.
  """

  # We could manually construct the ciphertext here, but for consistency we call
  # out to the real encrypt function with a fake PRG that always outputs zero.
  zero_rng = random_source.ZeroRng()
  empty_sk = rlwe.gen_key(scheme_params, zero_rng)
  return rlwe.encrypt(test_polynomial, empty_sk, zero_rng)


def gen_and_encrypt(
    cleartext_coefficients: jnp.ndarray,
    encoding_params: encoding.EncodingParameters,
    scheme_params: parameters.SchemeParameters,
) -> rlwe.RlweCiphertext:
  """Generates and encrypts a test polynomial for a lookup table.

  Args:
    cleartext_coefficients: the entries of the lookup table.
    encoding_params: the encoding parameters that define the message space size.
    scheme_params: the scheme parameters that define the polynomial modulus.

  Returns:
    An RlweCiphertext representing a padded test polynomial for the given
    look-up table.
  """
  test_poly = gen_test_polynomial(
      cleartext_coefficients, encoding_params, scheme_params
  )
  return trivial_encryption(test_poly, scheme_params)
