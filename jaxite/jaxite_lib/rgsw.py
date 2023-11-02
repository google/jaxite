"""RGSW encryption scheme."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jaxite.jaxite_lib import decomposition
from jaxite.jaxite_lib import encoding
from jaxite.jaxite_lib import jax_helpers
from jaxite.jaxite_lib import matrix_utils
from jaxite.jaxite_lib import parameters
from jaxite.jaxite_lib import random_source
from jaxite.jaxite_lib import rlwe


@dataclasses.dataclass
class RgswPlaintext:
  """An RGSW plaintext is an unsigned integer and a polynomial modulus degree.

  Note: in the TFHE paper, RGSW plaintext is a polynomial in Z[X] / (X^N + 1),
  where N = `modulus_degree` below. However, in TFHE the only use of RGSW is to
  encrypt the individual bits of an LWE secret key, a binary vector. In this
  case, the RgswPlaintext is in {0, 1} and we can safely restrict the plaintext
  type to an unsigned integer.
  """

  # The degree N of the ring modulus polynomial.
  modulus_degree: int

  # The unsigned integer representing the plaintext.
  message: jnp.uint32

  def __str__(self) -> str:
    return str(self.message)


@dataclasses.dataclass
class RgswCiphertext:
  """An RGSW ciphertext."""

  # A 3-dimensional array, where each row contains an RLWE encryption
  # of a particular bit of the message. Recall, an RLWE encryption is a list of
  # polynomials, so each (i,j) entry corresponds to a polynomial, with the third
  # dimension being the coefficients of the polynomial in order of increasing
  # degree.
  message: jnp.ndarray

  # the log of the modulus q of the polynomial coefficients
  log_coefficient_modulus: int

  # the degree N of the ring modulus polynomial.
  modulus_degree: int

  @property
  def coefficient_modulus(self) -> jnp.uint32:
    return jnp.uint32(2) ** self.log_coefficient_modulus

  def __len__(self):
    return self.message.shape[0]

  def pretty_print_polynomial(self, row: jnp.ndarray) -> str:
    # this does not need to be fast because it will only be used in development.
    s = ' + '.join(
        f'{coeff} x^{power}' for (power, coeff) in enumerate(row) if coeff != 0
    )
    return s if s else '0'

  def __str__(self) -> str:
    s = [
        [self.pretty_print_polynomial(e) + ',' for e in row]
        for row in self.message
    ]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    return '\n[\n  ' + '\n  '.join(table) + '\n]\n'

  def __repr__(self) -> str:
    return str(self)


@dataclasses.dataclass
class RgswSecretKey:
  """This is a wrapper around RlweSecretKey."""

  # the underlying RLWE secret key
  key: rlwe.RlweSecretKey

  def to_rlwe_secret_key(self) -> rlwe.RlweSecretKey:
    return self.key

  def data_at_index(self, i: int) -> jnp.uint32:
    return self.key.data[i]


def gen_key(
    params: parameters.SchemeParameters, prg: random_source.RandomSource
) -> RgswSecretKey:
  """Generate an RGSW secret key."""
  return RgswSecretKey(key=rlwe.gen_key(params, prg))


def key_from_rlwe(rlwe_key: rlwe.RlweSecretKey) -> RgswSecretKey:
  """Convert an RLWE secret key to the corresponding RGSW secret key."""
  return RgswSecretKey(key=rlwe_key)


def encrypt(
    plaintext: RgswPlaintext,
    sk: RgswSecretKey,
    decomposition_params: decomposition.DecompositionParameters,
    prg: random_source.RandomSource,
) -> RgswCiphertext:
  """Create an RGSW ciphertext."""
  k = num_blocks = sk.key.rlwe_dimension
  levels = decomposition_params.level_count
  rlwe_sk = sk.to_rlwe_secret_key()

  # Because of the way the vmaps are set up to map individual rows to each call
  # to encrypt_block and encrypt_and_modify_one_row, we organize the randomness
  # into a matching shape, so that the output of ai_samples[block, level] is in
  # the right shape to be used for a single RLWE encryption. Nb., it needs only
  # k random samples, not k+1 like the output array's shape, because the extra 1
  # comes from the b term of the RLWE encryption.
  ai_samples = prg.uniform(
      shape=(num_blocks + 1, levels, k, rlwe_sk.modulus_degree),
      dtype=jnp.uint32,
  )
  error_samples = prg.rounded_normal(
      shape=(num_blocks + 1, levels),
      dtype=jnp.uint32,
  )

  ciphertext = jit_encrypt(
      plaintext.message,
      rlwe_sk.data,
      ai_samples,
      error_samples,
      decomposition_params,
      num_blocks,
      rlwe_sk.log_coefficient_modulus,
      rlwe_sk.modulus_degree,
  )

  return RgswCiphertext(
      message=ciphertext,
      log_coefficient_modulus=rlwe_sk.log_coefficient_modulus,
      modulus_degree=plaintext.modulus_degree,
  )


@functools.partial(
    jax.jit,
    static_argnames=(
        'decomposition_params',
        'num_blocks',
        'log_coefficient_modulus',
        'modulus_degree',
    ),
)
def jit_encrypt(
    plaintext: jnp.ndarray,
    rlwe_sk: jnp.ndarray,
    ai_samples: jnp.ndarray,
    error_samples: jnp.ndarray,
    decomposition_params: decomposition.DecompositionParameters,
    num_blocks: int,
    log_coefficient_modulus: int,
    modulus_degree: int,
) -> jnp.ndarray:
  """Create an RGSW ciphertext."""
  levels = decomposition_params.level_count
  log_base = decomposition_params.log_base
  levels_range = jnp.arange(1, levels + 1, dtype=jnp.uint32)
  block_range = jnp.arange(num_blocks + 1, dtype=jnp.uint32)
  zero_to_encrypt = jnp.array([0], dtype=jnp.uint32)

  # In the Joye paper, p. 19, we're computing Z = m * G^T,
  # where G^T is defined on p. 17 as the block diagonal matrix
  #
  # ( 1/B                        )
  # ( 1/B^2                      )
  # ( ...                        )
  # ( 1/B^l                      )
  # (        1/B                 )
  # (        1/B^2               )
  # (        ...                 )
  # (        1/B^l               )
  # (               .            )
  # (                 .          )
  # (                   .        )
  # (                      1/B   )
  # (                      1/B^2 )
  # (                      ...   )
  # (                      1/B^l )
  #
  # However, in our case we're working in 32-bit uints, so each value is
  # multiplied by B^l == 2**(log_base*levels), which makes each block above
  # into (B^{l-1}, B^{l-2}, ..., 1).
  #
  # The elements of the matrix are polynomials, which here are represented as as
  # vectors of coefficients.  This requires us to make a 3-dimensional array.
  # The t-th row contains an RLWE encryption, which is a list of polynomials,
  # and the (t,s) entry contains the coefficients of a single polynomial.
  #
  # The jax magic below splits the generation of the matrix above into blocks
  # processed in the same order as in the diagram above.
  #
  #  - encrypt_and_modify_one_row computes the output for a single row of the
  #    matrix above, i.e., one RLWE encryption of zero + the decomposed
  #    plaintext.
  #  - encrypt_block computes vmaps each row computation for a block.
  #  - the final vmap maps across blocks.

  @jax.jit
  def encrypt_and_modify_one_row(
      ai_samples,
      error_sample,
      level,
      block,
      plaintext_message,
  ):
    # For each row of the matrix m * G^T above, we generate a fresh RLWE
    # encryption of zero, and because the message in our case is always an
    # integer, we only need to modify the constant term of the appropriate
    # polynomial, adding this decomposed_pt to it.
    decomposed_pt = plaintext_message << (
        log_coefficient_modulus - log_base * level
    )
    rlwe_zero = rlwe.jit_encrypt(
        zero_to_encrypt,
        rlwe_sk,
        ai_samples,
        error_sample,
        log_coefficient_modulus,
    )

    # Modify the constant coefficient by adding the decomposed plaintext.
    modified_poly = rlwe_zero[block]
    modfied_first_coeff = modified_poly[0] + decomposed_pt
    modified_poly = modified_poly.at[0].set(modfied_first_coeff)
    return rlwe_zero.at[block].set(modified_poly)

  @jax.jit
  def encrypt_block(ai_samples, error_samples, block, plaintext_message):
    return jax.vmap(
        encrypt_and_modify_one_row,
        in_axes=(0, 0, 0, None, None),
        out_axes=0,
    )(ai_samples, error_samples, levels_range, block, plaintext_message)

  # We use batch_vmap here because on GPU, the additional use of
  # i32_as_u8_matmul during encryption results in too much memory usage.
  # However, note that because key generation will typically not happen on a TPU
  # or GPU, this is mostly a mechanism to ensure we can run tests fast in CI.
  ciphertext = jax_helpers.batch_vmap(
      encrypt_block,
      in_axes=(0, 0, 0, None),
      out_axes=0,
      batch_size=1,
  )(ai_samples, error_samples, block_range, plaintext)

  return ciphertext.reshape(
      (levels * (num_blocks + 1), (num_blocks + 1), modulus_degree)
  )


def decrypt(
    ciphertext: RgswCiphertext,
    decomposition_params: decomposition.DecompositionParameters,
    sk: RgswSecretKey,
) -> RgswPlaintext:
  """Decrypts an RGSW ciphertext.

  The approach is to isolate the modified polynomial in a particular
  RgswCiphertext row, and use it to reconstruct the message m.

  - pick `row` = (a_1, ... a_j + mB^{s-1}, ..., a_k, b), * i.e., with entry `j`
      being the modified entry from the encryption step, and the row is chosen
      such that s = levels is maximal.
  - Let b  = sum_{i=1}^k s_i * a_i + e (since it was an RLWE encryption of 0)
  - Let b' = sum_{i=1}^k s_i * row_i = b - e + s_j m B^{s-1}
  - Note s_j must be 1 to recover m, which may affect the choice of row.
  - decomposed_pt_with_noise = b' - b = s_j m B^{s-1} - e = m B^{s-1} - e
                             = decomposed_pt - e
  - pt_with_noise = round(decomposed_pt_with_noise)
  - pt_guess = pt_with_noise[0] mod coeff_modulus
  - pt_guess ~~ plaintext

  Args:
    ciphertext: RGSW ciphertext to decrypt
    decomposition_params: parameters for decomposing the plaintext
    sk: RGSW secret key to use to decrypt

  Returns:
    RgswPlaintext: decrypted plaintext
  """
  k = sk.key.rlwe_dimension
  rlwe_sk = sk.to_rlwe_secret_key()
  log_base = decomposition_params.log_base
  omega = rlwe_sk.log_coefficient_modulus

  # this will fail if there is no such entry, which is necessary or else the
  # decryption will fail.
  try:
    sk_index = next(index for index in range(k) if rlwe_sk.data[index, 0] == 1)
  except StopIteration:
    raise ValueError(
        'Expected SK to have an entry with a nonzero '
        f'constant term, got: {rlwe_sk}'
    ) from None

  row_index = decomposition_params.level_count * sk_index
  ciphertext_row = ciphertext.message[row_index]

  # b = sum{i=0}^k s_i * a_i + e (since it was an RLWE encryption of 0)
  b = ciphertext_row[k]
  # b' = sum_{i=1}^k s_i * row_i
  inner_product = jnp.sum(
      jnp.array(
          [
              matrix_utils.poly_mul(a, s)
              for a, s, in zip(ciphertext_row[:k], rlwe_sk.data)
          ],
          dtype=jnp.uint32,
      ),
      axis=0,
  )
  # decomposed_pt_with_noise = b' - b
  #                          = decomposed_pt - e
  decomposed_pt_with_noise = inner_product - b

  # Round and shift back from the PowersOfB encoding of the original input.
  # the ciphertext row was chosen so that we're always working with the
  # largest block of the decomposition, i.e., corresponding to level=1
  # so this would be omega - log_base * level in the encryption step.
  shift_length = omega - log_base
  pt_with_noise = encoding.round_to_power_of_2(
      decomposed_pt_with_noise, shift_length
  )
  pt_with_noise = pt_with_noise >> shift_length
  # pt_guess = pt_with_noise[0] mod coeff_modulus
  pt_guess = jnp.mod(pt_with_noise[0], 2**log_base)

  return RgswPlaintext(message=pt_guess, modulus_degree=rlwe_sk.modulus_degree)
