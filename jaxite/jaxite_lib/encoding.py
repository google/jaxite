"""Logic for encoding and decoding a cleartext for use in TFHE."""

import dataclasses
from typing import Union

import jax.numpy as jnp
from jaxite.jaxite_lib import types


@dataclasses.dataclass
class EncodingParameters:
  """The parameters for encoding a cleartext for use in TFHE.

  See g.p.p.fhe.jaxite.jaxite_lib.encoding.encode for the structure of the
  encoding.
  """

  # total number of bits in a plaintext
  total_bit_length: int

  # the number of bits to use to store the message
  message_bit_length: int

  # The number of bits to use for padding, stored in the most sigificant bits
  # of a plaintext. This is needed to avoid the negation that occurs when
  # blind_rotate ends up indexing past the degree of the test polynomial.
  # See go/cggi#padding-and-blind-rotate for more details.
  # TODO(b/238643320): determine if we can/should automatically set this based
  # on the scheme parameters.
  padding_bit_length: int

  # the number of bits used for error; stored in the LSBs
  error_bit_length: int = dataclasses.field(init=False)

  # minimum allowed message value
  message_min: types.LweCleartext = dataclasses.field(init=False)

  # maximum allowed message value
  message_max: types.LweCleartext = dataclasses.field(init=False)

  # range of allowed message values
  message_range: types.LweCleartext = dataclasses.field(init=False)

  def __post_init__(self) -> None:
    # verify bit lengths are of appropriate size
    if self.total_bit_length < 1:
      raise ValueError('Total bit length must be >= 1.')
    if self.message_bit_length < 1:
      raise ValueError('Message bit length must be >= 1.')
    if self.padding_bit_length < 0:
      raise ValueError('Padding bit length must be >= 0.')

    # compute error bit length based on available space
    occupied_bits = self.message_bit_length + self.padding_bit_length
    remaining_bits = self.total_bit_length - occupied_bits

    if remaining_bits < 0:
      raise ValueError(f'Total bit length of {self.total_bit_length} exceeded.')

    self.error_bit_length = remaining_bits
    self.message_min = 0
    self.message_max = (1 << self.message_bit_length) - 1
    self.message_range = self.message_max - self.message_min


def encode(
    message: Union[types.LweCleartext, jnp.ndarray], params: EncodingParameters
) -> Union[types.LwePlaintext, jnp.ndarray]:
  """Encode a plaintext or array of plaintexts for use in a TFHE ciphertext.

  The bits are organized so that the top bits are padding for overflow,
  followed by message bits, followed by space left for noise. E.g.,

           00 0101101 00000
      padding message noise

  This method also works on arrays, for use in encoding the coefficients
  of a test polynomial.

  Args:
    message: the cleartext message.
    params: the parameters of the encoding.

  Returns:
    The encoded message or list of messages.

  Raises:
    ValueError: In the event that `message` is outside of the message bounds as
    specified in `params`.
  """
  dtype = types.LwePlaintext
  msg_arr = jnp.atleast_1d(message).astype(dtype)
  if (msg_arr > dtype(params.message_max)).any() or (
      msg_arr < dtype(params.message_min)
  ).any():
    raise ValueError(
        f'{message} is outside of allowable bounds '
        f'[{params.message_min}, {params.message_max}].'
    )

  return message << dtype(params.error_bit_length)


def decode(
    plaintext: types.LwePlaintext, params: EncodingParameters
) -> types.LweCleartext:
  """Decode a plaintext.

  Args:
    plaintext: the encoded plaintext message.
    params: the parameters of the encoding.

  Returns:
    The cleartext message.
  """
  shifted = remove_noise(plaintext, params) >> params.error_bit_length

  # In cases where the params.total_bit_length < 32, operations like modulus
  # switching can result in a plaintext that spills into the range of bits that
  # are outside the highest bit in the encoding. E.g., if the cleartext=0 and
  # message_bits=3, modulus switching can result in encrypting the "cleartext"
  # 8, which is only correct if you ignore the top bit. To deal with cases like
  # this (and really, this matters only for tests where we are not using all the
  # bits of the message space), we compute a final modulus with respect to the
  # message space size.
  message_space_mask = (1 << params.message_bit_length) - 1
  return types.LweCleartext(shifted & message_space_mask)


def remove_noise(
    inp: jnp.ndarray, encoding_params: EncodingParameters
) -> jnp.ndarray:
  """Uses encoding_params to round the inp to remove noise."""
  return round_to_power_of_2(inp, encoding_params.error_bit_length)


def round_to_power_of_2(arr: jnp.ndarray, log_pow_of_2: int) -> jnp.ndarray:
  """Rounds to the nearest multiple of a given power of 2."""
  # This bit determines whether to round up or down
  round_up_or_down_bit = log_pow_of_2 - 1
  lowest_unrounded_bit = log_pow_of_2

  # Shift down to clear all the bits that are rounded off, optionally add 1 to
  # round up, then shift back up.
  round_up = jnp.bitwise_and(jnp.right_shift(arr, round_up_or_down_bit), 1)
  return jnp.left_shift(
      jnp.right_shift(arr, lowest_unrounded_bit) + round_up,
      lowest_unrounded_bit,
  )


def extract_noise(
    plaintext: types.LwePlaintext, encoding_params: EncodingParameters
) -> int:
  """Extracts the noise bits of a plaintext as a (signed) int."""
  rounded = remove_noise(plaintext, encoding_params)
  return int(jnp.int32(plaintext)) - int(jnp.int32(rounded))
