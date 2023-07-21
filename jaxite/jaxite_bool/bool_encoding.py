"""A module to store hard-coded encoding information about jaxite_bool."""
from jaxite.jaxite_lib import encoding
from jaxite.jaxite_lib import types

ENCODING_PARAMS = encoding.EncodingParameters(
    # leaves 32 - 3 - 1 bits for noise
    total_bit_length=32,
    message_bit_length=3,
    padding_bit_length=1,
)
CLEARTEXT_FALSE = types.LweCleartext(0)
CLEARTEXT_TRUE = types.LweCleartext(1)

# This value is used for the "illegal" entry of a test polynomial, which
# bootstrap should never produce as output if our implementation is correct and
# the ciphertext error does not grow too large. The decryption function has an
# assertion that this value is never the final output of a circuit.
CLEARTEXT_UNUSED = types.LweCleartext(2)
