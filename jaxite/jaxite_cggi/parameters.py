"""Class encapsulating params for TFHE."""

import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class SchemeParameters:
  """Scheme parameters for TFHE."""

  # The dimension of the LWE secret key vector.
  # Note an encryption is (a_1, a_2, ..., a_n, b),
  # so an LweCiphertext has length lwe_dimension + 1
  lwe_dimension: int

  # the modulus to use for the LWE plaintext
  plaintext_modulus: int

  # dimension of the RLWE ciphertext
  # An RLWE encryption is composed of (a_1, ..., a_n, b),
  # where n = rlwe_dimension.
  rlwe_dimension: int

  # dimension of N in the polynomial modulus x^N + 1
  polynomial_modulus_degree: int

  # the log of polynomial_modulus_degree
  log_mod_degree: int = dataclasses.field(init=False)

  # the log of polynomial_modulus_degree
  log_plaintext_modulus: int = dataclasses.field(init=False)

  def __post_init__(self) -> None:
    object.__setattr__(
        self,
        'log_mod_degree',
        int(round(math.log2(self.polynomial_modulus_degree))),
    )
    object.__setattr__(
        self,
        'log_plaintext_modulus',
        int(round(math.log2(self.plaintext_modulus))),
    )
