"""Security parameters for binary encoding for CGGI schemes."""
import functools

from jaxite.jaxite_bool import bool_encoding
from jaxite.jaxite_bool import lut as bool_lut
from jaxite.jaxite_lib import decomposition
from jaxite.jaxite_lib import encoding
from jaxite.jaxite_lib import lwe
from jaxite.jaxite_lib import parameters
from jaxite.jaxite_lib import random_source
from jaxite.jaxite_lib import rlwe
from jaxite.jaxite_lib import types

ENCODING_PARAMS = bool_encoding.ENCODING_PARAMS


class Parameters:
  """TFHE Boolean gate shared parameters."""

  def __init__(
      self,
      scheme_params: parameters.SchemeParameters,
      ks_decomp_params: decomposition.DecompositionParameters,
      bs_decomp_params: decomposition.DecompositionParameters,
  ) -> None:
    self._scheme_params = scheme_params
    self._ks_decomp_params = ks_decomp_params
    self._bs_decomp_params = bs_decomp_params
    self._lut_cache = bool_lut.LutCache(self._scheme_params)

  @property
  def scheme_params(self) -> parameters.SchemeParameters:
    return self._scheme_params

  @property
  def ks_decomp_params(self) -> decomposition.DecompositionParameters:
    return self._ks_decomp_params

  @property
  def bs_decomp_params(self) -> decomposition.DecompositionParameters:
    return self._bs_decomp_params

  def _noiseless_embedding(
      self, cleartext: types.LweCleartext
  ) -> types.LweCiphertext:
    return lwe.noiseless_embedding(
        encoding.encode(cleartext, ENCODING_PARAMS),
        self.scheme_params.lwe_dimension,
    )

  @functools.cached_property
  def noiseless_true(self) -> types.LweCiphertext:
    return self._noiseless_embedding(bool_encoding.CLEARTEXT_TRUE)

  @functools.cached_property
  def noiseless_false(self) -> types.LweCiphertext:
    return self._noiseless_embedding(bool_encoding.CLEARTEXT_FALSE)

  def lut_poly(self, num_inputs: int, truth_table: int) -> rlwe.RlweCiphertext:
    return self._lut_cache.lut_poly(
        num_inputs=num_inputs, lut_as_int=truth_table
    )

  def lut_poly_by_name(self, name: str) -> rlwe.RlweCiphertext:
    return self._lut_cache.lut_poly_by_name(name)

  def lut(self, num_inputs: int, truth_table: int) -> bool_lut.LookUpTable:
    return self._lut_cache.lut(num_inputs=num_inputs, lut_as_int=truth_table)

  def lut_by_name(self, name: str) -> bool_lut.LookUpTable:
    return self._lut_cache.lut_by_name(name)


def get_params_for_128_bit_security() -> Parameters:
  """Returns boolean scheme params for 128 bit security."""
  return Parameters(
      SCHEME_PARAMS_128_BIT_SECURITY,
      KSK_DECOMP_PARAMS_128_BIT_SECURITY,
      BSK_DECOMP_PARAMS_128_BIT_SECURITY,
  )


def get_lwe_rng_for_128_bit_security(
    seed: int,
) -> random_source.PseudorandomSource:
  """Returns lwe rng for 128 bit security."""
  # This parameter was selected to satisfy 128-bit security while giving
  # efficient performance. While the performance measurements are for TPUs and
  # must be kept Google-internal, the general approach is outlined in this blog
  # post:
  # https://jeremykun.com/2022/12/28/estimating-the-security-of-ring-learning-with-errors-rlwe/
  return random_source.PseudorandomSource(seed=seed, normal_std=2**14)


def get_rlwe_rng_for_128_bit_security(
    seed: int,
) -> random_source.PseudorandomSource:
  """Returns rlwe rng for 128 bit security."""
  # This parameter was selected to satisfy 128-bit security while giving
  # efficient performance. While the performance measurements are for TPUs and
  # must be kept Google-internal, the general approach is outlined in this blog
  # post:
  # https://jeremykun.com/2022/12/28/estimating-the-security-of-ring-learning-with-errors-rlwe/
  return random_source.PseudorandomSource(seed=seed, normal_std=256)


# Scheme params and decomposition params for 128 bit security.
SCHEME_PARAMS_128_BIT_SECURITY = parameters.SchemeParameters(
    lwe_dimension=800,
    plaintext_modulus=4294967296,
    rlwe_dimension=2,
    polynomial_modulus_degree=512,
)
# Note: The following condition needs to be met for decomposition parameters
# in order to avoid loss of precision:
# (2^decomposition_log_base)^decomposition_level_count == plaintext_modulus.
BSK_DECOMP_PARAMS_128_BIT_SECURITY = decomposition.DecompositionParameters(
    log_base=4, level_count=6
)
# These parameters give an approximate decomposition. See go/keyswitch-error.
KSK_DECOMP_PARAMS_128_BIT_SECURITY = decomposition.DecompositionParameters(
    log_base=4, level_count=5
)


def get_params_for_test() -> Parameters:
  """Returns boolean scheme params for tests."""
  return Parameters(
      TEST_SCHEME_PARAMS, TEST_KSK_DECOMP_PARAMS, TEST_BSK_DECOMP_PARAMS
  )


def get_rng_for_test(seed: int) -> random_source.PseudorandomSource:
  """Returns rng for tests."""
  return random_source.PseudorandomSource(
      uniform_bounds=(0, 2**32), normal_std=1, seed=seed
  )


# Scheme params and decomposition params used in unit tests.
TEST_SCHEME_PARAMS = parameters.SchemeParameters(
    lwe_dimension=4,
    plaintext_modulus=2**32,
    rlwe_dimension=1,
    polynomial_modulus_degree=512,
)
# Note: The following condition needs to be met for decomposition parameters:
# (2^decomposition_log_base)^decomposition_level_count == plaintext_modulus.
TEST_BSK_DECOMP_PARAMS = decomposition.DecompositionParameters(
    log_base=4, level_count=8
)
TEST_KSK_DECOMP_PARAMS = TEST_BSK_DECOMP_PARAMS
