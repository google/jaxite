"""A helper module for constructing look-up tables (LUTs)."""
import functools
import itertools
from typing import Callable

import jax.numpy as jnp
from jaxite.jaxite_bool import bool_encoding
from jaxite.jaxite_lib import parameters
from jaxite.jaxite_lib import rlwe
from jaxite.jaxite_lib import test_polynomial
from jaxite.jaxite_lib import types


class LookUpTable:
  """A representation of a truth table.

  Attrs:
    num_inputs: the number of inputs to the boolean function represented.
    truth_table: an integer containing the truth table values, where the LSB is
      the output for the input "all false", the MSB is the output for the input
      "all true".
  """

  def __init__(self, num_inputs: int, truth_table: int):
    self.num_inputs = num_inputs
    self.truth_table = truth_table

  @functools.cached_property
  def _values_from_lsb_to_msb(self) -> tuple[int, ...]:
    return tuple(
        ((self.truth_table & (1 << i)) >> i)
        for i in range(2**self.num_inputs)
    )

  @functools.cached_property
  def as_cleartext_list(self) -> list[types.LweCleartext]:
    """Return the truth table as a list of LWE cleartexts."""
    return [
        bool_encoding.CLEARTEXT_TRUE if bit else bool_encoding.CLEARTEXT_FALSE
        for bit in self._values_from_lsb_to_msb
    ]

  @functools.lru_cache
  def as_rlwe_test_polynomial(
      self,
      scheme_params: parameters.SchemeParameters,
  ) -> rlwe.RlweCiphertext:
    """Return the truth table as an encrypted RLWE test polynomial."""
    msg_bits = bool_encoding.ENCODING_PARAMS.message_bit_length
    padding_required = 2**msg_bits - len(self.as_cleartext_list)

    if padding_required < 0:
      raise ValueError(
          f'Mismatch between LUT {self.truth_table} and '
          f'encoding params {bool_encoding.ENCODING_PARAMS}'
      )

    padded_coefficients = self.as_cleartext_list + (
        [bool_encoding.CLEARTEXT_UNUSED] * padding_required
    )
    test_poly_coefficients = jnp.array(padded_coefficients, dtype=jnp.uint32)
    return test_polynomial.gen_and_encrypt(
        test_poly_coefficients,
        bool_encoding.ENCODING_PARAMS,
        scheme_params,
    )

  def __str__(self) -> str:
    row_fmt = '{input} -> {value}'
    return '\n'.join(
        row_fmt.format(
            input=bin(input)[2:].rjust(self.num_inputs, '0'), value=bit
        )
        for (input, bit) in enumerate(self._values_from_lsb_to_msb)
    )

  def __repr__(self) -> str:
    return (
        'LookUpTable('
        f'num_inputs={self.num_inputs}, '
        f'truth_table={self.truth_table}, '
        ')'
    )

  def __hash__(self) -> int:
    return hash((self.num_inputs, self.truth_table))


def from_callable(
    num_inputs: int,
    # no way to type-annotate a Callable with any number of bool args
    fn: Callable[..., bool],
) -> LookUpTable:
  """Construct a LookUpTable from a callable of bools."""
  inputs = itertools.product([False, True], repeat=num_inputs)
  truth_table = 0
  bit = 0
  for input_bits in inputs:
    truth_table |= int(fn(*input_bits)) << bit
    bit += 1
  return LookUpTable(num_inputs=num_inputs, truth_table=truth_table)


FUNC_NAME_TO_LUT2 = {
    'and': from_callable(2, lambda x, y: x and y),
    'andny': from_callable(2, lambda x, y: not x and y),
    'andyn': from_callable(2, lambda x, y: x and not y),
    'nand': from_callable(2, lambda x, y: not (x and y)),
    'nor': from_callable(2, lambda x, y: not (x or y)),
    'or': from_callable(2, lambda x, y: x or y),
    'orny': from_callable(2, lambda x, y: not x or y),
    'oryn': from_callable(2, lambda x, y: x or not y),
    'xnor': from_callable(2, lambda x, y: x == y),
    'xor': from_callable(2, lambda x, y: x != y),
    'cmux': from_callable(3, lambda v1, v0, ctrl: v1 if ctrl else v0),
}


class LutCache:
  """A cache for LUT test polynomials for chosen scheme parameters."""

  def __init__(self, scheme_params: parameters.SchemeParameters):
    self.scheme_params = scheme_params
    self.test_poly_cache: dict[int, LookUpTable] = {}

  @functools.lru_cache
  def lut(self, num_inputs: int, lut_as_int: int) -> LookUpTable:
    self.test_poly_cache[lut_as_int] = LookUpTable(
        num_inputs=num_inputs, truth_table=lut_as_int
    )
    return self.test_poly_cache[lut_as_int]

  def lut_by_name(self, name: str) -> LookUpTable:
    if name not in FUNC_NAME_TO_LUT2:
      raise ValueError(
          f'Unknown gate name "{name}". Use `get_lut` or else use a '
          f'named 2-bit gate from {list(FUNC_NAME_TO_LUT2.keys())}'
      )
    key = FUNC_NAME_TO_LUT2[name].truth_table
    return self.lut(num_inputs=2, lut_as_int=key)

  def lut_poly(self, num_inputs: int, lut_as_int: int) -> rlwe.RlweCiphertext:
    return self.lut(num_inputs, lut_as_int).as_rlwe_test_polynomial(
        self.scheme_params
    )

  def lut_poly_by_name(self, name: str) -> rlwe.RlweCiphertext:
    """A helper for users that want to query the gate by name."""
    return self.lut_by_name(name).as_rlwe_test_polynomial(self.scheme_params)
