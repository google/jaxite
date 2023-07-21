"""Helpers for type conversions in jaxite_bool client code."""

from typing import List, Optional


def bit_slice_to_u8(bit_slice: List[bool]) -> int:
  """Given a bit slice of length 8, returns a base-10 int representation."""
  if len(bit_slice) != 8:
    raise ValueError(f'Expected an 8-bit representation but got: {bit_slice}.')
  result = 0
  for i in range(8):
    result |= (int(bit_slice[i])) << i
  return result


def u8_to_bit_slice(input_int: int) -> List[bool]:
  """Given an integer [0, 255], returns a bitwise representation."""
  if input_int < 0 or input_int > 255:
    raise ValueError(f'Expected a u8, but got: {input_int}.')
  result: List[bool] = [False] * 8
  for i in range(8):
    result[i] = ((input_int >> i) & 1) != 0
  return result


def u8_list_to_bit_slice(input_list: List[int]) -> List[bool]:
  """Given a list of u8 values, returns a flattened bitwise representation."""
  if not input_list:
    raise ValueError('Expected a non-empty list of u8 values.')
  result: List[bool] = []
  for i in input_list:
    result.extend(u8_to_bit_slice(i))
  return result


def bit_slice_to_bytes(bit_slice: List[bool]) -> bytes:
  """Given a bitwise representation, returns an ASCII bytes object."""
  if not bit_slice:
    raise ValueError('Expected a non-empty bit slice.')
  result: List[int] = []

  char_bit_slice: List[bool] = [False] * 8
  for i, entry in enumerate(bit_slice):
    if (i + 1) % 8 == 0:
      result.append(bit_slice_to_u8(char_bit_slice))
    char_bit_slice[i % 8] = entry

  return bytes(result)


def str_to_cleartext(
    text: str,
    static_len: Optional[int] = None,
    padding_byte: bytes = bytes(' ', 'ascii'),
) -> List[bool]:
  cleartext_bytes = list(bytes(text, 'ascii'))
  if static_len:
    cleartext_bytes.extend(list(padding_byte) * (static_len - len(text)))
  return u8_list_to_bit_slice(cleartext_bytes)


def cleartext_to_str(cleartext: List[bool]) -> str:
  return bit_slice_to_bytes(cleartext).decode('ascii')
