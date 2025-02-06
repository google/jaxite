"""Big integer classes for jaxite_ec."""

import gmpy2


class GMPBigInteger:
  """A class representing a big integer using gmpy2.

  This class provides basic arithmetic operations for big integers using gmpy2.
  """

  def __init__(self, value) -> None:
    if isinstance(value, (int, gmpy2.mpz)):
      self.value = gmpy2.mpz(value)
    elif isinstance(value, GMPBigInteger):
      self.value = value.value
    else:
      raise TypeError("Unsupported type for GMPBigInteger initialization")

  def __add__(self, other):
    if isinstance(other, (GMPBigInteger, int)):
      return GMPBigInteger(
          self.value
          + gmpy2.mpz(
              other.value if isinstance(other, GMPBigInteger) else other
          )
      )
    return NotImplemented

  def __sub__(self, other):
    if isinstance(other, (GMPBigInteger, int)):
      return GMPBigInteger(
          self.value
          - gmpy2.mpz(
              other.value if isinstance(other, GMPBigInteger) else other
          )
      )
    return NotImplemented

  def __mul__(self, other):
    if isinstance(other, (GMPBigInteger, int)):
      return GMPBigInteger(
          self.value
          * gmpy2.mpz(
              other.value if isinstance(other, GMPBigInteger) else other
          )
      )
    return NotImplemented

  def __truediv__(self, other):
    if isinstance(other, (GMPBigInteger, int)):
      if (
          gmpy2.mpz(other.value if isinstance(other, GMPBigInteger) else other)
          == 0
      ):
        raise ZeroDivisionError("division by zero")
      return GMPBigInteger(
          self.value
          // gmpy2.mpz(
              other.value if isinstance(other, GMPBigInteger) else other
          )
      )
    return NotImplemented

  def __mod__(self, other):
    if isinstance(other, (GMPBigInteger, int)):
      return GMPBigInteger(
          self.value
          % gmpy2.mpz(
              other.value if isinstance(other, GMPBigInteger) else other
          )
      )
    return NotImplemented

  def __eq__(self, other):
    if isinstance(other, (GMPBigInteger, int)):
      return self.value == gmpy2.mpz(
          other.value if isinstance(other, GMPBigInteger) else other
      )
    return NotImplemented

  def __ne__(self, other):
    if isinstance(other, (GMPBigInteger, int)):
      return self.value != gmpy2.mpz(
          other.value if isinstance(other, GMPBigInteger) else other
      )
    return NotImplemented

  def __lt__(self, other):
    if isinstance(other, (GMPBigInteger, int)):
      return self.value < gmpy2.mpz(
          other.value if isinstance(other, GMPBigInteger) else other
      )
    return NotImplemented

  def __le__(self, other):
    if isinstance(other, (GMPBigInteger, int)):
      return self.value <= gmpy2.mpz(
          other.value if isinstance(other, GMPBigInteger) else other
      )
    return NotImplemented

  def __gt__(self, other):
    if isinstance(other, (GMPBigInteger, int)):
      return self.value > gmpy2.mpz(
          other.value if isinstance(other, GMPBigInteger) else other
      )
    return NotImplemented

  def __ge__(self, other):
    if isinstance(other, (GMPBigInteger, int)):
      return self.value >= gmpy2.mpz(
          other.value if isinstance(other, GMPBigInteger) else other
      )
    return NotImplemented

  def __pow__(self, exponent, modulus=None):
    if isinstance(exponent, (GMPBigInteger, int, gmpy2.mpz)):
      if isinstance(exponent, GMPBigInteger):
        exponent = gmpy2.mpz(exponent.value)
      if isinstance(modulus, GMPBigInteger):
        modulus = gmpy2.mpz(modulus.value)
      if modulus is None:
        return GMPBigInteger(self.value**exponent)
      else:
        return GMPBigInteger(gmpy2.powmod(self.value, exponent, modulus))
    else:
      print(type(exponent))
      raise TypeError("Exponent must be an integer")

  def __lshift__(self, shift):
    """Left shift operator (<<)."""
    if isinstance(shift, GMPBigInteger):
      shift = shift.value
    return GMPBigInteger(self.value << shift)

  def __rshift__(self, shift):
    """Right shift operator (>>)."""
    if isinstance(shift, GMPBigInteger):
      shift = shift.value
    return GMPBigInteger(self.value >> shift)

  def __and__(self, other):
    if isinstance(other, (GMPBigInteger, int)):
      return GMPBigInteger(
          self.value
          & gmpy2.mpz(
              other.value if isinstance(other, GMPBigInteger) else other
          )
      )
    return NotImplemented

  def ceil_log2(self):
    """Calculate the base-2 logarithm of the GMPBigInteger."""
    if self.value <= 0:
      raise ValueError("log2 is only defined for positive integers")
    return GMPBigInteger(gmpy2.ceil(gmpy2.log2(self.value)))

  def __int__(self):
    return int(self.value)

  def __str__(self):
    return str(self.value)

  def __repr__(self):
    return f"GMPBigInteger({self.value})"

  def hex_value_str(self) -> str:
    return hex(self.value)
