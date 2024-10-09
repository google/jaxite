"""High-precision integer class for ZKP."""

import math

import gmpy2


class HPint:
  """An integer that supports arbitrary precision arithmetic."""

  def __init__(self, value) -> None:
    if isinstance(value, int):
      self.value = value
    elif isinstance(value, HPint):
      self.value = value.value
    else:
      raise TypeError("Unsupported type for HPint initialization")

  def __add__(self, other):
    if isinstance(other, (HPint, int)):
      return HPint(
          self.value + (other.value if isinstance(other, HPint) else other)
      )
    return NotImplemented

  def __sub__(self, other):
    if isinstance(other, (HPint, int)):
      return HPint(
          self.value - (other.value if isinstance(other, HPint) else other)
      )
    return NotImplemented

  def __mul__(self, other):
    if isinstance(other, (HPint, int)):
      return HPint(
          self.value * (other.value if isinstance(other, HPint) else other)
      )
    return NotImplemented

  def __truediv__(self, other):
    if isinstance(other, (HPint, int)):
      if (other.value if isinstance(other, HPint) else other) == 0:
        raise ZeroDivisionError("division by zero")
      return HPint(
          self.value // (other.value if isinstance(other, HPint) else other)
      )
    return NotImplemented

  def __mod__(self, other):
    if isinstance(other, (HPint, int)):
      return HPint(
          self.value % (other.value if isinstance(other, HPint) else other)
      )
    return NotImplemented

  def __eq__(self, other):
    if isinstance(other, (HPint, int)):
      return self.value == (other.value if isinstance(other, HPint) else other)
    return NotImplemented

  def __ne__(self, other):
    if isinstance(other, (HPint, int)):
      return self.value != (other.value if isinstance(other, HPint) else other)
    return NotImplemented

  def __lt__(self, other):
    if isinstance(other, (HPint, int)):
      return self.value < (other.value if isinstance(other, HPint) else other)
    return NotImplemented

  def __le__(self, other):
    if isinstance(other, (HPint, int)):
      return self.value <= (other.value if isinstance(other, HPint) else other)
    return NotImplemented

  def __gt__(self, other):
    if isinstance(other, (HPint, int)):
      return self.value > (other.value if isinstance(other, HPint) else other)
    return NotImplemented

  def __ge__(self, other):
    if isinstance(other, (HPint, int)):
      return self.value >= (other.value if isinstance(other, HPint) else other)
    return NotImplemented

  def __pow__(self, exponent, modulus=None):
    if isinstance(exponent, (HPint, int)):
      if isinstance(exponent, HPint):
        exponent = exponent.value
      if modulus is None:
        return HPint(pow(self.value, exponent))
      else:
        return HPint(pow(self.value, exponent, modulus))
    else:
      raise TypeError("Exponent must be an integer")

  def __lshift__(self, shift):
    """Left shift operator (<<)"""
    return HPint(self.value << shift)

  def __rshift__(self, shift):
    """Right shift operator (>>)"""
    if isinstance(shift, HPint):
      shift = shift.value
    return HPint(self.value >> shift)

  def __and__(self, other):
    if isinstance(other, (HPint, int)):
      return HPint(
          self.value & (other.value if isinstance(other, HPint) else other)
      )
    return NotImplemented

  def ceil_log2(self) -> float:
    """Calculate the base-2 logarithm of the HPint."""
    if self.value <= 0:
      raise ValueError("log2 is only defined for positive integers")
    return HPint(math.ceil(math.log2(self.value)))

  def __int__(self):
    return int(self.value)

  def __str__(self):
    return str(self.value)

  def __repr__(self):
    return f"HPint({self.value})"

  def hex_value_str(self) -> str:
    return hex(self.value)


class GMPHPint(HPint):

  def __init__(self, value) -> None:
    if isinstance(value, (int, gmpy2.mpz)):
      self.value = gmpy2.mpz(value)
    elif isinstance(value, HPint):
      self.value = gmpy2.mpz(value.value)
    else:
      raise TypeError("Unsupported type for GMPHPint initialization")

  def __add__(self, other):
    if isinstance(other, (GMPHPint, int)):
      return GMPHPint(
          self.value
          + gmpy2.mpz(other.value if isinstance(other, GMPHPint) else other)
      )
    return NotImplemented

  def __sub__(self, other):
    if isinstance(other, (GMPHPint, int)):
      return GMPHPint(
          self.value
          - gmpy2.mpz(other.value if isinstance(other, GMPHPint) else other)
      )
    return NotImplemented

  def __mul__(self, other):
    if isinstance(other, (GMPHPint, int)):
      return GMPHPint(
          self.value
          * gmpy2.mpz(other.value if isinstance(other, GMPHPint) else other)
      )
    return NotImplemented

  def __truediv__(self, other):
    if isinstance(other, (GMPHPint, int)):
      if gmpy2.mpz(other.value if isinstance(other, GMPHPint) else other) == 0:
        raise ZeroDivisionError("division by zero")
      return GMPHPint(
          self.value
          // gmpy2.mpz(other.value if isinstance(other, GMPHPint) else other)
      )
    return NotImplemented

  def __mod__(self, other):
    if isinstance(other, (GMPHPint, int)):
      return GMPHPint(
          self.value
          % gmpy2.mpz(other.value if isinstance(other, GMPHPint) else other)
      )
    return NotImplemented

  def __eq__(self, other):
    if isinstance(other, (GMPHPint, int)):
      return self.value == gmpy2.mpz(
          other.value if isinstance(other, GMPHPint) else other
      )
    return NotImplemented

  def __ne__(self, other):
    if isinstance(other, (GMPHPint, int)):
      return self.value != gmpy2.mpz(
          other.value if isinstance(other, GMPHPint) else other
      )
    return NotImplemented

  def __lt__(self, other):
    if isinstance(other, (GMPHPint, int)):
      return self.value < gmpy2.mpz(
          other.value if isinstance(other, GMPHPint) else other
      )
    return NotImplemented

  def __le__(self, other):
    if isinstance(other, (GMPHPint, int)):
      return self.value <= gmpy2.mpz(
          other.value if isinstance(other, GMPHPint) else other
      )
    return NotImplemented

  def __gt__(self, other):
    if isinstance(other, (GMPHPint, int)):
      return self.value > gmpy2.mpz(
          other.value if isinstance(other, GMPHPint) else other
      )
    return NotImplemented

  def __ge__(self, other):
    if isinstance(other, (GMPHPint, int)):
      return self.value >= gmpy2.mpz(
          other.value if isinstance(other, GMPHPint) else other
      )
    return NotImplemented

  def __pow__(self, exponent, modulus=None):
    if isinstance(exponent, (GMPHPint, int, gmpy2.mpz)):
      if isinstance(exponent, GMPHPint):
        exponent = gmpy2.mpz(exponent.value)
      if isinstance(modulus, GMPHPint):
        modulus = gmpy2.mpz(modulus.value)
      if modulus is None:
        return GMPHPint(self.value**exponent)
      else:
        return GMPHPint(gmpy2.powmod(self.value, exponent, modulus))
    else:
      print(type(exponent))
      raise TypeError("Exponent must be an integer")

  def __lshift__(self, shift):
    """Left shift operator (<<)"""
    if isinstance(shift, HPint):
      shift = shift.value
    return GMPHPint(self.value << shift)

  def __rshift__(self, shift):
    """Right shift operator (>>)"""
    if isinstance(shift, HPint):
      shift = shift.value
    return GMPHPint(self.value >> shift)

  def __and__(self, other):
    if isinstance(other, (GMPHPint, int)):
      return GMPHPint(
          self.value
          & gmpy2.mpz(other.value if isinstance(other, GMPHPint) else other)
      )
    return NotImplemented

  def ceil_log2(self):
    """Calculate the base-2 logarithm of the GMPHPint."""
    if self.value <= 0:
      raise ValueError("log2 is only defined for positive integers")
    return GMPHPint(gmpy2.ceil(gmpy2.log2(self.value)))

  def __str__(self):
    return str(self.value)

  def __repr__(self):
    return f"GMPHPint({self.value})"
