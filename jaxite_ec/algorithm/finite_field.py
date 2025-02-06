"""Finite field elements for elliptic curve cryptography."""

import copy
import math

from jaxite.jaxite_ec.algorithm import big_integer

BigInt = big_integer.GMPBigInteger


class FiniteFieldElement:
  """Finite field element for elliptic curve cryptography."""

  def __init__(self, value, prime):
    if isinstance(value, BigInt):
      self.value = value
    else:
      self.value = BigInt(value)

    if isinstance(prime, BigInt):
      self.prime = prime
    else:
      self.prime = BigInt(prime)

    if self.value < 0 or self.value >= self.prime:
      raise ValueError(f"Value {self.value} not in range 0 to {self.prime - 1}")

  def set_value(self, value):
    """Set the value of the finite field element, with validation."""
    if isinstance(value, BigInt):
      new_value = value
    else:
      new_value = BigInt(value)

    if new_value < BigInt(0) or new_value >= self.prime:
      raise ValueError(f"Value {new_value} not in range 0 to {self.prime - 1}")

    self.value = new_value

  def get_value(self) -> int:
    return int(self.value)

  def get_prime(self):
    return copy.deepcopy(self.prime)

  def copy(self, value=None, transform=False, reduction=False):
    """Create a deep copy of the current finite field element.

    transform and reduction are only place holders for unified API.

    Args:
        value (int): The value of the new finite field element. If None, the
          value of the current finite field element is used.
        transform (bool): Whether to transform the value of the new finite field
          element.
        reduction (bool): Whether to reduce the value of the new finite field
          element.

    Returns:
        FiniteFieldElement: A deep copy of the current finite field element.
    """
    obj = copy.copy(self)
    if not value:
      obj.value = BigInt(int(self.value))
    else:
      obj.value = BigInt(value)
    return obj

  def __add__(self, other):
    if self.prime != other.prime:
      raise ValueError("Cannot add two numbers in different Fields")
    result = (self.value + other.value) % self.prime
    return FiniteFieldElement(result, self.prime.value)

  def __sub__(self, other):
    if self.prime != other.prime:
      raise ValueError("Cannot subtract two numbers in different Fields")
    result = (self.value - other.value) % self.prime
    return FiniteFieldElement(result, self.prime.value)

  def __mul__(self, other):
    if self.prime != other.prime:
      raise ValueError("Cannot multiply two numbers in different Fields")
    result = (self.value * other.value) % self.prime
    return FiniteFieldElement(result, self.prime.value)

  def __truediv__(self, other):
    if self.prime != other.prime:
      raise ValueError("Cannot divide two numbers in different Fields")
    # Use Fermat's Little Theorem to find the inverse: a^(p-1) ≡ 1 (mod p)
    inverse = other.value.__pow__(self.prime.value - 2, self.prime.value)
    result = (self.value * inverse) % self.prime
    return FiniteFieldElement(result, self.prime.value)

  def __pow__(self, exponent):
    result = self.value.__pow__(exponent, self.prime.value)
    return FiniteFieldElement(result.value, self.prime.value)

  def __eq__(self, other):
    return self.value == other.value and self.prime == other.prime

  def __str__(self):
    return f"FieldElement_{self.prime.value}({self.value.value})"

  def __repr__(self):
    return (
        f"FiniteFieldElement(value={self.value.value},"
        f" prime={self.prime.value})"
    )

  def __hex__(self):
    return hex(int(self.value.value))

  def hex_value_str(self) -> str:
    return self.value.hex_value_str()


class FiniteFieldElementBarrett(FiniteFieldElement):
  """Finite field element for elliptic curve cryptography using Barrett reduction."""

  def __init__(self, value, prime, k=None):
    super().__init__(value, prime)
    if k == None:
      if isinstance(prime, BigInt):
        self.two_k = prime.ceil_log2() * 2
      else:
        self.two_k = BigInt(math.ceil(math.log2(prime))) * 2
    else:
      self.two_k = BigInt(2 * k)
    self.mu = BigInt(2) ** self.two_k / prime

  def barrett_reduction(self, x):
    # q = (x * mu) >> 2k
    q = (x * self.mu) >> self.two_k
    # r = x - q * prime
    r = x - q * self.prime
    if r >= self.prime:
      r -= self.prime
    return r

  def __add__(self, other):
    if self.prime != other.prime:
      raise ValueError("Cannot add two numbers in different Fields")
    result = self.value + other.value
    if result > self.prime:
      result -= self.prime
    new_instance = self.copy()
    new_instance.value = result
    return new_instance

  def __sub__(self, other):
    if self.prime != other.prime:
      raise ValueError("Cannot subtract two numbers in different Fields")
    if self.value < other.value:
      result = self.value + self.prime - other.value
    else:
      result = self.value - other.value
    new_instance = self.copy()
    new_instance.value = result
    return new_instance

  def __mul__(self, other):
    if self.prime != other.prime:
      raise ValueError("Cannot multiply two numbers in different Fields")
    result = self.value * other.value
    reduced_result = self.barrett_reduction(result)
    new_instance = self.copy()
    new_instance.value = reduced_result
    return new_instance

  def __truediv__(self, other):
    if self.prime != other.prime:
      raise ValueError("Cannot divide two numbers in different Fields")
    inverse = other.value.__pow__(self.prime.value - 2, self.prime.value)
    result = self.value * inverse
    reduced_result = self.barrett_reduction(result)
    new_instance = self.copy()
    new_instance.value = reduced_result
    return new_instance

  def copy(self, value=None, transform=False, reduction=False):
    """Create a deep copy of the current finite field element."""
    obj = copy.copy(self)
    if not value:
      obj.value = BigInt(self.value)
    else:
      if reduction:
        value = self.barrett_reduction(BigInt(value))
      obj.value = BigInt(value)
    return obj


class FiniteFieldElementMontgomery(FiniteFieldElement):
  """Finite field element for elliptic curve cryptography using Montgomery reduction."""

  def __init__(self, value, prime, k=None):
    super().__init__(value, prime)
    if not k:
      if isinstance(prime, BigInt):
        self.k = prime.ceil_log2()
      else:
        self.k = BigInt(math.ceil(math.log2(prime)))
    else:
      self.k = BigInt(k)

    self.r = BigInt(2) ** self.k
    # self.r_inverse = (self.r ** (self.prime - 2)) % self.prime
    self.r_inverse = self.r.__pow__(self.prime - 2, self.prime)
    self.n_prime = (self.r * self.r_inverse - 1) / self.prime
    self.r_mask = self.r - 1
    self.value = self.montgomeryize(self.value)
    self.montgomeryized = True
    self.one_bar = self.montgomeryize(BigInt(1))

  def montgomery_reduction(self, x):
    m = ((x & self.r_mask) * self.n_prime) & self.r_mask
    u = (x + m * self.prime) >> self.k
    if u >= self.prime:
      u -= self.prime
    return u

  def montgomeryize(self, x):
    x_bar = (x * self.r) % self.prime
    return x_bar

  def de_montgomeryize(self, x_bar):
    x = self.montgomery_reduction(x_bar)
    return x

  def change_montgomery_form(self):
    if self.montgomeryized:
      self.value = self.de_montgomeryize(self.value)
      self.montgomeryized = False
    else:
      self.value = self.montgomeryize(self.value)
      self.montgomeryized = True
    return self

  def __add__(self, other):
    if self.prime != other.prime:
      raise ValueError("Cannot add two numbers in different Fields")
    assert other.montgomeryized
    result = self.value + other.value
    if result > self.prime:
      result -= self.prime
    new_instance = self.copy()
    new_instance.value = result
    return new_instance

  def __sub__(self, other):
    if self.prime != other.prime:
      raise ValueError("Cannot subtract two numbers in different Fields")
    assert other.montgomeryized
    if self.value < other.value:
      result = self.value + self.prime - other.value
    else:
      result = self.value - other.value
    new_instance = self.copy()
    new_instance.value = result
    return new_instance

  def __mul__(self, other):
    if self.prime != other.prime:
      raise ValueError("Cannot multiply two numbers in different Fields")
    assert other.montgomeryized
    result = self.value * other.value
    reduced_result = self.montgomery_reduction(result)
    new_instance = self.copy()
    new_instance.value = reduced_result
    return new_instance

  def __truediv__(self, other):
    if self.prime != other.prime:
      raise ValueError("Cannot divide two numbers in different Fields")

    if other.montgomeryized:
      other_value = self.de_montgomeryize(other.value)
    else:
      other_value = other.value

    inverse = other_value.__pow__(self.prime.value - 2, self.prime.value)

    inverse_bar = self.montgomeryize(inverse)

    result = self.value * inverse_bar
    reduced_result = self.montgomery_reduction(result)
    new_instance = self.copy()
    new_instance.value = reduced_result
    return new_instance

  def copy(self, value=None, transform=False, reduction=False):
    """Create a deep copy of the current finite field element."""
    obj = copy.copy(self)
    if not value:
      obj.value = BigInt(self.value)
    else:
      if reduction:
        obj.value = self.montgomery_reduction(BigInt(value))
      elif transform:
        obj.value = self.montgomeryize(BigInt(value))
      else:
        obj.value = BigInt(value)

    return obj
