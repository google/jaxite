"""Modular Reduction with finite_field Operations for ZKP."""

from collections.abc import Sequence
import copy
import math

from absl import app

# from config_file import USE_GMP
USE_GMP = True
if USE_GMP:
  from hp_int import GMPHPint as HPint
else:
  from hp_int import HPint as HPint


class FiniteFieldElement:
  """Finite Field Element Operation Library."""

  def __init__(self, value, prime, k=None):
    if isinstance(value, HPint):
      self.value = value
    else:
      self.value = HPint(value)

    if isinstance(prime, HPint):
      self.prime = prime
    else:
      self.prime = HPint(prime)

    if value < 0 or value >= prime:
      raise ValueError(f"Value {value} not in range 0 to {prime - 1}")

  def set_value(self, value):
    """Set the value of the finite field element, with validation."""
    if isinstance(value, HPint):
      new_value = value
    else:
      new_value = HPint(value)

    if new_value < HPint(0) or new_value >= self.prime:
      raise ValueError(f"Value {new_value} not in range 0 to {self.prime - 1}")

    self.value = new_value

  def get_prime(self):
    return copy.deepcopy(self.prime)

  def copy(self, value=None, transform=False):
    """Create a deep copy of the current finite field element."""
    obj = copy.copy(self)
    if value == None:
      obj.value = HPint(self.value)
    else:
      obj.value = HPint(value)
    return obj

  def __add__(self, other):
    if self.prime != other.prime:
      raise ValueError("Cannot add two numbers in different Fields")
    result = (self.value + other.value) % self.prime
    return FiniteFieldElement(result.value, self.prime.value)

  def __sub__(self, other):
    if self.prime != other.prime:
      raise ValueError("Cannot subtract two numbers in different Fields")
    result = (self.value - other.value) % self.prime
    return FiniteFieldElement(result.value, self.prime.value)

  def __mul__(self, other):
    if self.prime != other.prime:
      raise ValueError("Cannot multiply two numbers in different Fields")
    result = (self.value * other.value) % self.prime
    return FiniteFieldElement(result.value, self.prime.value)

  def __truediv__(self, other):
    if self.prime != other.prime:
      raise ValueError("Cannot divide two numbers in different Fields")
    # Use Fermat's Little Theorem to find the inverse: a^(p-1) ≡ 1 (mod p)
    inverse = other.value.__pow__(self.prime.value - 2, self.prime.value)
    result = (self.value * inverse) % self.prime
    return FiniteFieldElement(result.value, self.prime.value)

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

  def __init__(self, value, prime, k=None):
    super().__init__(value, prime)
    if k == None:
      if isinstance(prime, HPint):
        self.two_k = prime.ceil_log2() * 2
      else:
        self.two_k = HPint(math.ceil(math.log2(prime))) * 2
    else:
      self.two_k = HPint(2 * k)
    self.mu = HPint(2) ** self.two_k / prime

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


class FiniteFieldElementMontgomery(FiniteFieldElement):

  def __init__(self, value, prime, k=None):
    super().__init__(value, prime, k)
    if k == None:
      if isinstance(prime, HPint):
        self.k = prime.ceil_log2()
      else:
        self.k = HPint(math.ceil(math.log2(prime)))
    else:
      self.k = HPint(k)

    self.r = HPint(2) ** self.k
    # self.r_inverse = (self.r ** (self.prime - 2)) % self.prime
    self.r_inverse = self.r.__pow__(self.prime - 2, self.prime)
    self.n_prime = (self.r * self.r_inverse - 1) / self.prime
    self.r_mask = self.r - 1
    self.value = self.montgomeryize(self.value)
    self.montgomeryized = True
    self.one_bar = self.montgomeryize(HPint(1))

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

  def copy(self, value=None, transform=False):
    """Create a deep copy of the current finite field element."""
    obj = copy.copy(self)
    if value == None:
      obj.value = HPint(self.value)
    else:
      if transform:
        obj.value = self.montgomeryize(HPint(value))
      else:
        obj.value = HPint(value)
    return obj


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")


if __name__ == "__main__":
  app.run(main)
