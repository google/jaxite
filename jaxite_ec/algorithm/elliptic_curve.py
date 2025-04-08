"""Elliptic curve coordinate systems and points."""

import abc
import copy
import enum
from typing import Dict, Generic, List, Optional, TypeVar, Union

from jaxite.jaxite_ec.algorithm import big_integer
from jaxite.jaxite_ec.algorithm import finite_field as finite_field_lib


BigInt = big_integer.GMPBigInteger

abstractmethod = abc.abstractmethod
ABC = abc.ABC
Auto = enum.auto
Enum = enum.Enum
T = TypeVar('T')
FieldEle = finite_field_lib.FiniteFieldElement


class CoordinateSystemType(Enum):
  """Enum to represent different types of coordinate systems for elliptic curves.

  AFFINE: The affine coordinate system, the standard coordinate system for
  elliptic curves: https://www.hyperelliptic.org/EFD/g1p/auto-shortw.html

  PROJECTIVE: The projective coordinate system,
  https://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#addition-madd-1998-cmo

  XYZZ: The XYZZ coordinate system:
  https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html
  """

  NONE = Auto()
  WEIERSTRASS_AFFINE = Auto()
  WEIERSTRASS_PROJECTIVE = Auto()
  WEIERSTRASS_XYZZ = Auto()


class ECPoint(Generic[T]):
  """Represents a point in an elliptic curve coordinate system."""

  def __init__(
      self,
      coordinates: List[T],
      coordinate_system: 'EllipticCurveCoordinateSystem',
      zero: bool = False,
  ) -> None:
    self.coordinate_system = coordinate_system
    self.zero = zero
    if self.zero:
      self.coordinates = None
    else:
      self.coordinates = self.coordinate_system.generate_formal_coordinates(
          coordinates
      )
    self.type = coordinate_system.get_type()

  def __getitem__(self, index: int) -> T:
    """Allows access to elements in self.coordinates using index."""
    return self.coordinates[index]

  def __setitem__(self, index: int, value: T) -> None:
    """Allows modification of elements in self.coordinates using index."""
    if self.coordinates is None:
      raise TypeError('unsupported operand type(s) for item assignment: None')
    self.coordinates[index] = value

  def __eq__(self, other: 'ECPoint') -> bool:
    if not isinstance(other, ECPoint):
      return NotImplemented

    return self.coordinates == other.coordinates

  def __add__(self, other: 'ECPoint') -> 'ECPoint':
    return self.coordinate_system.point_add(self, other)

  def __lshift__(self, shift) -> 'ECPoint':
    if self.zero:
      return self.copy()
    return self.coordinate_system.point_lshift(self, shift)

  def is_zero(self):
    return self.zero

  def get_type(self) -> CoordinateSystemType:
    return self.type

  def set_type(self, cs_type: CoordinateSystemType):
    self.type = cs_type

  def set_coordinate_system(
      self, coordinate_system: 'EllipticCurveCoordinateSystem'
  ):
    self.coordinate_system = coordinate_system
    self.type = coordinate_system.get_type()

  def append(self, coordinate: T):
    assert self.coordinates is not None
    self.coordinates.append(coordinate)

  def copy(self):
    obj = copy.copy(self)
    if not self.is_zero():
      obj.coordinates = self.coordinate_system.generate_formal_coordinates(
          self.coordinates
      )
    return obj

  def convert_to_affine(self):
    return self.coordinate_system.convert_to_affine(self)

  def __str__(self) -> str:
    if self.is_zero():
      return 'Point, O'
    ret = 'Point, '
    for i in range(len(self.coordinates)):
      ret += self.coordinates[i].hex_value_str() + ', '
    return ret


class EllipticCurveCoordinateSystem(ABC, Generic[T]):
  """Abstract base class for elliptic curve coordinate systems."""

  def __init__(self, config: Dict[str, Union[int, List[int]]]) -> None:
    """Initialize the coordinate system.

    Args:
      config: A dictionary containing the configuration of the coordinate
        system. The dictionary should contain the following keys: - ff_zero: The
        value "zero" in the finite field. - ff_one: The value "one" in the
        finite field. - type: The type of the coordinate system.
    """
    super().__init__()
    self.config = config
    self.ff_zero: FieldEle = FieldEle(0, config['prime'])
    self.ff_one: FieldEle = FieldEle(1, config['prime'])
    self.type = CoordinateSystemType.NONE

  @abstractmethod
  def generate_formal_coordinates(self, coordinates: List[T]) -> List[T]:
    pass

  def get_type(self):
    return self.type

  def generate_point(
      self,
      coordinates: Optional[Union[List[T], List[int]]] = None,
      zero: bool = False,
  ) -> ECPoint[T]:
    return ECPoint[T](coordinates, self, zero)

  @abstractmethod
  def point_add(self, point_a: ECPoint[T], point_b: ECPoint[T]):
    pass

  @abstractmethod
  def point_lshift(self, point_a: ECPoint[T], index):
    pass

  @abstractmethod
  def convert_to_affine(self, point_a: ECPoint[T]):
    pass


class ECCSWeierstrass(EllipticCurveCoordinateSystem[FieldEle]):
  """Base class for Weierstrass coordinate systems."""

  def __init__(self, config: Dict[str, Union[int, List[int]]]) -> None:
    """Initialize the EC Weierstrass coordinate systems.

    Weierstrass: y^2 = x^3 + a*x + b

    Args:
      config: A dictionary containing the configuration of the coordinate
        system. The dictionary should contain the following keys: - generator: A
        special point "G" in the Elliptic Curve. - order: order * generator == 0
        in the Elliptic Curve. - prime: The modulus of the Elliptic Curve. - a:
        The parameter a of the Elliptic Curve. - b: The parameter b of the
        Elliptic Curve.
    """
    super().__init__(config)
    self.prime = BigInt(config['prime'])
    self.order = BigInt(config['order'])

    self.generator: List[FieldEle] = []
    for coordinate in config['generator']:
      element = self.ff_zero.copy(coordinate)
      self.generator.append(element)
    self.a = self.ff_zero.copy(config['a'])
    self.b = self.ff_zero.copy(config['b'])

  def generate_formal_coordinates(
      self, coordinates: List[FieldEle]
  ) -> List[FieldEle]:
    formal_coordinate: List[FieldEle] = []
    for coordinate in coordinates:
      if isinstance(coordinate, FieldEle):
        formal_coordinate.append(coordinate)
      else:
        formal_coordinate.append(self.ff_zero.copy(coordinate))
    return formal_coordinate


class ECCSWeierstrassAffine(ECCSWeierstrass):
  """Weierstrass affine coordinate system.

  It has unique point add calculation. Details pleaase refer to:
  AFFINE: The affine coordinate system, the standard coordinate system for
  elliptic curves: https://www.hyperelliptic.org/EFD/g1p/auto-shortw.html
  """

  def __init__(self, config: Dict[str, Union[int, List[int]]]) -> None:
    super().__init__(config)
    self.type = CoordinateSystemType.WEIERSTRASS_AFFINE

  def add_general(self, point_a: ECPoint[FieldEle], point_b: ECPoint[FieldEle]):
    slope = (point_b[1] - point_a[1]) / (point_b[0] - point_a[0])
    cx = (slope * slope) - point_a[0] - point_b[0]
    cy = slope * (point_a[0] - cx) - point_a[1]
    return ECPoint[FieldEle]([cx, cy], self)

  def double_general(self, point_a: ECPoint[FieldEle]):
    x1 = point_a[0]
    y1 = point_a[1]
    slope = (x1 * x1 * 3 + self.a) / (y1 * 2)
    cx = slope * slope - x1 - x1
    cy = slope * (x1 - cx) - y1
    return ECPoint[FieldEle]([cx, cy], self)

  def point_add(self, point_a: ECPoint[FieldEle], point_b: ECPoint[FieldEle]):
    if point_b.is_zero():
      return point_a.copy()
    elif point_a.is_zero():
      return point_b.copy()

    if point_a == point_b:
      result = self.double_general(point_a)
    else:
      result = self.add_general(point_a, point_b)

    return result

  def point_lshift(self, point_a: ECPoint[FieldEle], shift: int):
    if point_a.is_zero():
      return point_a.copy()

    for _ in range(shift):
      point_a = self.double_general(point_a)
    return point_a

  def convert_to_affine(self, point_a: ECPoint[FieldEle]) -> ECPoint[FieldEle]:
    return point_a

  def generate_formal_coordinates(
      self, coordinates: List[FieldEle]
  ) -> List[FieldEle]:
    coordinate_length = len(coordinates)
    assert coordinate_length == 2 or coordinate_length == 3
    formal_coordinates: List[FieldEle] = []
    for coordinate in coordinates:
      if isinstance(coordinate, FieldEle):
        formal_coordinates.append(coordinate)
      else:
        formal_coordinates.append(self.ff_zero.copy(coordinate))
    if coordinate_length == 2:
      assert len(formal_coordinates) == 3
    return formal_coordinates


class ECCSWeierstrassProjective(ECCSWeierstrass):
  """Weierstrass projective coordinate system.

  PROJECTIVE: The projective coordinate system,
  https://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#addition-madd-1998-cmo
  """

  def __init__(self, config: Dict[str, Union[int, List[int]]]) -> None:
    super().__init__(config)
    self.prime = BigInt(config['prime'])
    self.order = BigInt(config['order'])
    self.type = CoordinateSystemType.WEIERSTRASS_PROJECTIVE

    self.generator: List[FieldEle] = []
    for coordinate in config['generator']:
      element = self.ff_zero.copy(coordinate)
      self.generator.append(element)

    self.a = self.ff_zero.copy(config['a'])
    self.b = self.ff_zero.copy(config['b'])

  def add_z2_eq_1(self, point_a: ECPoint[FieldEle], point_b: ECPoint[FieldEle]):
    if point_b[2] == self.ff_one:
      x1, y1, z1 = point_a
      x2, y2, _ = point_b
    elif point_a[2] == self.ff_one:
      x1, y1, z1 = point_b
      x2, y2, _ = point_a
    else:
      raise NotImplementedError

    u = y2 * z1 - y1
    uu = u * u
    v = x2 * z1 - x1
    vv = v * v
    vvv = v * vv
    r = vv * x1
    a = uu * z1 - vvv - (r + r)
    x3 = v * a
    y3 = u * (r - a) - vvv * y1
    z3 = vvv * z1
    return ECPoint[FieldEle]([x3, y3, z3], self)

  def add_general(self, point_a: ECPoint[FieldEle], point_b: ECPoint[FieldEle]):
    x1, y1, z1 = point_a
    x2, y2, z2 = point_b

    b3 = self.b * self.ff_zero.copy(3)
    a = self.a

    # Perform the operations
    t0 = x1 * x2
    t1 = y1 * y2
    t2 = z1 * z2
    t3 = (x1 + y1) * (x2 + y2)
    t4 = t0 + t1
    t3 = t3 - t4
    t4 = (x1 + z1) * (x2 + z2)
    t5 = t0 + t2
    t4 = t4 - t5
    t5 = (y1 + z1) * (y2 + z2)
    x3 = t1 + t2
    t5 = t5 - x3
    z3 = a * t4
    x3 = b3 * t2
    z3 = x3 + z3
    x3 = t1 - z3
    z3 = t1 + z3
    y3 = x3 * z3
    t1 = t0 + t0
    t1 = t1 + t0
    t2 = a * t2
    t4 = b3 * t4
    t1 = t1 + t2
    t2 = t0 - t2
    t2 = a * t2
    t4 = t4 + t2
    t0 = t1 * t4
    y3 = y3 + t0
    t0 = t5 * t4
    x3 = t3 * x3
    x3 = x3 - t0
    t0 = t3 * t1
    z3 = t5 * z3
    z3 = z3 + t0

    return ECPoint[FieldEle]([x3, y3, z3], self)

  def double_general(self, point_a: ECPoint):
    x1, y1, z1 = point_a
    a = self.a
    ff2 = self.ff_zero.copy(2)

    # Perform the operations based on the pseudocode
    xx = x1 * x1
    zz = z1 * z1
    w = a * zz + xx * self.ff_zero.copy(3)
    s = y1 * z1 * ff2
    ss = s * s
    sss = s * ss
    r = y1 * s
    rr = r * r
    b = (x1 + r) * (x1 + r) - xx - rr
    h = w * w - b * ff2
    x3 = h * s
    y3 = w * (b - h) - rr * ff2
    z3 = sss

    return ECPoint([x3, y3, z3], self)

  def point_add(
      self, point_a: ECPoint[FieldEle], point_b: ECPoint[FieldEle]
  ) -> ECPoint[FieldEle]:
    if point_b.is_zero():
      return point_a.copy()
    elif point_a.is_zero():
      return point_b.copy()

    if point_a == point_b:
      result = self.double_general(point_a)
    elif point_a[2] == self.ff_one or point_b[2] == self.ff_one:
      result = self.add_z2_eq_1(point_a, point_b)
    else:
      result = self.add_general(point_a, point_b)
    return result

  def point_lshift(self, point: ECPoint[FieldEle], shift: int):
    if point.is_zero():
      return point.copy()

    for _ in range(shift):
      point = self.double_general(point)
    return point

  def convert_from_affine(
      self, point_a: ECPoint[FieldEle]
  ) -> ECPoint[FieldEle]:
    assert point_a.get_type() == self.type
    new_point = point_a.copy()
    if new_point.coordinates is not None:
      new_point.coordinates.clear()
      z_invert = self.ff_one / point_a[2]
      new_point.append(point_a[0] * z_invert)
      new_point.append(point_a[1] * z_invert)
      new_point.append(self.ff_one)
      new_point.set_type(CoordinateSystemType.WEIERSTRASS_AFFINE)
      return new_point
    else:
      return new_point

  def convert_to_affine(self, point: ECPoint) -> ECPoint:
    assert point.get_type() == self.type
    new_point = point.copy()
    if new_point.coordinates is not None:
      new_point.coordinates.clear()
      z_invert = self.ff_one / point[2]
      new_point.append(point[0] * z_invert)
      new_point.append(point[1] * z_invert)
      new_point.append(self.ff_one)
      new_point.set_type(CoordinateSystemType.WEIERSTRASS_AFFINE)
      return new_point
    else:
      return new_point


class ECCSWeierstrassXYZZ(ECCSWeierstrass):
  """Weierstrass XYZZ coordinate system."""

  def __init__(self, config: Dict[str, Union[int, List[int]]]) -> None:
    super().__init__(config)
    self.type = CoordinateSystemType.WEIERSTRASS_XYZZ

  def generate_formal_coordinates(
      self, coordinates: List[FieldEle]
  ) -> List[FieldEle]:
    coordinate_length = len(coordinates)
    assert coordinate_length == 2 or coordinate_length == 4
    formal_coordinates: List[FieldEle] = []
    for coordinate in coordinates:
      if isinstance(coordinate, FieldEle):
        formal_coordinates.append(coordinate)
      else:
        formal_coordinates.append(self.ff_zero.copy(coordinate))
    if coordinate_length == 2:
      formal_coordinates.append(self.ff_one.copy())
      formal_coordinates.append(self.ff_one.copy())
    assert len(formal_coordinates) == 4
    return formal_coordinates

  def add_z2_eq_1(self, point_a: ECPoint[FieldEle], point_b: ECPoint[FieldEle]):
    """Add the general coordinates of two points in the XYZZ coordinate system.

    This function is not implemented for the XYZZ coordinate system.

    Args:
      point_a: The first point.
      point_b: The second point.

    Returns:
      The added general coordinates of the two points.
    """
    raise NotImplementedError
    # return ECPoint[FieldEle]([], self)

  def add_general(self, point_a: ECPoint[FieldEle], point_b: ECPoint[FieldEle]):
    x1, y1, zz1, zzz1 = point_a
    x2, y2, zz2, zzz2 = point_b

    u1 = x1 * zz2
    u2 = x2 * zz1
    s1 = y1 * zzz2
    s2 = y2 * zzz1
    p = u2 - u1
    r = s2 - s1
    pp = p * p
    ppp = p * pp
    q = u1 * pp
    x3 = r * r - ppp - (q + q)
    y3 = r * (q - x3) - s1 * ppp
    zz3 = zz1 * zz2 * pp
    zzz3 = zzz1 * zzz2 * ppp

    return ECPoint[FieldEle]([x3, y3, zz3, zzz3], self)

  def double_general(self, point_a: ECPoint[FieldEle]):
    """Double the general coordinates of a point in the XYZZ coordinate system.

    Args:
      point_a: The point to double the general coordinates of.

    Returns:
      The doubled general coordinates of the point.
    """
    x1, y1, zz1, zzz1 = point_a
    a = self.a

    # Perform the operations based on the pseudocode
    u = y1 + y1
    v = u * u
    w = u * v
    s = x1 * v
    x1_sq = x1 * x1
    m = (x1_sq + x1_sq + x1_sq) + a * (zz1 * zz1)
    x3 = (m * m) - (s + s)
    y3 = m * (s - x3) - w * y1
    zz3 = v * zz1
    zzz3 = w * zzz1

    return ECPoint[FieldEle]([x3, y3, zz3, zzz3], self)

  def point_add(
      self, point_a: ECPoint[FieldEle], point_b: ECPoint[FieldEle]
  ) -> ECPoint[FieldEle]:
    if point_b.is_zero():
      return point_a.copy()
    elif point_a.is_zero():
      return point_b.copy()

    if point_a == point_b:
      result = self.double_general(point_a)
    else:
      result = self.add_general(point_a, point_b)
    return result

  def point_lshift(self, point_a: ECPoint[FieldEle], shift: int):
    if point_a.is_zero():
      return point_a.copy()

    for _ in range(shift):
      point_a = self.double_general(point_a)
    return point_a

  def convert_from_affine(
      self, point_a: ECPoint[FieldEle]
  ) -> ECPoint[FieldEle]:
    assert (
        point_a.coordinate_system.get_type()
        == CoordinateSystemType.WEIERSTRASS_AFFINE
    )
    new_point = point_a.copy()
    new_point.append(self.ff_one)
    new_point.append(self.ff_one)
    new_point.set_coordinate_system(self)
    return new_point

  def convert_to_affine(self, point_a: ECPoint[FieldEle]) -> ECPoint[FieldEle]:
    assert point_a.get_type() == self.type
    new_point = point_a.copy()
    if new_point.coordinates is not None:
      new_point.coordinates.clear()
      a = self.ff_one / point_a[3]
      b = point_a[2] * a
      b_sq = b * b
      new_point.append(point_a[0] * b_sq)
      new_point.append(point_a[1] * a)
      new_point.append(self.ff_one)
      new_point.append(self.ff_one)
      new_point.set_type(CoordinateSystemType.WEIERSTRASS_AFFINE)
      return new_point
    else:
      return new_point
