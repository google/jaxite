"""elliptic_curve class implementation."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
import copy
from enum import Enum, auto
from typing import List

from absl import app
from traitlets import Bool

USE_GMP = True
USE_BARRETT = True
if USE_GMP:
  from hp_int import GMPHPint as HPint
else:
  from hp_int import HPint as HPint

if USE_BARRETT:
  from finite_field import FiniteFieldElementBarrett as FieldEle
else:
  from finite_field import FiniteFieldElement as FieldEle


class CoordinateSystemType(Enum):
  """Enum to represent different types of coordinate systems for elliptic curves."""

  NONE = auto()
  WEIERSTRASS_AFFINE = auto()
  WEIERSTRASS_PROJECTIVE = auto()


class ECCPoint:

  def __init__(
      self,
      coordinates: List,
      coordinate_system: 'EllipticCurveCoordinateSystem',
      zero: Bool = False,
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

  def __getitem__(self, index: int) -> FieldEle:
    """Allows access to elements in self.coordinates using index."""
    return self.coordinates[index]

  def __setitem__(self, index: int, value: FieldEle) -> None:
    """Allows modification of elements in self.coordinates using index."""
    self.coordinates[index] = value

  def __eq__(self, other: 'ECCPoint') -> bool:
    if not isinstance(other, ECCPoint):
      return NotImplemented

    # return (self.coordinates == other.coordinates and
    #         self.coordinate_system == other.coordinate_system)
    return self.coordinates == other.coordinates

  def __add__(self, other: 'ECCPoint') -> 'ECCPoint':
    return self.coordinate_system.point_add(self, other)

  def __lshift__(self, shift) -> 'ECCPoint':
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

  def append(self, coordinate: FieldEle):
    self.coordinates.append(coordinate)

  def copy(self):
    obj = copy.copy(self)
    if not self.is_zero():
      obj.coordinates = self.coordinate_system.generate_formal_coordinates(
          self.coordinates
      )
    return obj

  def convert_to_affine(self):
    self = self.coordinate_system.convert_to_affine(self)

  def __str__(self) -> str:
    if self.is_zero():
      return 'Point, O'
    ret = 'Point, '
    for i in range(len(self.coordinates)):
      ret += self.coordinates[i].hex_value_str() + ', '
    return ret


class EllipticCurveCoordinateSystem(ABC):

  def __init__(self, config: dict) -> None:
    super().__init__()
    self.config = config
    self.ff0: FieldEle = FieldEle(0, config['prime'])
    self.ff1: FieldEle = FieldEle(1, config['prime'])
    self.type = CoordinateSystemType.NONE

  @abstractmethod
  def generate_formal_coordinates(self, coordinates: List) -> List[FieldEle]:
    pass

  def get_type(self):
    return self.type

  def generate_point(self, coordinates: List, zero: bool = False) -> ECCPoint:
    return ECCPoint(coordinates, self, zero)

  @abstractmethod
  def point_add(self, pointA: ECCPoint, pointB: ECCPoint):
    pass

  @abstractmethod
  def point_lshift(self, pointA: ECCPoint, index):
    pass

  @abstractmethod
  def convert_to_affine(self, pointA: ECCPoint):
    pass


class ECCWeierstrassSystem(EllipticCurveCoordinateSystem):

  def __init__(self, config: dict) -> None:
    super().__init__(config)
    self.prime = BigInt(config['prime'])
    self.order = BigInt(config['order'])

    self.generator: List[FieldEle] = []
    for coordinate in config['generator']:
      element = self.ff0.copy(coordinate)
      self.generator.append(element)

    self.a = self.ff0.copy(config['a'])
    self.b = self.ff0.copy(config['b'])

  def generate_formal_coordinates(self, coordinates: List) -> List[FieldEle]:
    formal_coordinate: List[FieldEle] = []
    for coordinate in coordinates:
      if isinstance(coordinate, FieldEle):
        formal_coordinate.append(coordinate)
      else:
        formal_coordinate.append(self.ff0.copy(coordinate))
    return formal_coordinate


class ECCWeierstrassAffine(ECCWeierstrassSystem):

  def __init__(self, config: dict) -> None:
    super().__init__(config)
    self.type = CoordinateSystemType.WEIERSTRASS_AFFINE

  def add_general(self, pointA: ECCPoint, pointB: ECCPoint):
    slope = (pointB[1] - pointA[1]) / (pointB[0] - pointA[0])
    cx = (slope * slope) - pointA[0] - pointB[0]
    cy = slope * (pointA[0] - cx) - pointA[1]
    return ECCPoint([cx, cy], self)

  def double_general(self, pointA: ECCPoint):
    x1 = pointA[0]
    y1 = pointA[1]
    slope = (x1 * x1 * 3 + self.a) / (y1 * 2)
    cx = slope * slope - x1 - x1
    cy = slope * (x1 - cx) - y1
    return ECCPoint([cx, cy], self)

  def point_add(self, pointA: ECCPoint, pointB: ECCPoint):
    if pointB.is_zero():
      return pointA.copy()
    elif pointA.is_zero():
      return pointB.copy()

    if pointA == pointB:
      result = self.double_general(pointA)
    else:
      result = self.add_general(pointA, pointB)

    return result

  def point_lshift(self, pointA: ECCPoint, shift: int):
    if pointA.is_zero():
      return pointA.copy()

    for i in range(shift):
      pointA = self.double_general(pointA)
    return pointA

  def convert_to_affine(self, pointA: ECCPoint) -> ECCPoint:
    return pointA


class ECCWeierstrassProjective(ECCWeierstrassSystem):

  def __init__(self, config: dict) -> None:
    super().__init__(config)
    self.type = CoordinateSystemType.WEIERSTRASS_PROJECTIVE

  def generate_formal_coordinates(self, coordinates: List) -> List[FieldEle]:
    coordinate_length = len(coordinates)
    assert coordinate_length == 2 or coordinate_length == 3
    formal_coordinates: List[FieldEle] = []
    for coordinate in coordinates:
      if isinstance(coordinate, FieldEle):
        formal_coordinates.append(coordinate)
      else:
        formal_coordinates.append(self.ff0.copy(coordinate))
    if coordinate_length == 2:
      formal_coordinates.append(self.ff1.copy())
    assert len(formal_coordinates) == 3
    return formal_coordinates

  def add_z2_eq_1(self, pointA: ECCPoint, pointB: ECCPoint):
    if pointB[2] == self.ff1:
      X1, Y1, Z1 = pointA
      X2, Y2, Z2 = pointB
    elif pointA[2] == self.ff1:
      X1, Y1, Z1 = pointB
      X2, Y2, Z2 = pointA
    else:
      raise NotImplementedError

    u = Y2 * Z1 - Y1
    uu = u * u
    v = X2 * Z1 - X1
    vv = v * v
    vvv = v * vv
    R = vv * X1
    A = uu * Z1 - vvv - (R + R)
    X3 = v * A
    Y3 = u * (R - A) - vvv * Y1
    Z3 = vvv * Z1
    return ECCPoint([X3, Y3, Z3], self)

  def add_general(self, pointA: ECCPoint, pointB: ECCPoint):
    X1, Y1, Z1 = pointA
    X2, Y2, Z2 = pointB

    b3 = self.b * self.ff0.copy(3)
    a = self.a

    # Perform the operations
    t0 = X1 * X2
    t1 = Y1 * Y2
    t2 = Z1 * Z2
    t3 = (X1 + Y1) * (X2 + Y2)
    t4 = t0 + t1
    t3 = t3 - t4
    t4 = (X1 + Z1) * (X2 + Z2)
    t5 = t0 + t2
    t4 = t4 - t5
    t5 = (Y1 + Z1) * (Y2 + Z2)
    X3 = t1 + t2
    t5 = t5 - X3
    Z3 = a * t4
    X3 = b3 * t2
    Z3 = X3 + Z3
    X3 = t1 - Z3
    Z3 = t1 + Z3
    Y3 = X3 * Z3
    t1 = t0 + t0
    t1 = t1 + t0
    t2 = a * t2
    t4 = b3 * t4
    t1 = t1 + t2
    t2 = t0 - t2
    t2 = a * t2
    t4 = t4 + t2
    t0 = t1 * t4
    Y3 = Y3 + t0
    t0 = t5 * t4
    X3 = t3 * X3
    X3 = X3 - t0
    t0 = t3 * t1
    Z3 = t5 * Z3
    Z3 = Z3 + t0

    return ECCPoint([X3, Y3, Z3], self)

  def double_general(self, pointA: ECCPoint):
    X1, Y1, Z1 = pointA
    a = self.a
    ff2 = self.ff0.copy(2)

    # Perform the operations based on the pseudocode
    XX = X1 * X1
    ZZ = Z1 * Z1
    w = a * ZZ + XX * self.ff0.copy(3)
    s = Y1 * Z1 * ff2
    ss = s * s
    sss = s * ss
    R = Y1 * s
    RR = R * R
    B = (X1 + R) * (X1 + R) - XX - RR
    h = w * w - B * ff2
    X3 = h * s
    Y3 = w * (B - h) - RR * ff2
    Z3 = sss

    return ECCPoint([X3, Y3, Z3], self)

  def point_add(self, pointA: ECCPoint, pointB: ECCPoint) -> ECCPoint:
    if pointB.is_zero():
      return pointA.copy()
    elif pointA.is_zero():
      return pointB.copy()

    if pointA == pointB:
      result = self.double_general(pointA)
    elif pointA[2] == self.ff1 or pointB[2] == self.ff1:
      result = self.add_z2_eq_1(pointA, pointB)
    else:
      result = self.add_general(pointA, pointB)
    return result

  def point_lshift(self, pointA: ECCPoint, shift: int):
    if pointA.is_zero():
      return pointA.copy()

    for i in range(shift):
      pointA = self.double_general(pointA)
    return pointA

  def convert_from_affine(self, pointA: ECCPoint) -> ECCPoint:
    assert (
        pointA.coordinate_system.get_type()
        == CoordinateSystemType.WEIERSTRASS_AFFINE
    )
    new_point = pointA.copy()
    new_point.append(self.ff1)
    new_point.set_coordinate_system(self)
    return new_point

  def convert_to_affine(self, pointA: ECCPoint) -> ECCPoint:
    assert pointA.get_type() == self.type
    new_point = pointA.copy()
    new_point.coordinates.clear()
    Z_invert = self.ff1 / pointA[2]
    new_point.append(pointA[0] * Z_invert)
    new_point.append(pointA[1] * Z_invert)
    new_point.append(self.ff1)
    new_point.set_type(CoordinateSystemType.WEIERSTRASS_AFFINE)
    return new_point


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)
