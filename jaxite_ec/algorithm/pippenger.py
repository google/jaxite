"""The Pippenger algorithm.

This module implements the Pippenger algorithm for elliptic curve cryptography.
The Pippenger algorithm is a multi-step algorithm that is used to generate
elliptic curve points. The algorithm is based on the idea of using a multi-step
process to reduce the number of elliptic curve points that need to be generated.

The Pippenger algorithm has two main steps: the horizontal stream and the
vertical
stream. The horizontal stream is used to accumulate points from the input stream
into buckets. The vertical stream is used to merge the points from the buckets
into a single point.

The Pippenger algorithm is a powerful algorithm that can be used to generate
elliptic curve points very efficiently. It is also a relatively simple algorithm
to implement, which makes it a good choice for beginners to the field of
elliptic
curve cryptography.
"""

from typing import List

from jaxite.jaxite_ec.algorithm import elliptic_curve
from jaxite.jaxite_ec.algorithm import msm_reader as msm_reader_lib

MSMReader = msm_reader_lib.MSMReader
ECPoint = elliptic_curve.ECPoint
EllipticCurveCoordinateSystem = elliptic_curve.EllipticCurveCoordinateSystem


class PippengerBucket:
  """A bucket in the Pippenger algorithm.

  Attributes:
    slice_id: The ID of the slice that this bucket belongs to.
    point: The point that this bucket represents.
  """

  def __init__(
      self, coordinate_system: EllipticCurveCoordinateSystem, slice_id=0
  ):
    self.slice_id = slice_id
    # self.empty = True
    self.point = coordinate_system.generate_point([], True)

  def add_point(self, point: ECPoint):
    self.point += point

  def get_point(self):
    return self.point


class PippengerWindow:
  """A window in the Pippenger algorithm.

  Attributes:
    coordinate_system: The coordinate system that this window belongs to.
    slice_length: The length of the slice that this window represents.
    bucket_num: The number of buckets in this window.
    window_id: The ID of this window.
    buckets: A list of the buckets in this window.
    point: The point that this window represents.
  """

  def __init__(
      self,
      coordinate_system: EllipticCurveCoordinateSystem,
      slice_length,
      window_id=0,
  ) -> None:
    self.coordinate_system = coordinate_system
    self.slice_length = slice_length
    self.bucket_num = 2**slice_length
    self.window_id = window_id
    self.buckets: List[PippengerBucket] = [
        PippengerBucket(coordinate_system, i) for i in range(self.bucket_num)
    ]
    self.point = self.coordinate_system.generate_point(None, True)

  def bucket_reduction(self) -> ECPoint:
    window_sum: ECPoint = self.coordinate_system.generate_point(None, True)
    temp_sum: ECPoint = self.coordinate_system.generate_point(None, True)
    for i in range(self.bucket_num - 1, 0, -1):
      temp_sum += self.buckets[i].get_point()
      window_sum += temp_sum
    self.point = window_sum
    return window_sum

  def get_point(self):
    return self.point

  def __getitem__(self, index):
    return self.buckets[index]

  def __setitem__(self, index, value):
    self.buckets[index] = value


class PippengerMSM:
  """The Pippenger MSM algorithm.

  Attributes:
    coordinate_system: The coordinate system that this MSM belongs to.
    slice_length: The length of the slice that this MSM represents.
    window_num: The number of windows in this MSM.
    bucket_num_in_window: The number of buckets in each window.
    slice_mask: The mask that is used to extract the bucket ID from the scalar.
    bucket_num_all: The total number of buckets in this MSM.
    windows: A list of the windows in this MSM.
    result: The result of the MSM calculation.
  """

  def __init__(
      self,
      coordinate_system: EllipticCurveCoordinateSystem,
      slice_length,
      window_num,
  ) -> None:
    self.coordinate_system: EllipticCurveCoordinateSystem = coordinate_system
    self.slice_length = slice_length
    self.window_num = window_num
    self.bucket_num_in_window = 2**self.slice_length
    self.slice_mask = self.bucket_num_in_window - 1
    self.bucket_num_all = self.bucket_num_in_window * self.window_num
    self.windows: List[PippengerWindow] = [
        PippengerWindow(coordinate_system, self.slice_length, i)
        for i in range(self.window_num)
    ]
    self.result = None

  def msm_run(self, msm_reader: MSMReader):
    """Runs the Pippenger MSM algorithm.

    Args:
      msm_reader: The MSM reader to use to get the scalars and coordinates.

    Returns:
      The result of the MSM calculation.
    """
    scalar = msm_reader.get_next_scalar()
    coordinates = msm_reader.get_next_base()

    while scalar and coordinates:
      point = self.coordinate_system.generate_point(coordinates)
      self.msm_horizental_stream(scalar, point)
      scalar = msm_reader.get_next_scalar()
      coordinates = msm_reader.get_next_base()
    self.bucket_reduction()
    self.result = self.window_merge()
    self.result = self.result.convert_to_affine()
    return self.result

  def msm_horizental_stream(self, scalar: int, point: ECPoint):
    window_id = 0
    current_scalar = scalar
    while current_scalar != 0:
      bucket_id = current_scalar & self.slice_mask
      self.bucket_accumulation(window_id, bucket_id, point)
      current_scalar = current_scalar >> self.slice_length
      window_id += 1

  def bucket_accumulation(self, window_id, bucket_id, point: ECPoint):
    self.windows[window_id][bucket_id].add_point(point)

  def bucket_reduction(self):
    for i in range(0, self.window_num):
      self.windows[i].bucket_reduction()

  def window_merge(self):
    merged = self.coordinate_system.generate_point(None, True)
    for i in range(self.window_num - 1, -1, -1):
      point = self.windows[i].get_point()
      merged = (merged << self.slice_length) + point

    return merged
