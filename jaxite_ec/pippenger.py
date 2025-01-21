"""Pippenger algorithm for elliptic curves.

This module implements the Pippenger algorithm for elliptic curves. The
algorithm
is a generalization of the elliptic curve algorithm that can be used to find
elliptic curves of arbitrary order.

The Pippenger algorithm works by first finding a set of "scalars" and "points"
that lie on the elliptic curve. These scalars and points are then used to
construct
a "bucket" for each window of the elliptic curve. The buckets are then reduced
to a single point for each window. Finally, the points from all of the windows
are merged together to form the final elliptic curve.

The Pippenger algorithm is a powerful tool for finding elliptic curves, and it
has been used to find elliptic curves of arbitrary order and dimension.
"""

import copy
import math
import os
import sys
from typing import List, Optional

import jax.numpy as jnp
from jaxite.jaxite_ec.algorithm import msm_reader as msm_reader_lib
import jaxite.jaxite_ec.util as utils

MSM_Reader = msm_reader_lib.MSMReader
deepcopy = copy.deepcopy
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

COORDINATE_NUM = 4
BASE = 16
CHUNK_NUM = 24


class MSMPippenger:
  """Pippenger algorithm for elliptic curves.

  Attributes:
    coordinate_num: The number of coordinates in the elliptic curve.
    slice_length: The length of each slice in the elliptic curve.
    window_num: The number of windows in the elliptic curve.
    bucket_num_in_window: The number of buckets in each window.
    slice_mask: The mask for the slices in the elliptic curve.
    all_buckets: A list of lists of JAX arrays, where each array represents a
      bucket.
    bucket_zero_states: A list of lists of integers, where each integer
      represents the zero state of a bucket.
    window_sum: A list of JAX arrays, where each array represents the sum of the
      buckets in a window.
    result: The final elliptic curve.
    msm_length: The length of the MSM trace.
    scalars: A list of integers, where each integer represents an Orignal scalar
      from the trace.
    points: A list of JAX arrays, where each array represents an Orignal point
      from the trace.
  """

  def __init__(self, slice_length):
    self.coordinate_num = COORDINATE_NUM

    self.slice_length = slice_length
    self.window_num = int(math.ceil(255 / self.slice_length))
    self.bucket_num_in_window = 2**self.slice_length
    self.slice_mask = self.bucket_num_in_window - 1
    self.all_buckets: List[List[Optional[jnp.ndarray]]] = [
        [None for _ in range(self.bucket_num_in_window)]
        for _ in range(self.window_num)
    ]
    self.bucket_zero_states: List[List[int]] = [
        [0 for _ in range(self.bucket_num_in_window)]
        for _ in range(self.window_num)
    ]
    self.window_sum: List[Optional[jnp.ndarray]] = [
        None for _ in range(self.window_num)
    ]
    self.result = None
    self.msm_length = 0

    self.scalars: List[int] = []  # Orignal scalar from the trace
    # [Points, Points, ..., Points]
    self.points: List[jnp.ndarray] = []  # Orignal points from the trace

  def read_trace(self, reader: MSM_Reader):
    """Read the MSM trace and store the scalars and points.

    Args:
      reader: The MSM reader.
    """
    scalar = reader.get_next_scalar()
    coordinates = reader.get_next_base()
    length = 0
    while scalar:
      self.scalars.append(scalar)
      point_jax = utils.int_point_to_jax_point_pack(
          coordinates + [1] * (COORDINATE_NUM - len(coordinates))
      )
      self.points.append(point_jax)

      length += 1
      scalar = reader.get_next_scalar()
      coordinates = reader.get_next_base()

    self.msm_length = length

  def bucket_accumulation(self, jax_padd):
    """Accumulate the buckets for each window.

    Args:
      jax_padd: The JAX padding function.
    """
    for w in range(self.window_num):
      for i in range(self.msm_length):
        slice_index = (
            self.scalars[i] >> (w * self.slice_length)
        ) & self.slice_mask
        if slice_index != 0:
          if self.all_buckets[w][slice_index] == None:
            self.all_buckets[w][slice_index] = self.points[i]
          else:
            self.all_buckets[w][slice_index] = jax_padd(
                self.all_buckets[w][slice_index], self.points[i]
            )

  def bucket_reduction(self, jax_padd):
    """Reduce the buckets to a single point for each window.

    Args:
      jax_padd: The JAX padding function.
    """
    temp_sum: List[Optional[jnp.ndarray]] = [
        None for _ in range(self.window_num)
    ]

    for w in range(self.window_num):
      for i in range(self.bucket_num_in_window - 1, 0, -1):
        if temp_sum[w] == None:
          if self.all_buckets[w][i] != None:
            temp_sum[w] = self.all_buckets[w][i]
        else:
          if self.all_buckets[w][i] != None:
            temp_sum[w] = jax_padd(temp_sum[w], self.all_buckets[w][i])
        if self.window_sum[w] == None:
          self.window_sum[w] = temp_sum[w]
        else:
          self.window_sum[w] = jax_padd(self.window_sum[w], temp_sum[w])

  def window_merge(self, jax_padd, jax_pdul):
    self.result = self.window_sum[self.window_num - 1]
    for w in range(self.window_num - 2, -1, -1):
      for _ in range(self.slice_length):
        if self.result != None:
          self.result = jax_pdul(self.result)
      if self.result == None:
        self.result = self.window_sum[w]
      else:
        if self.window_sum[w] != None:
          self.result = jax_padd(self.result, self.window_sum[w])
