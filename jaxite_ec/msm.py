"""This file performs the Multiple Scalar Multiplication (MSM) operation.

1. The algorithm being used is Point Double and Addition [1].

2. The MSM is being performed via two stages:

- offline precomputation stage

**compute_base**: given a list of points P_i, it calculates P_i^{2x} for x in 0
to 255. The precomputed points are stored in self.base_points.

**construct_select** only self.base_points corresponding to the non-zero bit
in the scalar to the non-zero bit in the scalar. function constructs the
selection list for each pair of point and scalar, the point will be loaded only
if the corresponding bit in the scalar is 1.

- online computation stage

1) Point Accumulation: For each pair of point and scalar, we iterate each bit of
the scalar and load the point for accumulation.
2) Partial Sum Accumulation: the posted accumulated results for all pairs of 
scalars and points are accumulated into the final result.


[1] https://link.springer.com/chapter/10.1007/3-540-36400-5_41
"""

import copy
import functools
import math
from typing import List

import jax
import jax.numpy as jnp
from jaxite.jaxite_ec.algorithm import msm_reader as msm_reader_lib
import jaxite.jaxite_ec.elliptic_curve as jec
import jaxite.jaxite_ec.util as utils


MSM_Reader = msm_reader_lib.MSMReader
deepcopy = copy.deepcopy


class MSMDoubleAdd:
  """MSMDoubleAdd class is used to add points from MSM trace."""

  def __init__(self):
    self.length = 0

    self.scalars: List[int] = []  # Orignal scalar from the trace
    self.scalar_precision = 256
    # The bit precision of the scalar

    # [coordinates, coordinates, coordinates, coordinates]
    self.points: List[jnp.ndarray]  # Orignal points from the trace

    self.base_points: jnp.ndarray
    # 2^x power of points, for x in 0 to self.scalar_precision

    self.non_zero_states: jnp.ndarray
    self.overall_select: jnp.ndarray
    self.overall_non_zero_states: jnp.ndarray
    self.point_psum: jnp.ndarray
    self.result = None

  def initialize(self, scalars, points):
    """initialize MSM.

    Args:
      scalars: Orignal scalar from the trace.
      points: Orignal points from the trace.

    Returns:
      None

    Note that only base_point corresponding to the non-zero bit in the scalar
    will be loaded - so that we need to construct selection list based on scalar

    Further, any point when being accumulated with zero point will equal to
    itself. Such a special case needs special handling. Therefore, we construct
    an extra self.overall_non_zero_states to record the zero state of
    point_psum.

    Specifically, self.overall_non_zero_states has shape of (scalar_precision,
    length), where the value at [i, j] indicates whether the point_psum[j] is
    zero when starting to be accumulated with base_point[i] at bit i of scalar.

    if self.overall_non_zero_states[i, j] = True, means that point_psum[j] is
    non-zero when starting to be accumulated with base_point[i] at bit i of
    scalar.
    """
    # Initial internal selection from the scalar
    self.scalars = scalars
    self.length = len(scalars)
    self.psum_addition_length = int(math.log2(self.length))
    self.non_zero_states = jnp.full((self.length,), False, dtype=jnp.bool)
    self.construct_select()

    # Convert high-precision points into a vector of low-precision chunks
    self.points = [
        utils.int_list_to_2d_array(
            coordinates + [1, 1], utils.BASE, utils.U16_CHUNK_NUM
        )
        for coordinates in points
    ]  # pytype: disable=container-type-mismatch
    self.blank_point = jnp.array(
        utils.int_list_to_2d_array(
            [0, 0, 0, 0], utils.BASE, utils.U16_CHUNK_NUM
        ),
        dtype=jnp.uint16,
    )
    self.point_psum = jnp.array(
        [self.blank_point] * self.length, dtype=jnp.uint16
    )

  def compute_base(self):
    """Compute base points for MSM.

    Note that self.base_points comes with shape
    (scalar_precision, length, number of coordinates, number of chunks)

    Dim-0: Scalar_precision has temporal dependency, i.e. [i+1] = double([i])
    So we process it in a temporal loop with loop length.

    Dim-1: length dimension could be processed in full parallel
    So we vectorize at this dimension by calling jax.vmap().
    Note that self.base_points[i] is fed as input which comes with 3D dimension.
    Length dimension becomes index 0 -- so set jax.vmap with in_axes=0 and
    out_axes=0.
    """
    scaled_point = jnp.array(self.points, dtype=jnp.uint16)
    self.base_points = jnp.empty(
        (self.scalar_precision,) + scaled_point.shape, dtype=jnp.uint16
    )
    self.base_points = self.base_points.at[0].set(scaled_point)
    for i in range(1, self.scalar_precision):
      self.base_points = self.base_points.at[i].set(
          jax.vmap(jec.pdul_barrett_xyzz_pack, in_axes=0, out_axes=0)(
              self.base_points[i - 1]
          )
      )

  def construct_select(self):
    """Construct select list for each point.

    Only point corresponding to the non-zero bit in the scalar will be
    accumulated.
    """
    overall_select = []
    for i in range(0, self.scalar_precision):
      select_list = []
      for scalar in self.scalars:
        bit_value = ((scalar >> i) & 0x1) == 0x1
        select_list.append(bit_value)
      overall_select.append(select_list)
    self.overall_select = jnp.array(overall_select, dtype=jnp.bool)

    overall_non_zero_states = jnp.empty(
        (self.scalar_precision,) + (self.length,), dtype=jnp.bool
    )
    overall_non_zero_states.at[0].set(self.non_zero_states)
    for i in range(1, self.scalar_precision):
      overall_non_zero_states = overall_non_zero_states.at[i].set(
          self.update_zero_states(self.overall_select[i-1])
      )
    self.overall_non_zero_states = overall_non_zero_states

  def update_zero_states(self, select_list):
    """update zero states for each point partial sum.

    Args:
      select_list: select list for each point.

    Returns:
      non_zero_states: ndarray recording non-zero state of points.

      input,    bit in scalar,            output
      not_zero, bit_zero -> 1 || 0 = 1 -> not zero
      not_zero, bit_one  -> 1 || 1 = 1 -> not zero
      is_zero,  bit_zero -> 0 || 0 = 0 -> is zero
      is_zero,  bit_one  -> 0 || 1 = 1 -> not zero
    """
    self.non_zero_states = jnp.logical_or(self.non_zero_states, select_list)
    return self.non_zero_states


def selective_padd_with_zero(
    in_point_psum, in_base_point, select, is_in_point_psum_non_zero
):
  """Selective PADD considering zero in_point_psum and non-selected bit.

  Args:
    in_point_psum: The point to be accumulated.
    in_base_point: The base point to be accumulated with.
    select: A list of booleans indicating whether to accumulate the point.
    is_in_point_psum_non_zero: A list of booleans indicating whether the point
      is non-zero.

  Returns:
    The result of the accumulation.

    non_zero,  select                ->  returns
    not_zero, bit_zero -> 1 || 0 = 1 -> in_point_psum
    not_zero, bit_one  -> 1 || 1 = 1 -> in_point_psum + in_base_point
    is_zero,  bit_zero -> 0 || 0 = 0 -> in_point_psum
    is_zero,  bit_one  -> 0 || 1 = 1 -> in_base_point

  """
  result = jnp.where(
      select,
      jnp.where(
          is_in_point_psum_non_zero,
          jec.padd_barrett_xyzz_pack(in_point_psum, in_base_point),
          in_base_point,
      ),
      in_point_psum,
  )
  return result


# This compilation takes forever.
# @jax.named_call
# @functools.partial(
#     jax.jit,
#     static_argnames=(
#         "scalar_precision",
#     ),
# )
def point_accumulation(
    point_psum,
    base_points,
    overall_select,
    overall_non_zero_states,
    scalar_precision,
):
  """Accumulate each point based on the scalar value for MSM.

  Args:
    scalar_precision: The bit precision of the scalar.
    point_psum: The point to be accumulated.
    base_points: The base point to be accumulated with.
    overall_select: A list of booleans indicating whether to accumulate the
      point.
    overall_non_zero_states: A list of booleans indicating whether the point is
      non-zero.

  Returns:
    The result of the accumulation.

  Note that point_psum comes with shape
  (length, number of coordinates, number of chunks)

  Dim-0: length dimension could be processed in full parallel
  So we vectorize at this dimension by calling jax.vmap().
  Note that base_points[i] is fed as input which comes with 3D dimension.
  Length dimension becomes index 0 -- so set jax.vmap with in_axes=0 and
  out_axes=0.

  overall_select, overall_non_zero_states both have the shape
  (scalar_precision, length)

  Therefore, we need to specify the first index per iteration.
  """
  for i in range(0, scalar_precision):
    point_psum = jax.vmap(
        selective_padd_with_zero, in_axes=0, out_axes=0
    )(
        point_psum,
        base_points[i],
        overall_select[i, :],
        overall_non_zero_states[i, :],
    )
  return point_psum


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=("length"),
)
def psum_accumulation_algorithm(point_psum, length):
  """This function listed the algorithm of the psum_accumulation.
  """
  result = point_psum[0]
  for i in range(1, length):
    result = jec.padd_barrett_xyzz_pack(point_psum[i], result)
  return result


@jax.named_call
@functools.partial(
    jax.jit,
    static_argnames=("psum_addition_length", "length"),
)
def psum_accumulation(point_psum, psum_addition_length, length):
  """Tree based point accumulation for best efficiency -- dichotomy."""
  for i in range(psum_addition_length):
    full_length = length >> i
    half_length = length >> (i + 1)
    point_psum = jax.vmap(jec.padd_barrett_xyzz_pack, in_axes=0, out_axes=0)(
        point_psum[0:half_length],
        point_psum[half_length:full_length],
    )
  return point_psum[0]

