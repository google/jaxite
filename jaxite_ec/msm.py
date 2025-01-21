"""MSMDoubleAdd class is used to add points from MSM trace."""

import copy
from typing import List, Optional

import jax
import jax.numpy as jnp
from jaxite.jaxite_ec.algorithm import msm_reader as msm_reader_lib
import jaxite.jaxite_ec.elliptic_curve as jec
import jaxite.jaxite_ec.util as utils


MSM_Reader = msm_reader_lib.MSMReader
deepcopy = copy.deepcopy
COORDINATE_NUM = 4
BATCH_SIZE = 1
BASE = 16
CHUNK_NUM = 24


def selective_padd_with_zero(partial_sum, single_point, select, is_zero):
  new_partial_sum = jec.padd_barrett_xyzz_pack(partial_sum, single_point)
  sum_result = jnp.where(jnp.equal(select, 0), partial_sum, new_partial_sum)
  result = jnp.where(jnp.equal(is_zero, 0), sum_result, single_point)
  return result


class MSMDoubleAdd:
  """MSMDoubleAdd class is used to add points from MSM trace."""

  def __init__(self):
    self.length = 0

    self.scalars: List[int] = []  # Orignal scalar from the trace

    # [corrdinates, corrindates, corrindates, corrindates]
    self.points: List[jnp.ndarray] = []  # Orignal points from the trace

    # [Base1_Coords, Base2_Coords, Base4_Coords, Base8_Coords, ...,
    # Base255_Coords]
    self.base_points: List[jnp.ndarray] = (
        []
    )  # Preprocessed points with different magnitude (2^x)

    self.scalar_position = -1
    self.zero_states: List[int] = []
    self.buckets: Optional[jax.Array] = None
    self.result = None

  def read_trace(self, msm_reader: MSM_Reader):
    """Read MSM trace and store the data.

    Args:
      msm_reader: MSM_Reader object.
    """
    scalar = msm_reader.get_next_scalar()
    coordinates = msm_reader.get_next_base()
    points = [[] for _ in range(COORDINATE_NUM)]
    length = 0
    while scalar:
      self.scalars.append(scalar)
      coordinates = coordinates + [
          1 for _ in range(COORDINATE_NUM - len(coordinates))
      ]
      for i in range(COORDINATE_NUM):
        points[i].append(coordinates[i])
      length += 1
      scalar = msm_reader.get_next_scalar()
      coordinates = msm_reader.get_next_base()

    self.length = length
    self.zero_states = [1] * length
    for i in range(COORDINATE_NUM):
      self.points.append(utils.int_list_to_3d_array(points[i], BASE, CHUNK_NUM))

    self.blank_coordinate = utils.int_list_to_3d_array(
        [0] * length, BASE, CHUNK_NUM
    )
    self.buckets = jnp.array(
        [self.blank_coordinate] * COORDINATE_NUM, dtype=jnp.uint16
    )

  def compute_base(self, jax_dul):
    base_result = jnp.array(self.points, dtype=jnp.uint16)
    self.base_points.append(base_result)
    for i in range(1, 255):
      base_result = jax_dul(self.base_points[i - 1])
      self.base_points.append(base_result)

  def bucket_accumulation(self, jax_selective_padd):
    for i in range(0, 255):
      select_list = self.construct_select()
      select_jax = jnp.array(select_list, dtype=jnp.uint4)
      zero_states_jax = jnp.array(self.zero_states, dtype=jnp.uint4)
      self.buckets = jax_selective_padd(
          self.buckets, self.base_points[i], select_jax, zero_states_jax
      )
      self.update_zero_states(select_list)

  def bucket_merge(self, jax_add):
    buckets_jax = self.buckets
    length = self.length
    while length != 1:
      new_length = length >> 1
      half_a = buckets_jax[:, 0:new_length, 0:new_length, :]
      half_b = buckets_jax[:, new_length:length, new_length:length, :]
      buckets_jax = jax_add(half_a, half_b)
      length = new_length
    self.result = buckets_jax

  def construct_select(self):
    select_list = []
    self.scalar_position += 1
    index = 0
    for scalar in self.scalars:
      bit_value = (scalar >> self.scalar_position) & 0x1
      select_list.append(bit_value)

      index += 1
    # select = jnp.array(select_list, dtype=jnp.uint4)
    return select_list

  def update_zero_states(self, select_list):
    for i in range(0, len(self.zero_states)):
      self.zero_states[i] = int(self.zero_states[i] and (not select_list[i]))
      """
            not_zero, bit_zero -> 0 && 1 = 0 -> not zero
            not_zero, bit_one -> 0 && 0 = 0 -> not zero
            is_zero, bit_zero -> 1 && 1 = 1 -> is zero
            is_zero, bit_one -> 1 && 0 = 0 -> not zero
            """
