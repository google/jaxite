"""RNS Pippenger algorithm for elliptic curves.

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

Example Compiled Function:

bucket_accumulation_scan_jax_jit = jax.jit(
      bucket_accumulation_scan_jax,
      static_argnames='msm_length').lower(
        jax.ShapeDtypeStruct(
            (COORDINATE_NUM, WINDOW_NUM*BUCKET_NUM_PER_WINDOW, CHUNK_NUM),
            dtype=jnp.uint16
        ),
        jax.ShapeDtypeStruct(
            (MSM_LENGTH, COORDINATE_NUM, CHUNK_NUM),
            dtype=jnp.uint16
        ),
        jax.ShapeDtypeStruct(
            (MSM_LENGTH,WINDOW_NUM*BUCKET_NUM_PER_WINDOW),
            dtype=jnp.uint8
        ),
        jax.ShapeDtypeStruct(
            (MSM_LENGTH,WINDOW_NUM*BUCKET_NUM_PER_WINDOW),
            dtype=jnp.uint8
        ),
        MSM_LENGTH
      ).compile()

bucket_accumulation_index_scan_jax_jit = jax.jit(
    bucket_accumulation_index_scan_jax,
    static_argnames='msm_length').lower(
        jax.ShapeDtypeStruct(
            (COORDINATE_NUM, WINDOW_NUM,BUCKET_NUM_PER_WINDOW, CHUNK_NUM),
            dtype=jnp.uint16
        ),
        jax.ShapeDtypeStruct(
            (MSM_LENGTH, COORDINATE_NUM, CHUNK_NUM),
            dtype=jnp.uint16
        ),
        jax.ShapeDtypeStruct(
            (MSM_LENGTH,WINDOW_NUM),
            dtype=jnp.uint32
        ),
        jax.ShapeDtypeStruct(
            (MSM_LENGTH,WINDOW_NUM,BUCKET_NUM_PER_WINDOW),
            dtype=jnp.uint8
        ),
        MSM_LENGTH
      ).compile()

bucket_reduction_scan_jax_jit = jax.jit(
    bucket_reduction_scan_jax,
    static_argnames='bucket_num_in_window').lower(
        jax.ShapeDtypeStruct(
            (COORDINATE_NUM, WINDOW_NUM, BUCKET_NUM_PER_WINDOW, CHUNK_NUM),
            dtype=jnp.uint16
        ),
        jax.ShapeDtypeStruct(
            (COORDINATE_NUM, WINDOW_NUM, CHUNK_NUM),
            dtype=jnp.uint16
        ),
        jax.ShapeDtypeStruct(
            (COORDINATE_NUM, WINDOW_NUM, CHUNK_NUM),
            dtype=jnp.uint16
        ),
        jax.ShapeDtypeStruct(
            (BUCKET_NUM_PER_WINDOW, WINDOW_NUM),
            dtype=jnp.uint8
        ),
        jax.ShapeDtypeStruct(
            (BUCKET_NUM_PER_WINDOW, WINDOW_NUM),
            dtype=jnp.uint8
        ),
        jax.ShapeDtypeStruct(
            (BUCKET_NUM_PER_WINDOW, WINDOW_NUM,),
            dtype=jnp.uint8
        ),
        BUCKET_NUM_PER_WINDOW
      ).compile()

window_merge_scan_jax_jit = jax.jit(
      window_merge_scan_jax,
      static_argnames='slice_length').lower(
        jax.ShapeDtypeStruct((COORDINATE_NUM, WINDOW_NUM, CHUNK_NUM),
        dtype=jnp.uint16),
        SLICE_LENGTH
      ).compile()

selective_padd_with_zero_jit = jax.jit(
      selective_padd_with_zero).lower(
        jax.ShapeDtypeStruct(
          (COORDINATE_NUM, WINDOW_NUM*BUCKET_NUM_PER_WINDOW, COORDINATE_NUM),
          dtype=jnp.uint16
        ),
        jax.ShapeDtypeStruct(
            (COORDINATE_NUM, MSM_LENGTH, COORDINATE_NUM),
            dtype=jnp.uint16
        ),
        jax.ShapeDtypeStruct(
            (WINDOW_NUM*BUCKET_NUM_PER_WINDOW,),
            dtype=jnp.uint8
        ),
        jax.ShapeDtypeStruct(
            (WINDOW_NUM*BUCKET_NUM_PER_WINDOW,),
            dtype=jnp.uint8
        ),
      ).compile()
"""

import copy
import math
from typing import List

import jax
import jax.numpy as jnp
from jaxite.jaxite_ec import util
import jaxite.jaxite_ec.elliptic_curve as jec
import jaxite.jaxite_ec.finite_field as ff
import numpy as np


deepcopy = copy.deepcopy

"""
Example Parameters:
SLICE_LENGTH = 4
WINDOW_NUM = int(math.ceil(255 / SLICE_LENGTH))
BUCKET_NUM_PER_WINDOW = 2**SLICE_LENGTH
MSM_LENGTH = 1024
COORDINATE_NUM = 4
"""


def selective_padd_with_zero(partial_sum, single_point, select, is_zero):
  """Padd the partial sum with the single point, but only if the selection state is 1.

  Args:
    partial_sum: The partial sum.
    single_point: The single point.
    select: The selection state.
    is_zero: The zero states.

  Returns:
    The new partial sum.
  """
  _, batch_dim, _ = partial_sum.shape
  new_partial_sum = jec.padd_rns_xyzz_pack(partial_sum, single_point)

  cond_select = jnp.equal(select, 1).reshape(1, batch_dim, 1)
  sum_result = jnp.where(cond_select, new_partial_sum, partial_sum)

  cond_zero = jnp.equal(is_zero, 1).reshape(1, batch_dim, 1)
  cond_select_and_zero = jnp.logical_and(cond_select, cond_zero)
  result = jnp.where(cond_select_and_zero, single_point, sum_result)
  return result


def padd_with_zero(partial_sum, single_point, ps_is_zero, sp_is_zero):
  """Padd the partial sum with the single point.

  Check if the partial sum is equal to the single point first.

  Args:
    partial_sum: The partial sum.
    single_point: The single point.
    ps_is_zero: The zero states of the partial sum.
    sp_is_zero: The zero states of the single point.

  Returns:
    The new partial sum.
  """
  _, batch_dim, _ = partial_sum.shape
  new_partial_sum = jec.padd_rns_xyzz_pack(partial_sum, single_point)
  cond_sp_zero = jnp.equal(sp_is_zero, 1).reshape(1, batch_dim, 1)
  cond_ps_zero = jnp.equal(ps_is_zero, 1).reshape(1, batch_dim, 1)
  result_1 = jnp.where(cond_sp_zero, partial_sum, single_point)
  result_2 = jnp.where(
      jnp.logical_or(cond_sp_zero, cond_ps_zero), result_1, new_partial_sum
  )
  return result_2


def padd_with_zero_alter(partial_sum, single_point, ps_is_zero):

  _, batch_dim, _ = partial_sum.shape
  new_partial_sum = jec.padd_rns_xyzz_pack(partial_sum, single_point)
  cond_ps_zero = jnp.equal(ps_is_zero, 1).reshape(1, batch_dim, 1)
  result_2 = jnp.where(cond_ps_zero, single_point, new_partial_sum)
  return result_2


def padd_with_zero_and_pdul_check(
    partial_sum, single_point, ps_is_zero, sp_is_zero
):
  """Padd the partial sum with the single point.

  Check if the partial sum is equal to the single point first. If they are
  equal, then double the partial sum.

  Args:
    partial_sum: The partial sum.
    single_point: The single point.
    ps_is_zero: The zero states of the partial sum.
    sp_is_zero: The zero states of the single point.

  Returns:
    The new partial sum.
  """
  # coordinate_dim, batch_dim, precision_dim = partial_sum.shape
  _, batch_dim, _ = partial_sum.shape
  new_partial_sum = jec.padd_rns_xyzz_pack(partial_sum, single_point)
  double_partial_sum = jec.pdul_rns_xyzz_pack(partial_sum)
  cond_equal = jnp.all(partial_sum == single_point, axis=(0, 2)).reshape(
      1, batch_dim, 1
  )
  cond_sp_zero = jnp.equal(sp_is_zero, 1).reshape(1, batch_dim, 1)
  cond_ps_zero = jnp.equal(ps_is_zero, 1).reshape(1, batch_dim, 1)
  result_1 = jnp.where(cond_sp_zero, partial_sum, single_point)
  result_2 = jnp.where(
      jnp.logical_or(cond_sp_zero, cond_ps_zero), result_1, new_partial_sum
  )
  reuslt_3 = jnp.where(cond_equal, double_partial_sum, result_2)
  return reuslt_3


def bucket_accumulation_algorithm(
    all_buckets: jnp.ndarray,
    all_points: jnp.ndarray,
    selection_list: jnp.ndarray,
    zero_states_list: jnp.ndarray,
    msm_length: int,
):
  """Non-scan version BA."""
  coordinate_dim, buckets_dim, precision_dim = all_buckets.shape
  for i in range(msm_length):
    point = jax.lax.broadcast_in_dim(
        all_points[i], (coordinate_dim, buckets_dim, precision_dim), (0, 2)
    )
    all_buckets = selective_padd_with_zero(
        all_buckets, point, selection_list[i], zero_states_list[i]
    )
  return all_buckets


def bucket_accumulation_scan_algorithm(
    all_buckets: jnp.ndarray,
    all_points: jnp.ndarray,
    selection_list: jnp.ndarray,
    zero_states_list: jnp.ndarray,
    msm_length: int,
):
  """Scan version BA."""
  coordinate_dim, buckets_dim, precision_dim = all_buckets.shape

  def scan_body(buckets, point_with_cond_pack):
    point, selection, zero_states = point_with_cond_pack
    point = jax.lax.broadcast_in_dim(
        point, (coordinate_dim, buckets_dim, precision_dim), (0, 2)
    )
    all_buckets = selective_padd_with_zero(
        buckets, point, selection, zero_states
    )
    return all_buckets, None

  all_buckets, _ = jax.lax.scan(
      scan_body,
      all_buckets,
      (all_points, selection_list, zero_states_list),
      length=msm_length,
  )
  return all_buckets


def bucket_accumulation_index_algorithm(
    all_buckets: jnp.ndarray,
    all_points: jnp.ndarray,
    selection_index_list: jnp.ndarray,
    zero_states_list: jnp.ndarray,
    msm_length: int,
):
  """Non-scan version BA with index selection."""
  # buckets_dim is not used in the algorithm, changed it into _.
  coordinate_dim, window_dim, _, precision_dim = all_buckets.shape
  for i in range(msm_length):
    point = jax.lax.broadcast_in_dim(
        all_points[i], (coordinate_dim, window_dim, precision_dim), (0, 2)
    )
    selective_buckets = all_buckets[
        :, jnp.arange(window_dim), selection_index_list[i], :
    ]
    selective_zero_states = zero_states_list[
        i, jnp.arange(window_dim), selection_index_list[i]
    ]
    selective_update = padd_with_zero_alter(
        selective_buckets, point, selective_zero_states
    )
    all_buckets = all_buckets.at[
        :, jnp.arange(window_dim), selection_index_list[i], :
    ].set(selective_update)
  return all_buckets


def bucket_accumulation_index_scan_algorithm(
    all_buckets: jnp.ndarray,
    all_points: jnp.ndarray,
    selection_index_list: jnp.ndarray,
    zero_states_list: jnp.ndarray,
    msm_length: int,
):
  """Scan version BA with index selection."""
  # buckets_dim is not used in the algorithm, changed it into _.
  coordinate_dim, window_dim, _, precision_dim = all_buckets.shape

  def scan_body(buckets, point_with_cond_pack):
    point, selection_index, zero_states = point_with_cond_pack
    point = jax.lax.broadcast_in_dim(
        point, (coordinate_dim, window_dim, precision_dim), (0, 2)
    )
    selective_buckets = buckets[:, jnp.arange(window_dim), selection_index, :]
    selective_zero_states = zero_states[jnp.arange(window_dim), selection_index]
    selective_update = padd_with_zero_alter(
        selective_buckets, point, selective_zero_states
    )
    return (
        buckets.at[:, jnp.arange(window_dim), selection_index, :].set(
            selective_update
        ),
        None,
    )

  all_buckets, _ = jax.lax.scan(
      scan_body,
      all_buckets,
      (all_points, selection_index_list, zero_states_list),
      length=msm_length,
  )
  return all_buckets


def bucket_reduction_algorithm(
    all_buckets: jnp.ndarray,
    temp_sum: jnp.ndarray,
    window_sum: jnp.ndarray,
    bucket_zero_states_list: jnp.ndarray,
    temp_zero_states_list: jnp.ndarray,
    window_zero_states_list: jnp.ndarray,
    bucket_num_in_window: int,
):
  """Non-scan version BR."""
  all_buckets = jnp.flip(all_buckets.transpose(2, 0, 1, 3), axis=0)
  for i in range(bucket_num_in_window - 1):
    temp_sum = padd_with_zero(
        temp_sum,
        all_buckets[i],
        temp_zero_states_list[i],
        bucket_zero_states_list[i],
    )
    window_sum = padd_with_zero_and_pdul_check(
        window_sum,
        temp_sum,
        window_zero_states_list[i],
        temp_zero_states_list[i + 1],
    )
  return window_sum


def bucket_reduction_scan_algorithm(
    all_buckets: jnp.ndarray,
    temp_sum: jnp.ndarray,
    window_sum: jnp.ndarray,
    bucket_zero_states_list: jnp.ndarray,
    temp_zero_states_list: jnp.ndarray,
    window_zero_states_list: jnp.ndarray,
    bucket_num_in_window: int,
):
  """Scan version BR."""
  all_buckets = jnp.flip(all_buckets.transpose(2, 0, 1, 3), axis=0)

  def scan_body(temp_and_window_sum_pack, bucket_with_cond_pack):
    temp_sum, window_sum = temp_and_window_sum_pack
    (
        bucket,
        bucket_zero_states,
        temp_zero_states,
        temp_zero_states1,
        window_zero_states,
    ) = bucket_with_cond_pack
    temp_sum = padd_with_zero(
        temp_sum, bucket, temp_zero_states, bucket_zero_states
    )
    window_sum = padd_with_zero_and_pdul_check(
        window_sum, temp_sum, window_zero_states, temp_zero_states1
    )
    return (temp_sum, window_sum), None

  (_, window_sum), _ = jax.lax.scan(
      scan_body,
      (temp_sum, window_sum),
      (
          all_buckets[: bucket_num_in_window - 1],
          bucket_zero_states_list[: bucket_num_in_window - 1],
          temp_zero_states_list[: bucket_num_in_window - 1],
          temp_zero_states_list[1:],
          window_zero_states_list[: bucket_num_in_window - 1],
      ),
      length=bucket_num_in_window - 1,
  )

  return window_sum


def window_merge_algorithm(window_sum: jnp.ndarray, slice_length: int):
  """Non-scan version WM."""
  coordinate_dim, window_dim, precision_dim = window_sum.shape
  result = window_sum[:, window_dim - 1, :].reshape(
      (coordinate_dim, 1, precision_dim)
  )
  for w in range(window_dim - 2, -1, -1):
    for _ in range(slice_length):
      result = jec.pdul_rns_xyzz_pack(result)
    result = jec.padd_rns_xyzz_pack(
        result,
        window_sum[:, w, :].reshape((coordinate_dim, 1, util.NUM_MODULI)),
    )

  result = result.reshape((coordinate_dim, precision_dim))
  return result


def window_merge_scan_algorithm(window_sum: jnp.ndarray, slice_length: int):
  """Scan version WM."""
  coordinate_dim, window_dim, precision_dim = window_sum.shape
  window_sum = window_sum.transpose(1, 0, 2)
  result = window_sum[window_dim - 1, :, :].reshape(
      (coordinate_dim, 1, precision_dim)
  )

  def fori_loop_body(_, result):
    result = jec.pdul_rns_xyzz_pack(result)
    return result

  def scan_body(result, window_sum):
    result = jax.lax.fori_loop(0, slice_length, fori_loop_body, result)
    result = jec.padd_rns_xyzz_pack(
        result, window_sum.reshape((coordinate_dim, 1, util.NUM_MODULI))
    )
    return result, None

  result, _ = jax.lax.scan(
      scan_body,
      result,
      window_sum[: window_dim - 1, :, :],
      reverse=True,
      length=window_dim - 1,
  )
  result = result.reshape((coordinate_dim, precision_dim))
  return result


class MSMPippenger:
  """Pippenger algorithm for elliptic curves.

  Attributes:
    coordinate_num: The number of coordinates in the elliptic curve.
    slice_length: The length of each slice in the elliptic curve.
    window_num: The number of windows in the elliptic curve.
    bucket_num_per_window: The number of buckets in each window.
    bucket_num_in_window: The number of buckets in each window.
    slice_mask: The mask for the slices in the elliptic curve.
    blank_point: A JAX array of zeros, used to initialize the buckets.
    all_buckets: A JAX array of all the buckets in the elliptic curve.
    points: A list of JAX arrays, where each array represents an Orignal point
      from the trace.
    scalars: A list of integers, where each integer represents an Orignal scalar
      from the trace.
    all_points: A JAX array of all the points in the elliptic curve. from the
      trace.
    window_sum: A JAX array of the window sum.
    bucket_zero_states: A JAX array of the zero states for the buckets.
    temp_sum_zero_states: A JAX array of the zero states for the temp sum.
    window_sum_zero_states: A JAX array of the zero states for the window sum.
    zero_states_list: A JAX array of the zero states for the buckets.
    selection_list: A JAX array of the selection states for the buckets.
    selection_index_list: A JAX array of the selection index for the buckets.
    msm_length: The length of the MSM trace.
    result: The final elliptic curve.
    rns_mat: The lazy matrix used for padding and doubling.
  """

  def __init__(self, slice_length):
    self.coordinate_num = util.COORDINATE_NUM

    self.slice_length = slice_length
    self.window_num = int(math.ceil(254 / self.slice_length))  #
    self.bucket_num_per_window = 2**self.slice_length
    self.slice_mask = self.bucket_num_per_window - 1
    self.blank_point = util.int_list_to_array(
        [0, 0, 0, 0], util.BASE, util.NUM_MODULI
    ).reshape(self.coordinate_num, 1, util.NUM_MODULI)

    self.all_buckets = jnp.broadcast_to(
        self.blank_point.reshape(
            1, self.coordinate_num, 1, util.NUM_MODULI
        ).transpose(1, 0, 2, 3),
        (
            self.coordinate_num,
            self.window_num,
            self.bucket_num_per_window,
            util.NUM_MODULI,
        ),
    )

    self.window_sum: jnp.ndarray

    self.msm_length = 0
    self.bucket_zero_states: jnp.ndarray
    self.temp_sum_zero_states: jnp.ndarray
    self.window_sum_zero_states: jnp.ndarray
    self.zero_states_list: jnp.ndarray
    self.selection_list: jnp.ndarray
    self.selection_index_list: jnp.ndarray
    self.all_points: jnp.ndarray

    self.scalars: List[int] = []  # Orignal scalar from the trace
    # [Points, Points, ..., Points]
    self.points: List[jnp.ndarray] = []  # Orignal points from the trace
    self.rns_mat = util.construct_rns_matrix(util.MODULUS_377_INT)

    self.result = None

  def initialize(self, scalars, points):
    """Initialize the Pippenger algorithm.

    Args:
      scalars: A list of integers, where each integer represents an Orignal
        scalar from the trace.
      points: A list of JAX arrays, where each array represents an Orignal point
        from the trace.
    """
    # Initial internal selection from the scalar
    self.scalars = scalars
    self.msm_length = len(scalars)

    # Convert high-precision points into a vector of low-precision chunks
    self.points = [
        util.int_list_to_array_rns(coordinates + [1, 1])
        for coordinates in points
    ]  # pytype: disable=container-type-mismatch

    self.all_points = jnp.array(self.points).astype(jnp.uint16)

    # For BA
    zero_states_pylist, selection_pylist, selection_index_pylist = (
        self.construct_ba_zero_states_and_selection()
    )
    self.selection_list = jnp.array(selection_pylist, dtype=jnp.uint8).reshape(
        (-1, self.window_num * self.bucket_num_per_window)
    )
    # For index selection version BA
    self.zero_states_list = jnp.array(zero_states_pylist, dtype=jnp.uint8)
    self.selection_index_list = jnp.array(
        selection_index_pylist, dtype=jnp.uint32
    )

    # For BR
    (
        bucket_zero_states_py,
        temp_sum_zero_states_py,
        window_sum_zero_states_py,
    ) = self.construct_br_zero_states(
        zero_states_pylist[len(zero_states_pylist) - 1]
    )
    self.bucket_zero_states = jnp.array(bucket_zero_states_py, dtype=jnp.uint8)
    self.temp_sum_zero_states = jnp.array(
        temp_sum_zero_states_py, dtype=jnp.uint8
    )
    self.window_sum_zero_states = jnp.array(
        window_sum_zero_states_py, dtype=jnp.uint8
    )

  def bucket_accumulation(self, bucket_accumulation_index_func):
    """BA index selection version."""
    self.all_buckets = bucket_accumulation_index_func(
        self.all_buckets,
        self.all_points,
        self.selection_index_list,
        self.zero_states_list[: self.msm_length],
    )

    return self.all_buckets

  def bucket_reduction(self, bucket_reduction_func):
    """Reduce the buckets to a single point for each window."""
    temp_sum = jnp.broadcast_to(
        self.blank_point,
        (
            self.coordinate_num,
            self.window_num,
            util.NUM_MODULI,
        ),
    )
    window_sum = jnp.broadcast_to(
        self.blank_point,
        (
            self.coordinate_num,
            self.window_num,
            util.NUM_MODULI,
        ),
    )
    self.window_sum = bucket_reduction_func(
        self.all_buckets,
        temp_sum,
        window_sum,
        self.bucket_zero_states[: self.bucket_num_per_window],
        self.temp_sum_zero_states[: self.bucket_num_per_window],
        self.window_sum_zero_states[: self.bucket_num_per_window],
    )
    return self.window_sum

  def window_merge(self, window_merge_func):
    """Merge the windows to form the final elliptic curve.

    Args:
      window_merge_func: The function to merge the windows.

    Returns:
      The final elliptic curve.
    """
    self.result = window_merge_func(self.window_sum)
    return self.result

  def construct_ba_zero_states_and_selection(self):
    """Construct the zero states and selection for the bucket accumulation (BA) step.

    Returns:
      A tuple of two lists: the zero states for the bucket accumulation, and the
      selection for the bucket accumulation.
    """
    zero_states = [
        deepcopy([1] * self.bucket_num_per_window)
        for _ in range(self.window_num)
    ]
    zero_states_list = []
    zero_states_list.append(deepcopy(zero_states))
    selection_list = []
    selection_index_list = []  # Used for index selection
    for scalar in self.scalars:
      # Compute the zero states for each scalar by time dependence
      selection = [
          deepcopy(([0] * self.bucket_num_per_window))
          for _ in range(self.window_num)
      ]
      selection_index = []
      for w in range(self.window_num):
        slice_index = (scalar >> (w * self.slice_length)) & self.slice_mask
        zero_states[w][slice_index] = 0
        selection[w][slice_index] = 1
        selection_index.append(slice_index)

      selection_list.append(deepcopy(selection))
      zero_states_list.append(deepcopy(zero_states))
      selection_index_list.append(deepcopy(selection_index))
    return zero_states_list, selection_list, selection_index_list

  def construct_br_zero_states(self, bucket_zero_states):
    """Construct the zero states for the bucket reduction (BR) step.

    Args:
      bucket_zero_states: The zero states of the buckets.

    Returns:
      A tuple of three lists: the zero states for the bucket reduction, the zero
      states for the temporary sum, and the zero states for the window sum.
    """
    temp_sum_zero_states = np.array([1] * self.window_num)
    window_sum_zero_states = np.array([1] * self.window_num)
    temp_sum_zero_states_list = []
    window_sum_zero_states_list = []
    temp_sum_zero_states_list.append(temp_sum_zero_states)
    window_sum_zero_states_list.append(window_sum_zero_states)
    bucket_zero_states_list = np.flip(
        np.array(bucket_zero_states).transpose(1, 0), axis=0
    )
    for b in range(self.bucket_num_per_window):
      next_temp_sum_zero_states = (
          temp_sum_zero_states_list[b] & bucket_zero_states_list[b]
      )
      next_window_sum_zero_states = (
          window_sum_zero_states_list[b] & next_temp_sum_zero_states
      )
      temp_sum_zero_states_list.append(next_temp_sum_zero_states)
      window_sum_zero_states_list.append(next_window_sum_zero_states)
    return (
        bucket_zero_states_list,
        temp_sum_zero_states_list,
        window_sum_zero_states_list,
    )


#########################
# Functions for twisted curve
#########################


def padd(partial_sum, single_point):
  return jec.padd_rns_twisted_pack(partial_sum, single_point)


def padd_with_pdul_check(partial_sum, single_point):
  # coordinate_dim, batch_dim, precision_dim = partial_sum.shape
  _, batch_dim, _ = partial_sum.shape
  new_partial_sum = jec.padd_rns_twisted_pack(partial_sum, single_point)
  double_partial_sum = jec.pdul_rns_twisted_pack(partial_sum)
  cond_equal = jnp.all(partial_sum == single_point, axis=(0, 2)).reshape(
      1, batch_dim, 1
  )
  return jnp.where(cond_equal, double_partial_sum, new_partial_sum)


def bucket_accumulation_index_scan_parallel_algorithm_twisted(
    all_buckets: jnp.ndarray,
    all_points: jnp.ndarray,
    selection_index_list: jnp.ndarray,
    msm_length: int,
):
  """Scan version BA with index selection."""
  # buckets_dim is not used in the algorithm, changed it into _.
  coordinate_dim, batch_window_dim, _, precision_dim = all_buckets.shape
  _, _, parallel_dim, _ = (
      all_points.shape
  )  # (serial_dim, coordinate_dim, parallel_dim, precision_dim)
  single_window_dim = batch_window_dim // parallel_dim

  def scan_body(buckets, point_with_cond_pack):
    point, selection_index = point_with_cond_pack
    point = jax.lax.broadcast_in_dim(
        point,
        (coordinate_dim, parallel_dim, single_window_dim, precision_dim),
        (0, 1, 3),
    )
    point = point.reshape((coordinate_dim, batch_window_dim, precision_dim))
    selective_buckets = buckets[
        :, jnp.arange(batch_window_dim), selection_index, :
    ]
    selective_update = padd(selective_buckets, point)
    return (
        buckets.at[:, jnp.arange(batch_window_dim), selection_index, :].set(
            selective_update
        ),
        None,
    )

  all_buckets, _ = jax.lax.scan(
      scan_body,
      all_buckets,
      (all_points, selection_index_list),
      length=msm_length,
  )
  return all_buckets


def bucket_reduction_scan_algorithm_twisted(
    all_buckets: jnp.ndarray,
    temp_sum: jnp.ndarray,
    window_sum: jnp.ndarray,
    bucket_num_in_window: int,
):
  """Scan version BR."""
  all_buckets = jnp.flip(all_buckets.transpose(2, 0, 1, 3), axis=0)

  def scan_body(temp_and_window_sum_pack, buckets):
    temp_sum, window_sum = temp_and_window_sum_pack
    temp_sum = padd(temp_sum, buckets)
    window_sum = padd_with_pdul_check(window_sum, temp_sum)
    return (temp_sum, window_sum), None

  (_, window_sum), _ = jax.lax.scan(
      scan_body,
      (temp_sum, window_sum),
      all_buckets[:bucket_num_in_window],
      length=bucket_num_in_window,
  )

  return window_sum


def batch_window_summation_algorithm_twisted(
    batch_window_sum: jnp.ndarray,
    all_window_sum: jnp.ndarray,
    point_parallel: int,
):
  """Batch window summation algorithm for twisted curve."""
  # batch_window_dim is not used in the algorithm, changed it into _.
  coordinate_dim, _, precision_dim = all_window_sum.shape
  all_window_sum = all_window_sum.reshape(
      (coordinate_dim, point_parallel, -1, precision_dim)
  ).transpose(1, 0, 2, 3)

  def scan_body(batch_window_sum, single_window_sum):
    batch_window_sum = padd(batch_window_sum, single_window_sum)
    return batch_window_sum, None

  batch_window_sum, _ = jax.lax.scan(
      scan_body, batch_window_sum, all_window_sum, length=point_parallel
  )
  return batch_window_sum


def window_merge_scan_algorithm_twisted(
    window_sum: jnp.ndarray, slice_length: int
):
  """Scan version WM."""
  coordinate_dim, window_dim, precision_dim = window_sum.shape
  window_sum = window_sum.transpose(1, 0, 2)
  result = window_sum[window_dim - 1, :, :].reshape(
      (coordinate_dim, 1, precision_dim)
  )

  def fori_loop_body(_, result):
    result = jec.pdul_rns_twisted_pack(result)
    return result

  def scan_body(result, window_sum):
    result = jax.lax.fori_loop(0, slice_length, fori_loop_body, result)
    result = jec.padd_rns_twisted_pack(
        result, window_sum.reshape((coordinate_dim, 1, util.NUM_MODULI))
    )
    return result, None

  result, _ = jax.lax.scan(
      scan_body,
      result,
      window_sum[: window_dim - 1, :, :],
      reverse=True,
      length=window_dim - 1,
  )
  result = result.reshape((coordinate_dim, precision_dim))
  return result


class MSMPippengerTwisted:
  """Pippenger algorithm for elliptic curves with twisted points.

  Attributes:
    coordinate_num: The number of coordinates in the elliptic curve.
    slice_length: The length of each slice in the elliptic curve.
    point_parallel: The number of parallel points in the elliptic curve.
    window_num: The number of windows in the elliptic curve.
    batch_window_num: The number of batch windows in the elliptic curve.
    bucket_num_per_window: The number of buckets in each window.
    slice_mask: The mask for the slices in the elliptic curve.
    blank_point: A JAX array of zeros, used to initialize the buckets.
    all_buckets: A JAX array of all the buckets in the elliptic curve.
    points: A list of JAX arrays, where each array represents an Orignal point
      from the trace.
    scalars: A list of integers, where each integer represents an Orignal scalar
      from the trace.
    all_points: A JAX array of all the points in the elliptic curve. from the
      trace.
    window_sum: A JAX array of the window sum.
    br_temp_sum: A JAX array of the temp sum for bucket reduction.
    batch_window_sum: A JAX array of the batch window sum.
    selection_index_list: A JAX array of the selection index for the buckets.
    msm_length: The length of the MSM trace.
    result: The final elliptic curve.
    rns_mat: The lazy matrix used for padding and doubling.
  """

  def __init__(self, slice_length: int, point_parallel: int):

    self.coordinate_num = util.COORDINATE_NUM

    self.slice_length = slice_length
    self.point_parallel = point_parallel
    self.window_num = int(math.ceil(254 / self.slice_length))  #
    self.batch_window_num = self.window_num * self.point_parallel
    self.bucket_num_per_window = (
        2**self.slice_length - 1
    )  # Note: here remove the bucket_0
    self.slice_mask = 2**self.slice_length - 1
    self.blank_point = (
        util.int_list_to_array_rns([0, 1, 1, 0])
        .reshape(self.coordinate_num, 1, util.NUM_MODULI)
        .astype(jnp.uint16)
    )

    self.all_buckets = jnp.broadcast_to(
        self.blank_point.reshape(
            1, self.coordinate_num, 1, util.NUM_MODULI
        ).transpose(1, 0, 2, 3),
        (
            self.coordinate_num,
            self.window_num,
            self.bucket_num_per_window,
            util.NUM_MODULI,
        ),
    )

    self.all_buckets = jnp.tile(
        self.all_buckets, (1, self.point_parallel, 1, 1)
    )

    self.window_sum: jnp.ndarray
    self.br_temp_sum: jnp.ndarray
    self.batch_window_sum: jnp.ndarray

    self.msm_length = 0

    self.selection_index_list: jnp.ndarray
    self.all_points: jnp.ndarray

    self.scalars: List[int] = []  # Orignal scalar from the trace
    # [Points, Points, ..., Points]
    self.points: List[jnp.ndarray] = []  # Orignal points from the trace
    self.rns_mat = util.construct_rns_matrix(util.MODULUS_377_INT)

    self.result = None

  def initialize(self, scalars, points):
    """Initialize the Pippenger algorithm.

    Args:
      scalars: A list of integers, where each integer represents an Orignal
        scalar from the trace.
      points: A list of JAX arrays, where each array represents an Orignal point
        from the trace.
    """
    # Initial internal selection from the scalar
    self.scalars = scalars
    self.msm_length = len(scalars)

    # Convert high-precision points into a vector of low-precision chunks
    self.points = [
        util.int_list_to_array_rns(coordinates) for coordinates in points
    ]  # pytype: disable=container-type-mismatch

    self.all_points = jnp.array(self.points).astype(jnp.uint16)
    _, coordinate_dim, precision_dim = self.all_points.shape

    # For BA
    selection_index_pylist = self.construct_ba_selection()
    # Note: it contains uint(-1) for the bucket_0.
    # In BA, it may cause some undefined behavior when do bucket selection
    # it is correct now, because when setting buckets after the computation,
    # jax.numpy will ignore the index with uint(-1) out of index.
    self.selection_index_list = jnp.array(selection_index_pylist).astype(
        jnp.uint32
    )
    _, window_dim = self.selection_index_list.shape

    # Batch construction
    self.all_points = self.all_points.reshape(
        (-1, self.point_parallel, coordinate_dim, precision_dim)
    ).transpose(0, 2, 1, 3)
    self.selection_index_list = self.selection_index_list.reshape(
        (-1, window_dim * self.point_parallel)
    )
    self.br_temp_sum = jnp.broadcast_to(
        self.blank_point,
        (
            self.coordinate_num,
            self.batch_window_num,
            util.NUM_MODULI,
        ),
    )
    self.window_sum = jnp.broadcast_to(
        self.blank_point,
        (
            self.coordinate_num,
            self.batch_window_num,
            util.NUM_MODULI,
        ),
    )
    self.batch_window_sum = jnp.broadcast_to(
        self.blank_point,
        (
            self.coordinate_num,
            self.window_num,
            util.NUM_MODULI,
        ),
    )

  def bucket_accumulation(self, bucket_accumulation_index_func):
    """BA index selection version."""
    self.all_buckets = bucket_accumulation_index_func(
        self.all_buckets, self.all_points, self.selection_index_list
    )
    return self.all_buckets

  def bucket_reduction(self, bucket_reduction_func):
    """Reduce the buckets to a single point for each window."""
    temp_sum = jnp.broadcast_to(
        self.blank_point,
        (
            self.coordinate_num,
            self.batch_window_num,
            util.NUM_MODULI,
        ),
    )
    window_sum = jnp.broadcast_to(
        self.blank_point,
        (
            self.coordinate_num,
            self.batch_window_num,
            util.NUM_MODULI,
        ),
    )
    self.window_sum = bucket_reduction_func(
        self.all_buckets, temp_sum, window_sum
    )
    return self.window_sum

  def batch_window_summation(self, batch_window_summation_func):
    """Sum the batch windows to form the final window sum."""
    batch_window_sum = jnp.broadcast_to(
        self.blank_point,
        (
            self.coordinate_num,
            self.window_num,
            util.NUM_MODULI,
        ),
    )
    self.window_sum = batch_window_summation_func(
        batch_window_sum, self.window_sum
    )
    return self.window_sum

  def window_merge(self, window_merge_func):
    """Merge the windows to form the final elliptic curve."""
    self.result = window_merge_func(self.window_sum)
    return self.result

  def construct_ba_selection(self):
    selection_index_list = []  # Used for index selection
    for scalar in self.scalars:
      # Compute the zero states for each scalar by time dependence
      selection_index = []
      for w in range(self.window_num):
        slice_index = (
            (scalar >> (w * self.slice_length)) & self.slice_mask
        ) - 1
        selection_index.append(slice_index)
      selection_index_list.append(deepcopy(selection_index))
    return selection_index_list
