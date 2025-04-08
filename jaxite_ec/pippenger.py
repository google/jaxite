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
from typing import List

import jax
import jax.numpy as jnp
import jaxite.jaxite_ec.elliptic_curve as jec
import jaxite.jaxite_ec.finite_field as ff
import jaxite.jaxite_ec.util as utils
import numpy as np

deepcopy = copy.deepcopy


class MSMPippenger:
  """Pippenger algorithm for elliptic curves.

  Attributes:
    coordinate_num: The number of coordinates in the elliptic curve.
    slice_length: The length of each slice in the elliptic curve.
    window_num: The number of windows in the elliptic curve.
    bucket_num_in_window: The number of buckets in each window.
    slice_mask: The mask for the slices in the elliptic curve.
    blank_point: A JAX array of zeros, used to initialize the buckets.
    all_buckets_jax: A list of lists of JAX arrays, where each array represents
      a bucket.
    selection_list_jax: A list of lists of integers, where each integer
      represents the selection state of a bucket.
    zero_states_list_jax: A list of lists of integers, where each integer
      represents the zero state of a bucket.
    bucket_zero_states_jax: A list of lists of integers, where each integer
      represents the zero state of a bucket.
    temp_sum_zero_states_jax: A list of integers, where each integer represents
      the zero state of a bucket.
    window_sum_zero_states_jax: A list of integers, where each integer
      represents the zero state of the window sum.
    all_points_jax: A JAX array of all the points in the elliptic curve.
    scalars: A list of integers, where each integer represents an Orignal scalar
      from the trace.
    points: A list of JAX arrays, where each array represents an Orignal point
      from the trace.
    window_sum: A JAX array of the window sum.
    msm_length: The length of the MSM trace.
    result: The final elliptic curve.
    lazy_mat: The lazy matrix used for padding and doubling.
  """

  def __init__(self, slice_length):
    self.coordinate_num = utils.COORDINATE_NUM

    self.slice_length = slice_length
    self.window_num = int(math.ceil(255 / self.slice_length))
    self.bucket_num_in_window = 2**self.slice_length
    self.slice_mask = self.bucket_num_in_window - 1
    self.blank_point = utils.int_list_to_jax_array(
        [0] * self.coordinate_num, utils.BASE, utils.U16_EXT_CHUNK_NUM
    )

    self.all_buckets_jax = (
        jnp.array([
            [self.blank_point for _ in range(self.bucket_num_in_window)]
            for _ in range(self.window_num)
        ])
        .reshape((
            self.bucket_num_in_window * self.window_num,
            self.coordinate_num,
            utils.U16_EXT_CHUNK_NUM,
        ))
        .transpose(1, 0, 2)
    )

    self.window_sum: jnp.ndarray

    self.msm_length = 0
    self.zero_states_list_jax: jnp.ndarray
    self.selection_list_jax: jnp.ndarray
    self.all_points_jax: jnp.ndarray

    self.scalars: List[int] = []  # Orignal scalar from the trace
    # [Points, Points, ..., Points]
    self.points: List[jnp.ndarray] = []  # Orignal points from the trace
    self.lazy_mat = ff.construct_lazy_matrix(utils.MODULUS_377_INT)

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
        utils.int_list_to_2d_array(
            coordinates + [1, 1], utils.BASE, utils.U16_EXT_CHUNK_NUM
        )
        for coordinates in points
    ]  # pytype: disable=container-type-mismatch

    self.all_points_jax = jnp.array(self.points)

    # For BA
    zero_states_list, selection_list = (
        self.construct_ba_zero_states_and_selection()
    )
    self.zero_states_list_jax = jnp.array(zero_states_list).reshape(
        (-1, self.window_num * self.bucket_num_in_window)
    )
    self.selection_list_jax = jnp.array(selection_list).reshape(
        (-1, self.window_num * self.bucket_num_in_window)
    )

    # For BR
    bucket_zero_states, temp_sum_zero_states, window_sum_zero_states = (
        self.construct_bm_zero_states(
            zero_states_list[len(zero_states_list) - 1]
        )
    )
    self.bucket_zero_states_jax = jnp.array(bucket_zero_states)
    self.temp_sum_zero_states_jax = jnp.array(temp_sum_zero_states)
    self.window_sum_zero_states_jax = jnp.array(window_sum_zero_states)

  def selective_padd_with_zero(
      self, partial_sum, single_point, select, is_zero
  ):
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
    new_partial_sum = jec.padd_lazy_xyzz_pack(
        partial_sum, single_point, self.lazy_mat
    )

    cond_select = jnp.equal(select, 1).reshape(1, batch_dim, 1)
    sum_result = jnp.where(cond_select, new_partial_sum, partial_sum)

    cond_zero = jnp.equal(is_zero, 1).reshape(1, batch_dim, 1)
    cond_select_and_zero = jnp.logical_and(cond_select, cond_zero)
    result = jnp.where(cond_select_and_zero, single_point, sum_result)
    return result

  def padd_with_zero(self, partial_sum, single_point, ps_is_zero, sp_is_zero):
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
    new_partial_sum = jec.padd_lazy_xyzz_pack(
        partial_sum, single_point, self.lazy_mat
    )
    cond_sp_zero = jnp.equal(sp_is_zero, 1).reshape(1, batch_dim, 1)
    cond_ps_zero = jnp.equal(ps_is_zero, 1).reshape(1, batch_dim, 1)
    result_1 = jnp.where(cond_sp_zero, partial_sum, single_point)
    result_2 = jnp.where(
        jnp.logical_or(cond_sp_zero, cond_ps_zero), result_1, new_partial_sum
    )
    return result_2

  def padd_with_zero_and_pdul_check(
      self, partial_sum, single_point, ps_is_zero, sp_is_zero
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
    new_partial_sum = jec.padd_lazy_xyzz_pack(
        partial_sum, single_point, self.lazy_mat
    )
    double_partial_sum = jec.pdul_lazy_xyzz_pack(partial_sum, self.lazy_mat)
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

  def bucket_accumulation(self):
    """Accumulate the buckets for the MSM trace.

    Args: None

    Returns:
      None
    """
    all_buckets = self.all_buckets_jax
    coordinate_dim, buckets_dim, precision_dim = all_buckets.shape
    for i in range(self.msm_length):
      point = jax.lax.broadcast_in_dim(
          self.all_points_jax[i],
          (coordinate_dim, buckets_dim, precision_dim),
          (0, 2),
      )
      all_buckets = self.selective_padd_with_zero(
          all_buckets,
          point,
          self.selection_list_jax[i],
          self.zero_states_list_jax[i],
      )
    self.all_buckets_jax = all_buckets

  def bucket_reduction(self):
    """Reduce the buckets to a single point for each window."""
    temp_sum = jnp.array(
        [self.blank_point for _ in range(self.window_num)]
    ).transpose(1, 0, 2)
    window_sum = jnp.array(
        [self.blank_point for _ in range(self.window_num)]
    ).transpose(1, 0, 2)
    local_all_buckets_jax = self.all_buckets_jax.reshape((
        self.coordinate_num,
        self.window_num,
        self.bucket_num_in_window,
        utils.U16_EXT_CHUNK_NUM,
    ))
    local_all_buckets_jax = jnp.flip(
        local_all_buckets_jax.transpose(0, 2, 1, 3), axis=1
    )
    for i in range(self.bucket_num_in_window - 1):
      temp_sum = self.padd_with_zero(
          temp_sum,
          local_all_buckets_jax[:, i, :, :],
          self.temp_sum_zero_states_jax[i],
          self.bucket_zero_states_jax[i],
      )
      window_sum = self.padd_with_zero_and_pdul_check(
          window_sum,
          temp_sum,
          self.window_sum_zero_states_jax[i],
          self.temp_sum_zero_states_jax[i + 1],
      )
    self.window_sum = window_sum

  def window_merge(self):
    """Merge the windows to form the final elliptic curve."""
    self.result = self.window_sum[:, self.window_num - 1, :].reshape(
        (self.coordinate_num, 1, utils.U16_EXT_CHUNK_NUM)
    )
    for w in range(self.window_num - 2, -1, -1):
      for _ in range(self.slice_length):
        self.result = jec.pdul_lazy_xyzz_pack(self.result, self.lazy_mat)
      self.result = jec.padd_lazy_xyzz_pack(
          self.result,
          self.window_sum[:, w, :].reshape(
              (self.coordinate_num, 1, utils.U16_EXT_CHUNK_NUM)
          ),
          self.lazy_mat,
      )
    self.result = self.result.reshape(
        (self.coordinate_num, utils.U16_EXT_CHUNK_NUM)
    )
    return self.result

  def construct_ba_zero_states_and_selection(self):
    """Construct the zero states and selection for the bucket accumulation (BA) step.

    Returns:
      A tuple of two lists: the zero states for the bucket accumulation, and the
      selection for the bucket accumulation.
    """
    zero_states = [
        deepcopy([1] * self.bucket_num_in_window)
        for _ in range(self.window_num)
    ]
    zero_states_list = []
    zero_states_list.append(deepcopy(zero_states))
    selection_list = []
    for scalar in self.scalars:
      # Compute the zero states for each scalar by time dependence
      selection = [
          deepcopy(([0] * self.bucket_num_in_window))
          for _ in range(self.window_num)
      ]
      for w in range(self.window_num):
        slice_index = (scalar >> (w * self.slice_length)) & self.slice_mask
        zero_states[w][slice_index] = 0
        selection[w][slice_index] = 1

      selection_list.append(deepcopy(selection))
      zero_states_list.append(deepcopy(zero_states))
    return zero_states_list, selection_list

  def construct_bm_zero_states(self, bucket_zero_states):
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
    for b in range(self.bucket_num_in_window):
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
