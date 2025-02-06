"""Utility functions for jaxite_ec."""

from typing import List

import jax
import jax.numpy as jnp


BASE = 16
BASE_TYPE = jnp.uint16
U16_CHUNK_NUM = 24


def print_hex_values(int_list):
  hex_values = " ".join((hex(value)) for value in int_list)
  print(hex_values)


def bits_to_jnp_dtype(bits):
  if bits == 8:
    return jnp.uint8
  elif bits == 16:
    return jnp.uint16
  elif bits == 32:
    return jnp.uint32
  elif bits == 64:
    return jnp.uint64
  else:
    raise ValueError("Unsupported bit size. Use 8, 16, 32, or 64.")


def int_point_to_jax_point(coordinate_x, coordinate_y, z=None):
  x_array = int_to_array(coordinate_x, BASE, BASE_TYPE, U16_CHUNK_NUM)
  y_array = int_to_array(coordinate_y, BASE, BASE_TYPE, U16_CHUNK_NUM)
  if z:
    z_array = int_to_array(z, BASE, BASE_TYPE, U16_CHUNK_NUM)
    p = jnp.array([x_array, y_array, z_array])
  else:
    p = jnp.array([x_array, y_array])
  return p


def int_point_to_jax_point_pack(coordinates: List[int]):
  result = []
  for i in range(len(coordinates)):
    result.append(int_to_array(coordinates[i], BASE, BASE_TYPE, U16_CHUNK_NUM))
  return jnp.array(result)


def jax_point_pack_to_int_point(point: jax.Array):
  coordinate_num = point.shape[0]
  coordinates = []
  for i in range(coordinate_num):
    # print(point[i])
    c = array_to_int(point[i], BASE)
    coordinates.append(c)
  return coordinates


def jax_point_coordinates_pack_to_int_point(points_pack: jax.Array, base=BASE):
  """Converts a JAX array of point coordinates to a list of integer points.

  Args:
    points_pack: A JAX array of point coordinates. The array should have shape
      (num_points, coordinate_num, chunk_num).
    base: The base of the integer representation of the coordinates.

  Returns:
    A list of integer points. Each integer point is a list of coordinates.
  """
  coordinate_num = points_pack.shape[0]
  batch_size = points_pack.shape[1]
  corrdinates_ints = []
  for j in range(coordinate_num):
    corrdinates_ints.append(array_3d_to_int_list(points_pack[j], base))
  points: List[List[int]] = []
  for i in range(batch_size):
    point_coordinates = []
    for j in range(coordinate_num):
      point_coordinates.append(corrdinates_ints[j][i])
    points.append(point_coordinates)
  return points


def get_point_shape_dtype_structure(batch_size, coordinate_num):
  return jax.ShapeDtypeStruct(
      (batch_size, coordinate_num, U16_CHUNK_NUM), dtype=BASE_TYPE
  )


def array_to_int(jax_array: jax.Array, base) -> int:
  # Initialize the result as a Python integer
  result = 0

  # Iterate over the elements in the array
  for i, elem in enumerate(jax_array):
    # Convert each element to an int and shift it by the appropriate
    # number of bits
    result |= int(elem) << (i * base)

  return result


def int_to_array(python_int, base, dtype=None, array_size=None):
  """Converts a Python integer to a JAX array.

  Args:
    python_int: The Python integer to convert.
    base: The base of the integer representation of the coordinates.
    dtype: The data type of the JAX array. If None, the data type will be
      automatically determined based on the base.
    array_size: The size of the JAX array. If None, the array will have the
      minimum size necessary to store the integer.

  Returns:
    A JAX array representing the integer.
  """
  if dtype is None:
    dtype = bits_to_jnp_dtype(base)
  elements = []
  mask = (
      1 << base
  ) - 1  # Mask to extract the lower bits (e.g., 32 bits -> 0xFFFFFFFF)

  # Extract each element from the integer
  while python_int > 0:
    elements.append(python_int & mask)  # Extract the lower bits
    python_int >>= base  # Shift to remove the extracted bits

  # If array_size is provided, we pad or trim the result to match the
  # desired size
  if array_size is not None:
    assert array_size >= len(elements)
    elements = elements[:array_size] + [0] * (array_size - len(elements))

  # Convert the list to a JAX array
  return jnp.array(elements, dtype=dtype)


def int_list_to_jax_array(int_list):
  """Converts a list of integers to a JAX array."""
  result = []
  for int_value in int_list:
    result.append(int_to_array(int_value, BASE, array_size=24))
  return jnp.array(result, dtype=jnp.uint16)


def jax_array_to_int_list(jax_array, base):
  """Converts JAX array to single integer."""
  result_list = []
  for i in range(jax_array.shape[0]):
    value_vector = jax_array[i]
    value_int = array_to_int(value_vector, base)
    result_list.append(value_int)
  return result_list


def int_list_to_3d_array(int_list, base, array_size=None) -> jnp.ndarray:
  int_num = len(int_list)
  result_list = []
  i = 0
  for value in int_list:
    value_array = int_to_array(value, base, array_size=array_size).reshape(
        1, -1
    )
    value_array = jnp.pad(value_array, pad_width=((i, int_num - i - 1), (0, 0)))
    result_list.append(value_array)
    i += 1
  return jnp.array(result_list)


def array_3d_to_int_list(array: jnp.ndarray, base) -> List[int]:
  result_list = []
  for i in range(array.shape[0]):
    value_vector = array[i][i]
    value_int = array_to_int(value_vector, base)
    result_list.append(value_int)
  return result_list
