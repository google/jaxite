"""Utility functions for jaxite_ec.

Note that: All functions that directly take Python int as input cannot be jitted.
"""

from typing import List

import jax
import jax.numpy as jnp


BASE = 16
BASE_TYPE = jnp.uint16  # this type must match the BASE, i.e. jnp.uint<BASE>
U16_MASK = 0xFFFF
U32_MASK = 0xFFFFFFFF

U64_CHUNK_NUM = 6
U32_CHUNK_NUM = 12
U16_CHUNK_NUM = 24
U8_CHUNK_NUM = 48
U16_CHUNK_SHIFT_BITS = 16
U32_CHUNK_SHIFT_BITS = 32

MODULUS_377_INT = 0x01AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001
MU_377_INT = 0x98542343310183A5DB0F28160BBD3DCEEEB43799DDAC681ABCB52236169B40B43B5A1DE2710A9647E7F56317936BFF32
BARRETT_SHIFT_U8 = 95  # BARRETT Params for k = 380
CHUNK_MAX = 0xFF
CHUNK_PRECISION = 8


def print_hex_values(int_list):
  hex_values = " ".join((hex(value)) for value in int_list)
  print(hex_values)


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
  """Converts a JAX array to a single Python integer.
  """
  result = 0

  for i, elem in enumerate(jax_array):
    result |= int(elem) << (i * base)

  return result


def int_to_array(
    python_int, base=BASE, dtype=jnp.uint16, array_size=24
):
  """Chunk decompose a Python integer into a JAX array of fixed dtype and fixed size.

  Args:
    python_int: The Python integer to convert.
    base: The base of the integer representation of the coordinates.
    dtype: The data type of the JAX array. If None, the data type will be
      automatically determined based on the base.
    array_size: The size of the JAX array. If None, the array will have the
      minimum size necessary to store the integer.

  Note that: the default parameter is only for 384-bit data.

  Returns:
    A JAX array representing the integer.
  """
  mask = (1 << base) - 1

  # Chunk Decomposition
  elements = []
  while python_int > 0:
    elements.append(python_int & mask)  # Extract the lower bits
    python_int >>= base  # Shift to remove the extracted bits

  # we pad or trim the result to match the desired size
  if array_size is not None:
    assert array_size >= len(elements)
    elements = elements[:array_size] + [0] * (array_size - len(elements))

  return jnp.array(elements, dtype=dtype)


def int_list_to_jax_array(int_list, base=BASE, array_size=24):
  """Converts a list of integers to a JAX array."""
  result = []
  for int_value in int_list:
    result.append(int_to_array(int_value, base, array_size=array_size))
  return jnp.array(result, dtype=jnp.uint16)


def jax_array_to_int_list(jax_array, base):
  """Converts JAX array to single integer."""
  result_list = []
  for i in range(jax_array.shape[0]):
    value_vector = jax_array[i]
    value_int = array_to_int(value_vector, base)
    result_list.append(value_int)
  return result_list


def int_list_to_2d_array(int_list, base, array_size=None) -> jnp.ndarray:
  """Converts a list of integers to a 2D JAX array."""
  chunked_arrays = []
  for int_value in int_list:
    chunked_arrays.append(int_to_array(int_value, base, array_size=array_size))
  return jnp.array(chunked_arrays)


def int_list_to_3d_array(int_list, base, array_size=None) -> jnp.ndarray:
  int_num = len(int_list)
  result_list = []
  i = 0
  for value in int_list:
    value_array = int_to_array(
        value, base, array_size=array_size
    ).reshape(1, -1)
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
