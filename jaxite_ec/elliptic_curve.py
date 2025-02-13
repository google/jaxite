"""The jaxite_ec implementation of the Elliptic curve operations on TPU.

Detailed algorithms come from the following papers:
xyzz: https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html
affine: https://www.hyperelliptic.org/EFD/g1p/auto-shortw.html
projective:
https://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#addition-madd-1998-cmo

A non-TPU version of the same functions can be found in
jaxite_ec/algorithm/elliptic_curve.py

To test the functionalities of this library, please refer to
jaxite_ec/elliptic_curve_test.py
"""

import functools

import jax
import jax.numpy as jnp
from jaxite.jaxite_ec import finite_field
import jaxite.jaxite_ec.util as utils


add_3u16 = finite_field.add_3u16
add_2u16 = finite_field.add_2u16
cond_sub_2u16 = finite_field.cond_sub_2u16
cond_sub_mod_u16 = finite_field.cond_sub_mod_u16
mod_mul_barrett_2u16 = finite_field.mod_mul_barrett_2u16
mod_mul_lazy_2u16 = finite_field.mod_mul_lazy_2u16
mod_mul_rns_2u16 = finite_field.mod_mul_rns_2u16
add_rns_2u16 = finite_field.add_rns_2u16
add_rns_3u16 = finite_field.add_rns_3u16
add_sub_rns_var = finite_field.add_sub_rns_var
negate_rns_for_var_add = finite_field.negate_rns_for_var_add


# Barrett Reduction Based Functions
def padd_barret_xyzz(
    x1: jax.Array,
    y1: jax.Array,
    zz1: jax.Array,
    zzz1: jax.Array,
    x2: jax.Array,
    y2: jax.Array,
    zz2: jax.Array,
    zzz2: jax.Array,
):
  """PADD-BARRETT elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassXYZZ::add_general

  This function implements the PADD-BARRETT elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1: The first generator element.
    y1: The second generator element.
    zz1: The third generator element.
    zzz1: The third generator element.
    x2: The first generator element.
    y2: The second generator element.
    zz2: The third generator element.
    zzz2: The third generator element.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  u1 = mod_mul_barrett_2u16(x1, zz2)
  u2 = mod_mul_barrett_2u16(x2, zz1)
  s1 = mod_mul_barrett_2u16(y1, zzz2)
  s2 = mod_mul_barrett_2u16(y2, zzz1)
  zz1_zz2 = mod_mul_barrett_2u16(zz1, zz2)
  zzz1_zzz2 = mod_mul_barrett_2u16(zzz1, zzz2)

  p = cond_sub_2u16(u2, u1)
  r = cond_sub_2u16(s2, s1)

  pp = mod_mul_barrett_2u16(p, p)
  rr = mod_mul_barrett_2u16(r, r)

  ppp = mod_mul_barrett_2u16(pp, p)
  q = mod_mul_barrett_2u16(u1, pp)
  zz3 = mod_mul_barrett_2u16(zz1_zz2, pp)

  ppp_q_2 = add_3u16(ppp, q, q)
  ppp_q_2 = cond_sub_mod_u16(ppp_q_2)
  ppp_q_2 = cond_sub_mod_u16(ppp_q_2)

  x3 = cond_sub_2u16(rr, ppp_q_2)

  q_x3 = cond_sub_2u16(q, x3)
  s1_ppp = mod_mul_barrett_2u16(s1, ppp)
  zzz3 = mod_mul_barrett_2u16(zzz1_zzz2, ppp)

  y3 = mod_mul_barrett_2u16(r, q_x3)
  y3 = cond_sub_2u16(y3, s1_ppp)

  return jnp.array([x3, y3, zz3, zzz3])


def pdul_barret_xyzz(
    x1: jax.Array, y1: jax.Array, zz1: jax.Array, zzz1: jax.Array
):
  """PDUL-BARRET elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassXYZZ::double_general

  This function implements the PDUL-BARRET elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1: The first generator element.
    y1: The second generator element.
    zz1: The third generator element.
    zzz1: The third generator element.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  u = add_2u16(y1, y1)
  u = cond_sub_mod_u16(u)

  x1x1 = mod_mul_barrett_2u16(x1, x1)
  v = mod_mul_barrett_2u16(u, u)

  w = mod_mul_barrett_2u16(u, v)
  s = mod_mul_barrett_2u16(x1, v)

  s_2 = add_2u16(s, s)
  s_2 = cond_sub_mod_u16(s_2)

  m = add_3u16(x1x1, x1x1, x1x1)
  m = cond_sub_mod_u16(m)
  m = cond_sub_mod_u16(m)

  mm = mod_mul_barrett_2u16(m, m)
  w_y1 = mod_mul_barrett_2u16(w, y1)
  zz3 = mod_mul_barrett_2u16(v, zz1)
  zzz3 = mod_mul_barrett_2u16(w, zzz1)

  x3 = cond_sub_2u16(mm, s_2)

  s_x3 = cond_sub_2u16(s, x3)

  y3 = mod_mul_barrett_2u16(m, s_x3)
  y3 = cond_sub_2u16(y3, w_y1)

  return jnp.array([x3, y3, zz3, zzz3])


def pdul_barrett_xyzz_pack(x1_y1_zz1_zzz1: jax.Array):
  return pdul_barret_xyzz(
      x1_y1_zz1_zzz1[0], x1_y1_zz1_zzz1[1], x1_y1_zz1_zzz1[2], x1_y1_zz1_zzz1[3]
  )


def padd_barrett_xyzz_pack(
    x1_y1_zz1_zzz1: jax.Array, x2_y2_zz2_zzz2: jax.Array
):
  return padd_barret_xyzz(
      x1_y1_zz1_zzz1[0],
      x1_y1_zz1_zzz1[1],
      x1_y1_zz1_zzz1[2],
      x1_y1_zz1_zzz1[3],
      x2_y2_zz2_zzz2[0],
      x2_y2_zz2_zzz2[1],
      x2_y2_zz2_zzz2[2],
      x2_y2_zz2_zzz2[3],
  )


def pdul_barrett_xyzz_pack_batch_first(
    x1_y1_zz1_zzz1: jax.Array, transpose=(0, 1, 2)
):
  return pdul_barret_xyzz(
      x1_y1_zz1_zzz1[:, 0],
      x1_y1_zz1_zzz1[:, 1],
      x1_y1_zz1_zzz1[:, 2],
      x1_y1_zz1_zzz1[:, 3],
  ).transpose(transpose[0], transpose[1], transpose[2])


def padd_barrett_xyzz_pack_batch_first(
    x1_y1_zz1_zzz1: jax.Array, x2_y2_zz2_zzz2: jax.Array, transpose=(0, 1, 2)
):
  return padd_barret_xyzz(
      x1_y1_zz1_zzz1[:, 0],
      x1_y1_zz1_zzz1[:, 1],
      x1_y1_zz1_zzz1[:, 2],
      x1_y1_zz1_zzz1[:, 3],
      x2_y2_zz2_zzz2[:, 0],
      x2_y2_zz2_zzz2[:, 1],
      x2_y2_zz2_zzz2[:, 2],
      x2_y2_zz2_zzz2[:, 3],
  ).transpose(transpose[0], transpose[1], transpose[2])


# Lazy Reduction Based Functions
def padd_lazy_xyzz(
    x1: jax.Array,
    y1: jax.Array,
    zz1: jax.Array,
    zzz1: jax.Array,
    x2: jax.Array,
    y2: jax.Array,
    zz2: jax.Array,
    zzz2: jax.Array,
    lazy_mat: jax.Array,
):
  """PADD-LAZY elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassXYZZ::add_general

  This function implements the PADD-LAZY elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1: The first generator element.
    y1: The second generator element.
    zz1: The third generator element.
    zzz1: The third generator element.
    x2: The first generator element.
    y2: The second generator element.
    zz2: The third generator element.
    zzz2: The third generator element.
    lazy_mat: The lazy matrix.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  cond_sub_2u16_ext = functools.partial(
      cond_sub_2u16,
      modulus_377_int=utils.MODULUS_377_S16_INT,
      chunk_num_u16=utils.U16_EXT_CHUNK_NUM,
  )
  cond_sub_mod_u16_ext = functools.partial(
      cond_sub_mod_u16,
      modulus_377_int=utils.MODULUS_377_S16_INT,
      chunk_num_u16=utils.U16_EXT_CHUNK_NUM,
  )

  u1 = mod_mul_lazy_2u16(x1, zz2, lazy_mat)
  u2 = mod_mul_lazy_2u16(x2, zz1, lazy_mat)
  s1 = mod_mul_lazy_2u16(y1, zzz2, lazy_mat)
  s2 = mod_mul_lazy_2u16(y2, zzz1, lazy_mat)
  zz1_zz2 = mod_mul_lazy_2u16(zz1, zz2, lazy_mat)
  zzz1_zzz2 = mod_mul_lazy_2u16(zzz1, zzz2, lazy_mat)

  p = cond_sub_2u16_ext(u2, u1)
  r = cond_sub_2u16_ext(s2, s1)

  pp = mod_mul_lazy_2u16(p, p, lazy_mat)
  rr = mod_mul_lazy_2u16(r, r, lazy_mat)

  ppp = mod_mul_lazy_2u16(pp, p, lazy_mat)
  q = mod_mul_lazy_2u16(u1, pp, lazy_mat)
  zz3 = mod_mul_lazy_2u16(zz1_zz2, pp, lazy_mat)

  # Can be replaced by mod_add_lazy.
  ppp_q_2 = add_3u16(ppp, q, q)
  ppp_q_2 = cond_sub_mod_u16_ext(ppp_q_2)
  ppp_q_2 = cond_sub_mod_u16_ext(ppp_q_2)

  x3 = cond_sub_2u16_ext(rr, ppp_q_2)

  q_x3 = cond_sub_2u16_ext(q, x3)
  s1_ppp = mod_mul_lazy_2u16(s1, ppp, lazy_mat)
  zzz3 = mod_mul_lazy_2u16(zzz1_zzz2, ppp, lazy_mat)

  y3 = mod_mul_lazy_2u16(r, q_x3, lazy_mat)
  y3 = cond_sub_2u16_ext(y3, s1_ppp)

  return jnp.array([x3, y3, zz3, zzz3])


def pdul_lazy_xyzz(
    x1: jax.Array,
    y1: jax.Array,
    zz1: jax.Array,
    zzz1: jax.Array,
    lazy_mat: jax.Array,
):
  """PDUL-BARRET elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassXYZZ::double_general

  This function implements the PDUL-BARRET elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1: The first generator element.
    y1: The second generator element.
    zz1: The third generator element.
    zzz1: The third generator element.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  cond_sub_2u16_ext = functools.partial(
      cond_sub_2u16,
      modulus_377_int=utils.MODULUS_377_S16_INT,
      chunk_num_u16=utils.U16_EXT_CHUNK_NUM,
  )
  cond_sub_mod_u16_ext = functools.partial(
      cond_sub_mod_u16,
      modulus_377_int=utils.MODULUS_377_S16_INT,
      chunk_num_u16=utils.U16_EXT_CHUNK_NUM,
  )
  u = add_2u16(y1, y1)
  u = cond_sub_mod_u16_ext(u)

  x1x1 = mod_mul_lazy_2u16(x1, x1, lazy_mat)
  v = mod_mul_lazy_2u16(u, u, lazy_mat)

  w = mod_mul_lazy_2u16(u, v, lazy_mat)
  s = mod_mul_lazy_2u16(x1, v, lazy_mat)

  s_2 = add_2u16(s, s)
  s_2 = cond_sub_mod_u16_ext(s_2)

  m = add_3u16(x1x1, x1x1, x1x1)
  m = cond_sub_mod_u16_ext(m)
  m = cond_sub_mod_u16_ext(m)

  mm = mod_mul_lazy_2u16(m, m, lazy_mat)
  w_y1 = mod_mul_lazy_2u16(w, y1, lazy_mat)
  zz3 = mod_mul_lazy_2u16(v, zz1, lazy_mat)
  zzz3 = mod_mul_lazy_2u16(w, zzz1, lazy_mat)

  x3 = cond_sub_2u16_ext(mm, s_2)

  s_x3 = cond_sub_2u16_ext(s, x3)

  y3 = mod_mul_lazy_2u16(m, s_x3, lazy_mat)
  y3 = cond_sub_2u16_ext(y3, w_y1)

  return jnp.array([x3, y3, zz3, zzz3])


def padd_lazy_xyzz_pack(
    x1_y1_zz1_zzz1: jax.Array, x2_y2_zz2_zzz2: jax.Array, lazy_mat: jax.Array
):
  return padd_lazy_xyzz(
      x1_y1_zz1_zzz1[0],
      x1_y1_zz1_zzz1[1],
      x1_y1_zz1_zzz1[2],
      x1_y1_zz1_zzz1[3],
      x2_y2_zz2_zzz2[0],
      x2_y2_zz2_zzz2[1],
      x2_y2_zz2_zzz2[2],
      x2_y2_zz2_zzz2[3],
      lazy_mat,
  )


def pdul_lazy_xyzz_pack(x1_y1_zz1_zzz1: jax.Array, lazy_mat: jax.Array):
  return pdul_lazy_xyzz(
      x1_y1_zz1_zzz1[0],
      x1_y1_zz1_zzz1[1],
      x1_y1_zz1_zzz1[2],
      x1_y1_zz1_zzz1[3],
      lazy_mat,
  )


# RNS Based Functions
def padd_rns_xyzz(
    x1: jax.Array,
    y1: jax.Array,
    zz1: jax.Array,
    zzz1: jax.Array,
    x2: jax.Array,
    y2: jax.Array,
    zz2: jax.Array,
    zzz2: jax.Array,
    rns_mat: jax.Array,
):
  """PADD-RNS elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassXYZZ::add_general

  This function implements the PADD-RNS elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1: The first generator element.
    y1: The second generator element.
    zz1: The third generator element.
    zzz1: The third generator element.
    x2: The first generator element.
    y2: The second generator element.
    zz2: The third generator element.
    zzz2: The third generator element.
    rns_mat: The RNS matrix.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  u1 = mod_mul_rns_2u16(x1, zz2, rns_mat)
  u2 = mod_mul_rns_2u16(x2, zz1, rns_mat)
  s1 = mod_mul_rns_2u16(y1, zzz2, rns_mat)
  s2 = mod_mul_rns_2u16(y2, zzz1, rns_mat)
  zz1_zz2 = mod_mul_rns_2u16(zz1, zz2, rns_mat)
  zzz1_zzz2 = mod_mul_rns_2u16(zzz1, zzz2, rns_mat)

  p = add_sub_rns_var(u2, negate_rns_for_var_add(u1))
  r = add_sub_rns_var(s2, negate_rns_for_var_add(s1))

  pp = mod_mul_rns_2u16(p, p, rns_mat)
  rr = mod_mul_rns_2u16(r, r, rns_mat)

  ppp = mod_mul_rns_2u16(pp, p, rns_mat)
  q = mod_mul_rns_2u16(u1, pp, rns_mat)
  zz3 = mod_mul_rns_2u16(zz1_zz2, pp, rns_mat)

  # This implementation derives separate negation constants for different
  # combinations of signs
  x3 = add_sub_rns_var(
      rr,
      negate_rns_for_var_add(ppp),
      negate_rns_for_var_add(q),
      negate_rns_for_var_add(q),
  )
  q_x3 = add_sub_rns_var(ppp, q, q, negate_rns_for_var_add(rr), q)

  s1_ppp = mod_mul_rns_2u16(s1, ppp, rns_mat)
  zzz3 = mod_mul_rns_2u16(zzz1_zzz2, ppp, rns_mat)

  y3 = mod_mul_rns_2u16(r, q_x3, rns_mat)
  y3 = add_sub_rns_var(y3, negate_rns_for_var_add(s1_ppp))

  return jnp.array([x3, y3, zz3, zzz3])


def pdul_rns_xyzz(
    x1: jax.Array,
    y1: jax.Array,
    zz1: jax.Array,
    zzz1: jax.Array,
    rns_mat: jax.Array,
):
  """PDUL-RNS elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassXYZZ::double_general

  This function implements the PDUL-RNS elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1: The first generator element.
    y1: The second generator element.
    zz1: The third generator element.
    zzz1: The third generator element.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  u = add_rns_2u16(y1, y1)

  x1x1 = mod_mul_rns_2u16(x1, x1, rns_mat)
  v = mod_mul_rns_2u16(u, u, rns_mat)

  w = mod_mul_rns_2u16(u, v, rns_mat)
  s = mod_mul_rns_2u16(x1, v, rns_mat)

  # s_2 = add_rns_2u16(s, s)

  m = add_rns_3u16(x1x1, x1x1, x1x1)

  mm = mod_mul_rns_2u16(m, m, rns_mat)
  w_y1 = mod_mul_rns_2u16(w, y1, rns_mat)
  zz3 = mod_mul_rns_2u16(v, zz1, rns_mat)
  zzz3 = mod_mul_rns_2u16(w, zzz1, rns_mat)

  x3 = add_sub_rns_var(mm, negate_rns_for_var_add(s), negate_rns_for_var_add(s))

  s_x3 = add_sub_rns_var(s, negate_rns_for_var_add(mm), s, s)

  y3 = mod_mul_rns_2u16(m, s_x3, rns_mat)
  y3 = add_sub_rns_var(y3, negate_rns_for_var_add(w_y1))

  return jnp.array([x3, y3, zz3, zzz3])


def padd_rns_xyzz_pack(
    x1_y1_zz1_zzz1: jax.Array, x2_y2_zz2_zzz2: jax.Array, rns_mat: jax.Array
):
  return padd_rns_xyzz(
      x1_y1_zz1_zzz1[0],
      x1_y1_zz1_zzz1[1],
      x1_y1_zz1_zzz1[2],
      x1_y1_zz1_zzz1[3],
      x2_y2_zz2_zzz2[0],
      x2_y2_zz2_zzz2[1],
      x2_y2_zz2_zzz2[2],
      x2_y2_zz2_zzz2[3],
      rns_mat,
  )


def pdul_rns_xyzz_pack(x1_y1_zz1_zzz1: jax.Array, rns_mat: jax.Array):
  return pdul_rns_xyzz(
      x1_y1_zz1_zzz1[0],
      x1_y1_zz1_zzz1[1],
      x1_y1_zz1_zzz1[2],
      x1_y1_zz1_zzz1[3],
      rns_mat,
  )
