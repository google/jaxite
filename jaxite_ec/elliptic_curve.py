"""Elliptic curve operations for JAXite."""

import jax
import jax.numpy as jnp
from jaxite.jaxite_ec import finite_field

add_3u16 = finite_field.add_3u16
add_2u16 = finite_field.add_2u16
cond_sub_2u16 = finite_field.cond_sub_2u16
cond_sub_mod_u16 = finite_field.cond_sub_mod_u16
mod_mul_barrett_2u16 = finite_field.mod_mul_barrett_2u16


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

  x1 = x1_y1_zz1_zzz1[0]
  y1 = x1_y1_zz1_zzz1[1]
  zz1 = x1_y1_zz1_zzz1[2]
  zzz1 = x1_y1_zz1_zzz1[3]

  return pdul_barret_xyzz(x1, y1, zz1, zzz1)


def padd_barrett_xyzz_pack(
    x1_y1_zz1_zzz1: jax.Array, x2_y2_zz2_zzz2: jax.Array
):
  """PADD-BARRETT elliptic curve operation with packed arguments.

  This function implements the PADD-BARRETT elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1_y1_zz1_zzz1: The first generator element.
    x2_y2_zz2_zzz2: The second generator element.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """

  x1 = x1_y1_zz1_zzz1[0]
  y1 = x1_y1_zz1_zzz1[1]
  zz1 = x1_y1_zz1_zzz1[2]
  zzz1 = x1_y1_zz1_zzz1[3]

  x2 = x2_y2_zz2_zzz2[0]
  y2 = x2_y2_zz2_zzz2[1]
  zz2 = x2_y2_zz2_zzz2[2]
  zzz2 = x2_y2_zz2_zzz2[3]

  return padd_barret_xyzz(x1, y1, zz1, zzz1, x2, y2, zz2, zzz2)
