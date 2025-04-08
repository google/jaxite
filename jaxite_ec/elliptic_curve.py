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
rns_constant = finite_field.rns_constant


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
    lazy_mat: The lazy matrix.

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

  # u1 = mod_mul_rns_2u16(x1, zz2, rns_mat)
  # u2 = mod_mul_rns_2u16(x2, zz1, rns_mat)
  # s1 = mod_mul_rns_2u16(y1, zzz2, rns_mat)
  # s2 = mod_mul_rns_2u16(y2, zzz1, rns_mat)
  # zz1_zz2 = mod_mul_rns_2u16(zz1, zz2, rns_mat)
  # zzz1_zzz2 = mod_mul_rns_2u16(zzz1, zzz2, rns_mat)

  inputsl = jnp.vstack((x1, x2, y1, y2, zz1, zzz1))
  inputsr = jnp.vstack((zz2, zz1, zzz2, zzz1, zz2, zzz2))
  outputs = mod_mul_rns_2u16(inputsl, inputsr, rns_mat)
  u1, u2, s1, s2, zz1_zz2, zzz1_zzz2 = jnp.vsplit(outputs, 6)

  p = add_sub_rns_var(u2, negate_rns_for_var_add(u1))
  r = add_sub_rns_var(s2, negate_rns_for_var_add(s1))

  pp = mod_mul_rns_2u16(p, p, rns_mat)
  rr = mod_mul_rns_2u16(r, r, rns_mat)

  # ppp = mod_mul_rns_2u16(pp, p, rns_mat)
  # q = mod_mul_rns_2u16(u1, pp, rns_mat)
  # zz3 = mod_mul_rns_2u16(zz1_zz2, pp, rns_mat)

  inputsl = jnp.vstack((pp, u1, zz1_zz2))
  inputsr = jnp.vstack((p, pp, pp))
  outputs = mod_mul_rns_2u16(inputsl, inputsr, rns_mat)
  ppp, q, zz3 = jnp.vsplit(outputs, 3)

  # ppp_q_2 = add_rns_3u16(ppp, q, q)
  # x3 = sub_rns_2u16(rr, ppp_q_2)
  # q_x3 = sub_rns_2u16(q, x3)
  # x3 = rr - ppp - q - q
  # q - (rr - (ppp + q + q)) = ppp + q + q - rr + q
  # The alternative is to derive separate negation constants for different
  # combinations of signs
  x3 = add_sub_rns_var(
      rr,
      negate_rns_for_var_add(ppp),
      negate_rns_for_var_add(q),
      negate_rns_for_var_add(q),
  )
  q_x3 = add_sub_rns_var(ppp, q, q, negate_rns_for_var_add(rr), q)

  # s1_ppp = mod_mul_rns_2u16(s1, ppp, rns_mat)
  # zzz3 = mod_mul_rns_2u16(zzz1_zzz2, ppp, rns_mat)
  # y3 = mod_mul_rns_2u16(r, q_x3, rns_mat)

  inputsl = jnp.vstack((s1, zzz1_zzz2, r))
  inputsr = jnp.vstack((ppp, ppp, q_x3))
  outputs = mod_mul_rns_2u16(inputsl, inputsr, rns_mat)
  s1_ppp, zzz3, y3 = jnp.vsplit(outputs, 3)

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
    rns_mat: The RNS matrix.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  u = add_rns_2u16(y1, y1)

  # x1x1 = mod_mul_rns_2u16(x1, x1, rns_mat)
  v = mod_mul_rns_2u16(u, u, rns_mat)

  # w = mod_mul_rns_2u16(u, v, rns_mat)
  # s = mod_mul_rns_2u16(x1, v, rns_mat)

  inputsl = jnp.vstack((x1, u, x1))
  inputsr = jnp.vstack((x1, v, v))
  output = mod_mul_rns_2u16(inputsl, inputsr, rns_mat)
  x1x1, w, s = jnp.vsplit(output, 3)

  # s_2 = add_rns_2u16(s, s)

  m = add_rns_3u16(x1x1, x1x1, x1x1)

  # mm = mod_mul_rns_2u16(m, m, rns_mat)
  # w_y1 = mod_mul_rns_2u16(w, y1, rns_mat)
  # zz3 = mod_mul_rns_2u16(v, zz1, rns_mat)
  # zzz3 = mod_mul_rns_2u16(w, zzz1, rns_mat)

  inputsl = jnp.vstack((m, w, v, w))
  inputsr = jnp.vstack((m, y1, zz1, zzz1))
  output = mod_mul_rns_2u16(inputsl, inputsr, rns_mat)
  mm, w_y1, zz3, zzz3 = jnp.vsplit(output, 4)

  # x3 = sub_rns_2u16(mm, s_2)
  x3 = add_sub_rns_var(mm, negate_rns_for_var_add(s), negate_rns_for_var_add(s))

  # s_x3 = sub_rns_2u16(s, x3)
  s_x3 = add_sub_rns_var(s, negate_rns_for_var_add(mm), s, s)

  y3 = mod_mul_rns_2u16(m, s_x3, rns_mat)
  # y3 = sub_rns_2u16(y3, w_y1)
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


# RNS Based Functions
def padd_rns_t_xyzz(
    x1: jax.Array,
    y1: jax.Array,
    z1: jax.Array,
    t1: jax.Array,
    x2: jax.Array,
    y2: jax.Array,
    z2: jax.Array,
    t2: jax.Array,
    rns_mat: jax.Array,
    twist_d=utils.TWIST_D_RNS,
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
    z1: The third generator element.
    t1: The third generator element.
    x2: The first generator element.
    y2: The second generator element.
    z2: The third generator element.
    t2: The third generator element.
    rns_mat: The RNS matrix.
    twist_d: curve parameter.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  twist_d = jnp.array(twist_d, dtype=jnp.uint16)

  inputsl = jnp.vstack((x1, y1, z1, t1))
  inputsr = jnp.vstack((x2, y2, z2, t2))
  outputs = mod_mul_rns_2u16(inputsl, inputsr, rns_mat)
  a, b, d, c1 = jnp.vsplit(outputs, 4)

  e1 = add_rns_2u16(x1, y1)
  e2 = add_rns_2u16(x2, y2)

  inputsl = jnp.vstack((e1, c1))
  inputsr = jnp.vstack((e2, twist_d))
  outputs = mod_mul_rns_2u16(inputsl, inputsr, rns_mat)
  e3, c = jnp.vsplit(outputs, 2)

  e = add_sub_rns_var(e3, negate_rns_for_var_add(a), negate_rns_for_var_add(b))
  f = add_sub_rns_var(d, negate_rns_for_var_add(c))
  g = add_rns_2u16(d, c)
  h = add_rns_2u16(a, b)

  inputsl = jnp.vstack((e, g, f, e))
  inputsr = jnp.vstack((f, h, g, h))
  outputs = mod_mul_rns_2u16(inputsl, inputsr, rns_mat)
  x3, y3, z3, t3 = jnp.vsplit(outputs, 4)

  return jnp.array([x3, y3, z3, t3])


def pdul_rns_t_xyzz(
    x1: jax.Array,
    y1: jax.Array,
    z1: jax.Array,
    t1: jax.Array,
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
    z1: The third generator element.
    t1: The third generator element.
    rns_mat: The RNS matrix.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  et = add_rns_2u16(x1, y1)
  inputsl = jnp.vstack((x1, y1, z1, et))
  outputs = mod_mul_rns_2u16(inputsl, inputsl, rns_mat)
  a, b, ct, et2 = jnp.vsplit(outputs, 4)

  e = add_sub_rns_var(et2, negate_rns_for_var_add(a), negate_rns_for_var_add(b))
  g = add_sub_rns_var(b, negate_rns_for_var_add(a))
  f = add_sub_rns_var(g, negate_rns_for_var_add(ct), negate_rns_for_var_add(ct))
  h = add_sub_rns_var(negate_rns_for_var_add(a), negate_rns_for_var_add(b))

  inputsl = jnp.vstack((e, g, f, e))
  inputsr = jnp.vstack((f, h, g, h))
  outputs = mod_mul_rns_2u16(inputsl, inputsr, rns_mat)
  x3, y3, z3, t3 = jnp.vsplit(outputs, 4)

  return jnp.array([x3, y3, z3, t3])


def padd_rns_t_xyzz_pack(
    x1_y1_zz1_zzz1: jax.Array, x2_y2_zz2_zzz2: jax.Array, rns_mat: jax.Array
):
  return padd_rns_t_xyzz(
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


def pdul_rns_t_xyzz_pack(x1_y1_zz1_zzz1: jax.Array, rns_mat: jax.Array):
  return pdul_rns_t_xyzz(
      x1_y1_zz1_zzz1[0],
      x1_y1_zz1_zzz1[1],
      x1_y1_zz1_zzz1[2],
      x1_y1_zz1_zzz1[3],
      rns_mat,
  )


def rns_t_xyzz_zero():
  return jnp.array(
      [rns_constant(0), rns_constant(1), rns_constant(1), rns_constant(0)]
  )
