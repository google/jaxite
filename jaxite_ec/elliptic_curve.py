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
from jaxite.jaxite_ec import util


add_3u16 = finite_field.add_3u16
add_2u16 = finite_field.add_2u16
sub_2u16 = finite_field.sub_2u16
cond_sub_2u16 = finite_field.cond_sub_2u16
cond_sub_mod_u16 = finite_field.cond_sub_mod_u16
mod_mul_barrett_2u16 = finite_field.mod_mul_barrett_2u16
mod_mul_lazy_2u16 = finite_field.mod_mul_lazy_2u16
mod_mul_rns_2u16 = finite_field.mod_mul_rns_2u16
add_rns_2u16 = finite_field.add_rns_2u16
add_rns_3u16 = finite_field.add_rns_3u16
add_sub_rns_var = finite_field.add_sub_rns_var
negate_rns_for_var_add = finite_field.negate_rns_for_var_add
negate_rns_for_var_add_zero_check = (
    finite_field.negate_rns_for_var_add_zero_check
)
rns_constant = finite_field.rns_constant


# Barrett Reduction Based Functions
@jax.named_call
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


@jax.named_call
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


@jax.named_call
def pdul_barrett_xyzz_pack(x1_y1_zz1_zzz1: jax.Array):
  return pdul_barret_xyzz(
      x1_y1_zz1_zzz1[0], x1_y1_zz1_zzz1[1], x1_y1_zz1_zzz1[2], x1_y1_zz1_zzz1[3]
  )


@jax.named_call
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


@jax.named_call
def pdul_barrett_xyzz_pack_batch_first(
    x1_y1_zz1_zzz1: jax.Array, transpose=(0, 1, 2)
):
  return pdul_barret_xyzz(
      x1_y1_zz1_zzz1[:, 0],
      x1_y1_zz1_zzz1[:, 1],
      x1_y1_zz1_zzz1[:, 2],
      x1_y1_zz1_zzz1[:, 3],
  ).transpose(transpose[0], transpose[1], transpose[2])


@jax.named_call
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
@jax.named_call
def padd_lazy_xyzz(
    x1: jax.Array,
    y1: jax.Array,
    zz1: jax.Array,
    zzz1: jax.Array,
    x2: jax.Array,
    y2: jax.Array,
    zz2: jax.Array,
    zzz2: jax.Array,
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

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  cond_sub_2u16_ext = functools.partial(
      cond_sub_2u16,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num_u16=util.U16_EXT_CHUNK_NUM,
  )
  cond_sub_mod_u16_ext = functools.partial(
      cond_sub_mod_u16,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num_u16=util.U16_EXT_CHUNK_NUM,
  )

  u1 = mod_mul_lazy_2u16(x1, zz2)
  u2 = mod_mul_lazy_2u16(x2, zz1)
  s1 = mod_mul_lazy_2u16(y1, zzz2)
  s2 = mod_mul_lazy_2u16(y2, zzz1)
  zz1_zz2 = mod_mul_lazy_2u16(zz1, zz2)
  zzz1_zzz2 = mod_mul_lazy_2u16(zzz1, zzz2)

  p = cond_sub_2u16_ext(u2, u1)
  r = cond_sub_2u16_ext(s2, s1)

  pp = mod_mul_lazy_2u16(p, p)
  rr = mod_mul_lazy_2u16(r, r)

  ppp = mod_mul_lazy_2u16(pp, p)
  q = mod_mul_lazy_2u16(u1, pp)
  zz3 = mod_mul_lazy_2u16(zz1_zz2, pp)

  # Can be replaced by mod_add_lazy.
  ppp_q_2 = add_3u16(ppp, q, q)
  ppp_q_2 = cond_sub_mod_u16_ext(ppp_q_2)
  ppp_q_2 = cond_sub_mod_u16_ext(ppp_q_2)

  x3 = cond_sub_2u16_ext(rr, ppp_q_2)

  q_x3 = cond_sub_2u16_ext(q, x3)
  s1_ppp = mod_mul_lazy_2u16(s1, ppp)
  zzz3 = mod_mul_lazy_2u16(zzz1_zzz2, ppp)

  y3 = mod_mul_lazy_2u16(r, q_x3)
  y3 = cond_sub_2u16_ext(y3, s1_ppp)

  return jnp.array([x3, y3, zz3, zzz3])


@jax.named_call
def pdul_lazy_xyzz(
    x1: jax.Array,
    y1: jax.Array,
    zz1: jax.Array,
    zzz1: jax.Array,
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
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num_u16=util.U16_EXT_CHUNK_NUM,
  )
  cond_sub_mod_u16_ext = functools.partial(
      cond_sub_mod_u16,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num_u16=util.U16_EXT_CHUNK_NUM,
  )
  u = add_2u16(y1, y1)
  u = cond_sub_mod_u16_ext(u)

  x1x1 = mod_mul_lazy_2u16(x1, x1)
  v = mod_mul_lazy_2u16(u, u)

  w = mod_mul_lazy_2u16(u, v)
  s = mod_mul_lazy_2u16(x1, v)

  s_2 = add_2u16(s, s)
  s_2 = cond_sub_mod_u16_ext(s_2)

  m = add_3u16(x1x1, x1x1, x1x1)
  m = cond_sub_mod_u16_ext(m)
  m = cond_sub_mod_u16_ext(m)

  mm = mod_mul_lazy_2u16(m, m)
  w_y1 = mod_mul_lazy_2u16(w, y1)
  zz3 = mod_mul_lazy_2u16(v, zz1)
  zzz3 = mod_mul_lazy_2u16(w, zzz1)

  x3 = cond_sub_2u16_ext(mm, s_2)

  s_x3 = cond_sub_2u16_ext(s, x3)

  y3 = mod_mul_lazy_2u16(m, s_x3)
  y3 = cond_sub_2u16_ext(y3, w_y1)

  return jnp.array([x3, y3, zz3, zzz3])


@jax.named_call
def padd_lazy_xyzz_pack(x1_y1_zz1_zzz1: jax.Array, x2_y2_zz2_zzz2: jax.Array):
  return padd_lazy_xyzz(
      x1_y1_zz1_zzz1[0],
      x1_y1_zz1_zzz1[1],
      x1_y1_zz1_zzz1[2],
      x1_y1_zz1_zzz1[3],
      x2_y2_zz2_zzz2[0],
      x2_y2_zz2_zzz2[1],
      x2_y2_zz2_zzz2[2],
      x2_y2_zz2_zzz2[3],
  )


@jax.named_call
def pdul_lazy_xyzz_pack(x1_y1_zz1_zzz1: jax.Array):
  return pdul_lazy_xyzz(
      x1_y1_zz1_zzz1[0],
      x1_y1_zz1_zzz1[1],
      x1_y1_zz1_zzz1[2],
      x1_y1_zz1_zzz1[3],
  )


# Lazy Reduction Based Function
@jax.named_call
@functools.partial(jax.jit, static_argnames="twisted_d_chunk")
def padd_lazy_twisted(
    x1: jax.Array,
    y1: jax.Array,
    z1: jax.Array,
    t1: jax.Array,
    x2: jax.Array,
    y2: jax.Array,
    z2: jax.Array,
    t2: jax.Array,
    twisted_d_chunk=util.TWIST_D_INT_CHUNK,
):
  """PADD-LAZY elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassTwisted::add_general

  This function implements the PADD-LAZY elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1: The first generator element.
    y1: The second generator element.
    z1: The third generator element.
    t1: The fourth generator element.
    x2: The first generator element.
    y2: The second generator element.
    z2: The third generator element.
    t2: The fourth generator element.
    twisted_d_chunk: The twisted d parameter.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  cond_sub_2u16_ext = functools.partial(
      cond_sub_2u16,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num_u16=util.U16_EXT_CHUNK_NUM,
  )
  cond_sub_mod_u16_ext = functools.partial(
      cond_sub_mod_u16,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num_u16=util.U16_EXT_CHUNK_NUM,
  )

  twisted_d = jnp.asarray(twisted_d_chunk, dtype=jnp.uint16)
  twisted_d = jax.lax.broadcast(twisted_d, [x1.shape[0]])

  a = mod_mul_lazy_2u16(x1, x2)
  b = mod_mul_lazy_2u16(y1, y2)
  d = mod_mul_lazy_2u16(z1, z2)
  c = mod_mul_lazy_2u16(t1, t2)
  c = mod_mul_lazy_2u16(c, twisted_d)

  h = add_2u16(a, b)
  h = cond_sub_mod_u16_ext(h)
  e1 = add_2u16(x1, y1)
  e1 = cond_sub_mod_u16_ext(e1)
  e2 = add_2u16(x2, y2)
  e2 = cond_sub_mod_u16_ext(e2)
  e = mod_mul_lazy_2u16(e1, e2)

  e = cond_sub_2u16_ext(e, h)

  f = cond_sub_2u16_ext(d, c)
  g = add_2u16(d, c)
  g = cond_sub_mod_u16_ext(g)

  x3 = mod_mul_lazy_2u16(e, f)
  y3 = mod_mul_lazy_2u16(g, h)
  z3 = mod_mul_lazy_2u16(f, g)
  t3 = mod_mul_lazy_2u16(e, h)

  return jnp.array([x3, y3, z3, t3])


def pdul_lazy_twisted(
    x1: jax.Array,
    y1: jax.Array,
    z1: jax.Array,
    t1: jax.Array,
):
  """PDUL-LAZY elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassTwisted::double_general

  This function implements the PDUL-LAZY elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1: The first generator element.
    y1: The second generator element.
    z1: The third generator element.
    t1: The fourth generator element.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  cond_sub_2u16_ext = functools.partial(
      cond_sub_2u16,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num_u16=util.U16_EXT_CHUNK_NUM,
  )
  cond_sub_mod_u16_ext = functools.partial(
      cond_sub_mod_u16,
      modulus_377_int_chunk=util.MODULUS_377_S16_INT_CHUNK,
      chunk_num_u16=util.U16_EXT_CHUNK_NUM,
  )
  modulus_377_int_array = jnp.asarray(
      util.MODULUS_377_S16_INT_CHUNK, jnp.uint16
  )

  a = mod_mul_lazy_2u16(x1, x1)
  b = mod_mul_lazy_2u16(y1, y1)

  ct = mod_mul_lazy_2u16(z1, z1)
  ct2 = add_2u16(ct, ct)  #
  ct2 = cond_sub_mod_u16_ext(ct2)  #

  h = add_2u16(a, b)
  h = cond_sub_2u16_ext(modulus_377_int_array, h)  #

  et = add_2u16(x1, y1)  #
  et = cond_sub_mod_u16_ext(et)  #
  e = mod_mul_lazy_2u16(et, et)  #
  e = add_2u16(e, h)  #
  e = cond_sub_mod_u16_ext(e)  #

  g = cond_sub_2u16_ext(b, a)  #
  f = cond_sub_2u16_ext(g, ct2)  #
  x3 = mod_mul_lazy_2u16(e, f)  #
  y3 = mod_mul_lazy_2u16(g, h)
  z3 = mod_mul_lazy_2u16(f, g)
  t3 = mod_mul_lazy_2u16(e, h)
  return jnp.array([x3, y3, z3, t3])


def pneg_lazy_twisted(
    x1: jax.Array,
    y1: jax.Array,
    z1: jax.Array,
    t1: jax.Array,
):
  """PDUL-LAZY elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassTwisted::double_general

  This function implements the PDUL-LAZY elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1: The first generator element.
    y1: The second generator element.
    z1: The third generator element.
    t1: The fourth generator element.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """

  modulus_377_int_array = jnp.asarray(
      util.MODULUS_377_S16_INT_CHUNK, jnp.uint16
  )
  sub_2u16_ext = functools.partial(
      sub_2u16,
      chunk_num_u16=util.U16_EXT_CHUNK_NUM,
  )

  x3 = sub_2u16_ext(modulus_377_int_array, x1)
  y3 = y1
  z3 = z1
  t3 = sub_2u16_ext(modulus_377_int_array, t1)

  return jnp.array([x3, y3, z3, t3])


def padd_lazy_twisted_pack(x1_y1_z1_t1: jax.Array, x2_y2_z2_t2: jax.Array):
  return padd_lazy_twisted(
      x1_y1_z1_t1[0],
      x1_y1_z1_t1[1],
      x1_y1_z1_t1[2],
      x1_y1_z1_t1[3],
      x2_y2_z2_t2[0],
      x2_y2_z2_t2[1],
      x2_y2_z2_t2[2],
      x2_y2_z2_t2[3],
  )


def pdul_lazy_twisted_pack(x1_y1_z1_t1: jax.Array):
  return pdul_lazy_twisted(
      x1_y1_z1_t1[0],
      x1_y1_z1_t1[1],
      x1_y1_z1_t1[2],
      x1_y1_z1_t1[3],
  )


def pneg_lazy_twisted_pack(x1_y1_z1_t1: jax.Array):
  return pneg_lazy_twisted(
      x1_y1_z1_t1[0],
      x1_y1_z1_t1[1],
      x1_y1_z1_t1[2],
      x1_y1_z1_t1[3],
  )


# RNS Based Functions
@jax.named_call
@functools.partial(jax.jit, static_argnames="rns_mat")
def padd_rns_xyzz_pack(
    x1_y1_zz1_zzz1: jax.Array,
    x2_y2_zz2_zzz2: jax.Array,
    rns_mat=util.RNS_MAT,
):
  """PADD-RNS elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassXYZZ::add_general

  This function implements the PADD-RNS elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1_y1_zz1_zzz1: The first point.
    x2_y2_zz2_zzz2: The second point.
    rns_mat: The RNS matrix.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """

  # u1 = x1 * zz2
  # u2 = x2 * zz1
  # s1 = y1 * zzz2
  # s2 = y2 * zzz1
  # zz1_zz2 = zz1 * zz2
  # zzz1_zzz2 = zzz1 * zzz2
  num_moduli = x1_y1_zz1_zzz1.shape[-1]
  inputsl = jnp.vstack((x1_y1_zz1_zzz1, x1_y1_zz1_zzz1[2:])).reshape(
      -1, num_moduli
  )
  inputsr = jnp.vstack(
      (x2_y2_zz2_zzz2[2:], x2_y2_zz2_zzz2[2:], x2_y2_zz2_zzz2[:2])
  ).reshape(-1, num_moduli)
  outputs = mod_mul_rns_2u16(inputsl, inputsr, rns_mat)
  u1, s1, zz1_zz2, zzz1_zzz2, u2, s2 = jnp.vsplit(outputs, 6)

  # p = u2 - u1
  # r = s2 - s1
  p = add_sub_rns_var(u2, negate_rns_for_var_add(u1))
  r = add_sub_rns_var(s2, negate_rns_for_var_add(s1))

  # pp = p * p
  # rr = r * r
  pp = mod_mul_rns_2u16(p, p, rns_mat)
  rr = mod_mul_rns_2u16(r, r, rns_mat)

  # ppp = p * pp
  # q = u1 * pp
  # zz3 = zz1_zz2 * pp
  inputsl = jnp.vstack((p, u1, zz1_zz2))
  inputsr = jnp.vstack((pp, pp, pp))
  outputs = mod_mul_rns_2u16(inputsl, inputsr, rns_mat)
  ppp, q, zz3 = jnp.vsplit(outputs, 3)

  # x3 = r * r - ppp - (q + q)
  # q_x3 = q - x3
  x3 = add_sub_rns_var(
      rr,
      negate_rns_for_var_add(ppp),
      negate_rns_for_var_add(q),
      negate_rns_for_var_add(q),
  )
  q_x3 = add_sub_rns_var(q, negate_rns_for_var_add(x3))

  # s1_ppp = s1 * ppp
  # r_q_x3 = r * q_x3
  # zzz3 = zzz1_zzz2 * ppp
  # y3 = r_q_x3 - s1_ppp
  inputsl = jnp.vstack((s1, zzz1_zzz2, r))
  inputsr = jnp.vstack((ppp, ppp, q_x3))
  outputs = mod_mul_rns_2u16(inputsl, inputsr, rns_mat)
  s1_ppp, zzz3, r_q_x3 = jnp.vsplit(outputs, 3)

  y3 = add_sub_rns_var(r_q_x3, negate_rns_for_var_add(s1_ppp))

  return jnp.array([x3, y3, zz3, zzz3])


@jax.named_call
@functools.partial(jax.jit, static_argnames="rns_mat")
def pdul_rns_xyzz(
    x1: jax.Array,
    y1: jax.Array,
    zz1: jax.Array,
    zzz1: jax.Array,
    rns_mat=util.RNS_MAT,
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
  # u = y1 + y1
  u = add_rns_2u16(y1, y1)

  # v = u * u
  v = mod_mul_rns_2u16(u, u, rns_mat)

  # x1x1 = x1 * x1
  # w = u * v
  # s = x1 * v
  inputsl = jnp.vstack((x1, u, x1))
  inputsr = jnp.vstack((x1, v, v))
  output = mod_mul_rns_2u16(inputsl, inputsr, rns_mat)
  x1x1, w, s = jnp.vsplit(output, 3)

  # m = (x1x1 + x1x1 + x1x1) + a * (zz1 * zz1), Note: a = 0
  m = add_rns_3u16(x1x1, x1x1, x1x1)

  # mm = m * m
  # w_y1 = w * y1
  # zz3 = v * zz1
  # zzz3 = w * zzz1
  inputsl = jnp.vstack((m, w, v, w))
  inputsr = jnp.vstack((m, y1, zz1, zzz1))
  output = mod_mul_rns_2u16(inputsl, inputsr, rns_mat)
  mm, w_y1, zz3, zzz3 = jnp.vsplit(output, 4)

  # x3 = mm - (s + s)
  x3 = add_sub_rns_var(mm, negate_rns_for_var_add(s), negate_rns_for_var_add(s))

  # s_x3 = s - x3
  s_x3 = add_sub_rns_var(s, negate_rns_for_var_add(mm), s, s)

  # m_s_x3 = m * s_x3 - w_y1
  m_s_x3 = mod_mul_rns_2u16(m, s_x3, rns_mat)

  # y3 = m_s_x3 - w_y1
  y3 = add_sub_rns_var(m_s_x3, negate_rns_for_var_add(w_y1))

  return jnp.array([x3, y3, zz3, zzz3])


@jax.named_call
@functools.partial(jax.jit, static_argnames="rns_mat")
def pdul_rns_xyzz_pack(x1_y1_zz1_zzz1: jax.Array, rns_mat=util.RNS_MAT):
  return pdul_rns_xyzz(
      x1_y1_zz1_zzz1[0],
      x1_y1_zz1_zzz1[1],
      x1_y1_zz1_zzz1[2],
      x1_y1_zz1_zzz1[3],
      rns_mat,
  )


# RNS Based Functions
@jax.named_call
@functools.partial(jax.jit, static_argnames=("rns_mat", "twist_d"))
def padd_rns_twisted_pack(
    x1_y1_zz1_zzz1: jax.Array,
    x2_y2_zz2_zzz2: jax.Array,
    rns_mat=util.RNS_MAT,
    twist_d=util.TWIST_D_RNS,
):
  """PADD-RNS elliptic curve operation with packed arguments.

  As for the algorithm, pls refer to
  jaxite_ec/algorithm/elliptic_curve.py::ECCSWeierstrassXYZZ::add_general

  This function implements the PADD-RNS elliptic curve operation with packed
  arguments, which is used to compute the elliptic curve points of a given
  group.

  Args:
    x1_y1_zz1_zzz1: The first point.
    x2_y2_zz2_zzz2: The second point.
    rns_mat: The RNS matrix.
    twist_d: curve parameter.

  Returns:
    A tuple containing the third generator element and the elliptic curve points
    of the group.
  """
  twist_d = jnp.array(twist_d, dtype=jnp.uint16)

  inputsl = jnp.vstack(x1_y1_zz1_zzz1)
  inputsr = jnp.vstack(x2_y2_zz2_zzz2)
  outputs = mod_mul_rns_2u16(inputsl, inputsr, rns_mat)
  a, b, d, c = jnp.vsplit(outputs, 4)

  e1 = add_rns_2u16(x1_y1_zz1_zzz1[0], x1_y1_zz1_zzz1[1])
  e2 = add_rns_2u16(x2_y2_zz2_zzz2[0], x2_y2_zz2_zzz2[1])
  twist_d_here = jnp.broadcast_to(
      twist_d.reshape(-1, twist_d.shape[0]), c.shape
  )
  inputsl = jnp.vstack((e1, c))
  inputsr = jnp.vstack((e2, twist_d_here))
  outputs = mod_mul_rns_2u16(inputsl, inputsr, rns_mat)
  e3, c = jnp.vsplit(outputs, 2)

  # Issue happens here
  e = add_sub_rns_var(
      e3,
      negate_rns_for_var_add_zero_check(a),
      negate_rns_for_var_add_zero_check(b),
  )
  f = add_sub_rns_var(d, negate_rns_for_var_add_zero_check(c))
  g = add_rns_2u16(d, c)
  h = add_rns_2u16(a, b)

  inputsl = jnp.vstack((e, g, f, e))
  inputsr = jnp.vstack((f, h, g, h))
  outputs = mod_mul_rns_2u16(inputsl, inputsr, rns_mat)
  x3, y3, z3, t3 = jnp.vsplit(outputs, 4)

  return jnp.array([x3, y3, z3, t3])


@jax.named_call
@functools.partial(jax.jit, static_argnames="rns_mat")
def pdul_rns_twisted(
    x1: jax.Array,
    y1: jax.Array,
    z1: jax.Array,
    t1: jax.Array,
    rns_mat=util.RNS_MAT,
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


@jax.named_call
@functools.partial(jax.jit, static_argnames="rns_mat")
def pdul_rns_twisted_pack(x1_y1_zz1_zzz1: jax.Array, rns_mat=util.RNS_MAT):
  return pdul_rns_twisted(
      x1_y1_zz1_zzz1[0],
      x1_y1_zz1_zzz1[1],
      x1_y1_zz1_zzz1[2],
      x1_y1_zz1_zzz1[3],
      rns_mat,
  )


@jax.named_call
def rns_twist_zero():
  return jnp.array(
      [rns_constant(0), rns_constant(1), rns_constant(1), rns_constant(0)]
  )
