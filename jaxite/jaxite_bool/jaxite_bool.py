"""API for Boolean FHE operations over the discretized torus."""

import functools
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jaxite.jaxite_bool import bool_encoding
from jaxite.jaxite_bool import bool_params
from jaxite.jaxite_lib import bootstrap
from jaxite.jaxite_lib import encoding
from jaxite.jaxite_lib import key_switch
from jaxite.jaxite_lib import lwe
from jaxite.jaxite_lib import random_source
from jaxite.jaxite_lib import rgsw
from jaxite.jaxite_lib import rlwe
from jaxite.jaxite_lib import types

Parameters = bool_params.Parameters
ENCODING_PARAMS = bool_encoding.ENCODING_PARAMS
CLEARTEXT_TRUE = bool_encoding.CLEARTEXT_TRUE
CLEARTEXT_FALSE = bool_encoding.CLEARTEXT_FALSE
CLEARTEXT_UNUSED = bool_encoding.CLEARTEXT_UNUSED


class ClientKeySet:
  """A set of secret keys for client use."""

  @property
  def lwe_sk(self) -> lwe.LweSecretKey:
    return self._lwe_sk

  @property
  def rlwe_sk(self) -> rlwe.RlweSecretKey:
    return self._rlwe_sk

  def __init__(
      self,
      params: Parameters,
      lwe_rng: random_source.RandomSource,
      rlwe_rng: random_source.RandomSource,
  ) -> None:
    self._lwe_sk = lwe.gen_key(params.scheme_params, lwe_rng)
    self._rlwe_sk = rlwe.gen_key(params.scheme_params, rlwe_rng)


class ServerKeySet:
  """A set of public keys and relevant parameters for server/cloud use."""

  @property
  def bsk(self) -> bootstrap.BootstrappingKey:
    return self._bsk

  @property
  def ksk(self) -> key_switch.LweKeySwitchingKey:
    return self._ksk

  def __init__(
      self,
      client_key_set: ClientKeySet,
      params: Parameters,
      lwe_rng: random_source.RandomSource,
      rlwe_rng: random_source.RandomSource,
      bootstrap_callback: Optional[Callable[[str, Any], None]] = None,
  ) -> None:
    self.bootstrap_callback = bootstrap_callback
    rgsw_key = rgsw.key_from_rlwe(client_key_set.rlwe_sk)
    self._bsk = bootstrap.gen_bootstrapping_key(
        lwe_sk=client_key_set.lwe_sk,
        rgsw_sk=rgsw_key,
        decomposition_params=params.bs_decomp_params,
        prg=rlwe_rng,
    )

    in_key = rlwe.flatten_key(client_key_set.rlwe_sk)
    self._ksk = key_switch.gen_key(
        decomposition_params=params.ks_decomp_params,
        prg=lwe_rng,
        in_key=in_key,
        out_key=client_key_set.lwe_sk,
    )


def encrypt(
    value: bool, client_key_set: ClientKeySet, prg: random_source.RandomSource
) -> types.LweCiphertext:
  """Encrypts a Boolean value under a given LWE secret key."""
  cleartext = CLEARTEXT_TRUE if value else CLEARTEXT_FALSE
  plaintext = encoding.encode(cleartext, ENCODING_PARAMS)
  return lwe.encrypt(plaintext, client_key_set.lwe_sk, prg)


def decrypt(
    ciphertext: types.LweCiphertext, client_key_set: ClientKeySet
) -> bool:
  """Decrypts an `LweCiphertext` encryption of `True`/`False`."""
  plaintext = lwe.decrypt(ciphertext, client_key_set.lwe_sk, ENCODING_PARAMS)
  cleartext = encoding.decode(plaintext, ENCODING_PARAMS)
  assert cleartext != CLEARTEXT_UNUSED
  return cleartext != 0


def constant(value: bool, params: Parameters) -> types.LweCiphertext:
  return params.noiseless_true if value else params.noiseless_false


def not_(
    ciphertext: types.LweCiphertext, params: Parameters
) -> types.LweCiphertext:
  """Computes NOT of the input ciphertext."""
  return params.noiseless_true - ciphertext


def and_(
    lhs: types.LweCiphertext,
    rhs: types.LweCiphertext,
    server_key_set: ServerKeySet,
    params: Parameters,
) -> types.LweCiphertext:
  """Computes AND of lhs and rhs."""
  return bootstrap.bootstrap(
      2 * lhs + rhs,
      params.lut_poly_by_name('and'),
      server_key_set.bsk,
      server_key_set.ksk,
      params.ks_decomp_params,
      params.bs_decomp_params,
      params.scheme_params,
      callback=server_key_set.bootstrap_callback,
      callback_lut=params.lut_by_name('and').as_cleartext_list,
  )


def andny_(
    lhs: types.LweCiphertext,
    rhs: types.LweCiphertext,
    server_key_set: ServerKeySet,
    params: Parameters,
) -> types.LweCiphertext:
  """Computes ANDNY of lhs andny rhs."""
  return bootstrap.bootstrap(
      2 * lhs + rhs,
      params.lut_poly_by_name('andny'),
      server_key_set.bsk,
      server_key_set.ksk,
      params.ks_decomp_params,
      params.bs_decomp_params,
      params.scheme_params,
      callback=server_key_set.bootstrap_callback,
      callback_lut=params.lut_by_name('andny').as_cleartext_list,
  )


def andyn_(
    lhs: types.LweCiphertext,
    rhs: types.LweCiphertext,
    server_key_set: ServerKeySet,
    params: Parameters,
) -> types.LweCiphertext:
  """Computes ANDYN of lhs andyn rhs."""
  return bootstrap.bootstrap(
      2 * lhs + rhs,
      params.lut_poly_by_name('andyn'),
      server_key_set.bsk,
      server_key_set.ksk,
      params.ks_decomp_params,
      params.bs_decomp_params,
      params.scheme_params,
      callback=server_key_set.bootstrap_callback,
      callback_lut=params.lut_by_name('andyn').as_cleartext_list,
  )


def or_(
    lhs: types.LweCiphertext,
    rhs: types.LweCiphertext,
    server_key_set: ServerKeySet,
    params: Parameters,
) -> types.LweCiphertext:
  """Computes OR of lhs or rhs."""
  return bootstrap.bootstrap(
      2 * lhs + rhs,
      params.lut_poly_by_name('or'),
      server_key_set.bsk,
      server_key_set.ksk,
      params.ks_decomp_params,
      params.bs_decomp_params,
      params.scheme_params,
      callback=server_key_set.bootstrap_callback,
      callback_lut=params.lut_by_name('or').as_cleartext_list,
  )


def orny_(
    lhs: types.LweCiphertext,
    rhs: types.LweCiphertext,
    server_key_set: ServerKeySet,
    params: Parameters,
) -> types.LweCiphertext:
  """Computes ORNY of lhs orny rhs."""
  return bootstrap.bootstrap(
      2 * lhs + rhs,
      params.lut_poly_by_name('orny'),
      server_key_set.bsk,
      server_key_set.ksk,
      params.ks_decomp_params,
      params.bs_decomp_params,
      params.scheme_params,
      callback=server_key_set.bootstrap_callback,
      callback_lut=params.lut_by_name('orny').as_cleartext_list,
  )


def oryn_(
    lhs: types.LweCiphertext,
    rhs: types.LweCiphertext,
    server_key_set: ServerKeySet,
    params: Parameters,
) -> types.LweCiphertext:
  """Computes ORYN of lhs oryn rhs."""
  return bootstrap.bootstrap(
      2 * lhs + rhs,
      params.lut_poly_by_name('oryn'),
      server_key_set.bsk,
      server_key_set.ksk,
      params.ks_decomp_params,
      params.bs_decomp_params,
      params.scheme_params,
      callback=server_key_set.bootstrap_callback,
      callback_lut=params.lut_by_name('oryn').as_cleartext_list,
  )


def nand_(
    lhs: types.LweCiphertext,
    rhs: types.LweCiphertext,
    server_key_set: ServerKeySet,
    params: Parameters,
) -> types.LweCiphertext:
  """Computes NAND of lhs nand rhs."""
  return bootstrap.bootstrap(
      2 * lhs + rhs,
      params.lut_poly_by_name('nand'),
      server_key_set.bsk,
      server_key_set.ksk,
      params.ks_decomp_params,
      params.bs_decomp_params,
      params.scheme_params,
      callback=server_key_set.bootstrap_callback,
      callback_lut=params.lut_by_name('nand').as_cleartext_list,
  )


def nor_(
    lhs: types.LweCiphertext,
    rhs: types.LweCiphertext,
    server_key_set: ServerKeySet,
    params: Parameters,
) -> types.LweCiphertext:
  """Computes NOR of lhs nor rhs."""
  return bootstrap.bootstrap(
      2 * lhs + rhs,
      params.lut_poly_by_name('nor'),
      server_key_set.bsk,
      server_key_set.ksk,
      params.ks_decomp_params,
      params.bs_decomp_params,
      params.scheme_params,
      callback=server_key_set.bootstrap_callback,
      callback_lut=params.lut_by_name('nor').as_cleartext_list,
  )


def xor_(
    lhs: types.LweCiphertext,
    rhs: types.LweCiphertext,
    server_key_set: ServerKeySet,
    params: Parameters,
) -> types.LweCiphertext:
  """Computes XOR of lhs xor rhs."""
  return bootstrap.bootstrap(
      2 * lhs + rhs,
      params.lut_poly_by_name('xor'),
      server_key_set.bsk,
      server_key_set.ksk,
      params.ks_decomp_params,
      params.bs_decomp_params,
      params.scheme_params,
      callback=server_key_set.bootstrap_callback,
      callback_lut=params.lut_by_name('xor').as_cleartext_list,
  )


def xnor_(
    lhs: types.LweCiphertext,
    rhs: types.LweCiphertext,
    server_key_set: ServerKeySet,
    params: Parameters,
) -> types.LweCiphertext:
  """Computes XNOR of lhs xnor rhs."""
  return bootstrap.bootstrap(
      2 * lhs + rhs,
      params.lut_poly_by_name('xnor'),
      server_key_set.bsk,
      server_key_set.ksk,
      params.ks_decomp_params,
      params.bs_decomp_params,
      params.scheme_params,
      callback=server_key_set.bootstrap_callback,
      callback_lut=params.lut_by_name('xnor').as_cleartext_list,
  )


def lut1(
    a: types.LweCiphertext,
    truth_table: int,
    server_key_set: ServerKeySet,
    params: Parameters,
) -> types.LweCiphertext:
  """Computes a 1-LUT."""
  del server_key_set
  if truth_table == 0 or truth_table == 3:
    return constant(bool(truth_table), params)
  if truth_table == 1:
    return not_(a, params)
  if truth_table == 2:
    return a

  raise ValueError(f'Illegal truth table {truth_table} for 1-LUT')


def lut2(
    a: types.LweCiphertext,
    b: types.LweCiphertext,
    truth_table: int,
    server_key_set: ServerKeySet,
    params: Parameters,
) -> types.LweCiphertext:
  """Computes a 2-LUT.

  Over cleartext bits a, b, the operation computed by this function can be
  interpreted as

    truth_table >> {b, a}

  where {b, a} is the unsigned 2-bit integer with bits b, a from most
  significant bit to least significant bit.

  Args:
    a: the least significant bit of the value to lookup
    b: the most significant bit of the value to lookup
    truth_table: the truth table contents
    server_key_set: the FHE server keys
    params: the jaxite_bool parameters

  Returns:
    The LWE ciphertext encrypting the output of the 2-LUT.
  """
  return bootstrap.bootstrap(
      2 * b + a,
      params.lut_poly(num_inputs=2, truth_table=truth_table),
      server_key_set.bsk,
      server_key_set.ksk,
      params.ks_decomp_params,
      params.bs_decomp_params,
      params.scheme_params,
      callback=server_key_set.bootstrap_callback,
      callback_lut=params.lut(
          num_inputs=2, truth_table=truth_table
      ).as_cleartext_list,
  )


def lut3(
    a: types.LweCiphertext,
    b: types.LweCiphertext,
    c: types.LweCiphertext,
    truth_table: int,
    server_key_set: ServerKeySet,
    params: Parameters,
) -> types.LweCiphertext:
  """Computes a 3-LUT.

  Over cleartext bits a, b, c, the operation computed by this function can be
  interpreted as

    truth_table >> {c, b, a}

  where {c, b, a} is the unsigned 3-bit integer with bits c, b, a from most
  significant bit to least-significant bit.

  Args:
    a: the least significant bit of the value to lookup
    b: the middle bit of the value to lookup
    c: the most significant bit of the value to lookup
    truth_table: the truth table contents
    server_key_set: the FHE server keys
    params: the jaxite_bool parameters

  Returns:
    The LWE ciphertext encrypting the output of the 3-LUT.
  """
  return bootstrap.bootstrap(
      4 * c + 2 * b + a,
      params.lut_poly(num_inputs=3, truth_table=truth_table),
      server_key_set.bsk,
      server_key_set.ksk,
      params.ks_decomp_params,
      params.bs_decomp_params,
      params.scheme_params,
      callback=server_key_set.bootstrap_callback,
      callback_lut=params.lut(
          num_inputs=3, truth_table=truth_table
      ).as_cleartext_list,
  )


def cmux_(
    v1: types.LweCiphertext,
    v0: types.LweCiphertext,
    ctrl: types.LweCiphertext,
    server_key_set: ServerKeySet,
    params: Parameters,
) -> types.LweCiphertext:
  truth_table = params.lut_by_name('cmux').truth_table
  return lut3(ctrl, v0, v1, truth_table, server_key_set, params)


@functools.partial(
    jax.pmap,
    in_axes=(0, 0, 0, 0, None, None),
    out_axes=0,
    static_broadcasted_argnums=(4, 5),
)
def pmap_lut3_impl(
    a: types.LweCiphertext,
    b: types.LweCiphertext,
    c: types.LweCiphertext,
    truth_table: jnp.ndarray,
    server_key_set: ServerKeySet,
    params: Parameters,
) -> types.LweCiphertext:
  """A version of lut3 suitable for pmap."""
  return bootstrap.jit_bootstrap(
      4 * c + 2 * b + a,
      truth_table,
      server_key_set.bsk.encrypted_lwe_sk_bits,
      server_key_set.ksk.key_data,
      params.ks_decomp_params,
      params.bs_decomp_params,
      params.scheme_params,
  )


Lut3Args = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]


def pmap_lut3(
    inputs: list[Lut3Args], sks: ServerKeySet, params: Parameters
) -> jnp.ndarray:
  """Apply lut3 in parallel across all inputs.

  Args:
    inputs: A list of tuples (a, b, c, truth_table), the corresponding arguments
      to the `jaxite.lut3` function with the same semantics
    sks: the FHE server key set
    params: the jaxite_bool parameters

  Returns:
    The LWE ciphertexts encrypting the output of the 3-LUT,
    organized as rows of a jax array.
  """
  a_inputs = jnp.array([v[0] for v in inputs])
  b_inputs = jnp.array([v[1] for v in inputs])
  c_inputs = jnp.array([v[2] for v in inputs])

  truth_table_cts = jnp.array(
      [params.lut_poly(num_inputs=3, truth_table=v[3]).message for v in inputs]
  )

  return pmap_lut3_impl(
      a_inputs,
      b_inputs,
      c_inputs,
      truth_table_cts,
      sks,
      params,
  )


@functools.partial(
    jax.pmap,
    in_axes=(0, 0, 0, None, None),
    out_axes=0,
    static_broadcasted_argnums=(3, 4),
)
def pmap_lut2_impl(
    a: types.LweCiphertext,
    b: types.LweCiphertext,
    truth_table: jnp.ndarray,
    server_key_set: ServerKeySet,
    params: Parameters,
) -> types.LweCiphertext:
  """A version of lut3 suitable for pmap."""
  return bootstrap.jit_bootstrap(
      2 * b + a,
      truth_table,
      server_key_set.bsk.encrypted_lwe_sk_bits,
      server_key_set.ksk.key_data,
      params.ks_decomp_params,
      params.bs_decomp_params,
      params.scheme_params,
  )


Lut2Args = tuple[jnp.ndarray, jnp.ndarray, int]


def pmap_lut2(
    inputs: list[Lut2Args], sks: ServerKeySet, params: Parameters
) -> jnp.ndarray:
  """Apply lut2 in parallel across all inputs.

  Args:
    inputs: A list of tuples (a, b, truth_table), the corresponding arguments
      to the `jaxite.lut2` function with the same semantics
    sks: the FHE server key set
    params: the jaxite_bool parameters

  Returns:
    The LWE ciphertexts encrypting the output of the 2-LUT,
    organized as rows of a jax array.
  """
  a_inputs = jnp.array([v[0] for v in inputs])
  b_inputs = jnp.array([v[1] for v in inputs])

  truth_table_cts = jnp.array(
      [params.lut_poly(num_inputs=2, truth_table=v[2]).message for v in inputs]
  )

  return pmap_lut2_impl(
      a_inputs,
      b_inputs,
      truth_table_cts,
      sks,
      params,
  )
