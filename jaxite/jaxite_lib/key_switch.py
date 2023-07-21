"""LWE Key-switching utils."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jaxite.jaxite_lib import decomposition
from jaxite.jaxite_lib import lwe
from jaxite.jaxite_lib import random_source
from jaxite.jaxite_lib import types


@dataclasses.dataclass
class LweKeySwitchingKey:
  """A public key used to switch keys encrypted an LWE ciphertext."""

  # the q in Z/qZ, same as the LwePlaintext space
  modulus: jnp.uint32

  # the length of the sampled key_data vector s,
  # equal to len(LweCiphertext) - 1.
  # Encryption involves sampling a vector a = (a_1, ..., a_{lwe_dimension})
  # and computing a dot product <a, s> + error.
  lwe_dimension: int

  # the decomposition log base
  decomposition_log_base: int

  # the decomposition levels
  decomposition_level_count: int

  # size of switch_keys output plus one
  lwe_size: int

  # the key switching key data, with shape
  # (in_key.lwe_dimension, decomposition_level, out_key.lwe_dimension + 1)
  # note that for TFHE, the out_key.lwe_dimension = N * in_key.lwe_dimension,
  # where N is the polynomial modulus degree in the RLWE/RGSW parameters.
  key_data: jnp.ndarray


def gen_key(
    decomposition_params: decomposition.DecompositionParameters,
    prg: random_source.RandomSource,
    in_key: lwe.LweSecretKey,
    out_key: lwe.LweSecretKey,
) -> LweKeySwitchingKey:
  """Generate an LWE key switching key."""
  num_levels = decomposition_params.level_count
  n = in_key.lwe_dimension
  lwe_size = out_key.lwe_dimension + 1

  # For each bit of the input key (size n), decompose into num_levels pieces and
  # encrypt under the out_key (encryptions have size out_key.lwe_dimension + 1)
  #
  # in_key: length n
  # out_key: of length m
  #
  # "Powers of 2" each term of the in_key (n of these) into t levels, encrypt
  # under out_key to produce a m + 1 length vector
  #
  # Each index corresponds to the decomposition and encryption of one of the
  # bits of the input key.
  # The resulting shape of the key switching key is (n, num_levels, lwe_size)
  lwe_ai_samples = prg.uniform(
      shape=(n, num_levels, out_key.lwe_dimension),
      dtype=jnp.uint32,
  )
  lwe_error_samples = prg.rounded_normal(
      shape=(n, num_levels), dtype=jnp.uint32
  )
  key_data = jnp.zeros((n, num_levels, lwe_size), dtype=jnp.uint32)

  vmap_lwe_encrypt = jax.vmap(
      lwe.jit_encrypt, in_axes=(0, None, 0, 0, None), out_axes=0
  )

  def decompose_and_encrypt(in_key_bit, lwe_ai_samples, lwe_error_samples):
    """Decompose one bit of the input key, then encrypt each term."""
    powers_of_b_pts = decomposition.recomposition_summands(
        in_key_bit,
        base_log=decomposition_params.log_base,
        num_levels=num_levels,
    )
    return vmap_lwe_encrypt(
        powers_of_b_pts,
        out_key.key_data,
        lwe_ai_samples,
        lwe_error_samples,
        out_key.log_modulus,
    )

  key_data = jax.vmap(
      decompose_and_encrypt,
      in_axes=(0, 0, 0),
      out_axes=0,
  )(in_key.key_data, lwe_ai_samples, lwe_error_samples)

  return LweKeySwitchingKey(
      modulus=jnp.uint32(out_key.modulus),
      lwe_dimension=out_key.lwe_dimension,
      decomposition_log_base=decomposition_params.log_base,
      decomposition_level_count=num_levels,
      lwe_size=lwe_size,
      key_data=key_data,
  )


def switch_key(
    ksk: LweKeySwitchingKey, inp: types.LweCiphertext
) -> types.LweCiphertext:
  """Perform the key switch operation on an LWE ciphertext."""
  return jit_switch_key(
      ksk.key_data,
      inp,
      ksk.decomposition_level_count,
      ksk.decomposition_log_base,
  )


@jax.named_call
@functools.partial(jax.jit, static_argnames=("num_levels", "base_log"))
def jit_switch_key(
    ksk: jnp.ndarray,
    inp: types.LweCiphertext,
    num_levels: int,
    base_log: int,
) -> types.LweCiphertext:
  """Perform the key switch operation on an LWE ciphertext."""
  # Decompose the inp ciphertext into num_levels x lwe_size
  output_shape = jnp.shape(ksk[0][0])
  output_data = jnp.zeros(output_shape, dtype=jnp.uint32)
  # Set the b' value.
  output_data = output_data.at[-1].set(inp[-1])

  def decomp_dot(ksk_i, inp_i):
    decomp_i = decomposition.signed_decomposition(inp_i, base_log, num_levels)
    return jnp.dot(decomp_i, ksk_i)

  sum_data = jnp.sum(
      jax.vmap(decomp_dot, in_axes=(0, 0), out_axes=1)(ksk, inp[:-1]), axis=1
  )

  return output_data - sum_data
