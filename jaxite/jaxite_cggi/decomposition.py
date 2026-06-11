"""Bit-decomposition and recomposition functions."""

import dataclasses
import functools
from typing import NewType
import jax
import jax.numpy as jnp

DecomposedInt = NewType("DecomposedInt", jnp.ndarray)
GadgetMatrix = NewType("GadgetMatrix", jnp.ndarray)
GadgetDecomp = NewType("GadgetDecomp", jnp.ndarray)


@dataclasses.dataclass(frozen=True)
class DecompositionParameters:
  """The parameters to a bit decomposition subroutine.

  Let B = 2^log_base. A bit decomposition computes the first
  level_count digits of the base-B representation of a number,
  which also clears lower-order bits. In the context of TFHE, those lower-order
  bits contain noise and can be safely ignored.
  """

  log_base: int
  level_count: int
  total_bit_length: int = 32


# base_log, num_levels, and total_bit_length are static for the life of the
# program.
@functools.partial(
    jax.jit, static_argnames=("base_log", "num_levels", "total_bit_length")
)
def decompose(
    x: jnp.uint32,
    base_log: int = 4,
    num_levels: int = 3,
    total_bit_length: int = 32,
) -> DecomposedInt:
  """Decompose a number in a given power-of-2 base, up to a given level.

  Let B = 2^base_log. This function computes the first num_levels digits in the
  base-B representation of x, which corresponds to an approximate representation
  of x when the lower-order bits are ignored. In the context of TFHE, those
  lower-order bits contain noise and can be safely ignored.

  Because B is a power of 2, this method partitions a bit string into blocks of
  length base_log, and returns the first num_levels blocks. Here one block
  corresponds to one digit in the base-B representation of x.

  Args:
    x: The number to decompose.
    base_log: The base-2 logarithm of the radix B in which to represent the
      output.
    num_levels: The number of blocks (digits) to return.
    total_bit_length: The bit size of x.

  Returns:
    A DecomposedInt containing the first num_levels digits in the base-B
    representation.
  """
  # Set the most significant base_log many 1's in a bit-string of length
  # total_bit_length, with all other bits unset.  e.g., for base_log=4,
  # mask=0b11110...0
  mask = jnp.uint32((1 << base_log) - 1 << (total_bit_length - base_log))

  # The code that follows is a vectorized version of the following (more
  # legible) loop, in which we iteratively select the next block, shift the
  # resulting bits down to the lowest-order, and shift the mask.
  # Tests were first written with the loop implementation, then refactored
  # with the vectorized version.
  #
  # output = jnp.zeros((num_levels,), dtype=jnp.uint32)
  # for i in range(num_levels):
  #   # Shifts the top-most masked block to the lowest-order bits.
  #   # 1011_0000_0000  ->  0000_0000_1011
  #   digit = (mask & x) >> (total_bit_length - (i + 1) * base_log)
  #   output = jax.ops.index_update(output, i, digit)
  #   mask >>= base_log
  # return output
  output = jnp.repeat(x, num_levels).astype(jnp.uint32)
  masks = mask >> (jnp.arange(num_levels).astype(jnp.uint32) * base_log)
  return (output & masks) >> (  # pytype: disable=bad-return-type  # jnp-type
      total_bit_length
      - (1 + jnp.arange(num_levels).astype(jnp.uint32)) * base_log
  )


# base_log, num_levels, and total_bit_length are static for the life of the
# program.
@functools.partial(
    jax.jit, static_argnames=("base_log", "num_levels", "total_bit_length")
)
def recomposition_summands(
    x: jnp.uint32,
    base_log: int = 4,
    num_levels: int = 3,
    total_bit_length: int = 32,
) -> jnp.ndarray:
  """Outputs a list of summands of the digit decomposition.

  Let B = 2^base_log. This function computes the first num_levels digits in the
  base-B representation of x and then scales them with the appropriate power of
  B, so that when the array of recomposition summands are summed, we retrieve an
  approximation of x.

  Because B is a power of 2, this method partitions a bit string into blocks of
  length base_log, and returns the first num_levels blocks. Here one block
  corresponds to one summand in the base-B representation of x.

  Args:
    x: The number to decompose.
    base_log: The base-2 logarithm of the radix B in which to represent the
      output.
    num_levels: The number of blocks (digits) to return.
    total_bit_length: The bit size of x.

  Returns:
    A JAX array containing the first num_levels of summands in the base-B
    representation.
  """
  output = jnp.repeat(x, num_levels)
  return output << (
      total_bit_length
      - (1 + jnp.arange(num_levels, dtype=jnp.uint32)) * base_log
  )


# base_log and total_bit_length are static for the life of the program.
@functools.partial(jax.jit, static_argnames=("base_log", "total_bit_length"))
def recompose(
    digits: DecomposedInt, base_log: int = 4, total_bit_length: int = 32
) -> jnp.uint32:
  """The inverse of decompose.

  Note num_levels == len(digits).

  Args:
    digits: The digits of a bit-decomposed number.
    base_log: The base-2 logarithm of the radix B of the input representation.
    total_bit_length: The bit size of the output integer.

  Returns:
    An integer represented by the DecomposedInt.
  """
  shifted_digits = digits << (
      total_bit_length - (1 + jnp.arange(digits.shape[0])) * base_log
  )
  return jnp.sum(shifted_digits).astype(jnp.uint32)


@functools.partial(
    jax.jit, static_argnames=("base_log", "num_levels", "total_bit_length")
)
def signed_decomposition(
    x: jnp.uint32,
    base_log: jnp.uint32,
    num_levels: int,
    total_bit_length: int = 32,
) -> jnp.ndarray:
  """Compute the signed base-B digit decomposition of a number.

  This method is analogous to decomposition.decompose, however, it restricts
  the digits in the output to the range [-B/2, B/2). This requires extra care
  as follows.

  If one digit in the output is larger than B/2 - 1, then subtracting B from the
  result requires us to add 1 to the next most significant digit. This implies
  we must track a "carry" bit from one digit to the next. Cf.
  b/202561246#comment4 for more details.

  Args:
    x: The number to decompose.
    base_log: The base-2 logarithm of the radix B in which to represent the
      output.
    num_levels: The number of digits to return.
    total_bit_length: The bit size of x.

  Returns:
    A DecomposedInt containing the first num_levels digits in the base-B
    representation.
  """
  # Because a lower order bit might result in a cascading sequence of carry
  # bits, rather than just compute the top num_levels digits, we have to compute
  # all the digits and truncate the return value.
  result = jnp.zeros((total_bit_length // base_log,), dtype=jnp.uint32)
  base = jnp.uint32(1 << base_log)
  digit_mask = base - 1
  base_over_2_threshold = jnp.uint32(1 << (base_log - 1))
  carry = jnp.uint32(0)

  for i in range(len(result)):
    unsigned_digit = ((x >> (i * base_log)) & digit_mask) + carry

    # This mask tests for whether a digit is >= B/2, and hence requires a shift
    # and carry propagation
    carry_mask = unsigned_digit & base_over_2_threshold

    # a tricky way to optionally shift by B. The result is re-cast as a uint32,
    # but this is OK because future arithmetic operations as uint32 will wrap
    # around and treat 2**32-1 properly as -1.
    signed_digit = jnp.uint32(
        jnp.int32(unsigned_digit) - jnp.int32(carry_mask << 1)
    )

    # convert the carry mask to a 0/1 bit to carry to the next digit
    carry = carry_mask >> (base_log - 1)

    result = result.at[i].set(signed_digit)

  # reversed because the digits are computed from least-significant bit to
  # highest-significant bit..
  return jnp.flip(result, axis=-1)[:num_levels]


# Applies the signed decomposition to each coefficient of a polynomial
# independently, where the polynomial is represented as a jax array of uints.
# The second and later arguments are the same as `signed_decomposition`, and
# pass through unchanged to each invocation.
signed_decomposition_polynomial = jax.vmap(
    signed_decomposition, in_axes=(0, None, None, None)
)

# Applies the signed decomposition to each coefficient of a list of polynomials
# independently, where one polynomial is represented as a single row of a jax
# array of uints. The second and later arguments are the same as
# `signed_decomposition`, and pass through unchanged to each invocation.
signed_decomposition_polynomial_list = jax.vmap(
    signed_decomposition_polynomial, in_axes=(0, None, None, None)
)


@jax.named_call
@functools.partial(jax.jit, static_argnames="decomposition_params")
def decompose_rlwe_ciphertext(
    rlwe_ct: jnp.ndarray, decomposition_params: DecompositionParameters
) -> jnp.ndarray:
  """Bit-decompose and rearrange an RLWE ciphertext for external product.

  Args:
    rlwe_ct: The underlying message ((k+1, N)-shaped array) of an RLWE
      ciphertext.
    decomposition_params: The DecompositionParameters to use.

  Returns:
    A ((k+1)*L, N) shape ndarray A where A[i, j] is as follows: index i as
    (polynomial, level) from the natural enumeration of range(k+1) x range(L),
    and then A[(polynomial, level), j] is the level'th digit in the signed
    base-L decomposition of coefficient j of the polynomial.
  """
  log_base = decomposition_params.log_base
  level_count = decomposition_params.level_count

  decomposed_polys = signed_decomposition_polynomial_list(
      rlwe_ct, log_base, level_count, decomposition_params.total_bit_length
  )

  # At this point, decomposed_polys has indices in order of (polynomial,
  # coefficient, decomposition level), so we reorder them to (polynomial *
  # decomposition level, coefficient)) and then reshape it so that the rows
  # have dimension (k+1)*L, in major order of polynomial, then decomposition
  # level.
  new_shape = (
      decomposed_polys.shape[0] * decomposed_polys.shape[2],
      decomposed_polys.shape[1],
  )
  return jnp.transpose(decomposed_polys, (0, 2, 1)).reshape(new_shape)


def gadget_matrix(
    decomp_params: DecompositionParameters,
    vector_length: int,
    total_bit_length: int = 32,
) -> GadgetMatrix:
  """Construct a gadget matrix for the decomposition parameters.

  The output matrix is of the following form, where B is the base log of the
  decomposition and l is the number of levels in the decomposition.

  [
   [1/B, 1/B^2, ..., 1/B^l, 0  , 0    , ... , 0    ,        ...        , 0 ],
   [0  ,     ...   , 0    , 1/B, 1/B^2, ... , 1/B^l,        ...        , 0 ],
                                     .
                                        .
                                           .
   [0  ,                ...                 , 0    , 1/B, 1/B^2, ..., 1/B^l]
  ]

  It has the property that (for the same parameters implied),
  inverse_gadget(v).dot(gadget_matrix.T) ~ v, where inverse_gadget(v) is small.
  The approximation is due to the limited number of levels. If B^l ==
  2**total_bit_length, then it is an equality.

  Note, in the Joye paper the vector_length is defined as "n+1" because it
  corresponds to an (R)LWE encryption of size n, and the extra "1" corresponds
  to the last entry `b=sum(a_i s_i) + message + noise`.

  Args:
    decomp_params: The DecompositionParameters to use.
    vector_length: The length of the output vector when the gadget matrix is
      applied to it. If this is M, then the gadget matrix will have shape (M,
      (M*decomp_params.level_count)).
    total_bit_length: The total bit size of the input integers.

  Returns:
    The gadget matrix, in T_q^{(n+1) x (n+1)L}
  """
  base = 2**decomp_params.log_base
  levels = decomp_params.level_count

  if 2**total_bit_length % (base**levels) != 0:
    raise ValueError(
        "Bad params. base**levels must divide total_bit_length. "
        f"Instead found base={base}, levels={levels}, "
        f"total_bit_length={total_bit_length}"
    )

  powers = (jnp.float32(1 / base) ** jnp.arange(1, levels + 1)).reshape(
      (levels, 1)
  )
  return jnp.kron(jnp.identity((vector_length)), powers).T  # pytype: disable=bad-return-type  # jnp-type


def inverse_gadget(
    vector: jnp.ndarray,
    decomp_params: DecompositionParameters,
    total_bit_length: int = 32,
) -> GadgetDecomp:
  """Compute the inverse gadget decomposition for a vector.

  Args:
    vector: The vector to apply, with entries in [-1/2, 1/2)
    decomp_params: The DecompositionParameters to use
    total_bit_length: The total bit size of the input integers.

  Returns:
    A flattened vector, the output of the inverse gadget operation.
  """
  log_base = decomp_params.log_base
  base = 2**log_base
  levels = decomp_params.level_count

  scaled = vector * jnp.float32(base) ** levels
  scaled_rounded = jnp.rint(scaled).astype(jnp.uint32)
  return jnp.array(  # pytype: disable=bad-return-type  # jnp-type
      [decompose(x, log_base, levels, total_bit_length) for x in scaled_rounded]
  ).flatten()
