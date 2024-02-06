"""A module containing JAX helper code."""

import functools
from typing import Any, Callable, Sequence, TypeVar
import jax
import jax.numpy as jnp


tree_flatten = jax.tree_util.tree_flatten
tree_unflatten = jax.tree_util.tree_unflatten
tree_map = jax.tree_util.tree_map


def _tree_map_multi_output(f, *args):
  """Like tree_map, but for functions that return tuples."""
  leaves, treedefs = zip(*map(tree_flatten, args))
  if any(treedef != treedefs[0] for treedef in treedefs):
    raise ValueError(f'argument treedefs do not match {treedefs=}')
  outputs = zip(*map(f, *leaves))
  return tuple(tree_unflatten(treedefs[0], out) for out in outputs)


def _lax_map(f, *xs):
  """Like lax.map, but supports multiple arguments like the built-in map."""
  g = lambda _, x: ((), f(*x))
  _, ys = jax.lax.scan(g, (), xs)
  return ys


F = TypeVar('F', bound=Callable)


def batch_vmap(
    f: F,
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0,
    *,
    batch_size: int,
) -> F:
  """jax.vmap, but looping when the batch dimension exceeds batch_size."""

  def preprocess(x, in_axis):
    batch_count = x.shape[in_axis] // batch_size
    x = jnp.moveaxis(x, in_axis, 0)
    loop_elements = batch_count * batch_size
    x_loop = x[:loop_elements].reshape((batch_count, batch_size) + x.shape[1:])
    x_tail = x[loop_elements:]
    return x_loop, x_tail

  def postprocess(x_loop, x_tail, out_axis):
    shape = x_loop.shape
    x_loop = x_loop.reshape((shape[0] * shape[1],) + shape[2:])
    x = jnp.concatenate([x_loop, x_tail], axis=0)
    return jnp.moveaxis(x, 0, out_axis)

  def g(*args):
    if isinstance(in_axes, int) or in_axes is None:
      in_axes_tuple = (in_axes,) * len(args)
    else:
      in_axes_tuple = tuple(in_axes)

    unbatched = []
    loop_args = []
    tail_args = []
    for i, (arg, in_axis) in enumerate(zip(args, in_axes_tuple)):
      if in_axis is None:
        unbatched.append((i, arg))
      elif isinstance(in_axis, int):
        loop_arg, tail_arg = _tree_map_multi_output(
            functools.partial(preprocess, in_axis=in_axis), arg
        )
        loop_args.append(loop_arg)
        tail_args.append(tail_arg)
      else:
        loop_arg, tail_arg = _tree_map_multi_output(preprocess, arg, in_axis)
        loop_args.append(loop_arg)
        tail_args.append(tail_arg)

    def f2(*args):
      args2 = list(args)
      for i, arg in unbatched:
        args2.insert(i, arg)
      return f(*args2)

    loop_out = _lax_map(jax.vmap(f2), *loop_args)
    tail_out = jax.vmap(f2)(*tail_args)
    if isinstance(out_axes, int):
      out = tree_map(
          functools.partial(postprocess, out_axis=out_axes), loop_out, tail_out
      )
    else:
      out = tree_map(postprocess, loop_out, tail_out, out_axes)
    return out

  return g


def get_tpu_version() -> int:
  """Returns the numeric version of the TPU, or -1 if not on TPU."""
  kind = jax.devices()[0].device_kind
  if 'TPU' not in kind:
    return -1
  if kind.endswith(' lite'):
    kind = kind[: -len(' lite')]
  assert kind[:-1] == 'TPU v', kind
  return int(kind[-1])
