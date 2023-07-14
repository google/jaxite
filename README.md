# Jaxite

Jaxite is a fully homomorphic encryption backend targeting TPUs and GPUs,
written in [JAX](https://github.com/google/jax).

It implements the [CGGI cryptosystem](https://eprint.iacr.org/2018/421) with
some optimizations, and is a supported backend for [Google's FHE
compiler](https://github.com/google/fully-homomorphic-encryption).

## Quick start

### `jaxite_bool`

A program that shows how to use the `jaxite_bool` boolean gate API.

```python
from jaxite import jaxite_bool

bool_params = jaxite_bool.bool_params

lwe_rng = bool_params.get_lwe_rng_for_128_bit_security(seed=1)
rlwe_rng = bool_params.get_rlwe_rng_for_128_bit_security(seed=1)
params = bool_params.get_params_for_128_bit_security()
cks = jaxite_bool.ClientKeySet(
    params,
    lwe_rng=lwe_rng,
    rlwe_rng=rlwe_rng,
)
sks = jaxite_bool.ServerKeySet(
    cks,
    params,
    lwe_rng=lwe_rng,
    rlwe_rng=rlwe_rng,
    bootstrap_callback=callback,
)

ct_true = jaxite_bool.encrypt(True, cks, lwe_rng)
ct_false = jaxite_bool.encrypt(False, cks, lwe_rng)

not_false = jaxite_bool.not_(ct_false, params)
or_false = jaxite_bool.or_(not_false, ct_false, sks, params)
and_true = jaxite_bool.and_(or_false, ct_true, sks, params)
xor_true = jaxite_bool.xor_(and_true, ct_true, sks, params)
actual = jaxite_bool.decrypt(xor_true, cks)

expected = (((not False) or False) and True) != True
assert actual == expected
```

Jaxite also supports higher-bit-width gates, which are called look-up tables
(LUTs).

For example, this function is an 8-bit adder consisting entirely of `lut3`
gates.

```python
def add_i8(
    x: List[types.LweCiphertext],
    y: List[types.LweCiphertext],
    sks: jaxite_bool.ServerKeySet,
    params: jaxite_bool.Parameters) -> List[types.LweCiphertext]:
  temp_nodes: Dict[int, types.LweCiphertext] = {}
  false = jaxite_bool.constant(False, params)
  out = [None] * 8
  temp_nodes[0] = jaxite_bool.lut3(x[0], y[0], false, 8, sks, params)
  temp_nodes[1] = jaxite_bool.lut3(temp_nodes[0], x[1], y[1], 23, sks, params)
  temp_nodes[2] = jaxite_bool.lut3(temp_nodes[1], x[2], y[2], 43, sks, params)
  temp_nodes[3] = jaxite_bool.lut3(temp_nodes[2], x[3], y[3], 43, sks, params)
  temp_nodes[4] = jaxite_bool.lut3(temp_nodes[3], x[4], y[4], 43, sks, params)
  temp_nodes[5] = jaxite_bool.lut3(temp_nodes[4], x[5], y[5], 43, sks, params)
  temp_nodes[6] = jaxite_bool.lut3(temp_nodes[5], x[6], y[6], 43, sks, params)
  out[0] = jaxite_bool.lut3(x[0], y[0], false, 6, sks, params)
  out[1] = jaxite_bool.lut3(temp_nodes[0], x[1], y[1], 150, sks, params)
  out[2] = jaxite_bool.lut3(temp_nodes[1], x[2], y[2], 105, sks, params)
  out[3] = jaxite_bool.lut3(temp_nodes[2], x[3], y[3], 105, sks, params)
  out[4] = jaxite_bool.lut3(temp_nodes[3], x[4], y[4], 105, sks, params)
  out[5] = jaxite_bool.lut3(temp_nodes[4], x[5], y[5], 105, sks, params)
  out[6] = jaxite_bool.lut3(temp_nodes[5], x[6], y[6], 105, sks, params)
  out[7] = jaxite_bool.lut3(temp_nodes[6], x[7], y[7], 105, sks, params)
  return out
```

On a platform with parallelism, jaxite uses JAX's [`pmap`
API](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html) to allow
parallel evaluation of gates that have no interdependencies. E.g., the last
eight gate operations of the i8 adder above could be rewritten as

```python
 inputs = [
    (x[0], y[0], false, 6),            # out[0]
    (temp_nodes[0], x[1], y[1], 150),  # out[1]
    (temp_nodes[1], x[2], y[2], 105),  # out[2]
    (temp_nodes[2], x[3], y[3], 105),  # out[3]
    (temp_nodes[3], x[4], y[4], 105),  # out[4]
    (temp_nodes[4], x[5], y[5], 105),  # out[5]
    (temp_nodes[5], x[6], y[6], 105),  # out[6]
    (temp_nodes[6], x[7], y[7], 105),  # out[7]
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  return outputs
```

These circuits were generated with the jaxite support in [Google's Fully
Homomorphic Encryption Transpiler
project](https://github.com/google/fully-homomorphic-encryption), see
[transpiler/jaxite](https://github.com/google/fully-homomorphic-encryption/tree/main/transpiler/jaxite)
in that project for instructions.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.

## Disclaimer

This project is not an official Google project. It is not supported by Google
and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.
