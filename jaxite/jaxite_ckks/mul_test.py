"""Tests for multiplication and relinearization in CKKS."""

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import basis_conversion
from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import encrypt
from jaxite.jaxite_ckks import key_gen
from jaxite.jaxite_ckks import mul
from jaxite.jaxite_ckks import ntt
from jaxite.jaxite_ckks import rescale
from jaxite.jaxite_ckks import types
import numpy as np
from absl.testing import absltest

jax.config.update("jax_enable_x64", True)


class MulTest(absltest.TestCase):

  def test_encrypt_multiply_decrypt(self):
    q_towers = [536872097, 536870657, 536872001, 536870849, 536871233, 524353]
    p_towers = [1073741441, 1073740609]
    r, c = 4, 4
    degree = r * c
    dnum = 3
    scale = 281510041637249
    num_slots = 8

    pk, sk = key_gen.keygen(degree, q_towers + p_towers)
    encoder = encode.Encode(degree, q_towers, scale)
    encryptor = encrypt.Encrypt(pk)

    output_scale = (scale / q_towers[-1]) ** 2
    decoder = encode.Decode(scale=output_scale, num_slots=num_slots)

    sk_q = types.SecretKey(
        data=sk.data[:, : len(q_towers) - 1],
        moduli=sk.moduli[: len(q_towers) - 1],
    )
    evk = key_gen.gen_evaluation_key(sk_q, q_towers[:-1], p_towers, dnum)

    decryptor = encrypt.Decrypt(sk_q)

    slots1 = [0.25, 0.5, 0.75, 1, 2, 3, 4, 5]
    slots2 = [5, 4, 3, 2, 1, 0.75, 0.5, 0.25]
    expected_slots = [c1 * c2 for c1, c2 in zip(slots1, slots2)]

    pt1 = encoder.encode(slots1)
    pt2 = encoder.encode(slots2)

    ct1 = encryptor.encrypt(pt1)
    ct2 = encryptor.encrypt(pt2)

    rescaler = rescale.Rescale()
    rescaler.precompute_constants(q_towers, 1, r, c)

    drop_last_moduli = q_towers[:-1]
    extend_moduli = p_towers
    drop_last_extend_moduli = drop_last_moduli + extend_moduli

    control_indices = mul.Mul.compute_control_indices(
        drop_last_moduli, extend_moduli, dnum
    )

    bconv = basis_conversion.BasisConversionBarrett()
    bconv.precompute_constants(drop_last_extend_moduli, control_indices)

    full_ntt = ntt.NTTBarrett()
    full_ntt.precompute_constants(drop_last_extend_moduli, r, c)

    mul_kernel = mul.Mul(
        rescaler=rescaler,
        bconv=bconv,
        full_ntt=full_ntt,
    )
    mul_kernel.precompute_constants(
        q_towers, p_towers, dnum, r, c, composite_degree=1
    )

    ct_3elem = mul_kernel.tensor_multiply(ct1, ct2)
    res_ct = mul_kernel.relinearize(ct_3elem, evk)

    # We need to use the secret key on the rescaled ciphertext.
    # res_ct has moduli q_towers[:-1]
    pt_dec = decryptor.decrypt(res_ct)
    self.assertEqual(pt_dec.data.shape, (degree, len(q_towers) - 1))

    decoded_slots = decoder.decode(pt_dec, is_slot_form=False)

    np.testing.assert_allclose(
        np.array(decoded_slots),
        np.array(expected_slots),
        rtol=1e-5,
        atol=1e-5,
    )

  def test_relinearize_batched(self):
    q_towers = [536872097, 536870657, 536872001, 536870849, 536871233, 524353]
    p_towers = [1073741441, 1073740609]
    r, c = 4, 4
    degree = r * c
    dnum = 3
    scale = 281510041637249

    pk, sk = key_gen.keygen(degree, q_towers + p_towers)
    encoder = encode.Encode(degree, q_towers, scale)
    encryptor = encrypt.Encrypt(pk)

    sk_q = types.SecretKey(
        data=sk.data[:, : len(q_towers) - 1],
        moduli=sk.moduli[: len(q_towers) - 1],
    )
    evk = key_gen.gen_evaluation_key(sk_q, q_towers[:-1], p_towers, dnum)

    slots1 = [0.25, 0.5, 0.75, 1, 2, 3, 4, 5]
    slots2 = [5, 4, 3, 2, 1, 0.75, 0.5, 0.25]

    pt1 = encoder.encode(slots1)
    pt2 = encoder.encode(slots2)

    ct1 = encryptor.encrypt(pt1)
    ct2 = encryptor.encrypt(pt2)

    rescaler = rescale.Rescale()
    rescaler.precompute_constants(q_towers, 1, r, c)

    drop_last_moduli = q_towers[:-1]
    extend_moduli = p_towers
    drop_last_extend_moduli = drop_last_moduli + extend_moduli

    control_indices = mul.Mul.compute_control_indices(
        drop_last_moduli, extend_moduli, dnum
    )

    bconv = basis_conversion.BasisConversionBarrett()
    bconv.precompute_constants(drop_last_extend_moduli, control_indices)

    full_ntt = ntt.NTTBarrett()
    full_ntt.precompute_constants(drop_last_extend_moduli, r, c)

    mul_kernel = mul.Mul(
        rescaler=rescaler,
        bconv=bconv,
        full_ntt=full_ntt,
    )
    mul_kernel.precompute_constants(
        q_towers, p_towers, dnum, r, c, composite_degree=1
    )

    ct_3elem = mul_kernel.tensor_multiply(ct1, ct2)

    expected_res = mul_kernel.relinearize(ct_3elem, evk)

    # Create a batched ciphertext
    batched_data = jnp.stack([ct_3elem.data, ct_3elem.data])
    ct_3elem_batched = types.Ciphertext(
        data=batched_data, moduli=ct_3elem.moduli
    )

    res_ct = mul_kernel.relinearize(ct_3elem_batched, evk)

    self.assertEqual(res_ct.data.shape, (2, 2, degree, len(q_towers) - 1))

    np.testing.assert_allclose(res_ct.data[0], expected_res.data)
    np.testing.assert_allclose(res_ct.data[1], expected_res.data)

  def test_mul_pytree(self):
    q_towers = [536872097, 536870657, 536872001, 536870849, 536871233, 524353]
    p_towers = [1073741441, 1073740609]
    r, c = 4, 4
    dnum = 3

    mul_kernel = mul.Mul(
        rescaler=rescale.Rescale(),
        bconv=basis_conversion.BasisConversionBarrett(),
        ntt_factory=ntt.NTTBarrett,
    )
    mul_kernel.precompute_constants(
        q_towers, p_towers, dnum, r, c, composite_degree=1
    )

    children, aux_data = jax.tree_util.tree_flatten(mul_kernel)
    mul_unflattened = jax.tree_util.tree_unflatten(aux_data, children)

    self.assertTrue(mul_unflattened.is_initialized)
    self.assertEqual(
        mul_unflattened.original_moduli, mul_kernel.original_moduli
    )
    self.assertEqual(mul_unflattened.extend_moduli, mul_kernel.extend_moduli)
    self.assertIsNotNone(mul_unflattened.rescaler)
    self.assertIsNotNone(mul_unflattened.bconv)
    self.assertIsNotNone(mul_unflattened.ntt_current)
    self.assertIsNotNone(mul_unflattened.ntt_extend)
    self.assertIsNotNone(mul_unflattened.ks_ntt_kernels)


if __name__ == "__main__":
  absltest.main()
