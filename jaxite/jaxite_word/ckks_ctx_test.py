from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jaxite.jaxite_word.key_gen as kg
import jaxite.jaxite_word.ckks_ctx as ckks_ctx
import numpy as np
import jaxite.jaxite_word.util as util
from jaxite.jaxite_word.hemul import HEMul
from jaxite.jaxite_word.herot import HERot
from jaxite.jaxite_word.ciphertext import Ciphertext

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_traceback_filtering', 'off')

testing_params = [
  {
    'testcase_name': '0',
  }
]

@parameterized.named_parameters(testing_params)
class CKKSContextTest(parameterized.TestCase):
  def setUp(self):
    self.degree = 16
    self.num_slots = 8
    self.dnum = 3
    self.r, self.c = 4, 4

    self.scalingFactor = 563019763943521
    self.q_towers = [1073742881, 1073742721, 1073741441, 1073741857, 524353]
    self.p_towers = [1073740609, 1073739937, 1073739649]
    self.q = 696985728458547852910430465530901300664961
    self.qt = 1329229981441028949792278227703286337
    self.p = 30
    self.CKKS_M_FACTOR = 1
    self.noise_scale_degree = 1
    self.max_bits_in_word = 61
    self.sigma = 3.190000057220458984375
    self.degree_layout = (self.r, self.c)

    key_pair = kg.gen_pke_pair(self.q_towers, self.p_towers, self.degree)
    self.params = {
        "degree": self.degree,
        "num_slots": self.num_slots,
        "scalingFactor": self.scalingFactor,
        "output_scale": self.scalingFactor,
        "q_towers": self.q_towers,
        "p_towers": self.p_towers,
        "p": self.p,
        "CKKS_M_FACTOR": self.CKKS_M_FACTOR,
        "max_bits_in_word": self.max_bits_in_word,
        "noise_scale_degree": self.noise_scale_degree,
        "public_key": key_pair["public_key"],
        "secret_key": key_pair["secret_key"]
    }
    self.real_values_input_in1 = [
        complex(0.25, 0), complex(0.5, 0), complex(0.75, 0), complex(1, 0),
        complex(2, 0), complex(3, 0), complex(4, 0), complex(5, 0),
    ]
    self.real_values_input_in2 = [
        complex(5, 0), complex(4, 0), complex(3, 0), complex(2, 0),
        complex(1, 0), complex(0.75, 0), complex(0.5, 0), complex(0.25, 0),
    ]
    self.real_values_multiply_result = [
        complex(1.25, 0), complex(2, 0), complex(2.25, 0), complex(2, 0),
        complex(2, 0), complex(2.25, 0), complex(2, 0), complex(1.25, 0),
    ]
    self.real_values_rotate_result = [
        complex(0.5, 0), complex(0.75, 0), complex(1, 0), complex(2, 0),
        complex(3, 0), complex(4, 0), complex(5, 0), complex(0.25, 0),
    ]

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encode_decode(self):
    # Paramters Setup
    ctx = ckks_ctx.CKKSContext(self.params)
    # Step 1: Encoding
    encoded_ct = ctx.encode(self.real_values_input_in1)
    # Step 2: Decoding
    decoded_values = ctx.decode(encoded_ct, is_ntt=True)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_input_in1, decimal=3)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encrypt_decrypt(self):
    # Paramters Setup
    ctx = ckks_ctx.CKKSContext(self.params)
    # Step 1: Encoding
    encoded_ct = ctx.encode(self.real_values_input_in1)
    # Step 2: Encryption
    encrypted_ct = ctx.encrypt(encoded_ct)
    # Step 3: Decryption
    decrypted_ct = ctx.decrypt(encrypted_ct)
    # Step 4: Decoding
    decoded_values = ctx.decode(decrypted_ct)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_input_in1, decimal=3)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encrypt_rotate_decrypt(self):
    # Paramters Setup
    rotate_idx = 1
    coef_map = util.precompute_auto_map(self.degree, kg.find_automorphism_index_2n_complex(rotate_idx, self.degree))
    # initialization
    herot_obj = HERot(self.r, self.c, self.dnum, self.q_towers, self.p_towers)
    ek = kg.gen_rotation_key(self.params["secret_key"], self.q_towers, self.p_towers, rot_index=rotate_idx, dnum=self.dnum, noise_std=self.sigma, noise_scale=self.noise_scale_degree)
    herot_obj.setup_rotate(jnp.array(ek["a"], jnp.uint64).transpose(0,2,1).reshape(self.dnum,*self.degree_layout,-1), jnp.array(ek["b"], jnp.uint64).transpose(0,2,1).reshape(self.dnum,*self.degree_layout,-1), coef_map)
    herot_obj.control_gen(batch=1, degree_layout=self.degree_layout)
    ctx = ckks_ctx.CKKSContext(self.params)
    # Step 1: Encoding
    encoded_ct = ctx.encode(self.real_values_input_in1)
    # Step 2: Encryption
    encrypted_ct = ctx.encrypt(encoded_ct)
    # Step 3: Rotate
    result = herot_obj.rotate(encrypted_ct.ciphertext.reshape(1, 2, *self.degree_layout, len(self.q_towers)))
    encrypted_ct.set_batch_ciphertext(result.reshape(1,2,self.degree,len(self.q_towers)))
    # Step 4: Decryption
    decrypted_ct = ctx.decrypt(encrypted_ct)
    # Step 5: Decoding
    decoded_values = ctx.decode(decrypted_ct)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_rotate_result, decimal=3)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encrypt_rescale_decrypt(self):
    # Paramters Setup
    batch, num_elements, degree, num_moduli = 1, 2, 16, 5
    ct_shapes = {'batch': 1, 'num_elements': 2, 'degree': 16, 'num_moduli': 5, 'precision': 32, 'degree_layout': self.degree_layout}
    ct_params = {'moduli': self.q_towers, 'r': self.r, 'c': self.c}
    params = self.params.copy()
    params.update({
        "output_scale": (self.scalingFactor/self.q_towers[-1]),
    })
    # Initialization
    ctx = ckks_ctx.CKKSContext(params)
    ct = Ciphertext(ct_shapes, ct_params)
    ct.modulus_switch_control_gen(degree_layout=self.degree_layout)

    # Step 1: Encoding
    encoded_ct = ctx.encode(self.real_values_input_in1)
    # Step 2: Encryption
    encrypted_ct = ctx.encrypt(encoded_ct)
    # Step 3: Rescale
    ct.set_batch_ciphertext(encrypted_ct.ciphertext.reshape(batch, num_elements, *self.degree_layout, num_moduli))
    ct.rescale()
    # Step 4: Decryption
    ct.ciphertext = ct.ciphertext.reshape(batch, num_elements, degree, num_moduli-1)
    decrypted_ct = ctx.decrypt(ct)
    # Step 5: Decoding
    decoded_values = ctx.decode(decrypted_ct)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_input_in1, decimal=3)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encrypt_multiply_decrypt(self):
    """
    Test the encryption, multiplication, and decryption of the CKKS context.
    See hemul_test.py for the debugging version
    """
    # Paramters Setup
    r, c = 4, 4
    assert (r*c==self.degree)
    batch, num_elements, dnum, num_eval_mult = 1, 2, self.dnum, 1
    self.ek = kg.gen_evaluation_key(self.params["secret_key"], q=self.q_towers, P=self.p_towers, noise_std=self.sigma, noise_scale=1, dnum=3)
    eval_key_a, eval_key_b = jnp.array(self.ek["a"], dtype=jnp.uint32).transpose(0,2,1), jnp.array(self.ek["b"], dtype=jnp.uint32).transpose(0,2,1)
    params = self.params.copy()
    params.update({
        "evaluation_key": [eval_key_a, eval_key_b],
        "output_scale": (self.scalingFactor/self.q_towers[-1])**2,
    })
    # Initialization
    ctx = ckks_ctx.CKKSContext(params)
    he_mul = HEMul(batch, r, c, dnum, num_eval_mult, self.q_towers, self.p_towers)
    he_mul.control_gen(degree_layout=self.degree_layout)
    he_mul.setup_relinearization(eval_key_a, eval_key_b)
    # Step 1: Encoding
    encoded_ct1 = ctx.encode(self.real_values_input_in1)
    encoded_ct2 = ctx.encode(self.real_values_input_in2)
    # Step 2: Encryption
    encrypted_ct1 = ctx.encrypt(encoded_ct1)
    encrypted_ct2 = ctx.encrypt(encoded_ct2)
    # Step 3: Homomorphic Multiplication
    in_cts = jnp.concatenate([encrypted_ct1.ciphertext, encrypted_ct2.ciphertext], axis=1).reshape(batch, 2*num_elements, r, c, len(self.q_towers)).astype(jnp.uint32)
    encrypted_result = he_mul.mul(in_cts)
    # Step 4: Decryption
    encrypted_ct1.drop_last_modulus()
    encrypted_ct1.set_batch_ciphertext(encrypted_result.reshape(batch, 2, self.degree, len(self.q_towers)-1))
    decrypted_result = ctx.decrypt(encrypted_ct1)
    # Step 5: Decoding
    decoded_values = ctx.decode(decrypted_result, is_ntt=False)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_multiply_result, decimal=3)


if __name__ == "__main__":
  absltest.main()