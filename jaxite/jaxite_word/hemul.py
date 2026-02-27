from jaxite.jaxite_word.bconv import BConvBarrett
from jaxite.jaxite_word.ciphertext import Ciphertext
from jaxite.jaxite_word.finite_field import BarrettContext
import jax
import jax.numpy as jnp
import jaxite.jaxite_word.util as util

jax.config.update('jax_enable_x64', True)


class HEMul:

  def __init__(
      self, batch, r, c, dnum, num_eval_mult, original_moduli, extend_moduli
  ):
    self.batch = batch
    self.r = r
    self.c = c
    self.dnum = dnum
    self.num_eval_mult = num_eval_mult
    self.original_moduli = original_moduli
    self.drop_last_extend_moduli = original_moduli[:-1] + extend_moduli
    self.drop_last_moduli = original_moduli[:-1]
    self.extend_moduli = extend_moduli
    self.last_tower_moduli = original_moduli[-1]

  def control_gen(self, degree_layout=None, perf_test=False):
    if degree_layout is not None:
      self.degree_layout = degree_layout
    else:
      self.degree_layout = (self.r, self.c)

    self.perf_test = perf_test
    # ==========================================================================
    # 0. Configuration Derivation
    # ==========================================================================
    ring_dim = self.r * self.c
    original_moduli = self.original_moduli
    extend_moduli = self.extend_moduli

    sizeQ_in = len(original_moduli)
    overall_sizeQ_in = sizeQ_in
    overall_sizeP_in = len(extend_moduli)
    self.overall_sizeQ_in, self.overall_sizeP_in = (
        overall_sizeQ_in,
        overall_sizeP_in,
    )
    overall_sizeQ_in_no_last = sizeQ_in - 1

    # ==========================================================================
    # 1. Drop Last & Extend Moduli Arrays
    # ==========================================================================
    last_tower_psi = util.root_of_unity(2 * ring_dim, self.last_tower_moduli)
    power_of_last_tower_psi = jnp.array(
        [
            pow(last_tower_psi, i, self.last_tower_moduli)
            for i in range(ring_dim)
        ],
        jnp.uint64,
    )
    last_tower_inv_psi = pow(last_tower_psi, -1, self.last_tower_moduli)
    self.power_of_last_tower_inv_psi = jnp.array(
        [
            pow(last_tower_inv_psi, i, self.last_tower_moduli)
            for i in range(ring_dim)
        ],
        jnp.uint64,
    ).reshape(1, 1, -1)
    self.moduli_threshold = jnp.array(
        (self.last_tower_moduli + 1) // 2, dtype=jnp.uint32
    )

    # ==========================================================================
    # 2. Instantiate Ciphertext Object
    # Note: We must use precision=32 as per user request, matching the data type typically used here (uint32 for main storage)
    # ==========================================================================
    ct_refer_shapes = {
        'batch': self.batch,
        'num_elements': 4,
        'degree': ring_dim,
        'precision': 32,
        'num_moduli': overall_sizeQ_in,
        'degree_layout': self.degree_layout,
    }
    self.ct_refer = Ciphertext(
        ct_refer_shapes,
        parameters={
            'moduli': self.original_moduli,
            'finite_field_context': BarrettContext,
            'r': self.r,
            'c': self.c,
        },
    )
    self.ct_refer.modulus_switch_control_gen(degree_layout=self.degree_layout)

    ct_shapes = {
        'batch': self.batch,
        'num_elements': 4,
        'degree': self.r * self.c,
        'num_moduli': overall_sizeQ_in - 1,
        'precision': 32,
        'degree_layout': self.degree_layout,
    }
    self.ct_obj = Ciphertext(
        ct_shapes,
        parameters={
            'moduli': self.drop_last_moduli,
            'finite_field_context': BarrettContext,
            'r': self.r,
            'c': self.c,
        },
    )

    ct_last_limb_shapes = {
        'batch': self.batch,
        'num_elements': 4,
        'degree': ring_dim,
        'precision': 32,
        'num_moduli': 1,
        'degree_layout': self.degree_layout,
    }
    self.ct_last_limb = Ciphertext(
        ct_last_limb_shapes,
        parameters={
            'moduli': [self.last_tower_moduli],
            'finite_field_context': BarrettContext,
            'r': self.r,
            'c': self.c,
        },
    )

    idx_cur_last_tower = (
        overall_sizeQ_in - self.num_eval_mult
    )  # Calculate here for control_gen

    ct_extend_shapes = {
        'batch': self.batch,
        'num_elements': 2,
        'degree': ring_dim,
        'precision': 32,
        'num_moduli': overall_sizeP_in,
        'degree_layout': self.degree_layout,
    }
    self.ct_extend = Ciphertext(
        ct_extend_shapes, parameters={'moduli': self.extend_moduli}
    )

    # ==========================================================================
    # 3. Parameter Generation
    # ==========================================================================
    if perf_test:
      power_of_psi = util.random_parameters(
          (len(self.drop_last_extend_moduli), ring_dim),
          self.drop_last_extend_moduli,
          dtype=jnp.uint64,
      ).T
      power_of_inv_psi_approx_down = util.random_parameters(
          (len(self.drop_last_extend_moduli), ring_dim),
          self.drop_last_extend_moduli,
          dtype=jnp.uint64,
      ).T
    else:
      extend_psi = [
          util.root_of_unity(2 * ring_dim, q)
          for q in self.drop_last_extend_moduli
      ]
      power_of_psi = jnp.array(
          [
              [
                  pow(extend_psi[idx], i, self.drop_last_extend_moduli[idx])
                  for i in range(ring_dim)
              ]
              for idx in range(len(self.drop_last_extend_moduli))
          ],
          jnp.uint64,
      ).T
      extend_inv_psi = [
          pow(psi, -1, q)
          for (q, psi) in zip(self.drop_last_extend_moduli, extend_psi)
      ]
      power_of_inv_psi_approx_down = jnp.array(
          [
              [
                  pow(extend_inv_psi[idx], i, self.drop_last_extend_moduli[idx])
                  for i in range(ring_dim)
              ]
              for idx in range(len(self.drop_last_extend_moduli))
          ],
          jnp.uint64,
      ).T
    self.power_of_psi = power_of_psi[:, :idx_cur_last_tower]
    self.power_of_inv_psi_approx_down = power_of_inv_psi_approx_down[
        :, -overall_sizeP_in:
    ]
    self.bconv = BConvBarrett(self.drop_last_extend_moduli)
    control_indices_list = []
    rotate_indices = list(range(idx_cur_last_tower))
    extend_indices = list(
        range(idx_cur_last_tower, idx_cur_last_tower + overall_sizeP_in)
    )
    control_indices_list.append((extend_indices, rotate_indices))
    self.bconv.control_gen(control_indices_list, perf_test=perf_test)

    current_moduli = self.extend_moduli
    target_moduli = [
        item for item in self.drop_last_moduli if item not in current_moduli
    ]
    P = 1
    for moduli in current_moduli:
      P *= moduli
    PInvModq_approx_down = [util.modinv(P, q) for q in target_moduli]
    self.PInvModq = jnp.asarray(PInvModq_approx_down, dtype=jnp.uint32).reshape(
        idx_cur_last_tower
    )

    gammas, betas = util.gamma_beta_calculation(
        self.original_moduli, perf_test=perf_test
    )
    gammas_gen_power_of_psi = self.power_of_psi.T
    self.gammas_power_of_psi_no_last = (
        gammas[:, None].astype(jnp.uint64)
        * gammas_gen_power_of_psi.astype(jnp.uint64)
    ) % jnp.array(self.drop_last_moduli, jnp.uint64)[:, None]

    # ==========================================================================
    # 4. Parameter Reshape
    # ==========================================================================
    self.power_of_psi = self.power_of_psi.reshape(
        *self.degree_layout, idx_cur_last_tower
    )
    self.power_of_inv_psi_approx_down = (
        self.power_of_inv_psi_approx_down.reshape(
            *self.degree_layout, overall_sizeP_in
        )
    )
    self.gammas_power_of_psi_no_last = self.gammas_power_of_psi_no_last.T
    self.betas = jnp.array(betas, jnp.uint64).reshape(1, 1, 1, -1)
    self.drop_last_moduli_arr = jnp.array(
        self.drop_last_moduli, jnp.uint32
    ).reshape(1, 1, 1, -1)

    self.drop_last_extend_moduli_arr = jnp.array(
        self.drop_last_extend_moduli, jnp.uint32
    )
    self.q_correction = self.drop_last_extend_moduli_arr[
        :idx_cur_last_tower
    ].reshape(1, 1, 1, -1)
    self.post_rescale_shape = (
        self.batch,
        4,
        *self.degree_layout,
        len(self.drop_last_moduli),
    )
    # self.post_rescale_shape = (self.batch, 4, ring_dim, len(self.drop_last_moduli))

  def setup_relinearization(self, evalkey_a_vector, evalkey_b_vector):
    # Reshape keys to align with 4D structure for broadcasting
    # evalkey_a_vector: (K, D, M) -> (K, 1, D, M)
    self.evalkey_a_vector = evalkey_a_vector.astype(jnp.uint64)[
        :, None, *self.degree_layout
    ]
    self.evalkey_b_vector = evalkey_b_vector.astype(jnp.uint64)[
        :, None, *self.degree_layout
    ]

    # Delegate KS prep to Ciphertext
    # We use None for perf_test if not set in control_gen used self.perf_test (set in control_gen)
    # self.perf_test should be set in control_gen
    self.ct_obj.key_switch_control_gen(
        self.extend_moduli,
        self.dnum,
        evalkey_a_vector,
        evalkey_b_vector,
        perf_test=self.perf_test,
        selected_moduli=self.drop_last_moduli,
        degree_layout=self.degree_layout,
    )

  def mul(self, in_ciphertexts):
    # ---------- Step 1: Modulus Reduction (drop last tower) per-ciphertext ----------
    # Unpack static params (single tuple) in the same fixed order used in generation
    num_eval_mult = self.num_eval_mult
    overall_sizeQ_in, overall_sizeP_in = (
        self.overall_sizeQ_in,
        self.overall_sizeP_in,
    )
    idx_cur_last_tower = overall_sizeQ_in - num_eval_mult

    # ---------- Step 1: Rescale ----------
    self.ct_refer.set_batch_ciphertext(in_ciphertexts)
    temp_res = self.ct_refer.rescale()
    self.ct_obj.set_batch_ciphertext(temp_res.reshape(self.post_rescale_shape))

    # ---------- Step 2: Homomorphic multiplication core (inline) ----------
    ct_post_mult, last_ele_post_mult = self.ct_obj.ciphertext_mult()

    # ---------- Step 3 & 4: Key switch using Ciphertext methods ----------
    self.ct_obj.set_batch_ciphertext(last_ele_post_mult)
    self.ct_obj.key_switch()
    keyswitch_core_res = self.ct_obj.get_batch_ciphertext()

    # ---------- Step 5: Approximate modulus down (via Ciphertext) ----------
    result_ciphertext_list = []
    overall_moduli_jax = jnp.asarray(self.drop_last_moduli, dtype=jnp.uint32)
    approx_down_in_jax = jnp.asarray(keyswitch_core_res, dtype=jnp.uint32)

    self.ct_extend.set_batch_ciphertext(
        approx_down_in_jax[
            ..., idx_cur_last_tower : (idx_cur_last_tower + overall_sizeP_in)
        ]
    )
    self.ct_extend.to_coeffs_form()
    self.ct_extend.modmul(
        jnp.array(self.power_of_inv_psi_approx_down, jnp.uint64)
    )
    reduced_approx_down = self.ct_extend.get_batch_ciphertext()
    ct_new_basis_coef = self.bconv.basis_change_bat(
        reduced_approx_down, control_index=0
    ).astype(jnp.uint64)

    for element_index in range(ct_new_basis_coef.shape[1]):
      tower_new_basis_coef = ct_new_basis_coef[
          :, element_index : element_index + 1, ...
      ]
      self.ct_obj.set_batch_ciphertext(tower_new_basis_coef)
      self.ct_obj.modmul(self.power_of_psi)
      tower_new_basis_coef_scaled_muli_moduli_modq = (
          self.ct_obj.get_batch_ciphertext()
      )

      self.ct_obj.set_batch_ciphertext(
          tower_new_basis_coef_scaled_muli_moduli_modq
      )
      self.ct_obj.to_ntt_form()
      tower_new_basis_jax = self.ct_obj.get_batch_ciphertext()

      current_approx_down_in = approx_down_in_jax[
          :, element_index : element_index + 1, ..., :idx_cur_last_tower
      ]
      sub_result = jnp.where(
          current_approx_down_in < tower_new_basis_jax,
          current_approx_down_in + overall_moduli_jax - tower_new_basis_jax,
          current_approx_down_in - tower_new_basis_jax,
      )

      self.ct_obj.set_batch_ciphertext(sub_result)
      self.ct_obj.modmul(self.PInvModq)
      reduced_elem_modq = self.ct_obj.get_batch_ciphertext()

      result_ciphertext_list.append(reduced_elem_modq)

    approx_mod_down_custom = jnp.concatenate(result_ciphertext_list, axis=1)

    # ---------- Step 6: Add and return ----------
    result = ct_post_mult + approx_mod_down_custom
    val = jnp.where(
        result >= self.q_correction, result - self.q_correction, result
    )

    return val
