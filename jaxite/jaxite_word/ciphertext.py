import math
from typing import List, Optional, Tuple, Union
from jaxite.jaxite_word.bconv import BConvBarrett
import jaxite.jaxite_word.finite_field as ff_context
import jax.numpy as jnp
import jaxite.jaxite_word.ntt_mm as ntt
import jaxite.jaxite_word.util as util


########################
# Helper Functions
########################
def gen_power_of_inv_psi_arr(moduli, ring_dim):
  q_list = [moduli] if not isinstance(moduli, list) else moduli
  psi_list = [util.root_of_unity(2 * ring_dim, q) for q in q_list]
  inv_psi = [pow(psi, -1, q) for (q, psi) in zip(q_list, psi_list)]
  power_of_inv_psi_arr = [
      [pow(inv_psi[idx], i, q_list[idx]) for i in range(ring_dim)]
      for idx in range(len(psi_list))
  ]
  return jnp.array(power_of_inv_psi_arr, jnp.uint64).T.reshape(
      1, 1, ring_dim, -1
  )


class Ciphertext:

  def __init__(self, shapes: dict, parameters: Optional[dict] = None):
    """Initialize the Ciphertext object.

    Each ciphertext is a 4D tensor with shape (batch, num_elements, num_moduli,
    degree).

    Args:
        shapes (dict): A dictionary containing the shapes of the ciphertext. -
          batch: The batch size of the ciphertext. - num_elements: The number of
            elements in the ciphertext. - degree: The degree of the ciphertext.
            -
          num_moduli: The number of moduli in the ciphertext. - precision: The
            precision of the ciphertext.
        parameters (Optional[dict], optional): A dictionary containing the
          parameters of the ciphertext. - moduli: The moduli of the ciphertext.
          - If the moduli is a single integer, the ciphertext will be a single
          modulus. - If the moduli is a list of integers, the ciphertext will be
          a multi-modulus.
        finite_field_context (Optional[object], optional): The finite field
          context to use. - If not provided, a default BarrettContext will be
          created.
        r (Optional[int], optional): The r parameter for the NTT context.
        c (Optional[int], optional): The c parameter for the NTT context.
    """
    self.batch = shapes['batch']
    self.num_elements = shapes['num_elements']
    self.num_moduli = shapes['num_moduli']
    self.degree = shapes['degree']
    log_degree = int(math.log2(self.degree))
    self.precision = shapes['precision']
    if 'degree_layout' in shapes:
      self.degree_layout = shapes['degree_layout']
    else:
      self.degree_layout = (self.degree,)

    if len(self.degree_layout) == 2:
      self.r = self.degree_layout[0]
      self.c = self.degree_layout[1]
    else:
      self.r = 1 << (log_degree // 2)
      self.c = self.degree // self.r

    if self.precision <= 32:
      self.modulus_dtype = jnp.uint32
    else:
      self.modulus_dtype = jnp.uint64

    if parameters is not None and 'moduli' in parameters:
      self.moduli = parameters['moduli']
    else:
      self.moduli = util.find_moduli_ntt(
          self.num_moduli, self.precision, 2 * self.degree
      )

    # NTT Parameters
    if parameters is not None and 'finite_field_context' in parameters:
      finite_field_context = parameters['finite_field_context'](
          moduli=self.moduli
      )
    else:
      finite_field_context = ff_context.BarrettContext(moduli=self.moduli)

    ntt_params = {
        'r': self.r,
        'c': self.c,
        'finite_field_context': finite_field_context,
    }
    self.bit_reverse_indices = jnp.array(
        util.bit_reverse_indices(self.degree), jnp.uint32
    )

    self.shape_in_ntt_all_limbs = (-1, self.r, self.c, self.num_moduli)
    if (
        parameters is not None
        and 'BAT_lazy' in parameters
        and parameters['BAT_lazy']
    ):
      self.ntt_ctx = ntt.NTTCiphertextBATLazyContext(
          moduli=self.moduli, parameters=ntt_params
      )
    else:
      if isinstance(finite_field_context, ff_context.BarrettContext):
        self.ntt_ctx = ntt.NTTCiphertextBarrettContext(
            moduli=self.moduli, parameters=ntt_params
        )
      elif isinstance(finite_field_context, ff_context.MontgomeryContext):
        self.ntt_ctx = ntt.NTTCiphertextMontgomeryContext(
            moduli=self.moduli, parameters=ntt_params
        )
      elif isinstance(finite_field_context, ff_context.ShoupContext):
        self.ntt_ctx = ntt.NTTCiphertextShoupContext(
            moduli=self.moduli, parameters=ntt_params
        )
      else:
        raise ValueError(
            'Unsupported finite field context type:'
            f' {type(finite_field_context)}'
        )

    self.moduli_array = jnp.array(self.moduli, dtype=self.modulus_dtype)
    self.ciphertext = jnp.zeros(
        (self.batch, self.num_elements, self.degree, self.num_moduli),
        dtype=self.modulus_dtype,
    )
    self.extend_ciphertext = jnp.zeros(
        (self.batch, self.num_elements, self.degree, 1),
        dtype=self.modulus_dtype,
    )
    self.bconv = None
    self.bconv_indices_list = []

  def _create_bconv(self, moduli):
    if self.bconv is None:
      self.bconv = BConvBarrett(moduli)
    return self.bconv

  def random_init(self):
    self.ciphertext = util.random_batched_ciphertext(
        (self.batch, self.num_elements, *self.degree_layout, self.num_moduli),
        self.moduli,
        dtype=self.modulus_dtype,
    )

  #####################
  # Getter Functions
  #####################
  @property
  def shape(self):
    return self.ciphertext.shape

  def get_batch_ciphertext(self) -> jnp.ndarray:
    return self.ciphertext

  def get_ciphertext(self, batch_index) -> jnp.ndarray:
    return self.ciphertext[batch_index]

  def get_element(self, element_index) -> jnp.ndarray:
    return self.ciphertext[:, element_index]

  def get_limb(self, limb_index) -> jnp.ndarray:
    return self.ciphertext[..., limb_index]

  #####################
  # Setter Functions
  # Note set_ciphertext, set_element, set_limb are in place operations, not recommended in JAX
  #####################
  def set_batch_ciphertext(self, batch_ciphertext: jnp.ndarray) -> None:
    self.ciphertext = batch_ciphertext

  def set_ciphertext(
      self, batch_index: int, ciphertext: jnp.ndarray
  ) -> None:
    self.ciphertext = self.ciphertext.at[batch_index].set(ciphertext)

  def set_element(
      self, element_index: int, element: jnp.ndarray
  ) -> None:
    self.ciphertext = self.ciphertext.at[:, element_index].set(element)

  def set_limb(self, limb_index: int, limb: jnp.ndarray) -> None:
    self.ciphertext = self.ciphertext.at[..., limb_index].set(limb)

  def get_moduli_array(self) -> jnp.ndarray:
    return self.moduli_array

  def get_moduli(self) -> Union[List[int], int]:
    return self.moduli

  def get_modulus(self, index: int) -> int:
    return self.moduli[index]

  #####################
  # Domain Conversion Functions
  #####################
  def to_ntt_form(self):
    current_shape = self.ciphertext.shape
    reshaped_in = self.ciphertext.reshape(self.shape_in_ntt_all_limbs)
    ntt_result = self.ntt_ctx.ntt(reshaped_in)
    self.ciphertext = ntt_result.reshape(current_shape)

  def to_coeffs_form(self):
    current_shape = self.ciphertext.shape
    reshaped_in = self.ciphertext.reshape(self.shape_in_ntt_all_limbs)
    intt_result = self.ntt_ctx.intt(reshaped_in)
    self.ciphertext = intt_result.reshape(current_shape)

  def to_compute_format(self):
    self.ciphertext = self.ntt_ctx.to_computation_format(self.ciphertext)

  def to_original_format(self):
    self.ciphertext = self.ntt_ctx.to_original_format(self.ciphertext)

  #####################
  # Arithmetic Functions Entire Ciphertext
  #####################
  def add(self, other: Union['Ciphertext', jnp.ndarray]):
    other_array = other.ciphertext if isinstance(other, Ciphertext) else other
    self.ciphertext = self.ciphertext + other_array

  def sub(self, other: Union['Ciphertext', jnp.ndarray]):
    other_array = other.ciphertext if isinstance(other, Ciphertext) else other
    self.ciphertext = self.ciphertext - other_array

  def mul(self, other: Union['Ciphertext', jnp.ndarray]):
    other_array = other.ciphertext if isinstance(other, Ciphertext) else other
    self.ciphertext = self.ciphertext.astype(jnp.uint64) * other_array.astype(
        jnp.uint64
    )

  def modmul(self, other: Union['Ciphertext', jnp.ndarray]):
    other_array = other.ciphertext if isinstance(other, Ciphertext) else other
    temp = self.ciphertext.astype(jnp.uint64) * other_array.astype(jnp.uint64)
    reduced = self.ntt_ctx.ff_ctx.modular_reduction(temp)
    self.ciphertext = reduced.astype(self.modulus_dtype)

  def mod_reduce(self):
    reduced = self.ntt_ctx.ff_ctx.modular_reduction(
        self.ciphertext.astype(jnp.uint64)
    )
    self.ciphertext = reduced.astype(self.modulus_dtype)

  #####################
  # Modulus Dropping Functions
  #####################
  def drop_last_modulus(self) -> jnp.ndarray:
    if self.num_moduli <= 1:
      raise ValueError('Cannot drop modulus from a single-limb ciphertext.')

    # Drop ciphertext limb and track the new modulus set.
    self.ciphertext = self.ciphertext[..., :-1]
    self.moduli = self.moduli[:-1]
    self.moduli_array = self.moduli_array[:-1]
    self.num_moduli -= 1

    # Update finite field context and rebuild NTT context for the reduced limb set.
    self.shape_in_ntt_all_limbs = (-1, self.r, self.c, self.num_moduli)
    self.shape_in_ntt_last_limb = (-1, self.r, self.c)
    self.ntt_ctx.drop_last_modulus()
    return self.ciphertext

  #####################
  # Modulus Switching Helpers
  #####################
  def modulus_switch_control_gen(
      self, degree_layout: Optional[tuple] = None, perf_test: bool = False
  ):
    if degree_layout is None:
      degree_layout = (self.degree,)
    if self.num_moduli <= 1:
      raise ValueError(
          'Cannot perform modulus switch with fewer than two moduli.'
      )

    ring_dim = self.degree
    overall_psi = [util.root_of_unity(2 * ring_dim, q) for q in self.moduli]
    overall_power_of_psi = jnp.array(
        [
            [
                pow(overall_psi[idx], i, self.moduli[idx])
                for i in range(ring_dim)
            ]
            for idx in range(len(self.moduli))
        ],
        jnp.uint64,
    )

    gammas, betas = util.gamma_beta_calculation(
        self.moduli, perf_test=perf_test
    )
    gammas_power_of_psi_no_last = (
        gammas[: self.num_moduli - 1, None].astype(jnp.uint64)
        * overall_power_of_psi[: self.num_moduli - 1].astype(jnp.uint64)
    ) % jnp.array(self.moduli[: self.num_moduli - 1], jnp.uint64)[:, None]
    inv_psi_last = pow(overall_psi[-1], -1, self.moduli[-1])
    power_of_inv_psi_arr_last_tower = jnp.array(
        [pow(inv_psi_last, i, self.moduli[-1]) for i in range(ring_dim)],
        jnp.uint64,
    )

    ct_last_limb_shapes = {
        'batch': self.batch,
        'num_elements': 4,
        'degree': ring_dim,
        'precision': 32,
        'num_moduli': 1,
        'degree_layout': degree_layout,
    }
    self.ct_last_limb = Ciphertext(
        ct_last_limb_shapes,
        parameters={
            'moduli': [self.moduli[-1]],
            'finite_field_context': ff_context.BarrettContext,
            'r': self.r,
            'c': self.c,
        },
    )

    self.gammas_power_of_psi_no_last = jnp.array(
        gammas_power_of_psi_no_last, jnp.uint64
    ).T.reshape(1, *degree_layout, self.num_moduli - 1)
    self.betas = jnp.array(betas, jnp.uint64)[: self.num_moduli]

    # Reshape power_of_inv_psi_arr_last_tower to (1, 1, degree, 1) for broadcasting
    self.power_of_last_tower_inv_psi = jnp.array(
        power_of_inv_psi_arr_last_tower, jnp.uint64
    ).reshape(1, *degree_layout, 1)

    self.moduli_no_last = jnp.array(self.moduli[:-1], dtype=self.modulus_dtype)
    self.moduli_threshold = jnp.array(
        (self.moduli[-1] + 1) // 2, dtype=jnp.uint32
    )
    self.drop_last_moduli_arr = jnp.array(self.moduli[:-1], jnp.uint32)
    self.last_tower_moduli = self.moduli[-1]

    self.mod_switch_params = True  # Flag to indicate initialization
    self.drop_last_modulus()

  def rescale(self):
    """Rescale implementation using Ciphertext class methods."""
    # Ensure control parameters are available on the ciphertext
    if not hasattr(self, 'mod_switch_params'):
      self.modulus_switch_control_gen(degree_layout=self.degree_layout)

    # Capture current state before dropping modulus
    in_ciphertexts = self.get_batch_ciphertext()

    # 1. Extract Last Tower and Process logic similar to hemul.py
    last_towers = in_ciphertexts[..., -1:]
    self.ct_last_limb.set_batch_ciphertext(last_towers.astype(jnp.uint32))
    self.ct_last_limb.to_coeffs_form()
    self.ct_last_limb.modmul(self.power_of_last_tower_inv_psi)
    ct_last_limb_modred = self.ct_last_limb.get_batch_ciphertext().astype(
        jnp.uint32
    )

    condition = ct_last_limb_modred < self.moduli_threshold
    result_if_lt_threshold = ct_last_limb_modred
    result_if_ge_threshold = (
        jnp.array(self.drop_last_moduli_arr, jnp.uint64)
        - self.last_tower_moduli
        + ct_last_limb_modred
    )
    last_poly_switch_modulus_coef = jnp.where(
        condition, result_if_lt_threshold, result_if_ge_threshold
    )
    last_poly_switch_modulus_coef_twisted = (
        last_poly_switch_modulus_coef.astype(jnp.uint64)
        * self.gammas_power_of_psi_no_last
    )
    self.set_batch_ciphertext(last_poly_switch_modulus_coef_twisted)
    self.mod_reduce()
    self.to_ntt_form()
    mod_reduce_last_res_unreduced = self.get_batch_ciphertext()

    self.set_batch_ciphertext(in_ciphertexts[..., :-1])
    self.mul(self.betas)
    self.set_batch_ciphertext(
        self.get_batch_ciphertext()
        + mod_reduce_last_res_unreduced.astype(jnp.uint64)
    )
    self.mod_reduce()
    return self.get_batch_ciphertext()

  #####################
  # FHE Kernel Functions
  #####################
  def ciphertext_mult(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
    a0 = self.ciphertext[:, 0].astype(jnp.uint64)
    a1 = self.ciphertext[:, 1].astype(jnp.uint64)
    b0 = self.ciphertext[:, 2].astype(jnp.uint64)
    b1 = self.ciphertext[:, 3].astype(jnp.uint64)

    mul0_t = a0 * b0
    mul0 = self.ntt_ctx.ff_ctx.modular_reduction(mul0_t).astype(
        self.modulus_dtype
    )

    mul2_t = a1 * b1
    mul2 = self.ntt_ctx.ff_ctx.modular_reduction(mul2_t).astype(
        self.modulus_dtype
    )

    t1_t = a0 * b1 + a1 * b0
    mul1 = self.ntt_ctx.ff_ctx.modular_reduction(t1_t)

    return (
        jnp.concatenate([mul0[:, None], mul1[:, None]], axis=1),
        mul2[:, None],
    )

  #####################
  # Key Switching Functions
  #####################
  def key_switch_control_gen(
      self,
      extend_moduli: List[int],
      dnum: int,
      evalkey_a,
      evalkey_b,
      perf_test: bool = False,
      selected_moduli: List[int] | None = None,
      degree_layout: tuple | int | None = None,
  ):
    """Generates all parameters needed for key switching."""
    if degree_layout is None:
      degree_layout = (self.degree,)
    # 1. Configuration
    self.ks_params = {}
    current_moduli = self.moduli if selected_moduli is None else selected_moduli
    drop_last_extend_moduli = current_moduli + extend_moduli
    self.ks_params['drop_last_extend_moduli'] = drop_last_extend_moduli
    ring_dim = self.degree

    # Generate Drop Last Power of Psi
    drop_last_psi = [
        util.root_of_unity(2 * ring_dim, q) for q in drop_last_extend_moduli
    ]
    if perf_test:
      self.ks_params['drop_last_power_of_psi'] = util.random_parameters(
          (*self.degree_layout, len(drop_last_extend_moduli)),
          drop_last_extend_moduli,
          dtype=jnp.uint64,
      )
    else:
      self.ks_params['drop_last_power_of_psi'] = jnp.array(
          [
              [
                  pow(drop_last_psi[idx], i, drop_last_extend_moduli[idx])
                  for i in range(ring_dim)
              ]
              for idx in range(len(drop_last_extend_moduli))
          ],
          jnp.uint64,
      ).T.reshape(*self.degree_layout, len(drop_last_extend_moduli))

    # Tower Indices
    sizeQ_drop_last = len(current_moduli)
    alpha = (sizeQ_drop_last + dnum - 1) // dnum
    self.ks_params['alpha'] = alpha
    self.ks_params['numPartQl'] = (sizeQ_drop_last + alpha - 1) // alpha

    original_moduli_extract_index = []
    for i in range(sizeQ_drop_last):
      if i % alpha == 0:
        original_moduli_extract_index.append([i])
      else:
        original_moduli_extract_index[-1].append(i)

    (
        select_tower_index_overall,
        non_select_tower_index_overall,
        restore_indices,
    ) = ([], [], [])
    for part in range(self.ks_params['numPartQl']):
      select_tower_overall_index = original_moduli_extract_index[part]
      non_select_tower_overall_index = [
          i
          for i in range(len(drop_last_extend_moduli))
          if i not in select_tower_overall_index
      ]
      concat_order = select_tower_overall_index + non_select_tower_overall_index
      restore_index = [0] * len(concat_order)
      for pos, val in enumerate(concat_order):
        restore_index[val] = pos
      select_tower_index_overall.append(
          jnp.array(select_tower_overall_index, jnp.uint16)
      )
      non_select_tower_index_overall.append(
          jnp.array(non_select_tower_overall_index, jnp.uint16)
      )
      restore_indices.append(jnp.array(restore_index, jnp.uint16))

    self.ks_params['select_tower_index_overall'] = select_tower_index_overall
    self.ks_params['non_select_tower_index_overall'] = (
        non_select_tower_index_overall
    )
    self.ks_params['restore_indices'] = restore_indices

    # Helper function for inv psi
    def gen_power_of_inv_psi_arr(moduli):
      q_list = [moduli] if not isinstance(moduli, list) else moduli
      psi_list = [util.root_of_unity(2 * ring_dim, q) for q in q_list]
      inv_psi = [pow(psi, -1, q) for (q, psi) in zip(q_list, psi_list)]
      power_of_inv_psi_arr = [
          [pow(inv_psi[idx], i, q_list[idx]) for i in range(ring_dim)]
          for idx in range(len(psi_list))
      ]
      return jnp.array(power_of_inv_psi_arr, jnp.uint64).T.reshape(
          1, 1, *degree_layout, -1
      )

    if perf_test:
      self.ks_params['power_of_inv_psi_arr_drop_last'] = util.random_parameters(
          (len(current_moduli), ring_dim), current_moduli, dtype=jnp.uint64
      ).T.reshape(1, 1, *degree_layout, -1)
    else:
      self.ks_params['power_of_inv_psi_arr_drop_last'] = (
          gen_power_of_inv_psi_arr(current_moduli)
      )

    # Basis Change
    bconv_indices_list = []
    for part in range(self.ks_params['numPartQl']):
      bconv_indices_list.append((
          select_tower_index_overall[part].tolist(),
          non_select_tower_index_overall[part].tolist(),
      ))

    # Manage Global BConv
    self.ks_params['ks_control_start_idx'] = len(self.bconv_indices_list)
    self.bconv_indices_list.extend(bconv_indices_list)
    self._create_bconv(drop_last_extend_moduli)
    self.bconv.control_gen(self.bconv_indices_list, perf_test=perf_test)

    # KeySwitch Parts CTs
    self.ks_params['ct_ks_parts'] = []

    # Handle batch size for key switch
    ks_batch_size = self.batch

    ct_shapes_common = {
        'batch': ks_batch_size,
        'num_elements': 2,
        'degree': ring_dim,
        'precision': self.precision,
        'degree_layout': self.degree_layout,
    }
    ct_params_common = {
        'r': self.r,
        'c': self.c,
        'finite_field_context': ff_context.BarrettContext,
    }

    for part in range(self.ks_params['numPartQl']):
      target_indices = non_select_tower_index_overall[part].tolist()
      target_moduli = [drop_last_extend_moduli[i] for i in target_indices]
      shapes_part = ct_shapes_common.copy()
      shapes_part['num_moduli'] = len(target_moduli)
      params_part = ct_params_common.copy()
      params_part['moduli'] = target_moduli
      self.ks_params['ct_ks_parts'].append(Ciphertext(shapes_part, params_part))

    # CT for Drop Last + Extend (for ks_core result)
    shapes_dle = ct_shapes_common.copy()
    shapes_dle['num_moduli'] = len(drop_last_extend_moduli)
    params_dle = ct_params_common.copy()
    params_dle['moduli'] = drop_last_extend_moduli
    self.ks_params['ct_drop_last_extend'] = Ciphertext(shapes_dle, params_dle)

    # Keys Setup
    # Reshape keys
    idx_cur_last_tower = len(current_moduli)
    overall_sizeP = len(extend_moduli)

    self.ks_params['evk_a_precomp'] = jnp.concatenate(
        [evalkey_a[..., :idx_cur_last_tower], evalkey_a[..., -overall_sizeP:]],
        axis=-1,
    ).reshape(-1, *self.degree_layout, len(drop_last_extend_moduli))
    self.ks_params['evk_b_precomp'] = jnp.concatenate(
        [evalkey_b[..., :idx_cur_last_tower], evalkey_b[..., -overall_sizeP:]],
        axis=-1,
    ).reshape(-1, *self.degree_layout, len(drop_last_extend_moduli))
    self.ks_params['idx_cur_last_tower'] = idx_cur_last_tower
    self.ks_params['overall_sizeP'] = overall_sizeP

  def key_switch(self):
    """Performs key switching on the current ciphertext.

    Mutates self.ciphertext to the extended moduli basis.
    """
    if not hasattr(self, 'ks_params'):
      raise RuntimeError(
          'Key switch parameters not initialized. Call key_switch_control_gen'
          ' first.'
      )

    # Aliases
    params = self.ks_params
    power_of_inv_psi_arr_drop_last = params['power_of_inv_psi_arr_drop_last']
    select_tower_index_overall = params['select_tower_index_overall']
    non_select_tower_index_overall = params['non_select_tower_index_overall']
    ks_control_start_idx = params['ks_control_start_idx']
    ct_ks_parts = params['ct_ks_parts']
    numPartQl = params['numPartQl']
    ct_drop_last_extend = params['ct_drop_last_extend']
    evk_a_precomp = params['evk_a_precomp']
    evk_b_precomp = params['evk_b_precomp']
    restore_indices = params['restore_indices']

    # Save NTT input for reconstruction
    input_ntt = self.get_batch_ciphertext()

    # Step 3: Precompute
    self.to_coeffs_form()
    self.modmul(power_of_inv_psi_arr_drop_last)
    partCtCloneCoef = self.get_batch_ciphertext()

    res0 = None
    res1 = None
    for part in range(numPartQl):
      select_idxs = select_tower_index_overall[part]
      non_select_idxs = non_select_tower_index_overall[part]
      _power_of_psi_arr = jnp.take(
          params['drop_last_power_of_psi'], non_select_idxs, axis=-1
      )
      partCtCloneEval = self.bconv.basis_change_bat(
          jnp.take(partCtCloneCoef, select_idxs, axis=-1),
          control_index=ks_control_start_idx + part,
      ).astype(jnp.uint64)
      ct_part = ct_ks_parts[part]
      ct_part.ciphertext = partCtCloneEval
      ct_part.modmul(_power_of_psi_arr)
      partCtCloneEval_scaled_multi_moduli = ct_part.ciphertext
      ct_part.ciphertext = partCtCloneEval_scaled_multi_moduli
      ct_part.to_ntt_form()
      partsCtCompl_multi_moduli = ct_part.ciphertext

      partsCtExt_cur_part = jnp.concatenate(
          [
              jnp.take(input_ntt, select_idxs, axis=-1),
              partsCtCompl_multi_moduli,
          ],
          axis=-1,
      )
      partsCtExt_cur_part = jnp.take(
          partsCtExt_cur_part, restore_indices[part], axis=-1
      )

      if res0 is None:
        res0 = (
            partsCtExt_cur_part
            * evk_b_precomp.astype(jnp.uint64)[part][None, None, :, :]
        )
        res1 = (
            partsCtExt_cur_part
            * evk_a_precomp.astype(jnp.uint64)[part][None, None, :, :]
        )
      else:
        res0 += (
            partsCtExt_cur_part
            * evk_b_precomp.astype(jnp.uint64)[part][None, None, :, :]
        )
        res1 += (
            partsCtExt_cur_part
            * evk_a_precomp.astype(jnp.uint64)[part][None, None, :, :]
        )

    result = jnp.concatenate([res0, res1], axis=1)
    ct_drop_last_extend.set_batch_ciphertext(result)
    ct_drop_last_extend.mod_reduce()
    ks_result = ct_drop_last_extend.get_batch_ciphertext()
    self.set_batch_ciphertext(ks_result)
