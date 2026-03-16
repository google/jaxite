import jaxite.jaxite_word.bconv as bconv
from jaxite.jaxite_word.ciphertext import Ciphertext
import jax
import jax.numpy as jnp
import numpy as np
import jaxite.jaxite_word.util as util

# enable 64-bit computation in jax
jax.config.update('jax_enable_x64', True)


def is_power_of_two(x: int) -> bool:
  """Returns True if x is a power of two."""
  return x > 0 and (x & (x - 1)) == 0


def mat_1d_shuffle_to_2d(coefMap, r, c):
  """Memory Aligned Transformation

  Perform 1D data shuffing of O(N) in matrix fashion with O(sqrt(N)) memory
  cost.
  Precomputes the 2D indices.
  coefMap is the 1D permuted indices.

  Factor coefMap (length r*c) into row_perm (len r) and col_perm (len c) such
  that:
    coefMap.reshape(r,c)[i,j] == row_perm[i]*c + col_perm[j]

  Returns:
    row_perm: int32[r]
    col_perm: int32[c]
  """
  if coefMap.ndim != 1 or coefMap.shape[0] != r * c:
    raise ValueError(
        f'coefMap must be 1D of length r*c. Got shape {coefMap.shape},'
        f' r*c={r*c}.'
    )
  if r <= 0 or c <= 0:
    raise ValueError('r and c must be positive.')
  # (Recommended, since your degree is power-of-2)
  if not (is_power_of_two(r) and is_power_of_two(c)):
    raise ValueError('For your setting, r and c should be powers of two.')
  coef2d = coefMap.reshape(r, c)

  # If coef2d[i,j] = row_perm[i]*c + col_perm[j], then:
  row_perm = (coef2d[:, 0] // c).astype(jnp.int32)
  col_perm = (coef2d[0, :] % c).astype(jnp.int32)

  coef2d_h = np.asarray(jax.device_get(coef2d))
  row_h = np.asarray(jax.device_get(row_perm))
  col_h = np.asarray(jax.device_get(col_perm))
  expected_h = row_h[:, None] * c + col_h[None, :]
  if not np.array_equal(coef2d_h, expected_h):
    raise ValueError(
        'coefMap is NOT decomposable into a single global row permutation +'
        f' column permutation for r={r}, c={c}. (i.e., not P_row ⊗ P_col). Pick'
        ' a different (r,c) factorization, or keep using jnp.take(a, coefMap,'
        ' axis=2).'
    )

  return row_perm, col_perm


class HERot:

  def __init__(self, r, c, dnum, rotate_in_ciphertext_moduli, extend_moduli):
    self.r = r
    self.c = c
    self.dnum = dnum
    self.rotate_in_ciphertext_moduli = rotate_in_ciphertext_moduli
    self.extend_moduli = extend_moduli
    self.overall_moduli_init = (
        self.rotate_in_ciphertext_moduli + self.extend_moduli
    )
    # Instantiating a single class of BConvBarrett
    self.bconv = bconv.BConvBarrett(self.overall_moduli_init)
    self.evalkey_a_vector = None
    self.evalkey_b_vector = None

  def control_gen(self, batch=1, degree_layout=None, perf_test=False):
    if degree_layout is None:
      degree_layout = (self.r * self.c,)
    # degree_layout = (self.r, self.c)
    self.degree_layout = degree_layout
    sizeQl_in = len(self.rotate_in_ciphertext_moduli)
    sizeQlP_in = len(self.extend_moduli) + sizeQl_in
    alpha = (sizeQl_in + self.dnum - 1) // self.dnum
    ring_dim = self.r * self.c
    overall_moduli = self.rotate_in_ciphertext_moduli + self.extend_moduli
    self.perf_test = perf_test

    # External Input
    if perf_test:
      power_of_psi = util.random_parameters(
          (*degree_layout, len(overall_moduli)),
          overall_moduli,
          dtype=jnp.uint64,
      )
      power_of_inv_psi = util.random_parameters(
          (*degree_layout, len(overall_moduli)),
          overall_moduli,
          dtype=jnp.uint64,
      )
    else:
      original_psi = [
          util.root_of_unity(2 * ring_dim, q) for q in overall_moduli
      ]
      power_of_psi = jnp.array(
          [
              [
                  pow(original_psi[idx], i, overall_moduli[idx])
                  for i in range(ring_dim)
              ]
              for idx in range(len(overall_moduli))
          ],
          jnp.uint64,
      ).T.reshape(*degree_layout, len(overall_moduli))
      inv_psi = [
          pow(psi, -1, q) for (q, psi) in zip(overall_moduli, original_psi)
      ]
      power_of_inv_psi = jnp.array(
          [
              [
                  pow(inv_psi[idx], i, overall_moduli[idx])
                  for i in range(ring_dim)
              ]
              for idx in range(len(overall_moduli))
          ],
          jnp.uint64,
      ).T.reshape(*degree_layout, len(overall_moduli))

    ## parameters generation for approximation mod down
    current_moduli = self.extend_moduli
    target_moduli = [
        item for item in overall_moduli if item not in current_moduli
    ]

    P = 1
    for moduli in current_moduli:
      P *= moduli
    PInvModq_approx_down = [util.modinv(P, q) for q in target_moduli]

    self.overall_moduli = overall_moduli
    self.PInvModq = jnp.asarray(PInvModq_approx_down, dtype=jnp.uint32).reshape(
        sizeQl_in
    )
    self.power_of_psi = jnp.array(power_of_psi, jnp.uint64)
    self.power_of_inv_psi = power_of_inv_psi[..., :sizeQl_in]
    self.power_of_inv_psi_approx_down = power_of_inv_psi[
        ..., sizeQl_in:sizeQlP_in
    ]
    self.sizeQlP, self.sizeQl = sizeQlP_in, sizeQl_in
    self.batch = batch

    # BConv control generation
    original_moduli_extract_index = []
    for i in range(sizeQl_in):
      if i % alpha == 0:
        original_moduli_extract_index.append([i])
      else:
        original_moduli_extract_index[-1].append(i)
    numPartQl = (sizeQl_in + alpha - 1) // alpha

    control_indices_list = []

    self.select_tower_index = []
    self.non_select_tower_index = []

    # 1. Basis change for key switch decomposition
    for part in range(numPartQl):
      sel_index = original_moduli_extract_index[part]
      non_sel_index = [
          i for i in range(len(overall_moduli)) if i not in sel_index
      ]
      self.select_tower_index.append(sel_index)
      self.non_select_tower_index.append(non_sel_index)
      control_indices_list.append((sel_index, non_sel_index))

    # Precompute restore indices for scatter optimization
    self.restore_indices = []
    for part in range(numPartQl):
      sel_index = self.select_tower_index[part]
      non_sel_index = self.non_select_tower_index[part]

      # The resulting array after concatenation will have elements in this order:
      # [elements corresponding to sel_index, elements corresponding to non_sel_index]
      concat_order = sel_index + non_sel_index

      # We want to map this back to the natural order [0, 1, 2, ..., len(overall_moduli)-1]
      # restore_index[i] should be the position of 'i' in concat_order
      restore_index = [0] * len(concat_order)
      for pos, val in enumerate(concat_order):
        restore_index[val] = pos

      self.restore_indices.append(jnp.array(restore_index, dtype=jnp.uint16))

    # 2. Basis change for approximation modulus switch
    rotate_indices = list(range(sizeQl_in))
    extend_indices = list(range(sizeQl_in, sizeQlP_in))
    control_indices_list.append((extend_indices, rotate_indices))

    self.bconv.control_gen(control_indices_list, perf_test=perf_test)

    # Pre-allocate Ciphertext objects to amortize NTT context creation
    ct_in_shapes = {
        'batch': batch,
        'num_elements': 1,
        'degree': ring_dim,
        'precision': 32,
        'num_moduli': sizeQl_in,
        'degree_layout': degree_layout,
    }
    self.ct_in = Ciphertext(
        ct_in_shapes,
        parameters={'moduli': overall_moduli[:sizeQl_in], 'BAT_lazy': False},
    )

    self.ct_parts = []
    for part in range(numPartQl):
      _target_indices_list = self.non_select_tower_index[part]
      _target_moduli_list = [overall_moduli[i] for i in _target_indices_list]
      _num_moduli_part = len(_target_moduli_list)
      ct_part_shapes = {
          'batch': batch,
          'num_elements': 1,
          'degree': ring_dim,
          'precision': 32,
          'num_moduli': _num_moduli_part,
          'degree_layout': degree_layout,
      }
      self.ct_parts.append(
          Ciphertext(
              ct_part_shapes,
              parameters={'moduli': _target_moduli_list, 'BAT_lazy': False},
          )
      )

    ct_full_shapes = {
        'batch': batch,
        'num_elements': 1,
        'degree': ring_dim,
        'precision': 32,
        'num_moduli': sizeQlP_in,
        'degree_layout': degree_layout,
    }
    self.ct_full = Ciphertext(
        ct_full_shapes,
        parameters={'moduli': self.overall_moduli, 'BAT_lazy': False},
    )

    ct_approx_shapes = {
        'batch': batch,
        'num_elements': 1,
        'degree': ring_dim,
        'precision': 32,
        'num_moduli': sizeQlP_in - sizeQl_in,
        'degree_layout': degree_layout,
    }
    self.ct_approx = Ciphertext(
        ct_approx_shapes,
        parameters={'moduli': self.extend_moduli, 'BAT_lazy': False},
    )

  def setup_rotate(self, evalkey_a_vector, evalkey_b_vector, coefMap):
    self.evalkey_a_vector = evalkey_a_vector.astype(jnp.uint64)
    self.evalkey_b_vector = evalkey_b_vector.astype(jnp.uint64)
    self.coefMap = jnp.asarray(coefMap, dtype=jnp.int32)

  def rotate(
      self,
      in_ciphertexts,
  ):
    assert self.evalkey_a_vector is not None
    assert self.evalkey_b_vector is not None
    # -------------------------------
    # Inline of key_switch_precompute_core_28bit for in_ciphertexts[-1]
    # -------------------------------
    batch, r, c, dnum = self.batch, self.r, self.c, self.dnum
    sizeQlP, sizeQl = self.sizeQlP, self.sizeQl
    select_tower_index, non_select_tower_index = (
        self.select_tower_index,
        self.non_select_tower_index,
    )
    power_of_inv_psi = self.power_of_inv_psi
    power_of_psi = self.power_of_psi
    power_of_inv_psi_approx_down = self.power_of_inv_psi_approx_down
    sizeP = sizeQlP - sizeQl
    ring_dim = r * c

    overall_moduli_jax = jnp.asarray(self.overall_moduli, dtype=jnp.uint32)
    original_moduli = jnp.take(overall_moduli_jax, jnp.arange(sizeQl), axis=0)
    in_tower = in_ciphertexts[:, -1:, ..., :sizeQl]

    # ---------- Step 1: Keyswitch ----------
    self.ct_in.ciphertext = in_tower
    # self.ct_in.key_switch() # Rotate implements inline keyswitch for better performance.
    self.ct_in.to_coeffs_form()
    self.ct_in.modmul(jnp.array(power_of_inv_psi, jnp.uint64))
    partCtCloneCoef = self.ct_in.ciphertext
    partsCtExt = []
    res0 = None
    res1 = None
    for part in range(dnum):
      select_tower_index_arr = jnp.array(select_tower_index[part], jnp.uint16)
      non_select_tower_index_arr = jnp.array(
          non_select_tower_index[part], jnp.uint16
      )
      power_of_psi_arr_part = jnp.take(
          power_of_psi, non_select_tower_index_arr, axis=-1
      )

      input_for_bconv = jnp.take(
          partCtCloneCoef, select_tower_index_arr, axis=-1
      )
      partCtCloneEval = self.bconv.basis_change_bat(
          input_for_bconv, control_index=part
      ).astype(jnp.uint64)

      ct_part = self.ct_parts[part]
      ct_part.ciphertext = partCtCloneEval
      ct_part.modmul(power_of_psi_arr_part)
      partCtCloneEval_scaled_multi_moduli = ct_part.ciphertext

      ct_part.ciphertext = partCtCloneEval_scaled_multi_moduli
      ct_part.to_ntt_form()
      partsCtCompl_multi_moduli = ct_part.ciphertext

      partsCtExt_cur_part = jnp.concatenate(
          [
              jnp.take(in_tower, select_tower_index_arr, axis=-1),
              partsCtCompl_multi_moduli,
          ],
          axis=-1,
      )
      partsCtExt_cur_part = jnp.take(
          partsCtExt_cur_part, self.restore_indices[part], axis=-1
      )
      if res0 is None:
        res0 = (
            partsCtExt_cur_part * self.evalkey_b_vector[part][None, None, ...]
        )
        res1 = (
            partsCtExt_cur_part * self.evalkey_a_vector[part][None, None, ...]
        )
      else:
        res0 += (
            partsCtExt_cur_part * self.evalkey_b_vector[part][None, None, ...]
        )
        res1 += (
            partsCtExt_cur_part * self.evalkey_a_vector[part][None, None, ...]
        )

    keyswitch_core_res = jnp.concatenate([res0, res1], axis=1)
    self.ct_full.ciphertext = keyswitch_core_res
    self.ct_full.mod_reduce()
    keyswitch_core_res = self.ct_full.get_batch_ciphertext()
    # keyswitch_core_res = self.ct_in.get_batch_ciphertext()  # If use function

    # ---------- Step 3: Approximation modulus switch (inline) ----------
    result_ciphertext_list = []
    overall_moduli_jax = jnp.asarray(original_moduli, dtype=jnp.uint32)
    approx_down_in_jax = jnp.asarray(keyswitch_core_res, dtype=jnp.uint32)

    for element_index in range(keyswitch_core_res.shape[1]):
      _current_slice = approx_down_in_jax[
          :, element_index : element_index + 1, ..., sizeQl : (sizeQl + sizeP)
      ]

      self.ct_approx.ciphertext = _current_slice
      self.ct_approx.to_coeffs_form()
      self.ct_approx.modmul(jnp.array(power_of_inv_psi_approx_down, jnp.uint64))
      reduced_approx_down = self.ct_approx.ciphertext

      tower_new_basis_coef = self.bconv.basis_change_bat(
          reduced_approx_down, control_index=dnum
      ).astype(jnp.uint64)

      self.ct_in.ciphertext = tower_new_basis_coef
      self.ct_in.modmul(power_of_psi[..., :sizeQl])
      tower_new_basis_coef_scaled_muli_moduli_modq = self.ct_in.ciphertext

      self.ct_in.ciphertext = tower_new_basis_coef_scaled_muli_moduli_modq
      self.ct_in.to_ntt_form()
      tower_new_basis_jax = self.ct_in.ciphertext

      current_approx_down_in = approx_down_in_jax[
          :, element_index : element_index + 1, ..., :sizeQl
      ]
      sub_result = jnp.where(
          current_approx_down_in < tower_new_basis_jax,
          current_approx_down_in + overall_moduli_jax - tower_new_basis_jax,
          current_approx_down_in - tower_new_basis_jax,
      )

      self.ct_in.ciphertext = sub_result
      self.ct_in.modmul(self.PInvModq)
      reduced_elem_modq = self.ct_in.ciphertext

      result_ciphertext_list.append(reduced_elem_modq)

    mod_down_res = jnp.concatenate(result_ciphertext_list, axis=1)

    # ---------- Step 4: Add and return ----------
    base0 = jnp.asarray(in_ciphertexts[:, 0:1], dtype=jnp.uint32)
    q_b = overall_moduli_jax
    base0 = base0 + mod_down_res[:, 0:1]
    base0_modq = jnp.where(base0 >= q_b, base0 - q_b, base0)
    ks_results = mod_down_res.at[:, 0:1].set(base0_modq)

    # ---------- Step 5: Coefficient map ----------
    # coef_idx = self.coefMap
    ks_results = jnp.take(
        ks_results.reshape(-1, 2, ring_dim, sizeQl), self.coefMap, axis=2
    )

    return ks_results
