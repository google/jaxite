"""SHIP CKKS half-bootstrapping implementation."""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import add
from jaxite.jaxite_ckks import basis_conversion
from jaxite.jaxite_ckks import blind_rotate
from jaxite.jaxite_ckks import bootstrapping_utils as boot_utils
from jaxite.jaxite_ckks import conjugate
from jaxite.jaxite_ckks import key_gen
from jaxite.jaxite_ckks import key_switching
from jaxite.jaxite_ckks import mul
from jaxite.jaxite_ckks import ntt
from jaxite.jaxite_ckks import ntt_cpu
from jaxite.jaxite_ckks import random
from jaxite.jaxite_ckks import rescale
from jaxite.jaxite_ckks import types
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class SHIPLevelKernels:
  """Kernels and data precomputed for a single level of the product reduction tree."""

  mul_cc: mul.Mul
  rescale_lvl: rescale.Rescale
  has_odd: bool
  relk: types.EvaluationKeys

  def tree_flatten(self):
    children = (
        self.mul_cc,
        self.rescale_lvl,
        self.relk,
    )
    aux_data = (self.has_odd,)
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    (
        mul_cc,
        rescale_lvl,
        relk,
    ) = children
    (has_odd,) = aux_data
    return cls(
        mul_cc=mul_cc,
        rescale_lvl=rescale_lvl,
        has_odd=has_odd,
        relk=relk,
    )


def _multiply_and_relinearize_and_rescale(
    ct1: types.Ciphertext,
    ct2: types.Ciphertext,
    level_kernels: Any,
) -> types.Ciphertext:
  ct_3elem = level_kernels.mul_cc.tensor_multiply(ct1, ct2)
  ct_mul = level_kernels.mul_cc.relinearize(ct_3elem, level_kernels.relk)
  ct_mul = level_kernels.rescale_lvl.rescale(ct_mul)
  return ct_mul


@jax.tree_util.register_pytree_node_class
class SHIP:
  """Kernel for SHIP CKKS half-bootstrapping."""

  def __init__(self):
    self.r: int = 0
    self.c: int = 0
    self.degree: int = 0
    self.num_slots: int = 0
    self.ntt_q0: ntt.NTTBarrett = ntt.NTTBarrett()
    self.ntt_q: ntt.NTTBarrett = ntt.NTTBarrett()
    self.ntt_pq: ntt.NTTBarrett = ntt.NTTBarrett()
    self.ntt_p: ntt.NTTBarrett = ntt.NTTBarrett()
    self.tree_kernels: list[SHIPLevelKernels] = []
    self.conjugate_kernel: conjugate.Conjugation = conjugate.Conjugation()
    self.conj_bc_kernel: basis_conversion.BasisConversionBarrett = (
        basis_conversion.BasisConversionBarrett()
    )
    self.mul_kernel_lvl: mul.MulPlaintextCiphertextBarrett = (
        mul.MulPlaintextCiphertextBarrett()
    )
    self.rescale_kernel_lvl: rescale.Rescale = rescale.Rescale()
    self.add_kernel: add.AddModularSubtract = add.AddModularSubtract()
    self.mul_pt_ct_kernel_q: mul.MulPlaintextCiphertextBarrett = (
        mul.MulPlaintextCiphertextBarrett()
    )
    self.final_mul_pt_ct: mul.MulPlaintextCiphertextBarrett = (
        mul.MulPlaintextCiphertextBarrett()
    )
    self.final_rescale: rescale.Rescale = rescale.Rescale()
    self.final_ntt: ntt.NTTBarrett = ntt.NTTBarrett()
    self.brot_kernel: blind_rotate.BlindRotation = blind_rotate.BlindRotation()
    self.online_encoder: boot_utils.OnlineEncoder = boot_utils.OnlineEncoder()
    self.nonzero_idx: jax.Array = np.array([], dtype=np.uint32)  # pyrefly: ignore[bad-assignment]

    self.batched_cmkeys: types.Ciphertext = types.Ciphertext(
        jnp.empty((0,)), jnp.empty((0,))
    )
    self.batched_mmkeys: list[tuple[jax.Array, jax.Array]] = []
    self.bat_switcher: key_switching.BATKeySwitcher = (
        key_switching.BATKeySwitcher()
    )
    self.mmkeys_permutations: list[Any] = []
    self.conjk: types.EvaluationKeys = types.EvaluationKeys()
    self.dnum: int = 0
    self.dnum_conj: int = 0

  def tree_flatten(self):
    """Flattens the kernel into a JAX pytree."""
    children = (
        self.ntt_q0,
        self.ntt_q,
        self.ntt_pq,
        self.ntt_p,
        self.tree_kernels,
        self.conjugate_kernel,
        self.conj_bc_kernel,
        self.mul_kernel_lvl,
        self.rescale_kernel_lvl,
        self.add_kernel,
        self.mul_pt_ct_kernel_q,
        self.final_mul_pt_ct,
        self.final_rescale,
        self.final_ntt,
        self.brot_kernel,
        self.nonzero_idx,
        self.batched_cmkeys,
        self.batched_mmkeys,
        self.bat_switcher,
        self.mmkeys_permutations,
        self.conjk,
        self.online_encoder,
    )
    aux_data = (
        self.r,
        self.c,
        self.degree,
        self.num_slots,
        self.dnum,
        self.dnum_conj,
    )
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """Unflattens the kernel from a JAX pytree."""
    obj = cls()
    (
        obj.ntt_q0,
        obj.ntt_q,
        obj.ntt_pq,
        obj.ntt_p,
        obj.tree_kernels,
        obj.conjugate_kernel,
        obj.conj_bc_kernel,
        obj.mul_kernel_lvl,
        obj.rescale_kernel_lvl,
        obj.add_kernel,
        obj.mul_pt_ct_kernel_q,
        obj.final_mul_pt_ct,
        obj.final_rescale,
        obj.final_ntt,
        obj.brot_kernel,
        obj.nonzero_idx,
        obj.batched_cmkeys,
        obj.batched_mmkeys,
        obj.bat_switcher,
        obj.mmkeys_permutations,
        obj.conjk,
        obj.online_encoder,
    ) = children
    (
        obj.r,
        obj.c,
        obj.degree,
        obj.num_slots,
        obj.dnum,
        obj.dnum_conj,
    ) = aux_data
    return obj

  def __hash__(self):
    return id(self)

  def precompute_constants(
      self,
      q_limbs: list[int],
      p_limbs: list[int],
      degree: int,
      dnum: int,
      sk: types.SecretKey,
      theta: int,
      random_source: random.RandomSource,
  ):
    """Precomputes the NTT and tree constants needed for half-bootstrapping."""
    self.degree = degree
    self.num_slots = degree // 2

    # Factor degree N into r and c where r is the largest power of 2 <= sqrt(N)
    k = degree.bit_length() - 1
    r = 1 << (k // 2)
    c = degree // r
    self.r = r
    self.c = c

    # Convert sk to coefficient domain to find nonzero indices and Hamming
    # weight h.
    s_coeffs = ntt_cpu.intt_negacyclic_poly(
        np.array(sk.data[:, :1]), [int(sk.moduli[0])]
    )
    s_coeffs_flat = s_coeffs.flatten()
    nonzero_idx = np.flatnonzero(s_coeffs_flat).tolist()
    h = len(nonzero_idx)
    pq_limbs = q_limbs + p_limbs
    q0_limbs = [q_limbs[0]]

    self.ntt_q0.precompute_constants(q0_limbs, r, c)
    self.ntt_q.precompute_constants(q_limbs, r, c)
    self.ntt_pq.precompute_constants(pq_limbs, r, c)
    self.ntt_p.precompute_constants(p_limbs, r, c)

    fft_kernel = boot_utils.SpecialInverseFFT()
    fft_kernel.precompute_constants(degree)
    self.online_encoder.precompute_constants(
        degree=degree, fft_kernel=fft_kernel
    )

    # Precompute first-level plaintext-ciphertext multiplication kernel under Q
    self.mul_pt_ct_kernel_q.precompute_constants(q_limbs)

    # Precompute tree reduction constants
    # Level 0 starts with h leaves (only the rotated ciphertexts; ct_pt0 is
    # kept outside the tree).
    curr_nodes_len = h
    curr_moduli = q_limbs
    prev_moduli = q_limbs

    self.tree_kernels = []
    while curr_nodes_len > 1:
      dnum_lvl = min(dnum, len(prev_moduli))
      mul_cc = mul.Mul()
      mul_cc.precompute_constants(
          prev_moduli, p_limbs, dnum_lvl, r, c, composite_degree=0
      )
      rescale_lvl = rescale.Rescale()
      rescale_lvl.precompute_constants(curr_moduli, num_rescales=1, r=r, c=c)

      has_odd = curr_nodes_len % 2 == 1

      num_q_lvl = len(prev_moduli)
      sk_sliced = types.SecretKey(sk.data[:, :num_q_lvl], sk.moduli[:num_q_lvl])
      relk = key_gen.gen_evaluation_key(
          secret_key=sk_sliced,
          q_towers=prev_moduli,
          p_towers=p_limbs,
          dnum=dnum_lvl,
          random_source=random_source,
      )

      self.tree_kernels.append(
          SHIPLevelKernels(
              mul_cc=mul_cc,
              rescale_lvl=rescale_lvl,
              has_odd=has_odd,
              relk=relk,
          )
      )

      next_nodes_len = curr_nodes_len // 2
      if has_odd:
        next_nodes_len += 1
      curr_nodes_len = next_nodes_len
      curr_moduli = curr_moduli[:-1]
      prev_moduli = curr_moduli

    # Precompute the final Plaintext-Ciphertext multiplication level constants
    # (used to multiply the tree root with the constant term pt0)
    self.final_mul_pt_ct.precompute_constants(curr_moduli)
    self.final_rescale.precompute_constants(
        curr_moduli, num_rescales=1, r=r, c=c
    )
    self.final_ntt.precompute_constants(curr_moduli, r, c)

    curr_moduli = curr_moduli[:-1]

    # Precompute Homomorphic Conjugation and final summation constants
    # curr_moduli is the moduli of the final ciphertext ct_out
    self.dnum = dnum
    self.dnum_conj = 1

    conj_moduli = curr_moduli + p_limbs
    conj_control_indices = mul.Mul.compute_control_indices(
        curr_moduli, p_limbs, self.dnum_conj
    )
    self.conj_bc_kernel.precompute_constants(conj_moduli, conj_control_indices)

    ext_moduli_lvl = curr_moduli + p_limbs
    self.mul_kernel_lvl.precompute_constants(ext_moduli_lvl)

    self.rescale_kernel_lvl.precompute_constants(
        ext_moduli_lvl, num_rescales=1, r=r, c=c
    )

    self.add_kernel.precompute_constants(curr_moduli)

    # Precompute sub-kernels for blind rotation

    control_indices_list = []
    q_limbs_len = len(pq_limbs) - len(p_limbs)
    extend_indices = list(range(q_limbs_len, q_limbs_len + len(p_limbs)))
    rotate_indices = list(range(q_limbs_len))
    control_indices_list.append([rotate_indices, extend_indices])

    ks_control_indices = mul.Mul.compute_control_indices(
        q_limbs, p_limbs, dnum=1
    )
    control_indices_list.extend(ks_control_indices)

    self.brot_kernel.bc_kernel.precompute_constants(
        pq_limbs, control_indices_list
    )

    self.brot_kernel.mul_kernel.precompute_constants(pq_limbs)

    self.brot_kernel.rescale_kernel.precompute_constants(
        pq_limbs, num_rescales=len(p_limbs), r=r, c=c
    )

    # Populate blind rotation main kernel fields
    self.brot_kernel.q_limbs = jnp.array(q_limbs, dtype=jnp.uint64)
    self.brot_kernel.p_limbs = jnp.array(p_limbs, dtype=jnp.uint64)
    self.brot_kernel.all_moduli = jnp.array(pq_limbs, dtype=jnp.uint64)
    self.brot_kernel.q_limbs_u64_expanded = jnp.array(
        q_limbs, dtype=jnp.uint64
    ).reshape(1, 1, -1)
    self.brot_kernel.all_moduli_u64_expanded = jnp.array(
        pq_limbs, dtype=jnp.uint64
    ).reshape(1, 1, -1)
    self.brot_kernel.ntt_q = self.ntt_q
    self.brot_kernel.ntt_p = self.ntt_p

    num_q_conj = len(curr_moduli)
    sk_sliced = types.SecretKey(sk.data[:, :num_q_conj], sk.moduli[:num_q_conj])
    self.conjk = key_gen.gen_conjugate_key(
        sk=sk_sliced,
        q_limbs=curr_moduli,
        p_limbs=p_limbs,
        dnum=self.dnum_conj,
        random_source=random_source,
    )
    conj_key1 = key_gen.gen_key1(
        sk=sk_sliced,
        q_limbs=curr_moduli,
        p_limbs=p_limbs,
        random_source=random_source,
    )

    self.conjugate_kernel.precompute_constants(
        q_limbs=curr_moduli,
        p_limbs=p_limbs,
        r=r,
        c=c,
        bc_kernel=self.conj_bc_kernel,
        mul_kernel=self.mul_kernel_lvl,
        rescale_kernel=self.rescale_kernel_lvl,
        conj_key=self.conjk,
        key1=conj_key1,
        start_control_index=1,
    )

    self.nonzero_idx = jnp.array(nonzero_idx, dtype=jnp.uint32)
    cmkeys_list = []
    mmkeys_list = []
    for idx in nonzero_idx:
      cmkey_j, mmkey_j = key_gen.gen_hybrid_key(
          sk=sk,
          j=idx,
          idx=idx,
          s_j=1,
          theta=theta,
          q_limbs=q_limbs,
          p_limbs=p_limbs,
          random_source=random_source,
      )
      cmkeys_list.append(cmkey_j)
      mmkeys_list.append(mmkey_j)
    self.mmkeys_permutations = mmkeys_list[0].permutations

    # Initialize and precompute BAT switcher
    self.bat_switcher = key_switching.BATKeySwitcher()
    self.bat_switcher.precompute_constants(
        q_limbs=q_limbs,
        p_limbs=p_limbs,
        r=r,
        c=c,
        bc_kernel=self.brot_kernel.bc_kernel,
        mul_kernel=self.brot_kernel.mul_kernel,
        rescale_kernel=self.brot_kernel.rescale_kernel,
    )
    self.brot_kernel.bat_switcher = self.bat_switcher

    # Transform Mux keys to BAT format
    mmkeys_temp = []
    for mmkey in mmkeys_list:
      mmkey_bat_keys = []
      for hmrkey0, hmrkey1 in mmkey.keys:
        # Transform key0
        ksk0 = types.EvaluationKeys(
            a=hmrkey0.key0.data[1:2],
            b=hmrkey0.key0.data[0:1],
            moduli=hmrkey0.key0.moduli,
        )
        bat_key0 = self.bat_switcher.transform_key_to_bat(ksk0, hmrkey0.key1)

        # Transform key1
        ksk1 = types.EvaluationKeys(
            a=hmrkey1.key0.data[1:2],
            b=hmrkey1.key0.data[0:1],
            moduli=hmrkey1.key0.moduli,
        )
        bat_key1 = self.bat_switcher.transform_key_to_bat(ksk1, hmrkey1.key1)

        mmkey_bat_keys.append((bat_key0, bat_key1))
      mmkeys_temp.append(mmkey_bat_keys)

    # Stack the keys once and store them
    self.batched_cmkeys = jax.tree.map(
        lambda *args: jnp.stack(args), *cmkeys_list
    )
    self.batched_mmkeys = jax.tree.map(
        lambda *args: jnp.stack(args), *mmkeys_temp
    )

  def half_bootstrap(
      self,
      ct_in: types.Ciphertext,
      theta: int,
      scale: float,
      gamma: float = 1.0,
  ) -> types.Ciphertext:
    """Executes the SHIP (half) bootstrapping algorithm."""
    q0 = ct_in.moduli[0]

    # --- Step 1 & 2: Batched INTT of b and a to get coefficients ---
    ab_reshaped = ct_in.data.reshape(2, self.r, self.c, 1)
    ab_coeffs = self.ntt_q0.intt(ab_reshaped.astype(jnp.uint32))
    b_coeffs_flat = ab_coeffs[0].reshape(self.degree)
    a_coeffs_flat = ab_coeffs[1].reshape(self.degree)

    # --- Step 3: Plaintext Preprocessing and Batched Encoding ---
    v0_slots = boot_utils.compute_v0_slots(
        b_coeffs_flat=b_coeffs_flat,
        q0=q0,
        num_slots=self.num_slots,
        gamma=gamma,
    )
    w1_a_slots, w2_a_slots = boot_utils.compute_a_slots(
        a_lower=a_coeffs_flat[: self.num_slots],
        a_upper=a_coeffs_flat[self.num_slots :],
        q0=q0,
    )

    moduli_pq = jnp.array(self.ntt_pq.constants.moduli, dtype=jnp.uint32)
    all_a_slots = jnp.stack([
        w1_a_slots,
        jnp.conj(w1_a_slots),
        w2_a_slots,
        jnp.conj(w2_a_slots),
    ])
    pts_batched = jax.vmap(
        lambda s: self.online_encoder.encode(s, scale, moduli_pq, self.ntt_pq)
    )(all_a_slots)
    pts_tuple = tuple(
        types.Plaintext(data=pts_batched.data[i], moduli=pts_batched.moduli[i])
        for i in range(4)
    )

    vmapped_brot = jax.vmap(
        lambda cmkey, mmkey: self.brot_kernel.brot_hybrid(
            pts=pts_tuple,
            cmkey_hybrid=cmkey,
            mmkey_hybrid_bat=mmkey,
            permutations=self.mmkeys_permutations,
            theta=theta,
            control_index=2,
        ),
        in_axes=(0, 0),
    )
    ct_rot_batched = vmapped_brot(self.batched_cmkeys, self.batched_mmkeys)

    # ct_rot_batched is already batched: data shape (n, 2, degree, num_moduli)
    ct_current = ct_rot_batched
    n = ct_current.data.shape[0]
    current_scale = scale

    # --- Step 5: Binary Product Tree Multiplication ---
    # The product tree reduces the rotated ciphertexts.
    for k in range(len(self.tree_kernels)):
      level_kernels = self.tree_kernels[k]

      has_odd = n % 2 == 1

      limit = n - 1 if has_odd else n

      # Reshaping instead of step-slicing (which crashes XLA on TPU)
      data_limited = ct_current.data[:limit]
      moduli_limited = ct_current.moduli[:limit]

      data_reshaped = jnp.reshape(
          data_limited, (limit // 2, 2) + data_limited.shape[1:]
      )
      moduli_reshaped = jnp.reshape(
          moduli_limited, (limit // 2, 2) + moduli_limited.shape[1:]
      )

      data_a = data_reshaped[:, 0]
      data_b = data_reshaped[:, 1]
      moduli_a = moduli_reshaped[:, 0]
      moduli_b = moduli_reshaped[:, 1]

      ct_a = types.Ciphertext(data=data_a, moduli=moduli_a)
      ct_b = types.Ciphertext(data=data_b, moduli=moduli_b)

      vmapped_mul = jax.vmap(
          lambda c1, c2: _multiply_and_relinearize_and_rescale(
              c1, c2, level_kernels
          ),
          in_axes=(0, 0),
      )

      ct_mul_batched = vmapped_mul(ct_a, ct_b)

      if has_odd:
        # Extract odd node
        ct_odd = types.Ciphertext(
            data=ct_current.data[-1], moduli=ct_current.moduli[-1]
        )
        # Rescale odd node
        scaled_data = (
            (
                ct_odd.data.astype(jnp.uint64)
                * jnp.round(current_scale).astype(jnp.uint64)
            )
            % ct_odd.moduli.astype(jnp.uint64)
        ).astype(jnp.uint32)
        ct_odd_rescaled = types.Ciphertext(
            data=scaled_data, moduli=ct_odd.moduli
        )
        ct_odd_rescaled = level_kernels.rescale_lvl.rescale(ct_odd_rescaled)

        # Concatenate rescaled odd node
        ct_odd_data_expanded = jnp.expand_dims(ct_odd_rescaled.data, axis=0)
        ct_odd_moduli_expanded = jnp.expand_dims(ct_odd_rescaled.moduli, axis=0)

        new_data = jnp.concatenate(
            [ct_mul_batched.data, ct_odd_data_expanded], axis=0
        )
        new_moduli = jnp.concatenate(
            [ct_mul_batched.moduli, ct_odd_moduli_expanded], axis=0
        )
        ct_current = types.Ciphertext(data=new_data, moduli=new_moduli)
      else:
        ct_current = ct_mul_batched

      assert level_kernels.rescale_lvl.moduli is not None
      q_dropped = level_kernels.rescale_lvl.moduli[-1]
      current_scale = (current_scale**2) / q_dropped
      n = ct_current.data.shape[0]

    ct_prod = types.Ciphertext(
        data=ct_current.data[0], moduli=ct_current.moduli[0]
    )
    ct_prod_scale = current_scale
    q_final_dropped = ct_prod.moduli[-1]
    scale_pt0 = (q_final_dropped * scale) / ct_prod_scale
    # Perform the final Plaintext-Ciphertext multiplication with pt0.

    pt0 = self.online_encoder.encode(
        slots=v0_slots,
        scale=scale_pt0,
        moduli=ct_prod.moduli,
        ntt_kernel=self.final_ntt,
    )

    ct_out = self.final_mul_pt_ct.mul(ct_prod, pt0)
    ct_out = self.final_rescale.rescale(ct_out)

    # --- Step 6: Conjugation and Final Addition ---
    # Algorithm 1, Step 22: return ct0 + Conj(ct0)
    ct_conj = self.conjugate_kernel.conjugate(ct=ct_out)
    ct_final_data = self.add_kernel.add(ct_out.data, ct_conj.data)
    ct_final = types.Ciphertext(ct_final_data, ct_out.moduli)

    return ct_final
