"""SHIP CKKS half-bootstrapping implementation."""

import dataclasses
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import add
from jaxite.jaxite_ckks import basis_conversion
from jaxite.jaxite_ckks import blind_rotate
from jaxite.jaxite_ckks import bootstrapping_utils as boot_utils
from jaxite.jaxite_ckks import conjugate
from jaxite.jaxite_ckks import key_gen
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


@jax.tree_util.register_pytree_node_class
class SHIP:
  """Kernel for SHIP CKKS half-bootstrapping."""

  r: int = 0
  c: int = 0
  degree: int = 0
  num_slots: int = 0
  ntt_q0: ntt.NTTBarrett = ntt.NTTBarrett()
  ntt_q: ntt.NTTBarrett = ntt.NTTBarrett()
  ntt_pq: ntt.NTTBarrett = ntt.NTTBarrett()
  ntt_p: ntt.NTTBarrett = ntt.NTTBarrett()
  tree_kernels: list[SHIPLevelKernels] = []
  conjugate_kernel: conjugate.Conjugation = conjugate.Conjugation()
  conj_bc_kernel: basis_conversion.BasisConversionBarrett = (
      basis_conversion.BasisConversionBarrett()
  )
  mul_kernel_lvl: mul.MulPlaintextCiphertextBarrett = (
      mul.MulPlaintextCiphertextBarrett()
  )
  rescale_kernel_lvl: rescale.Rescale = rescale.Rescale()
  add_kernel: add.AddModularSubtract = add.AddModularSubtract()
  mul_pt_ct_kernel_q: mul.MulPlaintextCiphertextBarrett = (
      mul.MulPlaintextCiphertextBarrett()
  )
  final_mul_pt_ct: mul.MulPlaintextCiphertextBarrett = (
      mul.MulPlaintextCiphertextBarrett()
  )
  final_rescale: rescale.Rescale = rescale.Rescale()
  final_ntt: ntt.NTTBarrett = ntt.NTTBarrett()
  brot_kernel: blind_rotate.BlindRotation = blind_rotate.BlindRotation()
  online_encoder: boot_utils.OnlineEncoder = boot_utils.OnlineEncoder()
  nonzero_idx: jax.Array = np.array([], dtype=np.uint32)  # pytype: disable=annotation-type-mismatch
  cmkeys: list[list[list[types.Ciphertext]]] = []
  mmkeys: list[types.MuxRotationKey] = []
  conjk: types.EvaluationKeys = types.EvaluationKeys()
  dnum: int = 0
  dnum_conj: int = 0

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
        self.cmkeys,
        self.mmkeys,
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
        obj.cmkeys,
        obj.mmkeys,
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
    self.dnum_conj = min(dnum, len(curr_moduli))

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

    # Wire key_switcher manually to share NTT kernels
    self.brot_kernel.key_switcher.ntt_kernels_q = [self.ntt_q]
    self.brot_kernel.key_switcher.ntt_kernels_out = [self.ntt_p]
    p_mod_q_val = [1] * len(q_limbs)
    for i, q in enumerate(q_limbs):
      for p in p_limbs:
        p_mod_q_val[i] = (p_mod_q_val[i] * (p % q)) % q
    self.brot_kernel.key_switcher.p_mod_q = jnp.array(
        p_mod_q_val, dtype=jnp.uint64
    )
    self.brot_kernel.key_switcher.p_limbs = jnp.array(p_limbs, dtype=jnp.uint32)
    self.brot_kernel.key_switcher.bc_kernel = self.brot_kernel.bc_kernel
    self.brot_kernel.key_switcher.mul_kernel = self.brot_kernel.mul_kernel

    num_q_conj = len(curr_moduli)
    sk_sliced = types.SecretKey(sk.data[:, :num_q_conj], sk.moduli[:num_q_conj])
    self.conjk = key_gen.gen_conjugate_key(
        sk=sk_sliced,
        q_limbs=curr_moduli,
        p_limbs=p_limbs,
        dnum=self.dnum_conj,
        random_source=random_source,
    )

    self.conjugate_kernel.precompute_constants(
        q_limbs=curr_moduli,
        p_limbs=p_limbs,
        dnum=self.dnum_conj,
        r=r,
        c=c,
        bc_kernel=self.conj_bc_kernel,
        mul_kernel=self.mul_kernel_lvl,
        rescale_kernel=self.rescale_kernel_lvl,
        conj_key=self.conjk,
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
    self.cmkeys = cmkeys_list
    self.mmkeys = mmkeys_list

  def half_bootstrap(
      self,
      ct_in: types.Ciphertext,
      theta: int,
      scale: float,
      gamma: float = 1.0,
  ) -> types.Ciphertext:
    """Executes the SHIP (half) bootstrapping algorithm."""
    q0 = ct_in.moduli[0]

    # --- Step 1: INTT of b to get coefficients ---
    b_ntt = ct_in.data[0]
    b_reshaped = b_ntt.reshape(1, self.r, self.c, 1)
    b_coeffs = self.ntt_q0.intt(b_reshaped.astype(jnp.uint32))
    b_coeffs_flat = b_coeffs.reshape(self.degree)

    # --- Step 2: INTT of a to get coefficients ---
    a_ntt = ct_in.data[1]
    a_reshaped = a_ntt.reshape(1, self.r, self.c, 1)
    a_coeffs = self.ntt_q0.intt(a_reshaped.astype(jnp.uint32))
    a_coeffs_flat = a_coeffs.reshape(self.degree)

    # --- Step 3: Plaintext Preprocessing and Encoding ---

    # Algorithm 1, Step 1: pt0 <- Ecd( 1/(4*pi) * gamma * b_i )
    # Compute roots of unity modulo q0 using exact uint32 exponents.
    v0_slots = boot_utils.compute_v0_slots(
        b_coeffs_flat=b_coeffs_flat,
        q0=q0,
        num_slots=self.num_slots,
        gamma=gamma,
    )

    # Algorithm 1, Steps 2-3: Encoding pt1..pt4 for the hybrid blind rotation
    w1_a_slots, w2_a_slots = boot_utils.compute_a_slots(
        a_lower=a_coeffs_flat[: self.num_slots],
        a_upper=a_coeffs_flat[self.num_slots :],
        q0=q0,
    )

    # NOTE: pt0 (Algorithm 1, step 1) is computed later, after the product tree.

    # Algorithm 1, Step 2: pt1 <- Ecd( w * a_i ) (first half); pt2 <- pt1(X^-1)
    moduli_pq = jnp.array(self.ntt_pq.constants.moduli, dtype=jnp.uint32)
    pt1 = self.online_encoder.encode(
        w1_a_slots,
        scale,
        moduli_pq,
        self.ntt_pq,
    )
    pt2 = self.online_encoder.encode(
        jnp.conj(w1_a_slots),
        scale,
        moduli_pq,
        self.ntt_pq,
    )
    # Algorithm 1, Step 3: pt3 <- Ecd( w * a_i ) (second half); pt4 <- pt3(X^-1)
    pt3 = self.online_encoder.encode(w2_a_slots, scale, moduli_pq, self.ntt_pq)
    pt4 = self.online_encoder.encode(
        jnp.conj(w2_a_slots),
        scale,
        moduli_pq,
        self.ntt_pq,
    )

    # Stack the list of cmkeys and mmkeys to create batched PyTrees
    batched_cmkeys = jax.tree.map(lambda *args: jnp.stack(args), *self.cmkeys)
    batched_mmkeys = jax.tree.map(lambda *args: jnp.stack(args), *self.mmkeys)

    # Vectorized execution over the batch of keys
    vmapped_brot = jax.vmap(
        lambda cmkey, mmkey: self.brot_kernel.brot_hybrid(
            pts=(pt1, pt2, pt3, pt4),
            cmkey_hybrid=cmkey,
            mmkey_hybrid=mmkey,
            theta=theta,
            control_index=2,
        ),
        in_axes=(0, 0),
    )
    ct_rot_batched = vmapped_brot(batched_cmkeys, batched_mmkeys)

    # Un-batch the results back into a list of Ciphertexts for the binary
    # product tree.
    ct_list = [
        types.Ciphertext(
            data=ct_rot_batched.data[i], moduli=ct_rot_batched.moduli[i]
        )
        for i in range(len(self.cmkeys))
    ]

    # --- Step 5: Binary Product Tree Multiplication ---
    # The product tree reduces the rotated ciphertexts.
    nodes = list(ct_list)
    n = len(nodes)
    node_scales = [scale] * n

    # Algorithm 1, Steps 12-21: Product Tree loop
    for k in range(len(self.tree_kernels)):
      level_kernels = self.tree_kernels[k]
      next_nodes = []
      next_scales = []

      # Algorithm 1, Step 14: Loop over nodes
      for j in range(n // 2):
        n1 = nodes[2 * j]
        n2 = nodes[2 * j + 1]
        s1 = node_scales[2 * j]
        s2 = node_scales[2 * j + 1]

        # Algorithm 1, Step 15: ct <- CMult(ct_2j, ct_{2j+1})
        # Homomorphic Ciphertext-Ciphertext multiplication
        ct_3elem = level_kernels.mul_cc.tensor_multiply(n1, n2)
        ct_mul = level_kernels.mul_cc.relinearize(ct_3elem, level_kernels.relk)
        # Rescale
        level_kernels.rescale_lvl.rescale(ct_mul)
        next_nodes.append(ct_mul)

        assert level_kernels.rescale_lvl.moduli is not None
        q_dropped = level_kernels.rescale_lvl.moduli[-1]
        s_out = (s1 * s2) / q_dropped
        next_scales.append(s_out)

      if n % 2 == 1:
        # Drop the last modulus of the propagated odd node to match the
        # rescaled level of the other nodes.
        # We multiply the ciphertext by the scalar `scale` to match the scale
        # of the multiplied nodes, and then rescale it.
        ct_odd = nodes[-1]
        s_odd = node_scales[-1]
        scaled_data = (
            (
                ct_odd.data.astype(jnp.uint64)
                * jnp.round(s_odd).astype(jnp.uint64)
            )
            % ct_odd.moduli.astype(jnp.uint64)
        ).astype(jnp.uint32)
        ct_odd_mul = types.Ciphertext(data=scaled_data, moduli=ct_odd.moduli)
        level_kernels.rescale_lvl.rescale(ct_odd_mul)
        next_nodes.append(ct_odd_mul)

        assert level_kernels.rescale_lvl.moduli is not None
        q_dropped = level_kernels.rescale_lvl.moduli[-1]
        s_out = (s_odd * s_odd) / q_dropped
        next_scales.append(s_out)

      # Algorithm 1, Step 20: n <- ceil(n / 2)
      nodes = next_nodes
      node_scales = next_scales
      n = len(nodes)

    ct_prod = nodes[0]
    ct_prod_scale = node_scales[0]
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
    self.final_rescale.rescale(ct_out)

    # --- Step 6: Conjugation and Final Addition ---
    # Algorithm 1, Step 22: return ct0 + Conj(ct0)
    ct_conj = self.conjugate_kernel.conjugate(ct=ct_out)
    ct_final_data = self.add_kernel.add(ct_out.data, ct_conj.data)
    ct_final = types.Ciphertext(ct_final_data, ct_out.moduli)

    return ct_final
