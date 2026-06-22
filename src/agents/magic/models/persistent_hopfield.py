"""Persistent Hopfield State Cell (HSC) for MAGIC.

A GRU-style recurrent cell whose candidate vector is produced by Hopfield
prototype-attention rather than an affine transform.  The carry is a compact
(B, H) state — identical footprint to GRU — but the update mechanics pull
the state toward the *nearest learned prototype*, inducing a discrete set of
coordination modes as attractor fixed-points.

Design rationale
----------------
Standard GRU: candidate = tanh(W_c [h_{t-1}, x_t])
HSC candidate: prototype retrieval over K learned attractors in R^H
    q_t     = LN(Dense_H(x_t))          # query from current input
    scores  = β · LN(P) · q_t           # (K,)  Hopfield similarity
    attn    = softmax(scores)            # (K,)  convex combination
    c_t     = attn @ P                  # (H,)  blended attractor

Update gate (GRU-style): when to accept the new attractor
    z_t = sigmoid(Dense_H([h_{t-1}, x_t]))

State update:
    h_t = (1 − z) ⊙ h_{t-1} + z ⊙ c_t

Output:
    out_t = x_t + sigmoid(gate_scalar) · h_t   # residual skip, same as EH

The cell carry is (B, H) — same as GRU — so the TBPTT(1) side-buffer in
MAGICMAPPO works unchanged (no special LSTM-tuple path needed).

Literature basis
----------------
- Ramsauer et al. 2021 "Hopfield Networks Is All You Need" (ICLR 2021)
  §3.3: modern Hopfield retrieval = one-step softmax attention.
  We use their update rule with stored patterns = learned prototypes.

- Schlag et al. 2021 "Learning Associative Inference Using Fast Weight Programmers"
  (ICLR 2021) establishes that Hopfield/Fast-Weight style cells can serve as
  recurrent units — we adopt the residual output formulation from §4.

- Cho et al. 2014 "Learning Phrase Representations using RNN Encoder-Decoder"
  GRU update-gate structure imported directly: z controls how much of the
  carry to replace vs. retain.

- Goyal et al. 2021 "Recurrent Independent Mechanisms" (ICLR 2021)
  §3 motivates discrete, competition-based slot updates as an inductive bias
  for modular coordination — the K-prototype design is the minimal version of
  this idea.

Carry layout
------------
(B, H) float32 — persistent hidden state, identical to GRU.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class HopfieldStateCell(nn.Module):
    """GRU-structured cell with Hopfield prototype-attention as candidate.

    Parameters
    ----------
    hidden_size : int
        State/output width H.
    num_prototypes : int
        Number of learned prototype vectors K (coordination modes).
    beta_init : float
        Initial inverse temperature for prototype attention softmax.
    gate_init : float
        Initial output gate logit; sigmoid(0.0) = 0.5.
    """

    hidden_size: int
    num_prototypes: int = 16
    beta_init: float = 1.0
    gate_init: float = 0.0
    freeze_prototypes: bool = False
    """When True, stop_gradient is applied to the prototype bank so the
    attractor geometry is fixed during training.  Use for the frozen_proto
    transfer condition in the altruism-persistence experiment."""

    @nn.compact
    def __call__(self, carry: jax.Array, x: jax.Array):
        """One-step state update.

        Parameters
        ----------
        carry : (B, H) — persistent state from the previous step.
        x     : (B, in_dim) — current obs_enc from the policy obs encoder.

        Returns
        -------
        new_carry : (B, H)
        h_t       : (B, H)
        """
        H = self.hidden_size
        K = self.num_prototypes

        if x.shape[-1] != H:
            x = nn.Dense(H, name="input_proj")(x)

        # Learnable prototype bank — K coordination attractors in R^H.
        # lecun_normal keeps initial norms ≈ 1 in H dims so β=1 gives
        # reasonable softmax spread from step 0.
        prototypes = self.param(
            "prototypes",
            nn.initializers.lecun_normal(),
            (K, H),
        )
        beta = self.param("beta", nn.initializers.constant(self.beta_init), ())
        gate = self.param("gate", nn.initializers.constant(self.gate_init), ())

        # Shared LayerNorm: normalises query and keys to unit-ish magnitude so
        # β controls sharpness alone (Schlag et al. 2021 §3.2).
        # Freeze attractor geometry for the frozen_proto transfer condition.
        if self.freeze_prototypes:
            prototypes = jax.lax.stop_gradient(prototypes)

        ln = nn.LayerNorm(name="retrieval_ln")
        x_q = ln(x)  # (B, H)
        proto_k = ln(prototypes)  # (K, H)

        # Hopfield prototype retrieval (Ramsauer 2021 §3.3, Eq. 6):
        # scores = β · P_normed · q_t^T
        scores = beta * (x_q @ proto_k.T) / jnp.sqrt(float(H))  # (B, K)
        attn = jax.nn.softmax(scores, axis=-1)  # (B, K)
        # Retrieve from raw (non-normalised) prototypes — retains scale info.
        candidate = attn @ prototypes  # (B, H)

        # GRU update gate (Cho et al. 2014): decide how much to snap to attractor.
        gate_input = jnp.concatenate([x, carry], axis=-1)  # (B, 2H)
        z = jax.nn.sigmoid(nn.Dense(H, name="update_gate")(gate_input))  # (B, H)

        new_carry = (1.0 - z) * carry + z * candidate  # (B, H)

        # Residual output — same convention as EpisodicHopfieldEncoder:
        # output = input + gated state, letting obs_enc pass through unobstructed
        # while the persistent state provides an additive correction.
        h_t = x + jax.nn.sigmoid(gate) * new_carry  # (B, H)

        return new_carry, h_t
