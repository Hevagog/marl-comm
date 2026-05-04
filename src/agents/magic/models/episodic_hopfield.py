"""Episodic Hopfield encoder for MAGIC.

Replaces the LSTM/GRU cell at the §4.1 Eq.3 slot with a content-addressable
retrieval over a rolling buffer of the last T per-agent observation
encodings — i.e. a *modern* Hopfield network (Ramsauer et al. 2021 §3.3)
where the pattern matrix is the agent's own short-term episodic memory
rather than a static prototype bank.

Carry layout (flat for compatibility with the existing TBPTT(1) side-buffer):
    carry : (B, T * H) float32
    Reshaped internally to (B, T, H). Index 0 = oldest, T-1 = newest.

Update rule per step:
    h_query  = x_t                                  # (B, H)
    scores   = β · buffer · h_query                 # (B, T)
    attn     = softmax(scores, axis=-1)
    retrieved = einsum("bt,bth->bh", attn, buffer)
    h_t      = x_t + sigmoid(gate) · retrieved
    new_buffer = roll-and-append(buffer, x_t)       # drop oldest, push newest

Initialisation:
- gate ≡ 0 → sigmoid(gate) = 0.5 (NOT −2.0/sigmoid≈0.12 — the
  HopfieldSelfContext starvation pathology).
- β learnable scalar, init 1.0.
- buffer zeros at episode start. At t=0 retrieved = 0, so h_t = x_t.

The pattern matrix is non-learned (it's just stored values), but β and
gate provide the only differentiable knobs — gradients flow through the
current-step retrieval, mirroring TBPTT(1) for LSTM/GRU.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class EpisodicHopfieldEncoder(nn.Module):
    """Episodic modern-Hopfield retrieval over a rolling buffer.

    Parameters
    ----------
    hidden_size : int
        Per-slot embedding dim H.
    buffer_size : int
        Number of past steps T retained (defaults to 8).
    beta_init : float
        Initial inverse temperature for retrieval softmax.
    gate_init : float
        Initial gate logit; sigmoid(0.0)=0.5 keeps backbone and retrieval
        on equal footing from step 0.
    """

    hidden_size: int
    buffer_size: int = 8
    beta_init: float = 1.0
    gate_init: float = 0.0

    @nn.compact
    def __call__(self, carry: jax.Array, x: jax.Array):
        """One-step retrieval + buffer update.

        Parameters
        ----------
        carry : (B, T*H) flat episodic buffer (oldest..newest).
        x     : (B, in_dim) current obs encoding (projected to H internally
                so the cell can accept the policy's encoder output without
                tying ``recurrent_hidden_size`` to ``hidden_sizes[0]``).

        Returns
        -------
        new_carry : (B, T*H)
        h_t       : (B, H)
        """
        B = x.shape[0]
        H = self.hidden_size
        T = self.buffer_size
        # Project external input to the buffer's slot width — analogous to
        # the input gate inside `nn.LSTMCell` / `nn.GRUCell`.
        if x.shape[-1] != H:
            x = nn.Dense(H, name="input_proj")(x)
        buffer = carry.reshape(B, T, H)

        beta = self.param("beta", nn.initializers.constant(self.beta_init), ())
        gate = self.param("gate", nn.initializers.constant(self.gate_init), ())

        # Pre-retrieval LayerNorm (Schlag et al. 2021 §3.2). Without it the
        # query/key magnitudes drift and β does double duty (controlling both
        # noise rejection and attention sharpness). With LN, β is the only
        # knob for sharpness, and the retrieval cleanly interpolates between
        # boxcar averaging (β small) and winner-take-most retrieval (β large).
        norm = nn.LayerNorm(name="retrieval_ln")
        x_q = norm(x)
        # LayerNorm normalises over the last axis; apply to the whole
        # buffer of shape (B, T, H) — it normalises along H per slot.
        buffer_k = norm(buffer)

        # Retrieve over the *current* buffer (excludes x_t — consistent with
        # an episodic memory of strictly past observations).
        scores = beta * jnp.einsum("bth,bh->bt", buffer_k, x_q)  # (B, T)
        attn = jax.nn.softmax(scores, axis=-1)
        retrieved = jnp.einsum("bt,bth->bh", attn, buffer)  # (B, H)

        h_t = x + jax.nn.sigmoid(gate) * retrieved

        # Roll-and-append: drop the oldest slot, push x_t to the newest.
        # No int head index needed.
        new_buffer = jnp.concatenate([buffer[:, 1:, :], x[:, None, :]], axis=1)
        new_carry = new_buffer.reshape(B, T * H)
        return new_carry, h_t
