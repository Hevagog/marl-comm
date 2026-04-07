"""Hopfield Memory Bank for MAMHM agent.

Implements a lightweight Modern Hopfield Network memory module that augments
the MAM decoder with associative memory over learnable coordination patterns.

Architecture (single-pass scaled-dot-product attention over learnable prototypes):

    h_ln  = LayerNorm(h)                        # pre-norm (stabilises query scale)
    q     = W_q · h_ln                          # project hidden state
    K     = normalize(Xi)                       # unit-norm learnable keys
    V     = W_v · Xi                            # projected values (separate from keys)
    alpha = softmax(beta_scale · q @ K^T)
    m     = alpha @ V                           # memory retrieval
    gate  = sigmoid(gate_logit)                 # learnable scalar gate (init ≈ 0)
    h'    = h + gate · gamma · m                # gated residual update


**Learnable gate** (init near 0): At initialization, the memory patterns xi
   are random. Without a gate, gamma·m injects noise that disrupts the MAM
   backbone's learning (this caused v2's catastrophic regression). The gate
   starts near 0 (sigmoid(-3) ≈ 0.05) and opens as the patterns become useful.

**Separate W_v projection**: Per Ramsauer et al. Figure 5 (HopfieldLayer),
   keys and values should be independently parameterized. Using raw xi as both
   keys and values couples the lookup direction with the retrieved content,
   limiting representational capacity.

**Post-memory LayerNorm**: Stabilizes the output scale after the residual
   addition, preventing drift in ||h'|| that could destabilize downstream layers.

References
----------
- Ramsauer et al. 2021 "Hopfield Networks is All You Need" (Fig. 5, §3)
- Daniel et al. 2024 "Multi-Agent RL with Selective State-Space Models"
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class HopfieldMemoryBank(nn.Module):
    """Associative memory with learnable coordination patterns.

    Parameters
    ----------
    d_model : int
        Hidden dimension of input and memory patterns.
    num_memories : int
        Number of learnable memory prototypes (K).
    beta : float
        Inverse temperature multiplier on top of the 1/sqrt(d) scaling.
    gamma : float
        Maximum residual scaling factor. Actual contribution is gate * gamma * m.
    gate_init : float
        Initial logit for the learnable gate. sigmoid(gate_init) sets the
        initial memory contribution fraction. Default -3.0 → sigmoid ≈ 0.047.
    activation : str
        Attention activation: "softmax" (default) or "relu" (sparse).
    use_pre_ln : bool
        If True (default), apply LayerNorm to h before computing the query.
    """

    d_model: int
    num_memories: int = 64
    beta: float = 1.0
    gamma: float = 0.1
    gate_init: float = -3.0
    activation: str = "softmax"
    use_pre_ln: bool = True

    def setup(self) -> None:
        # Query projection (h → q)
        self.W_q = nn.Dense(self.d_model, use_bias=False)
        # Value projection (xi → v), separate from key space
        self.W_v = nn.Dense(self.d_model, use_bias=False)
        if self.use_pre_ln:
            self.pre_ln = nn.LayerNorm()
        # Post-memory LayerNorm for output stability
        self.post_ln = nn.LayerNorm()

        # Learnable memory patterns (num_memories, d_model)
        self.xi = self.param(
            "xi",
            nn.initializers.lecun_normal(),
            (self.num_memories, self.d_model),
        )

        # Learnable scalar gate — starts near 0 so memory is initially silent.
        # This prevents random-init patterns from disrupting the MAM backbone.
        self.gate_logit = self.param(
            "gate_logit",
            lambda _key, shape: jnp.full(shape, self.gate_init),
            (1,),
        )

        # Pre-compute scaled inverse temperature: beta / sqrt(d)
        self._beta_scale = self.beta / jnp.sqrt(
            jnp.asarray(self.d_model, dtype=jnp.float32)
        )

    def _scores(self, h: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute attention scores and projected values."""
        h_in = self.pre_ln(h) if self.use_pre_ln else h
        q = self.W_q(h_in)  # (batch, n_agent, d_model)

        # Unit-normalize memory patterns for stable key directions
        xi_keys = self.xi / (jnp.linalg.norm(self.xi, axis=-1, keepdims=True) + 1e-6)

        # Separate value projection (per Ramsauer Fig. 5)
        xi_values = self.W_v(self.xi)  # (num_memories, d_model)

        # matmul: (batch, n_agent, d_model) @ (d_model, K) → (batch, n_agent, K)
        scores = self._beta_scale * (q @ xi_keys.T)
        return scores, xi_values

    def __call__(self, h: jax.Array) -> jax.Array:
        """Memory-augmented forward pass.

        Parameters
        ----------
        h : (batch, n_agent, d_model)
            Hidden states from the decoder (post-CrossMamba).

        Returns
        -------
        h_out : (batch, n_agent, d_model)
            Memory-augmented hidden states with gated residual + post-LN.
        """
        scores, values = self._scores(h)

        if self.activation == "softmax":
            alpha = jax.nn.softmax(scores, axis=-1)
        elif self.activation == "relu":
            alpha = nn.relu(scores)
        else:
            raise ValueError(
                f"Unknown activation '{self.activation}'. Use 'softmax' or 'relu'."
            )

        # matmul: (batch, n_agent, K) @ (K, d_model) → (batch, n_agent, d_model)
        m = alpha @ values

        # Gated residual: gate starts near 0, grows as patterns become useful
        gate = jax.nn.sigmoid(self.gate_logit)  # scalar in (0, 1)
        h_out = h + gate * self.gamma * m

        return self.post_ln(h_out)

    def get_memory_attention(self, h: jax.Array) -> jax.Array:
        scores, _ = self._scores(h)
        if self.activation == "softmax":
            return jax.nn.softmax(scores, axis=-1)
        return nn.relu(scores)
