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
    gate  = sigmoid(W_gate · h_ln + b_gate)     # input-dependent gate
    h'    = LN(h + gate · gamma · m)            # gated residual update

v5 changes vs v4:
  - **Input-dependent gate**: replaces scalar gate_logit with Dense(1) projection
    from h_ln. The gate now varies per agent per timestep, providing strong
    gradient flow from the loss into the memory path. Bias initialized to
    gate_init so the initial gate value matches the v4 starting point.
  - **Diversity loss**: new method `diversity_loss()` returns mean squared cosine
    similarity between prototype pairs. Added to PPO loss to directly drive
    prototype specialization without relying on attenuated policy gradients.

References
----------
- Ramsauer et al. 2021 "Hopfield Networks is All You Need" (Fig. 5, S3)
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
        Initial bias for the gate projection. sigmoid(gate_init) sets the
        initial memory contribution fraction. Default -2.0 -> sigmoid ~ 0.12.
    activation : str
        Attention activation: "softmax" (default) or "relu" (sparse).
    use_pre_ln : bool
        If True (default), apply LayerNorm to h before computing the query.
    diversity_loss_scale : float
        Weight for the prototype diversity regularization loss.
    """

    d_model: int
    num_memories: int = 16
    beta: float = 1.5
    gamma: float = 0.25
    gate_init: float = -2.0
    activation: str = "softmax"
    use_pre_ln: bool = True
    diversity_loss_scale: float = 0.01

    def setup(self) -> None:
        # Query projection (h -> q)
        self.W_q = nn.Dense(self.d_model, use_bias=False)
        # Value projection (xi -> v), separate from key space
        self.W_v = nn.Dense(self.d_model, use_bias=False)
        if self.use_pre_ln:
            self.pre_ln = nn.LayerNorm()
        # Post-memory LayerNorm for output stability
        self.post_ln = nn.LayerNorm()

        # Input-dependent gate: Dense(1) with bias initialized to gate_init.
        # This replaces the scalar gate_logit from v4 — the gate now varies
        # per agent per timestep, giving the memory path strong gradients.
        self.gate_proj = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros,
            bias_init=lambda _key, shape, _dtype=None: jnp.full(shape, self.gate_init),
        )

        # Learnable memory patterns (num_memories, d_model)
        self.xi = self.param(
            "xi",
            nn.initializers.lecun_normal(),
            (self.num_memories, self.d_model),
        )

        # Pre-compute scaled inverse temperature: beta / sqrt(d)
        self._beta_scale = self.beta / jnp.sqrt(
            jnp.asarray(self.d_model, dtype=jnp.float32)
        )

    def _memory_keys(self) -> jax.Array:
        return self.xi * jax.lax.rsqrt(
            jnp.sum(jnp.square(self.xi), axis=-1, keepdims=True) + 1e-6
        )

    def _pre_norm(self, h: jax.Array) -> jax.Array:
        return self.pre_ln(h) if self.use_pre_ln else h

    def prepare_memory(self) -> tuple[jax.Array, jax.Array]:
        """Pre-compute keys and values (no gate — gate is now input-dependent)."""
        return (
            self._memory_keys(),
            self.W_v(self.xi),
        )

    def _scores(self, h: jax.Array, xi_keys: jax.Array) -> jax.Array:
        h_in = self._pre_norm(h)
        q = self.W_q(h_in)  # (batch, n_agent, d_model)
        return self._beta_scale * (q @ xi_keys.T)

    def _gate(self, h: jax.Array) -> jax.Array:
        """Input-dependent gating: sigmoid(W_gate @ h_ln + b_gate)."""
        h_in = self._pre_norm(h)
        return jax.nn.sigmoid(self.gate_proj(h_in))  # (batch, n_agent, 1)

    def _scores_and_gate(
        self, h: jax.Array, xi_keys: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Compute attention scores and gate in one pre-norm call."""
        h_in = self._pre_norm(h)  # compute LayerNorm once
        scores = self._beta_scale * (self.W_q(h_in) @ xi_keys.T)
        gate = jax.nn.sigmoid(self.gate_proj(h_in))
        return scores, gate

    def apply_memory(
        self,
        h: jax.Array,
        xi_keys: jax.Array,
        xi_values: jax.Array,
    ) -> jax.Array:
        scores, gate = self._scores_and_gate(h, xi_keys)

        if self.activation == "softmax":
            alpha = jax.nn.softmax(scores, axis=-1)
        elif self.activation == "relu":
            alpha = nn.relu(scores)
        else:
            raise ValueError(
                f"Unknown activation '{self.activation}'. Use 'softmax' or 'relu'."
            )

        m = alpha @ xi_values
        return self.post_ln(h + gate * self.gamma * m)

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
        return self.apply_memory(h, *self.prepare_memory())

    def diversity_loss(self) -> jax.Array:
        """Prototype diversity regularization.

        Returns mean squared cosine similarity between all pairs of prototypes.
        Minimizing this encourages prototypes to occupy distinct directions,
        preventing collapse where all queries retrieve the same pattern.
        """
        xi_norm = self.xi * jax.lax.rsqrt(
            jnp.sum(jnp.square(self.xi), axis=-1, keepdims=True) + 1e-6
        )
        sim = xi_norm @ xi_norm.T  # (K, K)
        # Upper triangle (exclude diagonal self-similarity)
        mask = jnp.triu(jnp.ones_like(sim), k=1)
        n_pairs = jnp.sum(mask)
        return jnp.sum(mask * jnp.square(sim)) / jnp.maximum(n_pairs, 1.0)

    def get_memory_attention(self, h: jax.Array) -> jax.Array:
        scores = self._scores(h, self._memory_keys())
        if self.activation == "softmax":
            return jax.nn.softmax(scores, axis=-1)
        return nn.relu(scores)
