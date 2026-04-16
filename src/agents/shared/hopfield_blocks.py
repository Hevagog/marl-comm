"""Hopfield and Energy Transformer encoder modules for MAMEncoderOnly.

Used for following agents:

1. HopfieldPooling  —  symmetric aggregation via learned query patterns
2. HopfieldLayer    —  prototype-based associative memory bank
3. ETEncoderBlock   —  recurrent energy minimisation (self-attention + HN)

All modules operate on agent representations of shape (batch, n_agent, d_model).

References
----------
- Ramsauer et al. 2021 "Hopfield Networks is All You Need" (ICLR 2021)
- Hoover et al. 2023 "Energy Transformer" (NeurIPS 2023)
- Official JAX impl: github.com/bhoov/energy-transformer-jax
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class HopfieldPooling(nn.Module):
    """Symmetric aggregation of agent representations via learned queries.

    K static learned queries retrieve coordination-relevant summaries from
    the full set of agent representations using the modern Hopfield update:

        z_k = H @ softmax(β * H^T @ q_k)

    The pooled vectors {z_k} are broadcast-concatenated with each agent's
    individual representation, providing a permutation-invariant context
    signal that eliminates BiMamba's positional gradient bias.

    Parameters
    ----------
    d_model : int
        Agent representation dimension.
    num_queries : int
        Number of learned pooling queries K.
    beta : float
        Inverse temperature for softmax retrieval.

    References
    ----------
    - Ramsauer et al. 2021, §3.3: HopfieldPooling module
    - Ramsauer et al. 2021, §5.1: Immune repertoire classification
    """

    d_model: int
    num_queries: int = 4
    beta: float = 1.0

    def setup(self) -> None:
        # Learned query patterns: (K, d_model)
        self.queries = self.param(
            "queries",
            nn.initializers.lecun_normal(),
            (self.num_queries, self.d_model),
        )

    def __call__(self, h: jax.Array) -> jax.Array:
        """Pool agent representations into K summary vectors.

        Parameters
        ----------
        h : (batch, n_agent, d_model)
            Per-agent representations from the encoder.

        Returns
        -------
        pooled : (batch, num_queries, d_model)
            K pooled coordination summaries.
        """
        # scores: (batch, K, n_agent) = β * queries @ h^T
        scores = self.beta * jnp.einsum("kd,bnd->bkn", self.queries, h)
        weights = jax.nn.softmax(scores, axis=-1)  # softmax over agents
        # pooled: (batch, K, d_model) = weights @ h
        return jnp.einsum("bkn,bnd->bkd", weights, h)


class HopfieldLayer(nn.Module):
    """Content-addressable memory bank with learned coordination prototypes.

    Each agent's representation queries a bank of P learned prototypes via:

        h_tilde_i = Xi @ softmax(β * Xi^T @ h_i)

    The retrieved prototype combination is added as a residual to the
    original representation, discretising the continuous BiMamba output
    toward one of P canonical coordination modes.

    Parameters
    ----------
    d_model : int
        Agent representation dimension.
    num_prototypes : int
        Number of stored prototype patterns P.
    beta : float
        Inverse temperature controlling retrieval sharpness.

    References
    ----------
    - Ramsauer et al. 2021, §3.4: HopfieldLayer (trainable lookup)
    """

    d_model: int
    num_prototypes: int = 8
    beta: float = 1.5

    def setup(self) -> None:
        # Prototype bank: (P, d_model) — unit-normalized at forward time
        self.prototypes = self.param(
            "prototypes",
            nn.initializers.lecun_normal(),
            (self.num_prototypes, self.d_model),
        )

    def __call__(self, h: jax.Array) -> jax.Array:
        """Retrieve from prototype bank and add as residual.

        Parameters
        ----------
        h : (batch, n_agent, d_model)
            Per-agent representations.

        Returns
        -------
        h_out : (batch, n_agent, d_model)
            Representations augmented with prototype retrieval (residual).
        """
        # Unit-normalize prototypes for stable retrieval
        xi = self.prototypes / (
            jnp.linalg.norm(self.prototypes, axis=-1, keepdims=True) + 1e-6
        )

        # similarities: (batch, n_agent, P) = h @ xi^T
        sims = self.beta * jnp.einsum("bnd,pd->bnp", h, xi)
        weights = jax.nn.softmax(sims, axis=-1)  # softmax over prototypes

        # retrieved: (batch, n_agent, d_model) = weights @ xi
        retrieved = jnp.einsum("bnp,pd->bnd", weights, xi)

        return h + retrieved


class _EnergySelfAttention(nn.Module):
    """Multi-head energy self-attention for the ET encoder.

    Computes the negative gradient of the self-attention energy:

        E_ATT = -(1/β) Σ_h Σ_i log(Σ_j exp(β K_j^T Q_i))

    Unlike the cross-attention in ETCrossBlock (decoder), here Q and K
    are both derived from the same agent token sequence (self-attention).

    References
    ----------
    - Hoover et al. 2023, §3.1: Energy attention
    - bhoov/energy-transformer-jax: architecture.py
    """

    d_model: int
    num_heads: int
    beta: float = 1.0

    def setup(self) -> None:
        head_dim = self.d_model // self.num_heads
        total = self.num_heads * head_dim
        self.W_Q = nn.Dense(total, use_bias=False)
        self.W_K = nn.Dense(total, use_bias=False)
        self.out_proj = nn.Dense(self.d_model, use_bias=False)

    def __call__(self, g: jax.Array) -> jax.Array:
        """Compute negative energy gradient for self-attention.

        Parameters
        ----------
        g : (batch, n_agent, d_model)
            LayerNorm'd agent representations.

        Returns
        -------
        update : (batch, n_agent, d_model)
            Negative attention energy gradient.
        """
        head_dim = self.d_model // self.num_heads
        B, N = g.shape[0], g.shape[1]
        H = self.num_heads

        Q = self.W_Q(g).reshape(B, N, H, head_dim)
        K = self.W_K(g).reshape(B, N, H, head_dim)

        # scores[b, i, h, j] = β * K[b,j,h,:] · Q[b,i,h,:] / sqrt(head_dim)
        scale = 1.0 / jnp.sqrt(jnp.asarray(head_dim, dtype=Q.dtype))
        scores = self.beta * scale * jnp.einsum("bjhd,bihd->bihj", K, Q)
        weights = jax.nn.softmax(scores, axis=-1)  # softmax over j (keys)

        # Weighted sum of keys: (B, N, H, head_dim)
        weighted_K = jnp.einsum("bihj,bjhd->bihd", weights, K)

        update = weighted_K.reshape(B, N, H * head_dim)
        return self.out_proj(update)


class _HopfieldMemory(nn.Module):
    """Hopfield associative memory for the ET encoder.

    Stores M learnable memory patterns and computes the negative gradient
    of the Hopfield energy:

        E_HN = -Σ_i Σ_μ G(ξ_μ^T g_i)

    where G is the integral of activation r.

    References
    ----------
    - Hoover et al. 2023, §3.2: Hopfield energy
    - Ramsauer et al. 2021: Modern Hopfield update
    """

    d_model: int
    num_memories: int = 64
    activation: str = "relu"

    def setup(self) -> None:
        self.xi = self.param(
            "xi",
            nn.initializers.lecun_normal(),
            (self.num_memories, self.d_model),
        )

    def __call__(self, g: jax.Array) -> jax.Array:
        """Compute negative Hopfield energy gradient.

        Parameters
        ----------
        g : (batch, n_agent, d_model)
            LayerNorm'd agent representations.

        Returns
        -------
        update : (batch, n_agent, d_model)
            Negative Hopfield energy gradient.
        """
        # Unit-normalize memories
        xi_norm = self.xi / (jnp.linalg.norm(self.xi, axis=-1, keepdims=True) + 1e-6)

        # similarities: (batch, n_agent, M)
        sims = jnp.einsum("bnd,md->bnm", g, xi_norm)

        if self.activation == "relu":
            acts = nn.relu(sims)
        elif self.activation == "softmax":
            acts = jax.nn.softmax(sims, axis=-1)
        else:
            raise ValueError(
                f"Unknown HN activation '{self.activation}'. Use 'relu' or 'softmax'."
            )

        # update: (batch, n_agent, d_model) = acts @ xi_norm
        return jnp.einsum("bnm,md->bnd", acts, xi_norm)


class ETEncoderBlock(nn.Module):
    """Recurrent Energy Transformer encoder block.

    Replaces BiMamba with a recurrent energy-minimization loop:

        x^(0) = Embed(o_1, ..., o_n)
        for t in range(T):
            g = LayerNorm(x)
            x = x + α * (-∂E_ATT/∂g + -∂E_HN/∂g)

    Supports test-time compute: num_steps can be increased at eval time
    for more refined representations (§6.4).

    Parameters
    ----------
    d_model : int
        Token/agent representation dimension.
    num_heads : int
        Number of self-attention heads.
    beta : float
        Inverse temperature for attention softmax.
    alpha : float
        Energy gradient step size.
    num_memories : int
        Number of Hopfield memory patterns.
    num_steps : int
        Number of energy minimization iterations (training default).
    hn_activation : str
        Hopfield activation: 'relu' or 'softmax'.

    References
    ----------
    - Hoover et al. 2023: Energy Transformer architecture
    - Hoover et al. 2023, Theorem 1: dE/dt ≤ 0 convergence guarantee
    """

    d_model: int
    num_heads: int = 4
    beta: float = 1.0
    alpha: float = 0.1
    num_memories: int = 64
    num_steps: int = 3
    hn_activation: str = "relu"

    def setup(self) -> None:
        self.energy_attn = _EnergySelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            beta=self.beta,
        )
        self.hopfield = _HopfieldMemory(
            d_model=self.d_model,
            num_memories=self.num_memories,
            activation=self.hn_activation,
        )
        self.ln = nn.LayerNorm()

    def __call__(self, x: jax.Array, num_steps: int | None = None) -> jax.Array:
        """Run T steps of energy minimization.

        Parameters
        ----------
        x : (batch, n_agent, d_model)
            Initial agent token representations.
        num_steps : int, optional
            Override number of ET steps (for test-time compute).
            Defaults to self.num_steps.

        Returns
        -------
        x : (batch, n_agent, d_model)
            Refined agent representations at the energy minimum.
        """
        T = num_steps if num_steps is not None else self.num_steps
        for _ in range(T):
            g = self.ln(x)
            attn_update = self.energy_attn(g)
            hn_update = self.hopfield(g)
            x = x + self.alpha * (attn_update + hn_update)
        return x

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute the total energy for monitoring convergence.

        This is a diagnostic method — not used during training.

        Parameters
        ----------
        x : (batch, n_agent, d_model)

        Returns
        -------
        total_energy : scalar
        """
        g = self.ln(x)

        # Attention energy: -1/β * Σ_i log(Σ_j exp(β * K_j^T Q_i))
        head_dim = self.d_model // self.num_heads
        B, N = g.shape[0], g.shape[1]
        H = self.num_heads

        Q = self.energy_attn.W_Q(g).reshape(B, N, H, head_dim)
        K = self.energy_attn.W_K(g).reshape(B, N, H, head_dim)
        scale = 1.0 / jnp.sqrt(jnp.asarray(head_dim, dtype=Q.dtype))
        scores = self.beta * scale * jnp.einsum("bjhd,bihd->bihj", K, Q)
        e_attn = -(1.0 / self.beta) * jax.nn.logsumexp(scores, axis=-1).sum()

        # Hopfield energy: -Σ_i Σ_μ G(ξ_μ^T g_i) where G = ReLU^2/2
        xi_norm = self.hopfield.xi / (
            jnp.linalg.norm(self.hopfield.xi, axis=-1, keepdims=True) + 1e-6
        )
        sims = jnp.einsum("bnd,md->bnm", g, xi_norm)
        if self.hopfield.activation == "relu":
            e_hn = -0.5 * nn.relu(sims).sum()
        else:
            e_hn = -0.5 * jax.nn.softmax(sims, axis=-1).sum()

        return e_attn + e_hn
