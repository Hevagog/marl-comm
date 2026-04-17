"""SyncMixer policy network.

Architecture
------------
Per-agent obs o_i (i=1..n) are embedded to d_model tokens. The encoder stacks
N_BLOCKS ``_SyncMixerBlock``s; each block applies:

    x := x + SymmetricSelfAttention(LN(x))        # token-mix, order-equivariant
    x := x + MLP(LN(x))                           # channel-mix, per-token

The symmetric self-attention uses no positional encoding, making the whole
stack permutation-equivariant over agent tokens (Vaswani et al. 2017, §3.1 —
attention without position is a symmetric set operator; see also Zaheer et
al. 2017 *Deep Sets*).

After encoding, a ``HopfieldPooling`` module (Ramsauer et al. 2021 §3.3) with
``K`` learned queries produces ``K`` coordination-context vectors of width
``pool_dim`` which are flattened and broadcast-concatenated to every agent
token. The fan-in to the per-agent head is ``d_model + K * pool_dim``; this
is held below 320 (with defaults d_model=128, K=4, pool_dim=32 → 256) to
avoid the head-inflation pathology reported in
``Hopfield_Energy_Transformer_for_MAMEncoderOnly.md`` §11.

References
----------
- Vaswani et al. 2017 "Attention Is All You Need"
- Zaheer et al. 2017 "Deep Sets"
- Ramsauer et al. 2021 "Hopfield Networks is All You Need" §3.3
- Tolstikhin et al. 2021 "MLP-Mixer" (the channel-mix/token-mix decomposition)
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import orthogonal

from skrl.models.jax import CategoricalMixin, Model

from agents.shared.hopfield_blocks import HopfieldPooling

_HIDDEN_GAIN = jnp.sqrt(2.0)
_OUTPUT_GAIN = 0.01


class _SymmetricSelfAttention(nn.Module):
    d_model: int
    num_heads: int

    def setup(self) -> None:
        assert self.d_model % self.num_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        )
        self.q_proj = nn.Dense(self.d_model, kernel_init=orthogonal(1.0))
        self.k_proj = nn.Dense(self.d_model, kernel_init=orthogonal(1.0))
        self.v_proj = nn.Dense(self.d_model, kernel_init=orthogonal(1.0))
        self.out_proj = nn.Dense(self.d_model, kernel_init=orthogonal(1.0))

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (batch, n_agent, d_model)
        b, n, _ = x.shape
        h = self.num_heads
        d_head = self.d_model // h

        def split(t):
            return t.reshape(b, n, h, d_head)

        q = split(self.q_proj(x))  # (b, n, h, d_head)
        k = split(self.k_proj(x))
        v = split(self.v_proj(x))

        scale = 1.0 / jnp.sqrt(d_head)
        scores = scale * jnp.einsum("bihd,bjhd->bhij", q, k)  # (b, h, n, n)
        weights = jax.nn.softmax(scores, axis=-1)
        ctx = jnp.einsum("bhij,bjhd->bihd", weights, v)  # (b, n, h, d_head)
        ctx = ctx.reshape(b, n, self.d_model)
        return self.out_proj(ctx)


class _ChannelMLP(nn.Module):
    d_model: int
    hidden_mult: int = 4

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        hidden = self.d_model * self.hidden_mult
        x = nn.Dense(hidden, kernel_init=orthogonal(_HIDDEN_GAIN))(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model, kernel_init=orthogonal(_OUTPUT_GAIN))(x)
        return x


class _SyncMixerBlock(nn.Module):
    d_model: int
    num_heads: int
    hidden_mult: int = 4

    def setup(self) -> None:
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.attn = _SymmetricSelfAttention(
            d_model=self.d_model, num_heads=self.num_heads
        )
        self.mlp = _ChannelMLP(d_model=self.d_model, hidden_mult=self.hidden_mult)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class _Encoder(nn.Module):
    n_block: int
    d_model: int
    num_heads: int
    hidden_mult: int

    def setup(self) -> None:
        self.obs_embed = nn.Sequential(
            [
                nn.Dense(self.d_model, kernel_init=orthogonal(_HIDDEN_GAIN)),
                nn.gelu,
            ]
        )
        self.ln = nn.LayerNorm()
        self.blocks = [
            _SyncMixerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                hidden_mult=self.hidden_mult,
            )
            for _ in range(self.n_block)
        ]

    def __call__(self, obs: jax.Array) -> jax.Array:
        """obs: (batch, n_agent, obs_dim) → rep: (batch, n_agent, d_model)."""
        h = self.ln(self.obs_embed(obs))
        for blk in self.blocks:
            h = blk(h)
        return h


class _Head(nn.Module):
    hidden_size: int
    action_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(_HIDDEN_GAIN))(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        return nn.Dense(self.action_dim, kernel_init=orthogonal(_OUTPUT_GAIN))(x)


class SyncMixerPolicyNet(CategoricalMixin, Model):
    """Symmetric mixer encoder + Hopfield coordination-context + per-agent MLP head.

    The actor is decentralised in execution (per-agent head on per-agent
    token) but centralised in representation (agent tokens exchange
    information via the symmetric self-attention blocks before the head).
    """

    d_model: int = 128
    n_block: int = 2
    num_heads: int = 4
    hidden_mult: int = 4
    num_agents: int = 2
    num_pool_queries: int = 4
    pool_dim: int = 32
    pool_beta: float = 2.0

    def __init__(
        self,
        observation_space,
        action_space,
        d_model: int = 128,
        n_block: int = 2,
        num_heads: int = 4,
        hidden_mult: int = 4,
        num_agents: int = 2,
        num_pool_queries: int = 4,
        pool_dim: int = 32,
        pool_beta: float = 2.0,
        unnormalized_log_prob: bool = True,
        device=None,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        CategoricalMixin.__init__(self, unnormalized_log_prob)
        object.__setattr__(self, "d_model", int(d_model))
        object.__setattr__(self, "n_block", int(n_block))
        object.__setattr__(self, "num_heads", int(num_heads))
        object.__setattr__(self, "hidden_mult", int(hidden_mult))
        object.__setattr__(self, "num_agents", int(num_agents))
        object.__setattr__(self, "num_pool_queries", int(num_pool_queries))
        object.__setattr__(self, "pool_dim", int(pool_dim))
        object.__setattr__(self, "pool_beta", float(pool_beta))

    def setup(self) -> None:
        act_dim = int(self.num_actions)
        self._encoder = _Encoder(
            n_block=self.n_block,
            d_model=self.d_model,
            num_heads=self.num_heads,
            hidden_mult=self.hidden_mult,
        )
        self._pool_proj = nn.Dense(
            self.pool_dim, kernel_init=orthogonal(_HIDDEN_GAIN)
        )
        self._pool = HopfieldPooling(
            d_model=self.pool_dim,
            num_queries=self.num_pool_queries,
            beta=self.pool_beta,
        )
        head_hidden = self.d_model + self.num_pool_queries * self.pool_dim
        self._head = _Head(hidden_size=head_hidden, action_dim=act_dim)

    def __call__(self, inputs: Mapping[str, Any], role: str = ""):
        x = inputs["states"]  # (B, obs_dim)
        n = self.num_agents
        b = x.shape[0]
        act_dim = int(self.num_actions)

        assert b >= n and b % n == 0, (
            f"SyncMixerPolicyNet requires batch_size ({b}) divisible by "
            f"num_agents ({n})."
        )

        groups = b // n
        obs_grouped = x.reshape(groups, n, -1)

        tokens = self._encoder(obs_grouped)  # (groups, n, d_model)

        # HopfieldPooling on a down-projected token space keeps the
        # context vector small enough that the head fan-in stays bounded
        # regardless of n. The pooled vectors are order-independent.
        pool_tokens = self._pool_proj(tokens)  # (groups, n, pool_dim)
        pooled = self._pool(pool_tokens)  # (groups, K, pool_dim)
        ctx = pooled.reshape(groups, 1, -1)  # (groups, 1, K*pool_dim)
        ctx = jnp.broadcast_to(ctx, (groups, n, ctx.shape[-1]))

        head_in = jnp.concatenate([tokens, ctx], axis=-1)
        logits = self._head(head_in)  # (groups, n, act_dim)
        return logits.reshape(b, act_dim), {}

    def act(
        self,
        inputs: Mapping[str, jax.Array | Any],
        role: str = "",
        params: jax.Array | None = None,
    ):
        actions, log_prob, outputs = super().act(inputs, role, params)
        # Entropy regularisation expects logits in outputs["stddev"]; see
        # agents/mappo/models/policy.py for the same pattern.
        outputs["stddev"] = outputs["net_output"]
        outputs["mean_actions"] = jnp.argmax(
            outputs["net_output"], axis=-1, keepdims=True
        )
        return actions, log_prob, outputs

    def init_state_dict(self, role: str, inputs=None, key=None) -> None:
        if inputs is None:
            obs_dim = self.observation_space.shape[0]
            dummy = jnp.zeros((self.num_agents, obs_dim))
            inputs = {"states": dummy}
        super().init_state_dict(role, inputs, key)

    @property
    def _modules(self):
        return {}
