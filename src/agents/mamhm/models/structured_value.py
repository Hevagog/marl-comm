"""Structured value network — entity-aware centralized critic.

Replaces the flat MLP critic with a structured architecture that preserves
spatial and relational inductive bias from the global state.

Architecture:
  1. Parse 1188-dim input → grid_features (12×16×6) + agent_features (4×8) + agent_id (4)
  2. Grid stream: small CNN → GlobalAvgPool → (32,) grid embedding
  3. Agent stream: shared per-agent MLP → mean-pool → (64,) team embedding
  4. Fusion: concat(grid_emb, team_emb, agent_id) → MLP → scalar value

This is a training-only change — it does NOT affect actor execution at all.

Global state layout (from warehouse_env.py):
  - grid_features: 12 × 16 × 6 = 1152 dims (cell types: shelf/treatment/goal/charger/repair/walkable)
  - agent_features: 4 × 8 = 32 dims (active/stranded/row/col/carrying/phase/locked/rescuing per agent)
  - Raw state_dim = 1184
  - Expanded with 4-dim agent_id one-hot = 1188 (appended during training by skrl)

References
----------
- Yu et al. 2021 "MAPPO" §5.2: centralized critic with full state.
- Architecture inspired by entity-based value decomposition (Rashid et al. 2020).
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp

from skrl.models.jax import DeterministicMixin, Model

_HIDDEN_GAIN = jnp.sqrt(2.0)
_OUTPUT_GAIN = 1.0


def parse_global_state(
    state: jax.Array,
    grid_h: int = 12,
    grid_w: int = 16,
    grid_channels: int = 6,
    num_agents: int = 4,
    agent_feat_dim: int = 8,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Parse the 1188-dim global state into structured components.

    Parameters
    ----------
    state : (..., 1188)
        Global state vector (raw 1184 + 4-dim agent_id one-hot).

    Returns
    -------
    grid_features : (..., grid_h, grid_w, grid_channels)
    agent_features : (..., num_agents, agent_feat_dim)
    agent_id : (..., num_agents)
    """
    # ASSUMPTION: verify against src/environments/warehouse_grid/warehouse_env.py
    # state[:1152] = grid (12×16×6), state[1152:1184] = agents (4×8), state[1184:1188] = agent_id
    grid_dim = grid_h * grid_w * grid_channels  # 1152
    agent_dim = num_agents * agent_feat_dim  # 32
    prefix = state.shape[:-1]

    grid_flat = state[..., :grid_dim]
    agent_flat = state[..., grid_dim : grid_dim + agent_dim]
    agent_id = state[..., grid_dim + agent_dim :]

    grid_features = grid_flat.reshape(*prefix, grid_h, grid_w, grid_channels)
    agent_features = agent_flat.reshape(*prefix, num_agents, agent_feat_dim)

    return grid_features, agent_features, agent_id


class StructuredValueNet(DeterministicMixin, Model):
    """Structured centralized value network with spatial/relational inductive bias.

    Same interface as MAMHMValueNet:
      __call__(inputs, role) → (value_scalar, {})
    """

    # Grid stream params
    grid_h: int = 12
    grid_w: int = 16
    grid_channels: int = 6
    grid_conv1_features: int = 16
    grid_conv2_features: int = 32

    # Agent stream params
    num_agents: int = 4
    agent_feat_dim: int = 8
    agent_mlp_dim: int = 64

    # Fusion params
    fusion_hidden: int = 128

    def __init__(
        self,
        observation_space,
        action_space,
        grid_h: int = 12,
        grid_w: int = 16,
        grid_channels: int = 6,
        grid_conv1_features: int = 16,
        grid_conv2_features: int = 32,
        num_agents: int = 4,
        agent_feat_dim: int = 8,
        agent_mlp_dim: int = 64,
        fusion_hidden: int = 128,
        device=None,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self)

        object.__setattr__(self, "grid_h", int(grid_h))
        object.__setattr__(self, "grid_w", int(grid_w))
        object.__setattr__(self, "grid_channels", int(grid_channels))
        object.__setattr__(self, "grid_conv1_features", int(grid_conv1_features))
        object.__setattr__(self, "grid_conv2_features", int(grid_conv2_features))
        object.__setattr__(self, "num_agents", int(num_agents))
        object.__setattr__(self, "agent_feat_dim", int(agent_feat_dim))
        object.__setattr__(self, "agent_mlp_dim", int(agent_mlp_dim))
        object.__setattr__(self, "fusion_hidden", int(fusion_hidden))

    @nn.compact
    def __call__(self, inputs: Mapping[str, Any], role: str = ""):
        x = inputs["states"]  # (B, 1188)

        # --- Parse global state ---
        grid_feat, agent_feat, agent_id = parse_global_state(
            x,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            grid_channels=self.grid_channels,
            num_agents=self.num_agents,
            agent_feat_dim=self.agent_feat_dim,
        )
        # grid_feat: (B, H, W, C), agent_feat: (B, N, 8), agent_id: (B, 4)

        # --- Grid stream: small 2D CNN ---
        g = nn.Conv(
            features=self.grid_conv1_features,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            name="grid_conv1",
        )(grid_feat)
        g = nn.gelu(g)

        g = nn.Conv(
            features=self.grid_conv2_features,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            name="grid_conv2",
        )(g)
        g = nn.gelu(g)

        # Global average pool: (B, H, W, 32) → (B, 32)
        grid_emb = jnp.mean(g, axis=(-3, -2))

        # --- Agent stream: shared per-agent MLP + mean pool ---
        # agent_feat: (B, N, 8) → per-agent MLP → (B, N, 64) → mean → (B, 64)
        a = nn.Dense(
            self.agent_mlp_dim,
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            name="agent_fc1",
        )(agent_feat)
        a = nn.gelu(a)
        a = nn.Dense(
            self.agent_mlp_dim,
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            name="agent_fc2",
        )(a)
        a = nn.gelu(a)

        # Mean-pool across agents
        team_emb = jnp.mean(a, axis=-2)  # (B, 64)

        # --- Fusion ---
        # concat(grid_emb[32], team_emb[64], agent_id[4]) → (100,)
        fused = jnp.concatenate([grid_emb, team_emb, agent_id], axis=-1)

        fused = nn.Dense(
            self.fusion_hidden,
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            name="fusion_fc1",
        )(fused)
        fused = nn.LayerNorm(name="fusion_ln")(fused)
        fused = nn.gelu(fused)

        fused = nn.Dense(
            self.fusion_hidden,
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            name="fusion_fc2",
        )(fused)
        fused = nn.gelu(fused)

        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(scale=_OUTPUT_GAIN),
            bias_init=nn.initializers.constant(0.0),
            name="value_head",
        )(fused)

        return value, {}

    @property
    def _modules(self):
        return {}
