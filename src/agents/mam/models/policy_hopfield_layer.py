"""MAM Encoder-Only + HopfieldLayer policy network.

Architecture:
  obs (B, obs_dim)
    → reshape (groups, n, obs_dim)
    → BiMamba encoder → h (groups, n, n_embd)
    → HopfieldLayer (residual) → h' (groups, n, n_embd)
    → MLP head → logits (groups, n, act_dim)
    → reshape (B, act_dim)

References
----------
- Ramsauer et al. 2021, §3.4: HopfieldLayer (trainable lookup)
- Daniel et al. 2024: MAM BiMamba encoder
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import orthogonal

from skrl.models.jax import CategoricalMixin, Model

from agents.shared.mamba_blocks import BiMamba
from agents.shared.hopfield_blocks import HopfieldLayer

_HIDDEN_GAIN = jnp.sqrt(2.0)
_OUTPUT_GAIN = 0.01


class _EncodeBlock(nn.Module):
    n_embd: int
    n_agent: int
    d_state: int
    d_conv: int
    delta_rank: int

    def setup(self) -> None:
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.bimamba = BiMamba(
            self.n_agent,
            self.n_embd,
            self.d_state,
            self.d_conv,
            self.delta_rank,
        )
        self.mlp = nn.Sequential(
            [
                nn.Dense(self.n_embd, kernel_init=orthogonal(_HIDDEN_GAIN)),
                nn.gelu,
                nn.Dense(self.n_embd, kernel_init=orthogonal(_OUTPUT_GAIN)),
            ]
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.bimamba(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class _Encoder(nn.Module):
    obs_dim: int
    n_block: int
    n_embd: int
    n_agent: int
    d_state: int
    d_conv: int
    delta_rank: int

    def setup(self) -> None:
        self.obs_encoder = nn.Sequential(
            [
                nn.Dense(self.n_embd, kernel_init=orthogonal(_HIDDEN_GAIN)),
                nn.gelu,
            ]
        )
        self.ln = nn.LayerNorm()
        self.blocks = nn.Sequential(
            [
                _EncodeBlock(
                    self.n_embd,
                    self.n_agent,
                    self.d_state,
                    self.d_conv,
                    self.delta_rank,
                )
                for _ in range(self.n_block)
            ]
        )

    def __call__(self, obs: jax.Array) -> jax.Array:
        """obs: (batch, n_agent, obs_dim) → rep: (batch, n_agent, n_embd)."""
        emb = self.obs_encoder(obs)
        return self.blocks(self.ln(emb))


class _Head(nn.Module):
    n_embd: int
    action_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.n_embd, kernel_init=orthogonal(_HIDDEN_GAIN))(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        return nn.Dense(self.action_dim, kernel_init=orthogonal(_OUTPUT_GAIN))(x)


class MAMHopfieldLayerPolicyNet(CategoricalMixin, Model):
    """BiMamba encoder + HopfieldLayer prototype bank + per-agent MLP head.

    The HopfieldLayer retrieves from a learned prototype bank and adds
    the result as a residual to each agent's BiMamba representation.
    This discretises the continuous representation toward canonical
    coordination modes, reducing action head variance.

    Training and rollout both use the same parallel forward pass.
    """

    n_embd: int = 128
    n_block: int = 1
    num_agents: int = 2
    d_state: int = 32
    d_conv: int = 4
    delta_rank: int = 128
    num_prototypes: int = 8
    proto_beta: float = 1.5

    def __init__(
        self,
        observation_space,
        action_space,
        n_embd: int = 128,
        n_block: int = 1,
        num_agents: int = 2,
        d_state: int = 32,
        d_conv: int = 4,
        delta_rank: int = 128,
        num_prototypes: int = 8,
        proto_beta: float = 1.5,
        unnormalized_log_prob: bool = True,
        device=None,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        CategoricalMixin.__init__(self, unnormalized_log_prob)
        object.__setattr__(self, "n_embd", int(n_embd))
        object.__setattr__(self, "n_block", int(n_block))
        object.__setattr__(self, "num_agents", int(num_agents))
        object.__setattr__(self, "d_state", int(d_state))
        object.__setattr__(self, "d_conv", int(d_conv))
        object.__setattr__(self, "delta_rank", int(delta_rank))
        object.__setattr__(self, "num_prototypes", int(num_prototypes))
        object.__setattr__(self, "proto_beta", float(proto_beta))

    def setup(self) -> None:
        obs_dim = self.observation_space.shape[0]
        act_dim = int(self.num_actions)
        self._encoder = _Encoder(
            obs_dim=obs_dim,
            n_block=self.n_block,
            n_embd=self.n_embd,
            n_agent=self.num_agents,
            d_state=self.d_state,
            d_conv=self.d_conv,
            delta_rank=self.delta_rank,
        )
        self._hopfield_layer = HopfieldLayer(
            d_model=self.n_embd,
            num_prototypes=self.num_prototypes,
            beta=self.proto_beta,
        )
        self._head = _Head(n_embd=self.n_embd, action_dim=act_dim)

    def __call__(self, inputs: Mapping[str, Any], role: str = ""):
        x = inputs["states"]  # (B, obs_dim)
        n = self.num_agents
        b = x.shape[0]
        act_dim = int(self.num_actions)

        assert b >= n and b % n == 0, (
            f"MAMHopfieldLayerPolicyNet requires batch_size ({b}) divisible by "
            f"num_agents ({n})."
        )

        groups = b // n
        obs_grouped = x.reshape(groups, n, -1)
        h = self._encoder(obs_grouped)  # (groups, n, n_embd)
        h = self._hopfield_layer(h)  # (groups, n, n_embd) — residual
        logits = self._head(h)  # (groups, n, act_dim)
        return logits.reshape(b, act_dim), {}

    def init_state_dict(self, role: str, inputs=None, key=None) -> None:
        if inputs is None:
            obs_dim = self.observation_space.shape[0]
            dummy = jnp.zeros((self.num_agents, obs_dim))
            inputs = {"states": dummy}
        super().init_state_dict(role, inputs, key)

    @property
    def _modules(self):
        return {}
