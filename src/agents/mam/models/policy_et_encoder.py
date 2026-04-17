"""MAM ET Encoder-Only policy network.

Replaces BiMamba with a recurrent Energy Transformer encoder:
  obs (B, obs_dim)
    → reshape (groups, n, obs_dim)
    → Embed → x^(0) (groups, n, n_embd)
    → ETEncoderBlock (T steps) → x^(T) (groups, n, n_embd)
    → MLP head → logits (groups, n, act_dim)
    → reshape (B, act_dim)

Supports test-time compute: more ET steps at eval for refined representations.

References
----------
- Hoover et al. 2023: Energy Transformer
- Hoover et al. 2023, Theorem 1: dE/dt ≤ 0 convergence
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import orthogonal

from skrl.models.jax import CategoricalMixin, Model

from agents.shared.hopfield_blocks import ETEncoderBlock

_HIDDEN_GAIN = jnp.sqrt(2.0)
_OUTPUT_GAIN = 0.01


class _Head(nn.Module):
    n_embd: int
    action_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.n_embd, kernel_init=orthogonal(_HIDDEN_GAIN))(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        return nn.Dense(self.action_dim, kernel_init=orthogonal(_OUTPUT_GAIN))(x)


class MAMETEncoderPolicyNet(CategoricalMixin, Model):
    """ET recurrent encoder + per-agent MLP head, no BiMamba.

    The ETEncoderBlock runs T steps of energy minimization over agent
    observation tokens, combining self-attention with Hopfield associative
    memory. This restores permutation equivariance (symmetric attention)
    and provides stored coordination prototypes.

    Supports test-time compute: set num_et_steps_eval > num_et_steps
    to allow more energy descent iterations at evaluation time.

    Training and rollout both use the same parallel forward pass.
    """

    n_embd: int = 128
    num_agents: int = 2
    num_heads: int = 4
    et_beta: float = 1.0
    et_alpha: float = 0.1
    num_memories: int = 64
    num_et_steps: int = 3
    num_et_steps_eval: int = 5
    hn_activation: str = "relu"
    stop_grad_intermediate: bool = False

    def __init__(
        self,
        observation_space,
        action_space,
        n_embd: int = 128,
        num_agents: int = 2,
        num_heads: int = 4,
        et_beta: float = 1.0,
        et_alpha: float = 0.1,
        num_memories: int = 64,
        num_et_steps: int = 3,
        num_et_steps_eval: int = 5,
        hn_activation: str = "relu",
        stop_grad_intermediate: bool = False,
        unnormalized_log_prob: bool = True,
        device=None,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        CategoricalMixin.__init__(self, unnormalized_log_prob)
        object.__setattr__(self, "n_embd", int(n_embd))
        object.__setattr__(self, "num_agents", int(num_agents))
        object.__setattr__(self, "num_heads", int(num_heads))
        object.__setattr__(self, "et_beta", float(et_beta))
        object.__setattr__(self, "et_alpha", float(et_alpha))
        object.__setattr__(self, "num_memories", int(num_memories))
        object.__setattr__(self, "num_et_steps", int(num_et_steps))
        object.__setattr__(self, "num_et_steps_eval", int(num_et_steps_eval))
        object.__setattr__(self, "hn_activation", str(hn_activation))
        object.__setattr__(self, "stop_grad_intermediate", bool(stop_grad_intermediate))

    def setup(self) -> None:
        act_dim = int(self.num_actions)

        # Observation embedding (replaces BiMamba input path)
        self._obs_embed = nn.Sequential(
            [
                nn.Dense(self.n_embd, kernel_init=orthogonal(_HIDDEN_GAIN)),
                nn.gelu,
            ]
        )
        self._embed_ln = nn.LayerNorm()

        # Recurrent ET encoder
        self._et_encoder = ETEncoderBlock(
            d_model=self.n_embd,
            num_heads=self.num_heads,
            beta=self.et_beta,
            alpha=self.et_alpha,
            num_memories=self.num_memories,
            num_steps=self.num_et_steps,
            hn_activation=self.hn_activation,
            stop_grad_intermediate=self.stop_grad_intermediate,
        )

        self._head = _Head(n_embd=self.n_embd, action_dim=act_dim)

    def __call__(self, inputs: Mapping[str, Any], role: str = ""):
        x = inputs["states"]  # (B, obs_dim)
        n = self.num_agents
        b = x.shape[0]
        act_dim = int(self.num_actions)

        assert b >= n and b % n == 0, (
            f"MAMETEncoderPolicyNet requires batch_size ({b}) divisible by "
            f"num_agents ({n})."
        )

        groups = b // n
        obs_grouped = x.reshape(groups, n, -1)

        # Embed observations
        emb = self._obs_embed(obs_grouped)
        emb = self._embed_ln(emb)

        # ET encoder: T steps of energy minimization
        # Use eval steps if provided via inputs (test-time compute)
        num_steps = inputs.get("et_steps", None)
        obs_rep = self._et_encoder(emb, num_steps=num_steps)

        logits = self._head(obs_rep)  # (groups, n, act_dim)
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
