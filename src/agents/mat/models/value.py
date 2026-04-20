"""MAT value network — centralized MLP critic (shared with CommFormer design).

The value network receives the global state + agent-ID one-hot, following
the MAPPO CTDE pattern (Yu et al. 2021 §5.2).  MAT's encoder is used only
in the actor; the critic keeps a simple MLP to avoid introducing transformer
complexity in the value function where it provides no proven benefit.

References
----------
- Yu et al. 2021 "MAPPO" §5.2: agent-ID appended to shared state.
- Wen et al. 2022 "MAT" §3: centralized critic, decentralized actor.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping, Sequence

import flax.linen as nn
import jax.numpy as jnp

from skrl.models.jax import DeterministicMixin, Model

_HIDDEN_GAIN = jnp.sqrt(2.0)
_OUTPUT_GAIN = 1.0


class MATValueNet(DeterministicMixin, Model):
    """Centralized MLP value network for MAT."""

    hidden_sizes: tuple = (256, 128)

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes: Sequence[int] = (256, 128),
        device=None,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self)
        object.__setattr__(self, "hidden_sizes", tuple(int(h) for h in hidden_sizes))

    @nn.compact
    def __call__(
        self,
        inputs: Mapping[str, Any],
        role: str = "",
    ):
        x = inputs["states"]
        for h in self.hidden_sizes:
            x = nn.tanh(
                nn.Dense(
                    int(h),
                    kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
                    bias_init=nn.initializers.constant(0.0),
                )(x)
            )
        return (
            nn.Dense(
                1,
                kernel_init=nn.initializers.orthogonal(scale=_OUTPUT_GAIN),
                bias_init=nn.initializers.constant(0.0),
            )(x),
            {},
        )

    @property
    def _modules(self):
        return {}
