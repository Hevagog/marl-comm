from __future__ import annotations

from typing import Any, Mapping
from collections.abc import Sequence

import flax.linen as nn

from skrl.models.jax import CategoricalMixin, Model


class PolicyNet(CategoricalMixin, Model):
    hidden_sizes: tuple = (64, 64)

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes: Sequence[int] = (64, 64),
        unnormalized_log_prob: bool = True,
        device=None,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        CategoricalMixin.__init__(self, unnormalized_log_prob)
        object.__setattr__(self, "hidden_sizes", tuple(int(h) for h in hidden_sizes))

    @nn.compact
    def __call__(
        self,
        inputs: Mapping[str, Any],
        role: str = "",
    ):
        x = inputs["states"]
        for h in self.hidden_sizes:
            x = nn.relu(nn.Dense(int(h))(x))
        return nn.Dense(int(self.num_actions))(x), {}  # type: ignore[arg-type]

    @property
    def _modules(self):
        """Dummy property to prevent skrl JAX multi-agent trainer from crashing."""
        return {}
