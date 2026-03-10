from __future__ import annotations

from typing import Any, Mapping, Sequence

import flax.linen as nn

from skrl.models.jax import DeterministicMixin, Model


class ValueNet(DeterministicMixin, Model):
    hidden_sizes: tuple = (64, 64)

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes: Sequence[int] = (64, 64),
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
            x = nn.relu(nn.Dense(int(h))(x))
        return nn.Dense(1)(x), {}

    @property
    def _modules(self):
        """Dummy property to prevent skrl JAX multi-agent trainer from crashing."""
        return {}
