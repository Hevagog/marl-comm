from __future__ import annotations

from typing import Any, Mapping, Sequence

import flax.linen as nn
import jax.numpy as jnp

from skrl.models.jax import DeterministicMixin, Model

_HIDDEN_GAIN = jnp.sqrt(2.0)
_OUTPUT_GAIN = 1.0


class ValueNetGRU(DeterministicMixin, Model):
    hidden_sizes: tuple = (64, 64)
    rnn_features: int = 256  # Size of the GRU hidden state

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes: Sequence[int] = (64, 64),
        rnn_features: int = 256,
        device=None,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self)
        object.__setattr__(self, "hidden_sizes", tuple(int(h) for h in hidden_sizes))
        object.__setattr__(self, "rnn_features", int(rnn_features))

    @nn.compact
    def __call__(
        self,
        inputs: Mapping[str, Any],
        role: str = "",
    ):
        x = inputs["states"]
        batch_size = x.shape[0]

        rnn_state = inputs.get("rnn", [jnp.zeros((batch_size, self.rnn_features))])[0]

        for h in self.hidden_sizes:
            x = nn.tanh(
                nn.Dense(
                    int(h),
                    kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
                    bias_init=nn.initializers.constant(0.0),
                )(x)
            )

        rnn_state, x = nn.GRUCell(
            features=self.rnn_features,
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
        )(rnn_state, x)

        output = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(scale=_OUTPUT_GAIN),
            bias_init=nn.initializers.constant(0.0),
        )(x)

        # Return the output AND the new rnn state packed in a list
        return output, {"rnn": [rnn_state]}

    @property
    def _modules(self):
        return {}
