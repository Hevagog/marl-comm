from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, Union
from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from skrl.models.jax import CategoricalMixin, Model

_HIDDEN_GAIN = jnp.sqrt(2.0)
_OUTPUT_GAIN = 0.01


class PolicyNetGRU(CategoricalMixin, Model):
    hidden_sizes: tuple = (64, 64)
    rnn_features: int = 256  # Size of the GRU hidden state

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes: Sequence[int] = (64, 64),
        rnn_features: int = 256,
        unnormalized_log_prob: bool = True,
        device=None,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        CategoricalMixin.__init__(self, unnormalized_log_prob)
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

        rnn_state, x = nn.GRUCell(features=self.rnn_features)(rnn_state, x)

        output = nn.Dense(
            int(self.num_actions),  # type: ignore[arg-type]
            kernel_init=nn.initializers.orthogonal(scale=_OUTPUT_GAIN),
            bias_init=nn.initializers.constant(0.0),
        )(x)

        # Return the output AND the new rnn state
        return output, {"rnn": [rnn_state]}

    def act(
        self,
        inputs: Mapping[str, Union[Union[np.ndarray, jax.Array], Any]],
        role: str = "",
        params: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, Union[jax.Array, None], Mapping[str, Union[jax.Array, Any]]]:
        actions, log_prob, outputs = super().act(inputs, role, params)
        # outputs["rnn"] is automatically populated by super().act() based on __call__
        outputs["stddev"] = outputs["net_output"]
        return actions, log_prob, outputs

    @property
    def _modules(self):
        return {}
