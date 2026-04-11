from __future__ import annotations

from typing import Any
from collections.abc import Mapping
from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from skrl.models.jax import CategoricalMixin, Model

# Orthogonal initialisation gains recommended by the MAPPO paper
_HIDDEN_GAIN = jnp.sqrt(2.0)
_OUTPUT_GAIN = 0.01


class PolicyNet(CategoricalMixin, Model):
    """Categorical policy that exposes logits as ``outputs["stddev"]``.

    skrl's MAPPO ``_update_policy`` computes entropy via
    ``get_entropy(outputs["stddev"])``.  The base ``CategoricalMixin`` fills
    ``stddev`` with NaN (discrete actions have no std-dev), breaking entropy
    regularisation.  This override stores the **logits** in that slot so that
    ``CategoricalMixin.get_entropy`` (which calls ``_entropy(logits)``)
    receives the correct input.

    Activations are ``tanh`` and weights are orthogonally initialised
    following Yu et al. 2021 ("The Surprising Effectiveness of PPO in
    Cooperative, Multi-Agent Games").
    """

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
            x = nn.tanh(
                nn.Dense(
                    int(h),
                    kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
                    bias_init=nn.initializers.constant(0.0),
                )(x)
            )
        return (
            nn.Dense(
                int(self.num_actions),  # type: ignore[arg-type]
                kernel_init=nn.initializers.orthogonal(scale=_OUTPUT_GAIN),
                bias_init=nn.initializers.constant(0.0),
            )(x),
            {},
        )

    def act(
        self,
        inputs: Mapping[str, np.ndarray | jax.Array | Any],
        role: str = "",
        params: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array | None, Mapping[str, jax.Array | Any]]:
        actions, log_prob, outputs = super().act(inputs, role, params)
        # Replace the NaN placeholder with actual logits so that
        # get_entropy(outputs["stddev"]) computes the correct categorical
        # entropy inside _update_policy.
        outputs["stddev"] = outputs["net_output"]
        # During training the
        # stochastic ``actions`` are used; during evaluation the trainer
        # selects ``mean_actions`` for cleaner behaviour.
        outputs["mean_actions"] = jnp.argmax(
            outputs["net_output"], axis=-1, keepdims=True
        )
        return actions, log_prob, outputs

    @property
    def _modules(self):
        """Dummy property to prevent skrl JAX multi-agent trainer from crashing."""
        return {}
