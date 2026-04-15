"""MAM encoder-only MAPPO agent (Ablations 1 & 2).

Stacks all agents' observations and calls MAMEncoderOnlyPolicyNet.
No autoregressive decode — all agents are processed in parallel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from agents.mappo.categorical_mappo import CategoricalMAPPO

if TYPE_CHECKING:
    from skrl.models.jax import Model


class MAMEncOnlyMAPPO(CategoricalMAPPO):
    """MAPPO with BiMamba encoder-only policy.

    Overrides ``act`` to stack all agents' observations before calling the
    shared policy, exactly like MAMMAPPO but without injecting ``ar_key``.
    The policy returns independent per-agent logits after BiMamba encoding.
    """

    def act(
        self,
        states: Mapping[str, np.ndarray | jax.Array],
        timestep: int,
        timesteps: int,
    ) -> tuple[
        dict[str, jax.Array],
        dict[str, jax.Array],
        dict[str, dict[str, Any]],
    ]:
        uid0 = self.possible_agents[0]
        policy: Model = self.policies[uid0]  # type: ignore[assignment]

        stacked_obs = jnp.concatenate(
            [
                self._state_preprocessor[uid](states[uid])
                for uid in self.possible_agents
            ],
            axis=0,
        )

        actions_all, log_prob_all, outputs_all = policy.act(
            {"states": stacked_obs},
            role="policy",
        )

        n = len(self.possible_agents)
        num_envs = stacked_obs.shape[0] // n
        actions: dict[str, jax.Array] = {}
        log_prob: dict[str, jax.Array] = {}
        outputs: dict[str, dict] = {}

        for i, uid in enumerate(self.possible_agents):
            s = slice(i * num_envs, (i + 1) * num_envs)
            actions[uid] = actions_all[s]
            log_prob[uid] = log_prob_all[s]
            outputs[uid] = {}
            for k, v in outputs_all.items():
                if isinstance(v, (jnp.ndarray, np.ndarray, jax.Array)) and v.ndim >= 1:
                    outputs[uid][k] = v[s]
                else:
                    outputs[uid][k] = v

        if not self._jax:
            actions = {uid: jax.device_get(a) for uid, a in actions.items()}
            log_prob = {uid: jax.device_get(lp) for uid, lp in log_prob.items()}

        self._current_log_prob = log_prob
        return actions, log_prob, outputs

    def _shuffle_buffer_indices(self, buffer_size: int) -> np.ndarray:
        """Same group-preserving shuffle as MAMMAPPO."""
        n = len(self.possible_agents)
        assert buffer_size % n == 0, (
            f"Enc-only pooled buffer ({buffer_size}) not divisible by num_agents ({n})."
        )
        M = buffer_size // n
        ts_perm = np.random.permutation(M)
        paired = np.empty(buffer_size, dtype=np.intp)
        for a in range(n):
            paired[a::n] = ts_perm + a * M
        return paired
