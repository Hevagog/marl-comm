"""MAT-enhanced MAPPO agent.

MAT keeps CommFormer's env-major grouping and timestep-preserving shuffle,
but its rollout path is now explicitly autoregressive. That distinction is
important: unlike CommFormer, MAT must pass an ``ar_key`` so the policy
samples actions sequentially in paper order.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from agents.commformer.commformer_mappo import CommFormerMAPPO


class MATAgent(CommFormerMAPPO):
    """MAT agent — CommFormerMAPPO without the communication graph.

    Overrides _shuffle_buffer_indices to add an explicit assertion that
    buffer_size is divisible by num_agents (fix for CC-B01 in memory/bugs.md).
    A buffer size that is not a multiple of N would silently truncate samples
    and produce incomplete agent groups in the mini-batch.
    """

    def act(
        self,
        states,
        timestep: int,
        timesteps: int,
    ) -> tuple[
        dict[str, jax.Array],
        dict[str, jax.Array],
        dict[str, dict[str, Any]],
    ]:
        """Autoregressive MAT rollout with env-major stacking and agent-major split."""
        uid0 = self.possible_agents[0]
        policy = self.policies[uid0]

        preprocessed = [
            self._state_preprocessor[uid](states[uid])
            for uid in self.possible_agents
        ]
        stacked_obs = jnp.stack(preprocessed, axis=1).reshape(
            -1, preprocessed[0].shape[-1]
        )

        with jax.default_device(policy.device):
            policy._c_i += 1  # type: ignore[attr-defined]
            ar_key = jax.random.fold_in(policy._c_key, policy._c_i)  # type: ignore[attr-defined]

        actions_all, log_prob_all, outputs_all = policy.act(
            {"states": stacked_obs, "ar_key": ar_key},
            role="policy",
        )
        assert log_prob_all is not None, "log_prob_all should not be None in AR mode"

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
        n = len(self.possible_agents)
        assert buffer_size % n == 0, (
            f"MATAgent._shuffle_buffer_indices: buffer_size={buffer_size} is not "
            f"divisible by num_agents={n}.  Adjust rollouts or memory size so that "
            f"rollouts * num_envs is a multiple of {n}."
        )
        return super()._shuffle_buffer_indices(buffer_size)
