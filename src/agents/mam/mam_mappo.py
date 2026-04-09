"""MAM-enhanced MAPPO agent with autoregressive action selection.

Extends CategoricalMAPPO to stack all agents' observations and call
the shared MAMPolicyNet, which uses Mamba-based encoder-decoder for
sequential decision making (multi-agent advantage decomposition).

During rollout: autoregressive action generation (each agent conditions
on previous agents' sampled actions via Mamba hidden states).

During training: parallel teacher-forced evaluation (standard PPO update).

References
----------
- Daniel et al. 2024 "Multi-Agent RL with Selective State-Space Models"
- Kuba et al. 2022: multi-agent advantage decomposition theorem
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


class MAMMAPPO(CategoricalMAPPO):
    """MAPPO agent with Multi-Agent Mamba policy.

    Overrides ``act`` to stack all agents' observations into a single
    batch before calling the shared policy, enabling the MAM encoder to
    process the full agent group via BiMamba and the decoder to generate
    actions autoregressively via Mamba + CrossMamba.
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
        """Stack all agents' observations and do autoregressive action selection.

        The base MAPPO.act() calls each agent's policy individually.
        By concatenating observations from all agents we get
        ``batch_size = num_agents * num_envs``, allowing the MAM
        encoder-decoder to run over the full agent group.

        An ``ar_key`` is injected into the inputs so MAMPolicyNet
        uses autoregressive decoding (sequential per-agent generation
        with Mamba hidden state propagation) rather than independent
        sampling.
        """
        uid0 = self.possible_agents[0]
        policy: Model = self.policies[uid0]  # type: ignore[assignment]

        # Stack observations from all agents: (num_agents * num_envs, obs_dim)
        stacked_obs = jnp.concatenate(
            [
                self._state_preprocessor[uid](states[uid])
                for uid in self.possible_agents
            ],
            axis=0,
        )

        # Generate autoregressive key for this timestep
        with jax.default_device(policy.device):
            policy._c_i += 1  # type: ignore[attr-defined]
            ar_key = jax.random.fold_in(policy._c_key, policy._c_i)  # type: ignore[attr-defined]

        actions_all, log_prob_all, outputs_all = policy.act(
            {"states": stacked_obs, "ar_key": ar_key},
            role="policy",
        )
        # In autoregressive mode (ar_key provided), log_prob_all is always returned
        assert log_prob_all is not None, "log_prob_all should not be None in AR mode"

        # Split results per agent
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
        """Shuffle indices at the timestep level, preserving agent groups.

        MAMPolicyNet's forward pass requires consecutive rows in the
        batch to belong to the same group of N agents (same timestep).
        This shuffle keeps same-timestep rows adjacent while randomising
        their order across mini-batches.

        The pooled buffer layout is::

            [agent_0_t0, agent_0_t1, ..., agent_{N-1}_t0, agent_{N-1}_t1, ...]

        We produce an interleaved permutation::

            [agent_0_tσ(0), agent_1_tσ(0), …, agent_0_tσ(1), agent_1_tσ(1), …]

        so each consecutive block of N rows is a valid agent group.
        """
        return self._shuffle_grouped_buffer_indices(buffer_size)
