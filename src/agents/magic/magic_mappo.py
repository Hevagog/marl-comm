from __future__ import annotations

from typing import Mapping, Union

import jax
import jax.numpy as jnp
import numpy as np

from agents.mappo import CategoricalMAPPO


_COMM_KEYS: frozenset[str] = frozenset(
    {"adj_matrices", "hard_adj", "messages", "agg_messages"}
)


class MAGICMAPPO(CategoricalMAPPO):
    """MAPPO variant with MAGIC communication protocol.

    Overrides two methods to enable correct communication during both
    rollout and training:

    - ``act`` (Bug 12 fix): stacks all agents' observations into one
      batch so the communication protocol activates (requires
      ``batch_size >= num_agents`` and ``batch_size % num_agents == 0``).

    - ``_shuffle_buffer_indices`` (Bug 13 fix): shuffles at the
      *timestep* level, keeping same-timestep agent observations
      adjacent.  This preserves correct agent pairing when the policy's
      ``__call__`` reshapes to ``(B // N, N, obs_dim)`` for the
      communication protocol.
    """

    def act(
        self,
        states: Mapping[str, Union[np.ndarray, jax.Array]],
        timestep: int,
        timesteps: int,
    ) -> tuple:
        """Stack all agents' observations and call the shared policy once.

        The base ``MAPPO.act`` calls each agent's policy individually
        (batch_size=1 per agent), so ``b < num_agents`` and the
        communication block in ``MAGICPolicyNet.__call__`` falls back to
        the no-communication path.

        By concatenating observations from all agents we get
        ``batch_size = num_agents``, allowing the communication graph
        (Scheduler + Message Processor) to operate correctly.

        Communication output tensors (``adj_matrices``, ``hard_adj``,
        ``messages``, ``agg_messages``) describe the **shared graph** for
        the whole agent group.  They are passed through unsliced to every
        agent's outputs dict so the analysis collector can read them from
        any agent's slot without losing shape information.
        """
        uid0 = self.possible_agents[0]
        policy = self.policies[uid0]

        # Stack observations from all agents: (num_agents * num_envs, obs_dim)
        stacked_obs = jnp.concatenate(
            [
                self._state_preprocessor[uid](states[uid])
                for uid in self.possible_agents
            ],
            axis=0,
        )

        # Call shared policy once with all agents' observations.
        # MAGICPolicyNet.act injects the Gumbel RNG automatically.
        actions_all, log_prob_all, outputs_all = policy.act(
            {"states": stacked_obs}, role="policy"
        )

        # Split results per agent.
        # Each agent's slice is of shape (num_envs, ...), typically (1, ...).
        n = len(self.possible_agents)
        num_envs = actions_all.shape[0] // n
        actions: dict[str, jax.Array] = {}
        log_prob: dict[str, jax.Array] = {}
        outputs: dict[str, dict] = {}
        for i, uid in enumerate(self.possible_agents):
            s = slice(i * num_envs, (i + 1) * num_envs)
            actions[uid] = actions_all[s]
            log_prob[uid] = log_prob_all[s]
            outputs[uid] = {}
            for k, v in outputs_all.items():
                if k in _COMM_KEYS:
                    # Comm tensors are shared — pass through unsliced so
                    # their shape is preserved for the analysis collector.
                    outputs[uid][k] = v
                elif isinstance(v, (jnp.ndarray, np.ndarray, jax.Array)):
                    outputs[uid][k] = v[s]
                else:
                    outputs[uid][k] = v

        if not self._jax:
            actions = {uid: jax.device_get(a) for uid, a in actions.items()}
            log_prob = {uid: jax.device_get(lp) for uid, lp in log_prob.items()}

        self._current_log_prob = log_prob
        return actions, log_prob, outputs

    def _shuffle_buffer_indices(self, buffer_size: int) -> np.ndarray:
        """Shuffle at the timestep level, preserving agent pairing.

        The pooled buffer has layout::

            [agent_0_t0, agent_0_t1, ..., agent_1_t0, agent_1_t1, ...]

        where each agent block has ``M = buffer_size // num_agents``
        rows.  Row ``t`` in agent_0's block and row ``t`` in agent_1's
        block correspond to the *same* environment timestep.

        We create an interleaved permutation::

            [agent_0_tσ(0), agent_1_tσ(0), agent_0_tσ(1), agent_1_tσ(1), ...]

        so consecutive groups of ``N`` rows are from the same timestep.
        The policy's ``__call__`` reshapes to ``(B // N, N, obs_dim)``
        and each group is a valid communication graph.
        """
        n = len(self.possible_agents)
        M = buffer_size // n  # timesteps per agent

        # Permute timestep indices
        ts_perm = np.random.permutation(M)

        # Interleave: for each shuffled timestep, include all agents
        # paired[0::n] = agent_0's indices, paired[1::n] = agent_1's, etc.
        paired = np.empty(buffer_size, dtype=np.intp)
        for a in range(n):
            paired[a::n] = ts_perm + a * M

        return paired
