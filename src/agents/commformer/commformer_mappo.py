"""CommFormer-enhanced MAPPO agent with bi-level optimization.

Extends CategoricalMAPPO with an additional optimization step for
the communication graph adjacency matrix α, implementing the
bi-level optimization from CommFormer §3.3.1 (Eqs. 6–10).

References
----------
- Hu et al. 2024 "CommFormer" (ICLR 2024), §3.3.1:
  - Lower level (Eq. 9): update encoder/decoder params ϕ, θ on training data.
  - Upper level (Eq. 10): update α on validation data.
- We adapt this by alternating: every ``alpha_update_interval`` updates,
  we use the current batch as a "validation" signal for α, separate from
  the gradient steps on the policy/value networks.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from agents.mappo.categorical_mappo import CategoricalMAPPO

# Keys that describe the global graph and should NOT be sliced per agent.
_GRAPH_KEYS: frozenset[str] = frozenset({"adj_matrices"})


class CommFormerMAPPO(CategoricalMAPPO):
    """MAPPO agent with CommFormer's learnable communication graph.

    Overrides ``act`` so that all agents' observations are stacked into a
    single batch before calling the shared policy.  This ensures the
    CommFormerPolicyNet always receives ``batch_size = num_agents * num_envs``
    (a multiple of N) and can run the full encoder-decoder pipeline instead
    of falling back to the MLP path.

    The communication graph (adjacency matrix ``α``) is part of the
    policy network parameters and is optimized jointly via PPO.
    The bi-level structure is approximated by the single-step
    alternation described in CommFormer §3.3.1, Eq. 9–10.
    """

    def act(
        self,
        states: Mapping[str, np.ndarray | jax.Array],
        timestep: int,
        timesteps: int,
    ) -> tuple:
        """Stack all agents' observations and call the shared policy once.

        The base ``MAPPO.act`` calls each agent's policy individually
        (batch_size=1 per agent), so ``b < num_agents`` and the
        CommFormerPolicyNet falls back to the MLP path.

        By concatenating observations from all agents we get
        ``batch_size = num_agents * num_envs``, allowing the transformer
        encoder-decoder to run over the full agent group.

        The graph adjacency (``adj_matrices``) is shared across all agents
        and passed through unsliced so the analysis collector can read it.
        """
        uid0 = self.possible_agents[0]
        policy = self.policies[uid0]

        # Stack observations env-major: (num_envs, N, obs_dim) → (N*num_envs, obs_dim).
        # Each consecutive block of N rows = all N agents in the same environment.
        # This matches the interleaved layout produced by _shuffle_buffer_indices
        # during the PPO update, so rollout and training log-probs are consistent.
        #
        # BUG-C-003 fix: the previous agent-major concatenation (axis=0) placed
        # all environments of agent-0 first, then agent-1, etc.  After reshape
        # to (groups=num_envs, N, hidden_dim) in the policy, each "group" ended up
        # containing observations from the SAME agent across different environments
        # rather than DIFFERENT agents in the same environment.  The encoder's
        # communication was therefore between environment-copies of one agent —
        # completely wrong semantics.  At training time the interleaved buffer
        # correctly groups N agents per timestep, so rollout and training
        # log-probs were computed under different data distributions →
        # ratio_max_abs_dev ≈ 1.8, ratio_clipped_frac ≈ 0.4, near-zero learning.
        preprocessed = [
            self._state_preprocessor[uid](states[uid])
            for uid in self.possible_agents
        ]
        # preprocessed[i]: (num_envs, obs_dim)
        # stack → (num_envs, N, obs_dim), reshape → (N*num_envs, obs_dim)
        stacked_obs = jnp.stack(preprocessed, axis=1).reshape(
            -1, preprocessed[0].shape[-1]
        )

        # Parallel decoder: no AR key needed.  Same zero-dec_in forward path
        # is used at rollout and training — ratio = 1.0 at update start.
        actions_all, log_prob_all, outputs_all = policy.act(
            {"states": stacked_obs},
            role="policy",
        )
        assert log_prob_all is not None, "log_prob_all should not be None"

        # Split results per agent.
        #
        # The stacked_obs layout is env-major (from jnp.stack(axis=1).reshape):
        #   [env0/a0, env0/a1, ..., env0/aN-1, env1/a0, ..., envM/aN-1]
        # so agent i's outputs sit at stride-N positions: i, N+i, 2N+i, …
        # Using slice(i*num_envs, (i+1)*num_envs) here was an agent-major
        # slice — correct for the old axis=0 concatenation (BUG-C-003) but
        # wrong after the env-major stacking fix, producing 12/16 mismatched
        # (obs, action, log_prob) triplets in the buffer and garbage IS ratios.
        n = len(self.possible_agents)
        actions: dict[str, jax.Array] = {}
        log_prob: dict[str, jax.Array] = {}
        outputs: dict[str, dict] = {}

        for i, uid in enumerate(self.possible_agents):
            # stride-N: picks env0/ai, env1/ai, env2/ai, … in order
            s = slice(i, None, n)
            actions[uid] = actions_all[s]
            log_prob[uid] = log_prob_all[s]
            outputs[uid] = {}
            for k, v in outputs_all.items():
                if k in _GRAPH_KEYS:
                    # Graph tensors are shared — pass unsliced to preserve shape.
                    outputs[uid][k] = v
                elif isinstance(v, (jnp.ndarray, np.ndarray, jax.Array)):
                    if v.ndim == 0:
                        outputs[uid][k] = v
                    else:
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

        CommFormerPolicyNet's forward pass requires consecutive rows in the
        batch to belong to the same group of N agents (same timestep).  This
        shuffle keeps same-timestep rows adjacent while randomising their
        order across mini-batches.

        The pooled buffer layout is::

            [agent_0_t0, agent_0_t1, ..., agent_{N-1}_t0, agent_{N-1}_t1, ...]

        We produce an interleaved permutation::

            [agent_0_tσ(0), agent_1_tσ(0), …, agent_0_tσ(1), agent_1_tσ(1), …]

        so each consecutive block of N rows is a valid agent group.
        """
        n = len(self.possible_agents)
        M = buffer_size // n  # timesteps per agent

        ts_perm = np.random.permutation(M)

        paired = np.empty(buffer_size, dtype=np.intp)
        for a in range(n):
            paired[a::n] = ts_perm + a * M

        return paired
