from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from agents.mappo import CategoricalMAPPO
from agents.magic.models.policy import _CommunicateBlock
from agents.magic.models.recurrent import (
    carry_batch_size,
    reset_carry_where,
    zero_carry,
)
from skrl.models.jax.categorical import _categorical


_COMM_KEYS: frozenset[str] = frozenset(
    {"adj_matrices", "hard_adj", "messages", "agg_messages"}
)


def _extract_comm_block_params(policy_state_dict) -> dict:
    """Extract the comm_block sub-parameters from a MAGICPolicyNet state_dict.

    The comm_block lives under ``params["params"]["comm_block"]`` in the
    Flax parameter tree (because ``MAGICPolicyNet`` is wrapped in
    ``CategoricalMixin`` / ``Model`` which adds an outer ``params`` key,
    and the ``nn.compact`` submodule is named ``"comm_block"``).
    """
    inner_params = policy_state_dict.params["params"]
    return {"params": inner_params["comm_block"]}


@functools.partial(jax.jit, static_argnames=("apply_fn",))
def _jit_encode_messages(apply_fn, params, obs, gumbel_key, subkey):
    """JIT'd version of MAGICPolicyNet.apply in encode_only mode."""
    inputs = {
        "states": obs,
        "key": subkey,
        "gumbel_rng": gumbel_key,
    }
    messages, extra = apply_fn(params, inputs, "encode_only")
    return messages, extra["obs_enc"]


@functools.partial(jax.jit, static_argnames=("apply_fn", "unnormalized_log_prob"))
def _jit_act_with_messages(
    apply_fn, unnormalized_log_prob, params, obs, agg_messages, gumbel_key, subkey
):
    """JIT'd version of MAGICPolicyNet.apply + _categorical for act_with_messages."""
    inputs = {
        "states": obs,
        "key": subkey,
        "gumbel_rng": gumbel_key,
        "external_messages": agg_messages,
    }
    net_output, outputs = apply_fn(params, inputs, "policy")
    actions, log_prob = _categorical(net_output, unnormalized_log_prob, None, subkey)
    outputs["net_output"] = net_output
    outputs["stddev"] = net_output
    return actions, log_prob, outputs


class MAGICMAPPO(CategoricalMAPPO):
    """MAPPO variant with MAGIC communication protocol.

    Overrides two methods to enable correct communication during both
    rollout and training:

    - ``act``: ensures the MAGIC communication block is activated correctly.
      **Homogeneous** agents: stacks all agents' observations into one batch
      so the shared policy's internal comm block runs over the full agent
      group.
      **Heterogeneous** agents: calls each agent's message encoder
      independently, stacks the fixed-size messages, runs the shared
      communication block, then feeds aggregated messages back to each
      agent's decoder + action head.

    - ``_shuffle_buffer_indices``: for the **homogeneous** (shared-policy)
      path, shuffles at the *timestep* level, keeping same-timestep agent
      observations adjacent.  For the **heterogeneous** path, falls back to
      the base class's random permutation since each agent's policy is
      updated on its own data.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Lazy-initialised cache for the heterogeneous comm block.
        # Avoids recreating nn.vmap(_CommunicateBlock) every step.
        self._cached_comm_module: nn.Module | None = None
        self._jit_comm_apply = None

        # Recurrent (LSTM/GRU) hidden-state buffer for MAGIC §4.1 Eq. 3.
        # Maintained per agent so heterogeneous and homogeneous paths share
        # the same plumbing. Lazily allocated on first `act()` call once we
        # know `num_envs` from the obs batch shape.
        # `None` if `cfg.magic.recurrent_type` is not set.
        uid0 = self.possible_agents[0]
        policy0 = self.policies[uid0]
        self._recurrent_type: str | None = getattr(policy0, "recurrent_type", None)
        # For LSTM/GRU this equals recurrent_hidden_size; for "hopfield" it is
        # T*H (rolling buffer flattened) — see MAGICPolicyNet.recurrent_carry_size.
        self._recurrent_hidden_size: int = int(
            getattr(
                policy0,
                "recurrent_carry_size",
                getattr(policy0, "recurrent_hidden_size", 0),
            )
        )
        # Live carry — propagated step-to-step at rollout, reset on done.
        self._hidden_state: dict[str, Any] = {}

        # TBPTT(1) side-buffer: stores the *input* carry h_{t-1} that the
        # cell consumed at slot t (alongside obs_t, action_t in skrl Memory).
        # Layout per agent:  GRU -> {"h": (memory_size, num_envs, hidden)},
        #                    LSTM -> {"c": ..., "h": ...}. Pool/concat across
        # agents during PPO update to mirror the standard tensor layout.
        self._rec_buffer: dict[str, dict[str, jax.Array]] = {}
        # Per-step write head, mirrors skrl Memory's index. Wraps at
        # `memory_size` exactly the same way (skrl `RandomMemory` uses a
        # ring buffer over `memory_size` slots).
        self._rec_step: int = 0
        # Snapshot of the carry that went *into* the cell at the current
        # rollout step. Set in `_act_homogeneous` immediately before
        # `policy.act()` so `record_transition` can persist it.
        self._current_input_carry: dict[str, Any] = {}

    def act(
        self,
        states: Mapping[str, np.ndarray | jax.Array],
        timestep: int,
        timesteps: int,
    ) -> tuple:
        # Compute current Gumbel temperature (linear annealing from start→end
        # over the first ``anneal_frac`` of training).  Enables dynamic
        # sharpening of the Scheduler's Gumbel-Softmax without JIT recompilation
        # (temperature is injected as a traced JAX scalar via inputs dict).
        magic_cfg = getattr(self, "cfg", {}).get("magic", {})
        t_start = float(magic_cfg.get("gumbel_temperature", 1.0))
        t_end = float(magic_cfg.get("gumbel_temperature_end", t_start))
        anneal_frac = float(magic_cfg.get("gumbel_temperature_anneal_fraction", 0.5))
        if timesteps > 0 and anneal_frac > 0.0 and t_end != t_start:
            progress = min(float(timestep) / (float(timesteps) * anneal_frac), 1.0)
            current_temp = t_start + progress * (t_end - t_start)
        else:
            current_temp = t_start

        if self._shared_policy:
            return self._act_homogeneous(states, current_temp)
        else:
            return self._act_heterogeneous(states)

    def _act_homogeneous(
        self,
        states: Mapping[str, np.ndarray | jax.Array],
        current_temp: float = 1.0,
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

        # Stack observations env-major: (num_agents * num_envs, obs_dim)
        # Layout: [a0_e0, a1_e0, ..., aN_e0, a0_e1, a1_e1, ..., aN_e(E-1)]
        # This matches the interleaved groups produced by _shuffle_buffer_indices
        # during training, so rollout log_probs align with training log_probs
        # and IS ratios stay near 1.0.  With num_envs=1 both orderings are
        # identical (no behavioural change).
        preprocessed = [
            self._state_preprocessor[uid](states[uid]) for uid in self.possible_agents
        ]
        # preprocessed[i]: (num_envs, obs_dim)
        # stack → (num_envs, num_agents, obs_dim) → reshape → (num_envs*num_agents, obs_dim)
        stacked_obs = jnp.stack(preprocessed, axis=1).reshape(
            -1, preprocessed[0].shape[-1]
        )

        # Call shared policy once with all agents' observations.
        # MAGICPolicyNet.act injects the Gumbel RNG automatically.
        # gumbel_temperature_override passes the annealed temperature without
        # triggering JIT recompilation (consumed as a traced JAX scalar in __call__).
        num_envs = preprocessed[0].shape[0]
        policy_inputs: dict[str, Any] = {
            "states": stacked_obs,
            "gumbel_temperature_override": current_temp,
        }
        if self._recurrent_type is not None:
            # Stack per-agent hidden state env-major to match `stacked_obs`
            # layout: (num_envs * num_agents, hidden_size). The agent index
            # is the inner dim, so row k = env * N + agent corresponds to
            # the obs at the same row.
            stacked_state = self._stack_hidden_state(num_envs)
            policy_inputs["recurrent_state"] = stacked_state
            # Snapshot the carry that *enters* the cell at this step. This
            # is the value we will store in the side-buffer in
            # `record_transition` and feed back during PPO update so the
            # IS ratio at training matches the rollout. Per agent: same
            # shape as live `self._hidden_state[uid]`.
            self._current_input_carry = {
                uid: self._clone_carry(self._hidden_state[uid])
                for uid in self.possible_agents
            }

        actions_all, log_prob_all, outputs_all = policy.act(
            policy_inputs,
            role="policy",
        )

        if self._recurrent_type is not None and "recurrent_state_new" in outputs_all:
            self._unstack_hidden_state(outputs_all["recurrent_state_new"], num_envs)

        # Split results per agent.
        # Env-major layout: rows i, i+n, i+2n, ... belong to agent i.
        n = len(self.possible_agents)
        actions: dict[str, jax.Array] = {}
        log_prob: dict[str, jax.Array] = {}
        outputs: dict[str, dict] = {}
        for i, uid in enumerate(self.possible_agents):
            actions[uid] = actions_all[i::n]
            log_prob[uid] = log_prob_all[i::n]
            outputs[uid] = {}
            for k, v in outputs_all.items():
                if k in _COMM_KEYS:
                    # Comm tensors are shared — pass through unsliced so
                    # their shape is preserved for the analysis collector.
                    outputs[uid][k] = v
                elif isinstance(v, (jnp.ndarray, np.ndarray, jax.Array)):
                    outputs[uid][k] = v[i::n]
                else:
                    outputs[uid][k] = v

        if not self._jax:
            actions = {uid: jax.device_get(a) for uid, a in actions.items()}
            log_prob = {uid: jax.device_get(lp) for uid, lp in log_prob.items()}

        self._current_log_prob = log_prob
        return actions, log_prob, outputs

    def _act_heterogeneous(
        self,
        states: Mapping[str, np.ndarray | jax.Array],
    ) -> tuple:
        """Generate actions for heterogeneous agents with cross-agent communication.

        Each agent has its own ``MAGICPolicyNet`` with a potentially different
        observation dimension.  Communication still works because:

        1. Each agent's message encoder maps obs → fixed-size ``message_dim``
           embedding.  This is agent-specific (different obs_dim input) but
           produces a uniform output.

        2. Messages from all agents are stacked and passed through a shared
           communication block (Scheduler + MessageProcessor from agent 0's
           policy params).  This block operates purely in message space and
           is independent of obs_dim.

        3. Aggregated messages are distributed back to each agent's policy,
           which uses them (via ``act_with_messages``) together with its own
           obs encoding to produce actions.

        This implements CTDE communication: local obs encoding is decentralised,
        but message passing is centralised across all agents.
        """
        uid0 = self.possible_agents[0]
        n = len(self.possible_agents)

        # Encode messages per agent (JIT'd)
        # Each call runs the agent-specific obs_encoder + msg_encoder.
        per_agent_messages: dict[str, jax.Array] = {}
        per_agent_obs_enc: dict[str, jax.Array] = {}
        for uid in self.possible_agents:
            policy = self.policies[uid]
            preprocessed_obs = self._state_preprocessor[uid](states[uid])
            with jax.default_device(policy.device):
                policy._c_i += 1
                subkey = jax.random.fold_in(policy._c_key, policy._c_i)
                gumbel_key = jax.random.fold_in(
                    policy._c_key, policy._c_i + 1_000_000_000
                )
            messages, obs_enc = _jit_encode_messages(
                policy.apply,
                policy.state_dict.params,
                preprocessed_obs,
                gumbel_key,
                subkey,
            )
            per_agent_messages[uid] = messages  # (num_envs, message_dim)
            per_agent_obs_enc[uid] = obs_enc  # (num_envs, hidden_size)

        # Stack messages and run shared comm block
        # All messages are (num_envs, message_dim) — same dim regardless of
        # obs_dim.  Stack to (num_envs, num_agents, message_dim) for the
        # communication block.
        num_envs = per_agent_messages[uid0].shape[0]
        # Shape: (num_envs, num_agents, message_dim)
        all_messages = jnp.stack(
            [per_agent_messages[uid] for uid in self.possible_agents],
            axis=1,
        )

        # Run the communication block using agent 0's comm_block parameters.
        # The _CommunicateBlock takes (N, message_dim) per group and we vmap
        # over groups (= num_envs here).
        policy0 = self.policies[uid0]
        comm_block_params = _extract_comm_block_params(policy0.state_dict)

        # Generate a Gumbel RNG key for the comm block.
        with jax.default_device(policy0.device):
            policy0._c_i += 1
            gumbel_rng = jax.random.fold_in(
                policy0._c_key, policy0._c_i + 1_000_000_000
            )

        # vmap over groups (num_envs) — each group has N agents.
        group_keys = jax.random.split(gumbel_rng, num_envs)

        magic_cfg = (
            getattr(self, "cfg", {}).get("magic", {}) if hasattr(self, "cfg") else {}
        )
        message_dim = int(per_agent_messages[uid0].shape[-1])
        num_comm_rounds = int(magic_cfg.get("num_comm_rounds", policy0.num_comm_rounds))
        gumbel_temperature = float(
            magic_cfg.get("gumbel_temperature", policy0.gumbel_temperature)
        )
        num_heads = int(magic_cfg.get("num_heads", policy0.num_heads))

        # Lazy-init: cache the VmappedComm module and a JIT'd apply fn
        # so we don't recreate nn.vmap(_CommunicateBlock) every step.
        if self._cached_comm_module is None:
            VmappedComm = nn.vmap(
                _CommunicateBlock,
                variable_axes={"params": None},
                split_rngs={"params": False},
                in_axes=0,
                out_axes=0,
            )
            self._cached_comm_module = VmappedComm(
                message_dim=message_dim,
                num_comm_rounds=num_comm_rounds,
                gumbel_temperature=gumbel_temperature,
                num_heads=num_heads,
                name="comm_block",
            )
            self._jit_comm_apply = jax.jit(self._cached_comm_module.apply)

        # all_messages: (num_envs, N, message_dim) — already in the right shape.
        # group_keys: (num_envs,) — one PRNG key per group.
        processed_grouped, adjs_grouped = self._jit_comm_apply(
            comm_block_params,
            all_messages,
            group_keys,
        )
        # processed_grouped: (num_envs, N, message_dim)
        # adjs_grouped: (num_envs, num_comm_rounds, N, N)

        # Reorder adjs to (num_comm_rounds, num_envs, N, N) for consistency.
        adj_matrices = jnp.transpose(jnp.asarray(adjs_grouped), (1, 0, 2, 3))
        hard_adj = (adj_matrices[-1] > 0.5).astype(jnp.float32)
        # raw messages: (num_envs, N, message_dim)
        raw_messages = all_messages

        # Distribute aggregated messages and generate actions
        actions: dict[str, jax.Array] = {}
        log_prob: dict[str, jax.Array] = {}
        outputs: dict[str, dict] = {}

        for i, uid in enumerate(self.possible_agents):
            policy = self.policies[uid]
            preprocessed_obs = self._state_preprocessor[uid](states[uid])

            # Extract this agent's aggregated messages: (num_envs, message_dim)
            agent_agg_messages = processed_grouped[:, i, :]

            with jax.default_device(policy.device):
                policy._c_i += 1
                subkey = jax.random.fold_in(policy._c_key, policy._c_i)
                gumbel_key = jax.random.fold_in(
                    policy._c_key, policy._c_i + 1_000_000_000
                )

            agent_actions, agent_log_prob, agent_outputs = _jit_act_with_messages(
                policy.apply,
                policy._c_unnormalized_log_prob,
                policy.state_dict.params,
                preprocessed_obs,
                agent_agg_messages,
                gumbel_key,
                subkey,
            )

            actions[uid] = agent_actions
            log_prob[uid] = agent_log_prob
            # Override comm tensors to be the shared ones (same for all agents).
            agent_outputs["adj_matrices"] = adj_matrices
            agent_outputs["hard_adj"] = hard_adj
            agent_outputs["messages"] = raw_messages
            agent_outputs["agg_messages"] = processed_grouped
            outputs[uid] = agent_outputs

        if not self._jax:
            actions = {uid: jax.device_get(a) for uid, a in actions.items()}
            log_prob = {uid: jax.device_get(lp) for uid, lp in log_prob.items()}

        self._current_log_prob = log_prob
        return actions, log_prob, outputs

    # ------------------------------------------------------------------
    # Recurrent (LSTM/GRU) hidden-state plumbing.
    #
    # The state is laid out env-major to mirror `_act_homogeneous`'s
    # `stacked_obs` reshape:
    #     stacked_obs.row[k] ↔ env e=k//N, agent a=k%N
    # Per-agent state buffer is `(num_envs, hidden_size)` (or a tuple for
    # LSTM); stacking produces `(num_envs * num_agents, hidden_size)`.
    # ------------------------------------------------------------------

    def _ensure_hidden_state(self, num_envs: int) -> None:
        """Allocate zero hidden state on first use or when num_envs changes."""
        if self._recurrent_type is None:
            return
        for uid in self.possible_agents:
            existing = self._hidden_state.get(uid)
            if existing is None or carry_batch_size(existing) != num_envs:
                self._hidden_state[uid] = zero_carry(
                    self._recurrent_type, num_envs, self._recurrent_hidden_size
                )

    def _stack_hidden_state(self, num_envs: int) -> Any:
        """Stack per-agent state env-major: (E, N, H) → (E*N, H)."""
        self._ensure_hidden_state(num_envs)
        per_agent = [self._hidden_state[uid] for uid in self.possible_agents]
        if self._recurrent_type == "lstm":
            cs = jnp.stack([s[0] for s in per_agent], axis=1).reshape(
                num_envs * len(self.possible_agents), -1
            )
            hs = jnp.stack([s[1] for s in per_agent], axis=1).reshape(
                num_envs * len(self.possible_agents), -1
            )
            return (cs, hs)
        # GRU
        return jnp.stack(per_agent, axis=1).reshape(
            num_envs * len(self.possible_agents), -1
        )

    def _unstack_hidden_state(self, stacked: Any, num_envs: int) -> None:
        """Inverse of `_stack_hidden_state`; writes back to `self._hidden_state`."""
        n = len(self.possible_agents)
        if self._recurrent_type == "lstm":
            cs, hs = stacked
            cs = cs.reshape(num_envs, n, -1)
            hs = hs.reshape(num_envs, n, -1)
            for i, uid in enumerate(self.possible_agents):
                self._hidden_state[uid] = (cs[:, i, :], hs[:, i, :])
        else:
            stacked = stacked.reshape(num_envs, n, -1)
            for i, uid in enumerate(self.possible_agents):
                self._hidden_state[uid] = stacked[:, i, :]

    @staticmethod
    def _clone_carry(carry: Any) -> Any:
        """JAX arrays are immutable; this is a no-op alias used to make the
        snapshot intent explicit in `_act_homogeneous`."""
        return carry

    def _ensure_rec_buffer(self, num_envs: int) -> None:
        """Allocate the TBPTT(1) side-buffer once we know `num_envs`."""
        if self._recurrent_type is None or self._rec_buffer:
            return
        memory_size = int(self.memories[self.possible_agents[0]].memory_size)
        H = self._recurrent_hidden_size
        for uid in self.possible_agents:
            if self._recurrent_type == "lstm":
                self._rec_buffer[uid] = {
                    "c": jnp.zeros((memory_size, num_envs, H), dtype=jnp.float32),
                    "h": jnp.zeros((memory_size, num_envs, H), dtype=jnp.float32),
                }
            else:
                self._rec_buffer[uid] = {
                    "h": jnp.zeros((memory_size, num_envs, H), dtype=jnp.float32),
                }

    def _store_input_carry(self) -> None:
        """Write the snapshotted *input* carry into slot `_rec_step` for each
        agent and advance the write head with skrl's ring-buffer wrap.
        """
        if self._recurrent_type is None or not self._current_input_carry:
            return
        uid0 = self.possible_agents[0]
        carry0 = self._current_input_carry[uid0]
        num_envs = carry_batch_size(carry0)
        self._ensure_rec_buffer(num_envs)
        memory_size = int(self.memories[uid0].memory_size)
        slot = self._rec_step % memory_size
        for uid in self.possible_agents:
            carry = self._current_input_carry[uid]
            buf = self._rec_buffer[uid]
            if self._recurrent_type == "lstm":
                c, h = carry
                buf["c"] = buf["c"].at[slot].set(c)
                buf["h"] = buf["h"].at[slot].set(h)
            else:
                buf["h"] = buf["h"].at[slot].set(carry)
        self._rec_step += 1

    def _sample_recurrent_minibatch(self, idx: np.ndarray) -> Any:
        """Slice the pooled side-buffer with the same `idx` the policy uses.

        Pool layout matches `_update_shared_policy`'s tensor pool: each
        agent contributes `(memory_size * num_envs, hidden)` rows in order
        `[agent_0, agent_1, ..., agent_{N-1}]`. The interleaving applied by
        `_shuffle_buffer_indices` is purely on `idx`, so identical slicing
        gives the carry that matches each obs/action row.
        """
        if self._recurrent_type is None or not self._rec_buffer:
            return None
        per_agent_h = []
        per_agent_c = []
        for uid in self.possible_agents:
            buf = self._rec_buffer[uid]
            h = buf["h"].reshape(-1, buf["h"].shape[-1])  # (M*E, H)
            per_agent_h.append(h)
            if self._recurrent_type == "lstm":
                c = buf["c"].reshape(-1, buf["c"].shape[-1])
                per_agent_c.append(c)
        pooled_h = jnp.concatenate(per_agent_h, axis=0)
        if self._recurrent_type == "lstm":
            pooled_c = jnp.concatenate(per_agent_c, axis=0)
            return (pooled_c[idx], pooled_h[idx])
        return pooled_h[idx]

    def _reset_hidden_state(self, dones_per_agent: dict[str, jax.Array]) -> None:
        """Zero rows of the hidden state where the env terminated/truncated.

        Called from `record_transition`. Mirrors the standard PyTorch
        ppo_rnn convention of resetting carries on episode boundaries
        (skrl-torch `ppo_rnn._update`).
        """
        if self._recurrent_type is None or not self._hidden_state:
            return
        for uid in self.possible_agents:
            mask = dones_per_agent.get(uid)
            if mask is None:
                continue
            mask = jnp.asarray(mask).reshape(-1).astype(bool)
            self._hidden_state[uid] = reset_carry_where(self._hidden_state[uid], mask)

    def record_transition(
        self,
        states,
        actions,
        rewards,
        next_states,
        terminated,
        truncated,
        infos,
        timestep,
        timesteps,
    ):
        super().record_transition(
            states,
            actions,
            rewards,
            next_states,
            terminated,
            truncated,
            infos,
            timestep,
            timesteps,
        )
        if self._recurrent_type is not None:
            # 1) Persist h_{t-1} (the carry that just produced this action)
            #    into the side-buffer before we reset on dones. Order
            #    matters: the reset wipes `_hidden_state` for *next* step;
            #    `_current_input_carry` was already snapshotted in
            #    `_act_homogeneous` and remains untouched.
            self._store_input_carry()
            dones = {
                uid: jnp.asarray(terminated[uid]) | jnp.asarray(truncated[uid])
                for uid in self.possible_agents
            }
            self._reset_hidden_state(dones)

    def _shuffle_buffer_indices(self, buffer_size: int) -> np.ndarray:
        """Shuffle indices for one training epoch.

        **Homogeneous** (shared policy): shuffles at the *timestep* level,
        keeping same-timestep agent observations adjacent.  The pooled
        buffer has layout::

            [agent_0_t0, agent_0_t1, ..., agent_1_t0, agent_1_t1, ...]

        where each agent block has ``M = buffer_size // num_agents`` rows.
        Row ``t`` in agent_0's block and row ``t`` in agent_1's block
        correspond to the *same* environment timestep.

        We create an interleaved permutation::

            [agent_0_tσ(0), agent_1_tσ(0), agent_0_tσ(1), agent_1_tσ(1), ...]

        so consecutive groups of ``N`` rows are from the same timestep.
        The policy's ``__call__`` reshapes to ``(B // N, N, obs_dim)``
        and each group is a valid communication graph.

        **Heterogeneous** (per-agent policies): each agent's policy is
        updated on its own data independently, so there is no need to
        preserve agent pairing.  Falls back to a plain random permutation.
        """
        if not self._shared_policy:
            # Heterogeneous: no agent-pairing constraint.
            return np.random.permutation(buffer_size)

        # Homogeneous: timestep-level shuffle preserving agent groups.
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
