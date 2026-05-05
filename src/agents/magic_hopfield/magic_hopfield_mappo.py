"""MAGICHopfieldMAPPO: MAPPO agent with MAGIC+Hopfield self-context communication.

Inherits MAGICMAPPO for all communication mechanics (homogeneous stacking,
heterogeneous cross-comm, buffer shuffling).  The policy class is
MAGICHopfieldPolicyNet, which uses _CommBlockHopfieldSelf: per-agent Hopfield
prototype retrieval replaces the forced self-loop, output dim stays D (not 2D).
"""

from __future__ import annotations
import functools

import flax.linen as nn
import jax
import jax.numpy as jnp

from agents.magic.magic_mappo import MAGICMAPPO
from agents.magic_hopfield.models.policy import (
    MAGICHopfieldPolicyNet,
    _CommBlockHopfieldSelf,
)
from skrl.models.jax.categorical import _categorical


def _extract_comm_block_params(policy_state_dict) -> dict:
    inner_params = policy_state_dict.params["params"]
    return {"params": inner_params["comm_block"]}


@functools.partial(jax.jit, static_argnames=("apply_fn",))
def _jit_encode_messages(apply_fn, params, obs, gumbel_key, subkey):
    inputs = {"states": obs, "key": subkey, "gumbel_rng": gumbel_key}
    messages, extra = apply_fn(params, inputs, "encode_only")
    return messages, extra["obs_enc"]


@functools.partial(jax.jit, static_argnames=("apply_fn", "unnormalized_log_prob"))
def _jit_act_with_messages(
    apply_fn, unnormalized_log_prob, params, obs, agg_messages, gumbel_key, subkey
):
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


class MAGICHopfieldMAPPO(MAGICMAPPO):
    """MAPPO with MAGIC+Hopfield self-context communication protocol.

    Overrides _act_heterogeneous to use _CommBlockHopfieldSelf.
    The homogeneous path is inherited unchanged from MAGICMAPPO —
    it calls the shared MAGICHopfieldPolicyNet which handles everything internally.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cached_hopfield_comm_module: nn.Module | None = None
        self._jit_hopfield_comm_apply = None

    def _act_heterogeneous(self, states, **kwargs):
        """Heterogeneous path using _CommBlockHopfieldSelf (output dim D, not 2D)."""
        uid0 = self.possible_agents[0]

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
            per_agent_messages[uid] = messages
            per_agent_obs_enc[uid] = obs_enc

        num_envs = per_agent_messages[uid0].shape[0]
        all_messages = jnp.stack(
            [per_agent_messages[uid] for uid in self.possible_agents],
            axis=1,
        )  # (num_envs, N, D)

        policy0 = self.policies[uid0]
        assert isinstance(policy0, MAGICHopfieldPolicyNet)
        comm_block_params = _extract_comm_block_params(policy0.state_dict)

        with jax.default_device(policy0.device):
            policy0._c_i += 1
            gumbel_rng = jax.random.fold_in(
                policy0._c_key, policy0._c_i + 1_000_000_000
            )

        group_keys = jax.random.split(gumbel_rng, num_envs)

        magic_cfg = getattr(self, "cfg", {}).get("magic_hopfield", {})
        message_dim = int(per_agent_messages[uid0].shape[-1])
        num_comm_rounds = int(magic_cfg.get("num_comm_rounds", policy0.num_comm_rounds))
        gumbel_temperature = float(
            magic_cfg.get("gumbel_temperature", policy0.gumbel_temperature)
        )
        num_heads = int(magic_cfg.get("num_heads", policy0.num_heads))
        hopfield_num_prototypes = int(
            magic_cfg.get("hopfield_num_prototypes", policy0.hopfield_num_prototypes)
        )
        hopfield_beta = float(magic_cfg.get("hopfield_beta", policy0.hopfield_beta))
        hopfield_gate_init = float(
            magic_cfg.get("hopfield_gate_init", policy0.hopfield_gate_init)
        )

        if self._cached_hopfield_comm_module is None:
            VmappedComm = nn.vmap(
                _CommBlockHopfieldSelf,
                variable_axes={"params": None},
                split_rngs={"params": False},
                in_axes=0,
                out_axes=0,
            )
            self._cached_hopfield_comm_module = VmappedComm(
                message_dim=message_dim,
                num_comm_rounds=num_comm_rounds,
                gumbel_temperature=gumbel_temperature,
                num_heads=num_heads,
                hopfield_num_prototypes=hopfield_num_prototypes,
                hopfield_beta=hopfield_beta,
                hopfield_gate_init=hopfield_gate_init,
                name="comm_block",
            )
            self._jit_hopfield_comm_apply = jax.jit(
                self._cached_hopfield_comm_module.apply
            )

        # Stochastic rollout path
        processed_grouped, adjs_grouped = self._jit_hopfield_comm_apply(
            comm_block_params,
            all_messages,
            group_keys,
            jnp.ones(num_envs, dtype=jnp.float32),  # gumbel_scale=1.0
            jnp.full((num_envs,), gumbel_temperature),
        )
        # processed_grouped: (num_envs, N, D)
        # adjs_grouped:      (num_envs, R, N, N)

        adj_matrices = jnp.transpose(jnp.asarray(adjs_grouped), (1, 0, 2, 3))
        hard_adj = (adj_matrices[-1] > 0.5).astype(jnp.float32)
        raw_messages = all_messages

        actions: dict[str, jax.Array] = {}
        log_prob: dict[str, jax.Array] = {}
        outputs: dict[str, dict] = {}

        for i, uid in enumerate(self.possible_agents):
            policy = self.policies[uid]
            preprocessed_obs = self._state_preprocessor[uid](states[uid])
            agent_agg_messages = processed_grouped[:, i, :]  # (num_envs, D)

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
