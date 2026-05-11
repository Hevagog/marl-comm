"""Factory function for MAGIC models.

Creates policy and value networks for all agents, following the same
pattern as ``create_mappo_models`` but using MAGIC's communication-
enhanced policy.
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from agents.magic.models.policy import MAGICPolicyNet
from agents.magic.models.value import MAGICValueNet


def _obs_dim(space) -> int:
    return int(np.prod(space.shape))


def _spaces_homogeneous(spaces: dict[str, Any]) -> bool:
    dims = [_obs_dim(s) for s in spaces.values()]
    return len(set(dims)) == 1


def _actions_homogeneous(spaces: dict[str, Any]) -> bool:
    if not spaces:
        return True
    first = next(iter(spaces.values()))
    for s in spaces.values():
        if type(s) != type(first):
            return False
        if hasattr(s, "n") and s.n != first.n:
            return False
        if hasattr(s, "shape") and s.shape != first.shape:
            return False
    return True


def create_magic_models(
    possible_agents: list[str],
    observation_spaces: dict[str, Any],
    action_spaces: dict[str, Any],
    shared_observation_spaces: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Instantiate MAGIC policy and value networks for all agents.

    **Homogeneous agents** (all agents share the same observation shape and
    action space): a single shared ``MAGICPolicyNet`` is created and reused
    for every agent.  During ``MAGICMAPPO.act``, all agents' observations are
    stacked into one batch, which activates the full MAGIC communication
    protocol (Scheduler + MessageProcessor).

    **Heterogeneous agents** (different observation shapes or action spaces):
    a separate ``MAGICPolicyNet`` is created per agent so that each network's
    observation encoder has the correct input dimension.  The communication
    protocol still operates: during ``MAGICMAPPO.act``, each agent's
    message encoder produces a fixed-size embedding (``message_dim``), these
    embeddings are gathered, the *first* agent's shared Scheduler +
    MessageProcessor is applied to them, and the aggregated messages are fed
    back to each agent's decoder + action head.  Architecturally this matches
    MAGIC §4 — the communication layers are shared across agents (parameter
    sharing in message space), while the obs-to-message encoder and the
    action head may differ.

    The value network receives the shared state + one-hot agent-ID
    (same as MAPPO, Yu et al. 2021 §5.2).
    """
    policy_cfg = cfg.get("policy", {})
    value_cfg = cfg.get("value", {})
    magic_cfg = cfg.get("magic", {})

    hidden_sizes_policy: list[int] = policy_cfg.get("hidden_sizes", [128, 128])
    unnormalized_log_prob: bool = policy_cfg.get("unnormalized_log_prob", True)
    hidden_sizes_value: list[int] = value_cfg.get("hidden_sizes", [128, 128])

    message_dim: int = magic_cfg.get("message_dim", 64)
    num_comm_rounds: int = magic_cfg.get("num_comm_rounds", 1)
    num_heads: int = magic_cfg.get("num_heads", 1)
    gumbel_temperature: float = magic_cfg.get("gumbel_temperature", 1.0)
    recurrent_type = magic_cfg.get("recurrent_type", None)
    recurrent_hidden_size: int = int(magic_cfg.get("recurrent_hidden_size", 64))
    hopfield_num_prototypes: int = int(magic_cfg.get("hopfield_num_prototypes", 16))
    hopfield_beta_init: float = float(magic_cfg.get("hopfield_beta_init", 1.0))
    hopfield_gate_init: float = float(magic_cfg.get("hopfield_gate_init", 0.0))

    first_agent = possible_agents[0]
    act_space = action_spaces[first_agent]
    shared_obs_space = shared_observation_spaces[first_agent]

    num_agents = len(possible_agents)
    orig_dim = shared_obs_space.shape[0]
    expanded_dim = orig_dim + num_agents
    expanded_shared_obs_space = gymnasium.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(expanded_dim,),
        dtype=np.float32,
    )

    # Value network (always shared — takes global state)
    shared_value = MAGICValueNet(
        observation_space=expanded_shared_obs_space,
        action_space=act_space,
        hidden_sizes=hidden_sizes_value,
    )
    shared_value.init_state_dict(role="value")

    # Policy network(s)
    obs_homogeneous = _spaces_homogeneous(observation_spaces)
    act_homogeneous = _actions_homogeneous(action_spaces)
    use_shared_policy = obs_homogeneous and act_homogeneous

    if use_shared_policy:
        # Homogeneous: one shared MAGICPolicyNet for all agents.
        # The communication block is activated in MAGICMAPPO.act by
        # stacking all agents' observations into a single batch.
        obs_space = observation_spaces[first_agent]
        shared_policy = MAGICPolicyNet(
            observation_space=obs_space,
            action_space=act_space,
            hidden_sizes=hidden_sizes_policy,
            message_dim=message_dim,
            num_comm_rounds=num_comm_rounds,
            num_heads=num_heads,
            gumbel_temperature=gumbel_temperature,
            num_agents=num_agents,
            unnormalized_log_prob=unnormalized_log_prob,
            recurrent_type=recurrent_type,
            recurrent_hidden_size=recurrent_hidden_size,
            hopfield_num_prototypes=hopfield_num_prototypes,
            hopfield_beta_init=hopfield_beta_init,
            hopfield_gate_init=hopfield_gate_init,
        )
        shared_policy.init_state_dict(role="policy")

        models: dict[str, dict[str, Any]] = {}
        for agent in possible_agents:
            models[agent] = {"policy": shared_policy, "value": shared_value}
    else:
        # Heterogeneous: per-agent MAGICPolicyNet.
        # Each agent's obs encoder + action head operates on that agent's obs
        # space.  The Scheduler and MessageProcessor (which operate purely in
        # message space) are architecturally independent of obs_dim, but each
        # MAGICPolicyNet instance has its own copy of these layers.
        # MAGICMAPPO.act handles cross-agent communication explicitly
        # (see MAGICMAPPO.act docstring for heterogeneous case).
        models = {}
        for agent in possible_agents:
            obs_space = observation_spaces[agent]
            agent_act_space = action_spaces[agent]

            policy = MAGICPolicyNet(
                observation_space=obs_space,
                action_space=agent_act_space,
                hidden_sizes=hidden_sizes_policy,
                message_dim=message_dim,
                num_comm_rounds=num_comm_rounds,
                num_heads=num_heads,
                gumbel_temperature=gumbel_temperature,
                num_agents=num_agents,
                unnormalized_log_prob=unnormalized_log_prob,
                recurrent_type=recurrent_type,
                recurrent_hidden_size=recurrent_hidden_size,
                hopfield_num_prototypes=hopfield_num_prototypes,
                hopfield_beta_init=hopfield_beta_init,
                hopfield_gate_init=hopfield_gate_init,
            )
            policy.init_state_dict(role="policy")
            models[agent] = {"policy": policy, "value": shared_value}

    return models
