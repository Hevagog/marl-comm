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


def create_magic_models(
    possible_agents: list[str],
    observation_spaces: dict[str, Any],
    action_spaces: dict[str, Any],
    shared_observation_spaces: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Instantiate MAGIC policy and value networks for all agents.

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

    first_agent = possible_agents[0]
    obs_space = observation_spaces[first_agent]
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
    )

    shared_value = MAGICValueNet(
        observation_space=expanded_shared_obs_space,
        action_space=act_space,
        hidden_sizes=hidden_sizes_value,
    )

    shared_policy.init_state_dict(role="policy")
    shared_value.init_state_dict(role="value")

    models: dict[str, dict[str, Any]] = {}
    for agent in possible_agents:
        models[agent] = {"policy": shared_policy, "value": shared_value}

    return models
