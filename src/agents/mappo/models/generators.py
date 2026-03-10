from __future__ import annotations

from typing import Any

from agents.mappo.models.policy import PolicyNet
from agents.mappo.models.value import ValueNet


def create_mappo_models(
    possible_agents: list[str],
    observation_spaces: dict[str, Any],
    action_spaces: dict[str, Any],
    shared_observation_spaces: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Instantiate MAPPO policy and value networks for all agents."""
    policy_cfg = cfg["policy"]
    value_cfg = cfg["value"]

    hidden_sizes_policy: list[int] = policy_cfg["hidden_sizes"]
    unnormalized_log_prob: bool = policy_cfg["unnormalized_log_prob"]
    hidden_sizes_value: list[int] = value_cfg["hidden_sizes"]

    first_agent = possible_agents[0]
    obs_space = observation_spaces[first_agent]
    act_space = action_spaces[first_agent]
    shared_obs_space = shared_observation_spaces[first_agent]

    shared_policy = PolicyNet(
        observation_space=obs_space,
        action_space=act_space,
        hidden_sizes=hidden_sizes_policy,
        unnormalized_log_prob=unnormalized_log_prob,
    )

    shared_value = ValueNet(
        observation_space=shared_obs_space,
        action_space=act_space,
        hidden_sizes=hidden_sizes_value,
    )

    shared_policy.init_state_dict(role="policy")
    shared_value.init_state_dict(role="value")

    models: dict[str, dict[str, Any]] = {}
    for agent in possible_agents:
        models[agent] = {"policy": shared_policy, "value": shared_value}

    return models
