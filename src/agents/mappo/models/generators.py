from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from agents.mappo.models.policy import PolicyNet
from agents.mappo.models.policy_memory import PolicyNetGRU
from agents.mappo.models.value import ValueNet
from agents.mappo.models.value_memory import ValueNetGRU


def create_mappo_models(
    possible_agents: list[str],
    observation_spaces: dict[str, Any],
    action_spaces: dict[str, Any],
    shared_observation_spaces: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Instantiate MAPPO policy and value networks for all agents.

    The value network receives the shared state **plus** a one-hot agent-ID
    vector (dim = ``len(possible_agents)``).  This allows a single shared
    critic to distinguish which agent's value it is estimating, as
    recommended in the MAPPO paper (Yu et al. 2021, §5.2).
    """
    policy_cfg = cfg["policy"]
    value_cfg = cfg["value"]

    hidden_sizes_policy: list[int] = policy_cfg["hidden_sizes"]
    unnormalized_log_prob: bool = policy_cfg["unnormalized_log_prob"]
    hidden_sizes_value: list[int] = value_cfg["hidden_sizes"]

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

    if cfg["policy"]["use_memory"]:
        shared_policy = PolicyNetGRU(
            observation_space=obs_space,
            action_space=act_space,
            hidden_sizes=hidden_sizes_policy,
            rnn_features=cfg["policy"].get("rnn_features", 256),
            unnormalized_log_prob=unnormalized_log_prob,
        )
    else:
        shared_policy = PolicyNet(
            observation_space=obs_space,
            action_space=act_space,
            hidden_sizes=hidden_sizes_policy,
            unnormalized_log_prob=unnormalized_log_prob,
        )

    if cfg["value"]["use_memory"]:
        shared_value = ValueNetGRU(
            observation_space=expanded_shared_obs_space,
            action_space=act_space,
            hidden_sizes=hidden_sizes_policy,
            rnn_features=cfg["value"].get("rnn_features", 256),
        )

    else:
        shared_value = ValueNet(
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
