from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from agents.mat.models.policy import MATPolicyNet
from agents.mat.models.value import MATValueNet


def create_mat_models(
    possible_agents: list[str],
    observation_spaces: dict[str, Any],
    action_spaces: dict[str, Any],
    shared_observation_spaces: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Instantiate MAT policy and value networks for all agents.

    All agents share one policy and one value network (homogeneous agents).
    The value network receives the global shared state + one-hot agent-ID,
    following the MAPPO CTDE pattern (Yu et al. 2021 §5.2).
    """
    policy_cfg = cfg.get("policy", {})
    value_cfg = cfg.get("value", {})
    mat_cfg = cfg.get("mat", {})

    hidden_sizes_value: list[int] = value_cfg.get("hidden_sizes", [256, 128])
    unnormalized_log_prob: bool = policy_cfg.get("unnormalized_log_prob", True)

    hidden_dim: int = mat_cfg.get("hidden_dim", 256)
    num_blocks: int = mat_cfg.get("num_blocks", 2)
    num_heads: int = mat_cfg.get("num_heads", 4)
    head_dim: int = mat_cfg.get("head_dim", 64)
    mlp_dim: int = mat_cfg.get("mlp_dim", 512)

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

    shared_policy = MATPolicyNet(
        observation_space=obs_space,
        action_space=act_space,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        head_dim=head_dim,
        mlp_dim=mlp_dim,
        num_agents=num_agents,
        unnormalized_log_prob=unnormalized_log_prob,
    )

    shared_value = MATValueNet(
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
