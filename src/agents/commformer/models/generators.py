from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from agents.commformer.models.policy import CommFormerPolicyNet
from agents.commformer.models.value import CommFormerValueNet


def create_commformer_models(
    possible_agents: list[str],
    observation_spaces: dict[str, Any],
    action_spaces: dict[str, Any],
    shared_observation_spaces: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Instantiate CommFormer policy and value networks for all agents.

    The value network receives the shared state + one-hot agent-ID.
    """
    policy_cfg = cfg.get("policy", {})
    value_cfg = cfg.get("value", {})
    cf_cfg = cfg.get("commformer", {})

    hidden_sizes_value: list[int] = value_cfg.get("hidden_sizes", [128, 128])
    unnormalized_log_prob: bool = policy_cfg.get("unnormalized_log_prob", True)

    hidden_dim: int = cf_cfg.get("hidden_dim", 64)
    num_blocks: int = cf_cfg.get("num_blocks", 1)
    num_heads: int = cf_cfg.get("num_heads", 1)
    head_dim: int = cf_cfg.get("head_dim", 64)
    mlp_dim: int = cf_cfg.get("mlp_dim", 128)
    sparsity: float = cf_cfg.get("sparsity", 0.4)

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

    shared_policy = CommFormerPolicyNet(
        observation_space=obs_space,
        action_space=act_space,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        head_dim=head_dim,
        mlp_dim=mlp_dim,
        num_agents=num_agents,
        sparsity=sparsity,
        unnormalized_log_prob=unnormalized_log_prob,
    )

    shared_value = CommFormerValueNet(
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
