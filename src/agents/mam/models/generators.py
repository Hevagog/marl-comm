from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from agents.mam.models.policy import MAMPolicyNet
from agents.mam.models.value import MAMValueNet


def create_mam_models(
    possible_agents: list[str],
    observation_spaces: dict[str, Any],
    action_spaces: dict[str, Any],
    shared_observation_spaces: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Instantiate MAM policy and value networks for all agents.

    Uses parameter sharing: one policy and one value network shared
    across all agents (homogeneous assumption, same as CommFormer).
    The value network receives shared_state + one-hot agent-ID.
    """
    policy_cfg = cfg.get("policy", {})
    value_cfg = cfg.get("value", {})
    mam_cfg = cfg.get("mam", {})

    # Value network config
    hidden_sizes_value: list[int] = value_cfg.get("hidden_sizes", [128, 128])
    unnormalized_log_prob: bool = policy_cfg.get("unnormalized_log_prob", True)

    # MAM architecture hyperparameters
    n_embd: int = mam_cfg.get("n_embd", 128)
    n_block: int = mam_cfg.get("n_block", 1)
    d_state: int = mam_cfg.get("d_state", 32)
    d_conv: int = mam_cfg.get("d_conv", 4)
    delta_rank: int = mam_cfg.get("delta_rank", 128)

    first_agent = possible_agents[0]
    obs_space = observation_spaces[first_agent]
    act_space = action_spaces[first_agent]
    shared_obs_space = shared_observation_spaces[first_agent]

    num_agents = len(possible_agents)

    # Expand shared obs space with agent-ID one-hot
    orig_dim = shared_obs_space.shape[0]
    expanded_dim = orig_dim + num_agents
    expanded_shared_obs_space = gymnasium.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(expanded_dim,),
        dtype=np.float32,
    )

    shared_policy = MAMPolicyNet(
        observation_space=obs_space,
        action_space=act_space,
        n_embd=n_embd,
        n_block=n_block,
        num_agents=num_agents,
        d_state=d_state,
        d_conv=d_conv,
        delta_rank=delta_rank,
        unnormalized_log_prob=unnormalized_log_prob,
    )

    shared_value = MAMValueNet(
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
