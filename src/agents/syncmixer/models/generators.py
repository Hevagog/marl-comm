from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from agents.mam.models.value import MAMValueNet
from agents.syncmixer.models.policy import SyncMixerPolicyNet


def create_syncmixer_models(
    possible_agents: list[str],
    observation_spaces: dict[str, Any],
    action_spaces: dict[str, Any],
    shared_observation_spaces: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Instantiate SyncMixer policy + shared MLP critic (MAM-style)."""
    policy_cfg = cfg.get("policy", {})
    value_cfg = cfg.get("value", {})
    sm_cfg = cfg.get("syncmixer", {})

    hidden_sizes_value: list[int] = value_cfg.get("hidden_sizes", [256, 128])
    unnormalized_log_prob: bool = policy_cfg.get("unnormalized_log_prob", True)

    d_model: int = sm_cfg.get("d_model", 128)
    n_block: int = sm_cfg.get("n_block", 2)
    num_heads: int = sm_cfg.get("num_heads", 4)
    hidden_mult: int = sm_cfg.get("hidden_mult", 4)
    num_pool_queries: int = sm_cfg.get("num_pool_queries", 4)
    pool_dim: int = sm_cfg.get("pool_dim", 32)
    pool_beta: float = sm_cfg.get("pool_beta", 2.0)

    first_agent = possible_agents[0]
    obs_space = observation_spaces[first_agent]
    act_space = action_spaces[first_agent]
    shared_obs_space = shared_observation_spaces[first_agent]
    num_agents = len(possible_agents)

    # Critic sees global state + one-hot agent ID — same expansion as MAPPO.
    orig_dim = shared_obs_space.shape[0]
    expanded_dim = orig_dim + num_agents
    expanded_shared_obs_space = gymnasium.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(expanded_dim,),
        dtype=np.float32,
    )

    shared_policy = SyncMixerPolicyNet(
        observation_space=obs_space,
        action_space=act_space,
        d_model=d_model,
        n_block=n_block,
        num_heads=num_heads,
        hidden_mult=hidden_mult,
        num_agents=num_agents,
        num_pool_queries=num_pool_queries,
        pool_dim=pool_dim,
        pool_beta=pool_beta,
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
