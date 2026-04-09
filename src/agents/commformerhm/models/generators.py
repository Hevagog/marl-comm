from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from agents.commformerhm.models.policy import CommFormerHMPolicyNet
from agents.mamhm.models.structured_value import StructuredValueNet
from agents.commformer.models.value import CommFormerValueNet


def create_commformerhm_models(
    possible_agents: list[str],
    observation_spaces: dict[str, Any],
    action_spaces: dict[str, Any],
    shared_observation_spaces: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Instantiate CommFormerHM policy and value networks for all agents."""
    policy_cfg = cfg.get("policy", {})
    value_cfg = cfg.get("value", {})
    cf_cfg = cfg.get("commformerhm", {})

    hidden_sizes_value: list[int] = value_cfg.get("hidden_sizes", [512, 512])
    unnormalized_log_prob: bool = policy_cfg.get("unnormalized_log_prob", True)

    # CommFormer architecture hyperparameters
    hidden_dim: int = cf_cfg.get("hidden_dim", 256)
    num_blocks: int = cf_cfg.get("num_blocks", 2)
    num_heads: int = cf_cfg.get("num_heads", 4)
    head_dim: int = cf_cfg.get("head_dim", 64)
    mlp_dim: int = cf_cfg.get("mlp_dim", 512)
    sparsity: float = cf_cfg.get("sparsity", 0.5)

    # Task Hopfield params
    use_task_hopfield: bool = cf_cfg.get("use_task_hopfield", True)
    task_hopfield_num_heads: int = cf_cfg.get("task_hopfield_num_heads", 4)
    task_hopfield_beta: float = cf_cfg.get("task_hopfield_beta", 2.0)
    task_hopfield_gate_init: float = cf_cfg.get("task_hopfield_gate_init", -3.0)
    execution_mode: str = cf_cfg.get("execution_mode", "ctde")

    # Critic choice
    use_structured_critic: bool = cf_cfg.get("use_structured_critic", True)

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

    shared_policy = CommFormerHMPolicyNet(
        observation_space=obs_space,
        action_space=act_space,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        head_dim=head_dim,
        mlp_dim=mlp_dim,
        num_agents=num_agents,
        sparsity=sparsity,
        use_task_hopfield=use_task_hopfield,
        task_hopfield_num_heads=task_hopfield_num_heads,
        task_hopfield_beta=task_hopfield_beta,
        task_hopfield_gate_init=task_hopfield_gate_init,
        execution_mode=execution_mode,
        unnormalized_log_prob=unnormalized_log_prob,
    )

    if use_structured_critic:
        shared_value = StructuredValueNet(
            observation_space=expanded_shared_obs_space,
            action_space=act_space,
        )
    else:
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
