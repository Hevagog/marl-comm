from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from agents.mamhm.models.policy import MAMHMPolicyNet
from agents.mamhm.models.value import MAMHMValueNet
from agents.mamhm.models.structured_value import StructuredValueNet


def create_mamhm_models(
    possible_agents: list[str],
    observation_spaces: dict[str, Any],
    action_spaces: dict[str, Any],
    shared_observation_spaces: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Instantiate MAMHM policy and value networks for all agents.

    Uses parameter sharing: one policy and one value network shared
    across all agents (homogeneous assumption, same as MAM/CommFormer).
    The value network receives shared_state + one-hot agent-ID.
    """
    policy_cfg = cfg.get("policy", {})
    value_cfg = cfg.get("value", {})
    mamhm_cfg = cfg.get("mamhm", {})

    # Value network
    hidden_sizes_value: list[int] = value_cfg.get("hidden_sizes", [128, 128])
    unnormalized_log_prob: bool = policy_cfg.get("unnormalized_log_prob", True)

    # MAM architecture hyperparameters
    n_embd: int = mamhm_cfg.get("n_embd", 128)
    n_block: int = mamhm_cfg.get("n_block", 1)
    d_state: int = mamhm_cfg.get("d_state", 32)
    d_conv: int = mamhm_cfg.get("d_conv", 4)
    delta_rank: int = mamhm_cfg.get("delta_rank", 128)

    # Hopfield Memory Bank hyperparameters
    num_memories: int = mamhm_cfg.get("num_memories", 64)
    memory_beta: float = mamhm_cfg.get("memory_beta", 1.0)
    memory_gamma: float = mamhm_cfg.get("memory_gamma", 0.1)
    memory_gate_init: float = mamhm_cfg.get("memory_gate_init", -3.0)
    memory_activation: str = mamhm_cfg.get("memory_activation", "softmax")
    memory_use_pre_ln: bool = mamhm_cfg.get("memory_use_pre_ln", True)
    memory_diversity_loss_scale: float = mamhm_cfg.get("diversity_loss_scale", 0.01)

    # Upstream Hopfield pooling
    use_task_hopfield: bool = mamhm_cfg.get("use_task_hopfield", False)
    use_entity_hopfield: bool = mamhm_cfg.get("use_entity_hopfield", False)
    task_hopfield_num_heads: int = mamhm_cfg.get("task_hopfield_num_heads", 4)
    task_hopfield_beta: float = mamhm_cfg.get("task_hopfield_beta", 2.0)
    task_hopfield_gate_init: float = mamhm_cfg.get("task_hopfield_gate_init", -3.0)

    # Post-decoder Hopfield (legacy)
    use_post_decoder_hopfield: bool = mamhm_cfg.get("use_post_decoder_hopfield", False)

    # Structured critic
    use_structured_critic: bool = mamhm_cfg.get("use_structured_critic", False)

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

    shared_policy = MAMHMPolicyNet(
        observation_space=obs_space,
        action_space=act_space,
        n_embd=n_embd,
        n_block=n_block,
        num_agents=num_agents,
        d_state=d_state,
        d_conv=d_conv,
        delta_rank=delta_rank,
        # Upstream Hopfield
        use_task_hopfield=use_task_hopfield,
        use_entity_hopfield=use_entity_hopfield,
        task_hopfield_num_heads=task_hopfield_num_heads,
        task_hopfield_beta=task_hopfield_beta,
        task_hopfield_gate_init=task_hopfield_gate_init,
        # Post-decoder Hopfield (legacy)
        use_post_decoder_hopfield=use_post_decoder_hopfield,
        num_memories=num_memories,
        memory_beta=memory_beta,
        memory_gamma=memory_gamma,
        memory_gate_init=memory_gate_init,
        memory_activation=memory_activation,
        memory_use_pre_ln=memory_use_pre_ln,
        memory_diversity_loss_scale=memory_diversity_loss_scale,
        unnormalized_log_prob=unnormalized_log_prob,
    )

    if use_structured_critic:
        shared_value = StructuredValueNet(
            observation_space=expanded_shared_obs_space,
            action_space=act_space,
        )
    else:
        shared_value = MAMHMValueNet(
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
