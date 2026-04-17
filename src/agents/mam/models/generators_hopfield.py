"""Model factories for Hopfield/ET encoder-only architectures.

Three factory functions, one per architecture variant:
- create_hopfield_pooling_models
- create_hopfield_layer_models
- create_et_encoder_models

Each follows the same pattern as ``create_enc_only_models`` in
``generators_ablations.py``.
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from .value import MAMValueNet


def create_hopfield_pooling_models(
    possible_agents: list[str],
    observation_spaces: dict[str, Any],
    action_spaces: dict[str, Any],
    shared_observation_spaces: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Instantiate HopfieldPooling encoder-only policy + shared value net."""
    from agents.mam.models.policy_hopfield_pooling import MAMHopfieldPoolingPolicyNet

    policy_cfg = cfg.get("policy", {})
    value_cfg = cfg.get("value", {})
    mam_cfg = cfg.get("mam", {})

    hidden_sizes_value: list[int] = value_cfg.get("hidden_sizes", [128, 128])
    unnormalized_log_prob: bool = policy_cfg.get("unnormalized_log_prob", True)

    n_embd: int = mam_cfg.get("n_embd", 128)
    n_block: int = mam_cfg.get("n_block", 1)
    d_state: int = mam_cfg.get("d_state", 32)
    d_conv: int = mam_cfg.get("d_conv", 4)
    delta_rank: int = mam_cfg.get("delta_rank", 128)
    num_pool_queries: int = mam_cfg.get("num_pool_queries", 4)
    pool_beta: float = mam_cfg.get("pool_beta", 1.0)

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

    shared_policy = MAMHopfieldPoolingPolicyNet(
        observation_space=obs_space,
        action_space=act_space,
        n_embd=n_embd,
        n_block=n_block,
        num_agents=num_agents,
        d_state=d_state,
        d_conv=d_conv,
        delta_rank=delta_rank,
        num_pool_queries=num_pool_queries,
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


def create_hopfield_layer_models(
    possible_agents: list[str],
    observation_spaces: dict[str, Any],
    action_spaces: dict[str, Any],
    shared_observation_spaces: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Instantiate HopfieldLayer encoder-only policy + shared value net."""
    from agents.mam.models.policy_hopfield_layer import MAMHopfieldLayerPolicyNet

    policy_cfg = cfg.get("policy", {})
    value_cfg = cfg.get("value", {})
    mam_cfg = cfg.get("mam", {})

    hidden_sizes_value: list[int] = value_cfg.get("hidden_sizes", [128, 128])
    unnormalized_log_prob: bool = policy_cfg.get("unnormalized_log_prob", True)

    n_embd: int = mam_cfg.get("n_embd", 128)
    n_block: int = mam_cfg.get("n_block", 1)
    d_state: int = mam_cfg.get("d_state", 32)
    d_conv: int = mam_cfg.get("d_conv", 4)
    delta_rank: int = mam_cfg.get("delta_rank", 128)
    num_prototypes: int = mam_cfg.get("num_prototypes", 8)
    proto_beta: float = mam_cfg.get("proto_beta", 1.5)

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

    shared_policy = MAMHopfieldLayerPolicyNet(
        observation_space=obs_space,
        action_space=act_space,
        n_embd=n_embd,
        n_block=n_block,
        num_agents=num_agents,
        d_state=d_state,
        d_conv=d_conv,
        delta_rank=delta_rank,
        num_prototypes=num_prototypes,
        proto_beta=proto_beta,
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


def create_et_encoder_models(
    possible_agents: list[str],
    observation_spaces: dict[str, Any],
    action_spaces: dict[str, Any],
    shared_observation_spaces: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Instantiate ET Encoder-only policy + shared value net."""
    from agents.mam.models.policy_et_encoder import MAMETEncoderPolicyNet

    policy_cfg = cfg.get("policy", {})
    value_cfg = cfg.get("value", {})
    mam_cfg = cfg.get("mam", {})

    hidden_sizes_value: list[int] = value_cfg.get("hidden_sizes", [128, 128])
    unnormalized_log_prob: bool = policy_cfg.get("unnormalized_log_prob", True)

    n_embd: int = mam_cfg.get("n_embd", 128)
    num_heads: int = mam_cfg.get("num_heads", 4)
    et_beta: float = mam_cfg.get("et_beta", 1.0)
    et_alpha: float = mam_cfg.get("et_alpha", 0.1)
    num_memories: int = mam_cfg.get("num_memories", 64)
    num_et_steps: int = mam_cfg.get("num_et_steps", 3)
    num_et_steps_eval: int = mam_cfg.get("num_et_steps_eval", 5)
    hn_activation: str = mam_cfg.get("hn_activation", "relu")
    stop_grad_intermediate: bool = mam_cfg.get("stop_grad_intermediate", False)

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

    shared_policy = MAMETEncoderPolicyNet(
        observation_space=obs_space,
        action_space=act_space,
        n_embd=n_embd,
        num_agents=num_agents,
        num_heads=num_heads,
        et_beta=et_beta,
        et_alpha=et_alpha,
        num_memories=num_memories,
        num_et_steps=num_et_steps,
        num_et_steps_eval=num_et_steps_eval,
        hn_activation=hn_activation,
        stop_grad_intermediate=stop_grad_intermediate,
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
