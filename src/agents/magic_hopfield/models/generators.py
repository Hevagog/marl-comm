from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from agents.magic_hopfield.models.policy import MAGICHopfieldPolicyNet
from agents.magic_hopfield.models.value import MAGICHopfieldValueNet


def _obs_dim(space) -> int:
    return int(np.prod(space.shape))


def _spaces_homogeneous(spaces: dict[str, Any]) -> bool:
    dims = [_obs_dim(s) for s in spaces.values()]
    return len(set(dims)) == 1


def _actions_homogeneous(spaces: dict[str, Any]) -> bool:
    if not spaces:
        return True
    first = next(iter(spaces.values()))
    for s in spaces.values():
        if type(s) != type(first):
            return False
        if hasattr(s, "n") and s.n != first.n:
            return False
        if hasattr(s, "shape") and s.shape != first.shape:
            return False
    return True


def create_magic_hopfield_models(
    possible_agents: list[str],
    observation_spaces: dict[str, Any],
    action_spaces: dict[str, Any],
    shared_observation_spaces: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Instantiate MAGICHopfield policy and value networks."""
    policy_cfg = cfg.get("policy", {})
    value_cfg = cfg.get("value", {})
    magic_cfg = cfg.get("magic_hopfield", {})

    hidden_sizes_policy: list[int] = policy_cfg.get("hidden_sizes", [128, 128])
    unnormalized_log_prob: bool = policy_cfg.get("unnormalized_log_prob", True)
    hidden_sizes_value: list[int] = value_cfg.get("hidden_sizes", [128, 128])

    message_dim: int = magic_cfg.get("message_dim", 64)
    num_comm_rounds: int = magic_cfg.get("num_comm_rounds", 1)
    num_heads: int = magic_cfg.get("num_heads", 1)
    gumbel_temperature: float = magic_cfg.get("gumbel_temperature", 1.0)
    hopfield_num_prototypes: int = magic_cfg.get("hopfield_num_prototypes", 8)
    hopfield_beta: float = magic_cfg.get("hopfield_beta", 1.0)
    hopfield_gate_init: float = magic_cfg.get("hopfield_gate_init", 0.0)

    first_agent = possible_agents[0]
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

    shared_value = MAGICHopfieldValueNet(
        observation_space=expanded_shared_obs_space,
        action_space=act_space,
        hidden_sizes=hidden_sizes_value,
    )
    shared_value.init_state_dict(role="value")

    obs_homogeneous = _spaces_homogeneous(observation_spaces)
    act_homogeneous = _actions_homogeneous(action_spaces)
    use_shared_policy = obs_homogeneous and act_homogeneous

    if use_shared_policy:
        obs_space = observation_spaces[first_agent]
        shared_policy = MAGICHopfieldPolicyNet(
            observation_space=obs_space,
            action_space=act_space,
            hidden_sizes=hidden_sizes_policy,
            message_dim=message_dim,
            num_comm_rounds=num_comm_rounds,
            num_heads=num_heads,
            gumbel_temperature=gumbel_temperature,
            num_agents=num_agents,
            hopfield_num_prototypes=hopfield_num_prototypes,
            hopfield_beta=hopfield_beta,
            hopfield_gate_init=hopfield_gate_init,
            unnormalized_log_prob=unnormalized_log_prob,
        )
        shared_policy.init_state_dict(role="policy")
        models: dict[str, dict[str, Any]] = {
            agent: {"policy": shared_policy, "value": shared_value}
            for agent in possible_agents
        }
    else:
        models = {}
        for agent in possible_agents:
            obs_space = observation_spaces[agent]
            agent_act_space = action_spaces[agent]
            policy = MAGICHopfieldPolicyNet(
                observation_space=obs_space,
                action_space=agent_act_space,
                hidden_sizes=hidden_sizes_policy,
                message_dim=message_dim,
                num_comm_rounds=num_comm_rounds,
                num_heads=num_heads,
                gumbel_temperature=gumbel_temperature,
                num_agents=num_agents,
                hopfield_num_prototypes=hopfield_num_prototypes,
                hopfield_beta=hopfield_beta,
                unnormalized_log_prob=unnormalized_log_prob,
            )
            policy.init_state_dict(role="policy")
            models[agent] = {"policy": policy, "value": shared_value}

    return models
