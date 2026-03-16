from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from agents.mappo.models.policy import PolicyNet
from agents.mappo.models.policy_memory import PolicyNetGRU
from agents.mappo.models.value import ValueNet
from agents.mappo.models.value_memory import ValueNetGRU


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


def create_mappo_models(
    possible_agents: list[str],
    observation_spaces: dict[str, Any],
    action_spaces: dict[str, Any],
    shared_observation_spaces: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Instantiate MAPPO policy and value networks for all agents.

    **Homogeneous agents** (all agents share the same observation shape and
    action space): a single shared policy and a single shared value network
    are created and reused for every agent.  This is the standard MAPPO
    parameter-sharing setup (Yu et al. 2021, §5).

    **Heterogeneous agents** (agents have different observation shapes or
    action spaces): a separate policy network is created per agent so that
    each network's first Dense layer has the correct input dimension.  The
    value network always receives the **global shared state** (same for all
    agents) and therefore remains shared regardless of heterogeneity.

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

    # Value network (always shared — takes global state)
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

    if cfg["value"].get("use_memory", False):
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
    shared_value.init_state_dict(role="value")

    # Policy network(s)
    obs_homogeneous = _spaces_homogeneous(observation_spaces)
    act_homogeneous = _actions_homogeneous(action_spaces)
    use_shared_policy = obs_homogeneous and act_homogeneous

    if use_shared_policy:
        # Standard parameter-sharing: one shared policy for all agents.
        obs_space = observation_spaces[first_agent]

        if policy_cfg.get("use_memory", False):
            shared_policy = PolicyNetGRU(
                observation_space=obs_space,
                action_space=act_space,
                hidden_sizes=hidden_sizes_policy,
                rnn_features=policy_cfg.get("rnn_features", 256),
                unnormalized_log_prob=unnormalized_log_prob,
            )
        else:
            shared_policy = PolicyNet(
                observation_space=obs_space,
                action_space=act_space,
                hidden_sizes=hidden_sizes_policy,
                unnormalized_log_prob=unnormalized_log_prob,
            )
        shared_policy.init_state_dict(role="policy")

        models: dict[str, dict[str, Any]] = {}
        for agent in possible_agents:
            models[agent] = {"policy": shared_policy, "value": shared_value}
    else:
        # Heterogeneous agents: per-agent policy networks.
        # Each agent gets its own network with the correct input size.
        models = {}
        for agent in possible_agents:
            obs_space = observation_spaces[agent]
            agent_act_space = action_spaces[agent]

            if policy_cfg.get("use_memory", False):
                policy = PolicyNetGRU(
                    observation_space=obs_space,
                    action_space=agent_act_space,
                    hidden_sizes=hidden_sizes_policy,
                    rnn_features=policy_cfg.get("rnn_features", 256),
                    unnormalized_log_prob=unnormalized_log_prob,
                )
            else:
                policy = PolicyNet(
                    observation_space=obs_space,
                    action_space=agent_act_space,
                    hidden_sizes=hidden_sizes_policy,
                    unnormalized_log_prob=unnormalized_log_prob,
                )
            policy.init_state_dict(role="policy")
            models[agent] = {"policy": policy, "value": shared_value}

    return models
