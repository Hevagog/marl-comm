"""HighwayEnv multi-agent intersection wrapper (PettingZoo parallel API).

Multiple vehicles approach a 2-D intersection from different directions.
Each vehicle is independently controlled. The goal is to maximize throughput
(vehicles successfully crossing) while avoiding collisions.

The wrapper converts HighwayEnv's single-agent Gymnasium interface
(with ``controlled_vehicles > 1``) into a PettingZoo parallel API by
extracting per-vehicle observations and distributing the joint reward.

Observation
-----------
Each agent receives a Kinematics observation: a flattened vector of
``(vehicles_count, features)`` where features are
``[presence, x, y, vx, vy]``.  The observation is ego-centric: nearby
vehicles are listed relative to the controlled vehicle.

Action space
------------
``DiscreteMetaAction`` with 5 actions:
0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Mapping, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

import highway_env  # noqa: F401 — registers gymnasium envs

from .config import IntersectionConfig


class IntersectionPettingZooEnv:
    """PettingZoo parallel-API wrapper for HighwayEnv intersection.

    Converts the single-agent Gymnasium interface (with multiple controlled
    vehicles) into a multi-agent PettingZoo parallel environment where each
    controlled vehicle is an independent agent.

    Parameters
    ----------
    config : IntersectionConfig
        Environment configuration.
    render_mode : str or None
        "human" for Pygame window, "rgb_array" for numpy frames.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "intersection_v0"}

    def __init__(
        self,
        config: IntersectionConfig | None = None,
        render_mode: Literal["human", "rgb_array"] | None = None,
    ):
        if config is None:
            config = IntersectionConfig()
        self._config = config
        self._render_mode = render_mode

        # Build highway-env config dict
        self._highway_config = {
            "controlled_vehicles": config.num_agents,
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": config.vehicles_count,
                    "features": config.features,
                    "flatten": True,
                    "observe_intentions": False,
                },
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "lateral": True,
                    "longitudinal": True,
                },
            },
            "duration": config.duration,
            "initial_vehicle_count": config.initial_vehicle_count,
            "spawn_probability": config.spawn_probability,
            "collision_reward": config.collision_reward,
            "arrived_reward": config.arrived_reward,
            "high_speed_reward": config.high_speed_reward,
            "reward_speed_range": config.reward_speed_range,
            "normalize_reward": config.normalize_reward,
            "simulation_frequency": config.simulation_frequency,
            "policy_frequency": config.policy_frequency,
        }

        self._env = gymnasium.make(
            "intersection-v1",
            render_mode=render_mode,
            config=self._highway_config,
        )

        # Agent metadata
        self._agent_names = [f"vehicle_{i}" for i in range(config.num_agents)]
        self._possible_agents = list(self._agent_names)
        self._agents = list(self._possible_agents)

        # Observation space per agent: flattened kinematics
        obs_dim = config.vehicles_count * len(config.features)
        self._obs_dim = obs_dim
        self._obs_spaces = {
            a: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for a in self._possible_agents
        }

        # Action space: DiscreteMetaAction has 5 actions
        self._num_actions = 5
        self._act_spaces = {
            a: spaces.Discrete(self._num_actions) for a in self._possible_agents
        }

        # State space (concatenation of all agent observations)
        self._state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim * config.num_agents,),
            dtype=np.float32,
        )

        self._last_obs = None

    @property
    def possible_agents(self) -> list[str]:
        return self._possible_agents

    @property
    def agents(self) -> list[str]:
        return list(self._agents)

    @property
    def num_agents(self) -> int:
        return len(self._agents)

    @property
    def observation_spaces(self) -> Mapping[str, gymnasium.Space]:
        return self._obs_spaces

    @property
    def action_spaces(self) -> Mapping[str, gymnasium.Space]:
        return self._act_spaces

    @property
    def state_spaces(self) -> Mapping[str, gymnasium.Space]:
        return {a: self._state_space for a in self._possible_agents}

    def observation_space(self, agent: str) -> gymnasium.Space:
        return self._obs_spaces[agent]

    def action_space(self, agent: str) -> gymnasium.Space:
        return self._act_spaces[agent]

    def state(self) -> np.ndarray:
        """Return global state (concatenation of all agent observations)."""
        if self._last_obs is None:
            return np.zeros(self._state_space.shape, dtype=np.float32)
        return np.concatenate(
            [self._last_obs[a] for a in self._possible_agents], axis=-1
        ).astype(np.float32)

    def reset(
        self, seed: int | None = None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        raw_obs, info = self._env.reset(seed=seed, **kwargs)
        self._agents = list(self._possible_agents)
        obs = self._split_observations(raw_obs)
        self._last_obs = obs
        infos = {a: {} for a in self._agents}
        return obs, infos

    def step(
        self, actions: Mapping[str, int | np.integer]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        # Build tuple action for highway-env multi-agent
        action_tuple = tuple(int(actions[a]) for a in self._possible_agents)

        raw_obs, reward, terminated, truncated, info = self._env.step(action_tuple)

        # Split observations
        obs = self._split_observations(raw_obs)
        self._last_obs = obs

        # Distribute reward: shared across all agents
        # highway-env returns a single scalar reward for all controlled vehicles
        shared_reward = float(reward)
        rewards = {a: shared_reward for a in self._possible_agents}

        # Per-agent crashed status from info if available
        per_agent_info = {}
        crashed_vehicles = info.get("crashed", False)
        for i, agent in enumerate(self._possible_agents):
            agent_info = {"shared_reward": shared_reward}
            # Check if individual vehicle crashed
            try:
                vehicles = self._env.unwrapped.controlled_vehicles
                if i < len(vehicles):
                    agent_info["crashed"] = vehicles[i].crashed
                    agent_info["arrived"] = getattr(vehicles[i], "arrived", False)
            except Exception:
                pass
            per_agent_info[agent] = agent_info

        terminated_dict = {a: bool(terminated) for a in self._possible_agents}
        truncated_dict = {a: bool(truncated) for a in self._possible_agents}

        if terminated or truncated:
            self._agents = []

        return obs, rewards, terminated_dict, truncated_dict, per_agent_info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

    def _split_observations(self, raw_obs: Any) -> Dict[str, np.ndarray]:
        """Split highway-env multi-agent observation into per-agent dicts."""
        obs = {}
        raw = np.asarray(raw_obs)

        if raw.ndim == 1:
            # Already flattened single observation — replicate for all agents
            flat = raw.astype(np.float32)
            if flat.shape[0] == self._obs_dim * self._config.num_agents:
                # Concatenated per-agent observations
                for i, agent in enumerate(self._possible_agents):
                    start = i * self._obs_dim
                    obs[agent] = flat[start : start + self._obs_dim]
            else:
                # Single agent obs replicated
                for agent in self._possible_agents:
                    if flat.shape[0] >= self._obs_dim:
                        obs[agent] = flat[: self._obs_dim]
                    else:
                        obs[agent] = np.pad(flat, (0, self._obs_dim - flat.shape[0]))
        elif isinstance(raw_obs, tuple):
            # Tuple of per-agent observations (MultiAgentObservation)
            for i, agent in enumerate(self._possible_agents):
                if i < len(raw_obs):
                    agent_obs = np.asarray(raw_obs[i], dtype=np.float32).flatten()
                    if agent_obs.shape[0] >= self._obs_dim:
                        obs[agent] = agent_obs[: self._obs_dim]
                    else:
                        obs[agent] = np.pad(
                            agent_obs, (0, self._obs_dim - agent_obs.shape[0])
                        )
                else:
                    obs[agent] = np.zeros(self._obs_dim, dtype=np.float32)
        elif raw.ndim == 2:
            # (num_agents, obs_dim) or (vehicles_count, features)
            if raw.shape[0] == self._config.num_agents:
                for i, agent in enumerate(self._possible_agents):
                    flat = raw[i].flatten().astype(np.float32)
                    if flat.shape[0] >= self._obs_dim:
                        obs[agent] = flat[: self._obs_dim]
                    else:
                        obs[agent] = np.pad(flat, (0, self._obs_dim - flat.shape[0]))
            else:
                # Single kinematic matrix — same for all agents
                flat = raw.flatten().astype(np.float32)
                for agent in self._possible_agents:
                    if flat.shape[0] >= self._obs_dim:
                        obs[agent] = flat[: self._obs_dim]
                    else:
                        obs[agent] = np.pad(flat, (0, self._obs_dim - flat.shape[0]))
        elif raw.ndim == 3:
            # (num_agents, vehicles_count, features)
            for i, agent in enumerate(self._possible_agents):
                if i < raw.shape[0]:
                    flat = raw[i].flatten().astype(np.float32)
                    if flat.shape[0] >= self._obs_dim:
                        obs[agent] = flat[: self._obs_dim]
                    else:
                        obs[agent] = np.pad(flat, (0, self._obs_dim - flat.shape[0]))
                else:
                    obs[agent] = np.zeros(self._obs_dim, dtype=np.float32)
        else:
            # Fallback: zero observations
            for agent in self._possible_agents:
                obs[agent] = np.zeros(self._obs_dim, dtype=np.float32)

        return obs


def make_intersection_env(
    config: IntersectionConfig | None = None,
    render_mode: Literal["human", "rgb_array"] | None = None,
) -> IntersectionPettingZooEnv:
    """Create a HighwayEnv intersection with PettingZoo parallel API.

    Args:
        config: Environment configuration. Uses defaults if None.
        render_mode: Rendering mode.

    Returns:
        PettingZoo-compatible parallel environment instance.
    """
    if config is None:
        config = IntersectionConfig()
    return IntersectionPettingZooEnv(config=config, render_mode=render_mode)
