from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import gymnasium
import numpy as np
from gymnasium import spaces

from .config import (
    FLATLAND_STATUS_DIM,
    FlatlandConfig,
    flatland_observation_dim,
    flatland_state_dim,
)


class FlatlandPettingZooEnv:
    """PettingZoo-style parallel wrapper around Flatland's RailEnv."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "flatland_v0"}

    def __init__(
        self,
        config: FlatlandConfig | None = None,
        render_mode: Literal["human", "rgb_array"] | None = None,
    ) -> None:
        if config is None:
            config = FlatlandConfig()
        self._config = config
        self._render_mode = render_mode

        from flatland.envs.line_generators import sparse_line_generator
        from flatland.envs.malfunction_generators import (
            MalfunctionParameters,
            ParamMalfunctionGen,
        )
        from flatland.envs.observations import TreeObsForRailEnv
        from flatland.envs.predictions import ShortestPathPredictorForRailEnv
        from flatland.envs.rail_env import RailEnv
        from flatland.envs.rail_generators import sparse_rail_generator
        from flatland.envs.step_utils.states import TrainState

        self._TrainState = TrainState
        self._tree_actions = tuple(TreeObsForRailEnv.tree_explored_actions_char)

        predictor = ShortestPathPredictorForRailEnv(max_depth=config.prediction_depth)
        obs_builder = TreeObsForRailEnv(max_depth=config.tree_depth, predictor=predictor)

        malfunction_generator = None
        if config.use_malfunctions:
            malfunction_params = MalfunctionParameters(
                malfunction_rate=config.malfunction_rate,
                min_duration=config.malfunction_min_duration,
                max_duration=config.malfunction_max_duration,
            )
            malfunction_generator = ParamMalfunctionGen(malfunction_params)

        self._env = RailEnv(
            width=config.width,
            height=config.height,
            rail_generator=sparse_rail_generator(
                max_num_cities=config.max_num_cities,
                grid_mode=config.grid_mode,
                max_rails_between_cities=config.max_rails_between_cities,
                max_rail_pairs_in_city=config.max_rail_pairs_in_city,
                seed=config.random_seed,
            ),
            line_generator=sparse_line_generator(
                speed_ratio_map=dict(config.speed_ratio_map),
                seed=config.random_seed or 1,
            ),
            number_of_agents=config.num_agents,
            obs_builder_object=obs_builder,
            malfunction_generator=malfunction_generator,
            remove_agents_at_target=config.remove_agents_at_target,
            random_seed=config.random_seed,
        )

        self._possible_agents = [f"agent_{idx}" for idx in range(config.num_agents)]
        self._agents = list(self._possible_agents)
        self._agent_ids = {
            agent_name: agent_idx
            for agent_idx, agent_name in enumerate(self._possible_agents)
        }

        self._obs_dim = flatland_observation_dim(config.tree_depth)
        self._state_dim = flatland_state_dim(config.num_agents, config.tree_depth)
        self._obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self._state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._state_dim,),
            dtype=np.float32,
        )
        self._observation_spaces = {
            agent: self._obs_space for agent in self._possible_agents
        }
        self._action_spaces = {
            agent: spaces.Discrete(5) for agent in self._possible_agents
        }

        self._last_obs = {
            agent: np.zeros(self._obs_dim, dtype=np.float32)
            for agent in self._possible_agents
        }
        self._last_info = {
            agent: self._empty_agent_info() for agent in self._possible_agents
        }
        self._episode_steps = 0
        self._episode_return = 0.0
        self._renderer = None
        self._renderer_failed = False

    @property
    def possible_agents(self) -> list[str]:
        return list(self._possible_agents)

    @property
    def agents(self) -> list[str]:
        return list(self._agents)

    @property
    def num_agents(self) -> int:
        return len(self._possible_agents)

    @property
    def observation_spaces(self) -> Mapping[str, gymnasium.Space]:
        return self._observation_spaces

    @property
    def action_spaces(self) -> Mapping[str, gymnasium.Space]:
        return self._action_spaces

    @property
    def state_spaces(self) -> Mapping[str, gymnasium.Space]:
        return {agent: self._state_space for agent in self._possible_agents}

    def observation_space(self, agent: str) -> gymnasium.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> gymnasium.Space:
        return self._action_spaces[agent]

    def state(self) -> np.ndarray:
        pieces: list[np.ndarray] = []
        for agent_name in self._possible_agents:
            pieces.append(self._last_obs[agent_name])
            pieces.append(self._status_features(agent_name))
        return np.concatenate(pieces, axis=0).astype(np.float32)

    def reset(
        self, seed: int | None = None, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        random_seed = seed if seed is not None else self._config.random_seed
        raw_obs, raw_info = self._env.reset(
            regenerate_rail=self._config.regenerate_rail_on_reset,
            regenerate_schedule=self._config.regenerate_schedule_on_reset,
            random_seed=random_seed,
        )
        if self._config.max_episode_steps is not None:
            self._env._max_episode_steps = int(self._config.max_episode_steps)

        self._agents = list(self._possible_agents)
        self._episode_steps = 0
        self._episode_return = 0.0

        obs = self._build_obs_dict(raw_obs)
        infos = self._build_infos(raw_info)
        self._last_obs = obs
        self._last_info = infos
        return obs, infos

    def step(
        self, actions: Mapping[str, int | np.integer]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        from flatland.envs.rail_env import RailEnvActions

        raw_actions = {
            self._agent_ids[agent_name]: RailEnvActions(
                int(actions.get(agent_name, RailEnvActions.DO_NOTHING))
            )
            for agent_name in self._possible_agents
        }

        raw_obs, raw_rewards, raw_dones, raw_info = self._env.step(raw_actions)
        self._episode_steps += 1

        obs = self._build_obs_dict(raw_obs)
        rewards = {
            agent_name: float(raw_rewards.get(self._agent_ids[agent_name], 0.0))
            for agent_name in self._possible_agents
        }
        self._episode_return += float(sum(rewards.values()))

        infos = self._build_infos(raw_info)
        completion_count = sum(
            1 for info in infos.values() if bool(info.get("reached_target", False))
        )
        all_completed = completion_count == self.num_agents
        global_done = bool(raw_dones.get("__all__", False))
        global_truncated = global_done and not all_completed

        terminated = {}
        truncated = {}
        for agent_name in self._possible_agents:
            info = infos[agent_name]
            reached_target = bool(info["reached_target"])
            agent_done = bool(raw_dones.get(self._agent_ids[agent_name], False))

            terminated[agent_name] = agent_done and reached_target
            truncated[agent_name] = (
                global_truncated and not terminated[agent_name]
            ) or (agent_done and not reached_target)

        self._last_obs = obs
        self._last_info = infos
        return obs, rewards, terminated, truncated, infos

    def render(self):
        if self._render_mode is None or self._renderer_failed:
            return None

        try:
            if self._renderer is None:
                from flatland.utils.rendertools import RenderTool

                self._renderer = RenderTool(self._env, gl="PILSVG")

            image = self._renderer.render_env(
                show=self._render_mode == "human",
                show_observations=False,
                return_image=self._render_mode == "rgb_array",
            )
            if self._render_mode == "rgb_array":
                if image is not None:
                    return np.asarray(image)
                return np.asarray(self._renderer.get_image())
            return None
        except Exception:
            # Flatland rendering is brittle across versions / NumPy builds.
            self._renderer_failed = True
            return None

    def close(self) -> None:
        if self._renderer is not None:
            close_window = getattr(self._renderer, "close_window", None)
            if callable(close_window):
                close_window()
        self._renderer = None

    def _empty_agent_info(self) -> dict[str, Any]:
        return {
            "active": False,
            "ready_to_depart": False,
            "done": False,
            "reached_target": False,
            "malfunction": 0,
            "speed": 0.0,
            "position": None,
            "direction": 0,
            "completion_count": 0,
            "completion_ratio": 0.0,
            "episode_length": 0,
            "total_return": 0.0,
        }

    def _build_obs_dict(self, raw_obs: Mapping[int, Any]) -> dict[str, np.ndarray]:
        return {
            agent_name: self._flatten_tree_observation(raw_obs.get(agent_idx))
            for agent_name, agent_idx in self._agent_ids.items()
        }

    def _build_infos(self, raw_info: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        info_action_required = raw_info.get("action_required", {})
        info_malfunction = raw_info.get("malfunction", {})
        info_speed = raw_info.get("speed", {})
        info_state = raw_info.get("state", {})

        infos: dict[str, dict[str, Any]] = {}
        completion_count = 0
        for agent_name, agent_idx in self._agent_ids.items():
            env_agent = self._env.agents[agent_idx]
            state = info_state.get(agent_idx, env_agent.state)
            ready_to_depart = self._is_ready_state(state)
            active = self._is_active_state(state)
            done = self._is_done_state(state)
            reached_target = done
            completion_count += int(reached_target)

            position = env_agent.position
            if position is not None:
                position = (int(position[0]), int(position[1]))

            malfunction = int(
                info_malfunction.get(
                    agent_idx,
                    getattr(env_agent.malfunction_handler, "_malfunction_down_counter", 0),
                )
            )
            speed = float(
                info_speed.get(
                    agent_idx,
                    getattr(env_agent.speed_counter, "speed", 0.0),
                )
            )

            infos[agent_name] = {
                "active": active,
                "ready_to_depart": ready_to_depart,
                "action_required": bool(info_action_required.get(agent_idx, False)),
                "done": done,
                "reached_target": reached_target,
                "malfunction": malfunction,
                "speed": speed,
                "position": position,
                "direction": int(env_agent.direction or 0),
            }

        completion_ratio = completion_count / max(1, self.num_agents)
        for agent_name in self._possible_agents:
            infos[agent_name]["completion_count"] = completion_count
            infos[agent_name]["completion_ratio"] = completion_ratio
            infos[agent_name]["episode_length"] = self._episode_steps
            infos[agent_name]["total_return"] = self._episode_return

        return infos

    def _flatten_tree_observation(self, node: Any) -> np.ndarray:
        values: list[float] = []

        def _append_subtree(current: Any, depth: int) -> None:
            if not hasattr(current, "_fields"):
                values.extend([0.0] * 12)
                if depth < self._config.tree_depth:
                    for _ in self._tree_actions:
                        _append_subtree(None, depth + 1)
                return

            for field_name in current._fields:
                if field_name == "childs":
                    continue
                value = float(getattr(current, field_name))
                if not np.isfinite(value):
                    value = 0.0
                values.append(value)

            if depth < self._config.tree_depth:
                for action_name in self._tree_actions:
                    _append_subtree(current.childs.get(action_name), depth + 1)

        _append_subtree(node, 0)
        obs = np.asarray(values, dtype=np.float32)
        if obs.shape != (self._obs_dim,):
            raise ValueError(
                f"Flatland observation shape mismatch: got {obs.shape}, expected {(self._obs_dim,)}"
            )
        return obs

    def _status_features(self, agent_name: str) -> np.ndarray:
        info = self._last_info.get(agent_name, self._empty_agent_info())
        return np.asarray(
            [
                float(info.get("ready_to_depart", False)),
                float(info.get("active", False)),
                float(info.get("done", False)),
                float(info.get("direction", 0)),
                float(info.get("malfunction", 0)),
                float(info.get("speed", 0.0)),
                float(info.get("reached_target", False)),
            ],
            dtype=np.float32,
        )

    def _is_ready_state(self, state: Any) -> bool:
        return state == self._TrainState.READY_TO_DEPART

    def _is_done_state(self, state: Any) -> bool:
        return state == self._TrainState.DONE

    def _is_active_state(self, state: Any) -> bool:
        if hasattr(state, "is_on_map_state"):
            return bool(state.is_on_map_state())
        return state in {
            self._TrainState.MOVING,
            self._TrainState.STOPPED,
            self._TrainState.MALFUNCTION,
        }


def make_flatland_env(
    config: FlatlandConfig | None = None,
    render_mode: Literal["human", "rgb_array"] | None = None,
) -> FlatlandPettingZooEnv:
    return FlatlandPettingZooEnv(config=config, render_mode=render_mode)
