from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import gymnasium
import numpy as np
from gymnasium import spaces

from .config import (
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
        obs_builder = TreeObsForRailEnv(
            max_depth=config.tree_depth, predictor=predictor
        )

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

        # Dense-reward shaping state.
        # ``_distance_map`` is (num_agents, H, W, 4) distance-to-target, clamped
        # to a finite max.  ``_prev_distances`` is the last per-agent d value so
        # we can compute a one-step potential delta.  ``_deadlock_fired`` is a
        # one-shot flag so the deadlock penalty fires exactly once per agent
        # per episode (otherwise a stuck agent would bleed -1 per step forever
        # and drown the gradient again).
        max_cells = float(config.height * config.width)
        self._max_distance = max_cells
        self._distance_map = np.zeros(
            (self.num_agents, config.height, config.width, 4), dtype=np.float32
        )
        self._prev_distances = np.full(self.num_agents, max_cells, dtype=np.float32)
        self._deadlock_fired = np.zeros(self.num_agents, dtype=bool)
        self._completion_ratio = 0.0

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
        # Ensure we never pass None to the underlying RailEnv.reset
        if random_seed is None:
            random_seed = 0
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

        if self._renderer is not None:
            self._renderer.reset()

        obs = self._build_obs_dict(raw_obs)
        infos = self._build_infos(raw_info)
        self._last_obs = obs
        self._last_info = infos

        # Refresh distance tracking once the new rail is in place.  Flatland
        # rebuilds its internal distance map on every reset when regenerate_*
        # is True, so we always re-pull it here.
        self._refresh_distance_map()
        self._deadlock_fired[:] = False
        for idx in range(self.num_agents):
            self._prev_distances[idx] = self._agent_distance(idx)
        self._completion_ratio = 0.0
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

        # Dense shaping must run *before* the episode return is accumulated
        # (so the logged return reflects the reward the agent actually sees)
        # and *before* the terminated flags are propagated, because the
        # completion bonus is conditioned on per-agent terminal state which
        # we derive from raw_dones here.
        per_agent_terminated = {
            agent_name: bool(raw_dones.get(self._agent_ids[agent_name], False))
            for agent_name in self._possible_agents
        }
        rewards = self._shape_rewards(rewards, per_agent_terminated)
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
                # Patch flatland for numpy 2.x compatibility before initializing RenderTool
                import flatland.core.grid.grid4 as grid4

                _orig_set_transitions = grid4.fast_grid4_set_transitions

                def _patched_set_transitions(cell_transition, orientation, block_tuple):
                    mask = block_tuple[0]
                    new_transitions = block_tuple[1]
                    negmask = (~mask) & 0xFFFF
                    return (cell_transition & negmask) | (
                        new_transitions << ((3 - orientation) * 4)
                    )

                grid4.fast_grid4_set_transitions = _patched_set_transitions

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
                    getattr(
                        env_agent.malfunction_handler, "_malfunction_down_counter", 0
                    ),
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
        # Mirror on the instance so the vectorized wrapper can read the
        # latest completion ratio for curriculum scheduling without having
        # to dig through info dicts.
        self._completion_ratio = float(completion_ratio)
        for agent_name in self._possible_agents:
            infos[agent_name]["completion_count"] = completion_count
            infos[agent_name]["completion_ratio"] = completion_ratio
            infos[agent_name]["episode_length"] = self._episode_steps
            infos[agent_name]["total_return"] = self._episode_return

        return infos

    @property
    def completion_ratio(self) -> float:
        """Latest per-episode completion ratio (fraction of agents done)."""
        return self._completion_ratio

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

    def _refresh_distance_map(self) -> None:
        """Snapshot Flatland's per-agent distance-to-target map.

        ``RailEnv.distance_map.get()`` returns a ``(num_agents, H, W, 4)``
        array of minimum rail-path lengths from every (row, col, direction)
        cell to each agent's target, computed from the current schedule.
        Cells with no feasible path hold ``inf`` — we clamp them to
        ``H*W`` so that (a) the progress delta stays finite when an agent
        crosses that boundary and (b) the deadlock detector can treat a
        ``max_d`` reading as "no exit from this cell".
        """
        dmap = self._env.distance_map.get()
        dmap = np.asarray(dmap, dtype=np.float32)
        self._distance_map = np.where(
            np.isfinite(dmap), dmap, self._max_distance
        ).astype(np.float32)

    def _agent_distance(self, idx: int) -> float:
        """Current shortest-path distance-to-target for agent ``idx``.

        Uses the live ``RailEnv`` agent pose (position + direction) rather
        than the cached obs so we don't pick up a stale value after step.
        Before an agent has left its initial cell ``agent.position`` is
        ``None``; in that case we fall back to ``initial_position`` / the
        initial direction so the very first ``reset`` prev-distance is sane.
        """
        ag = self._env.agents[idx]
        pos = ag.position if ag.position is not None else ag.initial_position
        if pos is None:
            return self._max_distance
        r, c = int(pos[0]), int(pos[1])
        d = int(ag.direction if ag.direction is not None else ag.initial_direction or 0)
        return float(self._distance_map[idx, r, c, d])

    def _is_agent_deadlocked(self, idx: int) -> bool:
        """Best-effort deadlock detection without Flatland's internal API.

        We classify an agent as deadlocked when it is on the map, not yet
        done, not currently malfunctioning, and all four outgoing distances
        from its current cell are clamped to ``max_distance`` — i.e. there
        is no feasible path to its target from here.  Malfunctioning agents
        are excluded so we don't double-penalise a transient freeze.
        """
        ag = self._env.agents[idx]
        if ag.position is None or self._is_done_state(ag.state):
            return False
        # Skip agents currently frozen by a malfunction.
        mal = getattr(ag.malfunction_handler, "_malfunction_down_counter", 0)
        if mal and mal > 0:
            return False
        r, c = int(ag.position[0]), int(ag.position[1])
        return bool((self._distance_map[idx, r, c, :] >= self._max_distance).all())

    def _shape_rewards(
        self,
        raw_rewards: dict[str, float],
        terminated: dict[str, bool],
    ) -> dict[str, float]:
        """Potential-based shaping + step / deadlock / completion terms.

        Mathematical form per agent per step::

            r_shaped = r_raw
                     + α * clip(d_{t-1} - d_t, -C, +C)   # progress potential
                     - β * 1{active ∧ ¬done}              # step penalty
                     - γ * 1{first deadlock}              # one-shot deadlock
                     + ρ * 1{terminated (reached target)} # completion bonus

        Potential-based shaping is policy-invariant for the optimal policy
        set (Ng, Harada, Russell 1999), but drastically speeds learning by
        handing out per-step credit the critic can actually latch onto.
        ``_deadlock_fired`` guarantees the deadlock penalty fires *once*
        per agent per episode — otherwise a permanently-stuck agent would
        re-bleed -1 every step and drown the rest of the signal.
        """
        cfg = self._config
        if not cfg.use_shaped_reward:
            return raw_rewards

        shaped: dict[str, float] = {}
        for agent_name, agent_idx in self._agent_ids.items():
            curr_d = self._agent_distance(agent_idx)
            prev_d = float(self._prev_distances[agent_idx])
            progress = cfg.progress_coeff * (prev_d - curr_d)
            progress = float(np.clip(progress, -cfg.progress_clip, cfg.progress_clip))

            is_done = bool(terminated[agent_name])
            step_pen = -cfg.step_penalty if not is_done else 0.0

            dead_pen = 0.0
            if not is_done and not self._deadlock_fired[agent_idx]:
                if self._is_agent_deadlocked(agent_idx):
                    dead_pen = -cfg.deadlock_penalty
                    self._deadlock_fired[agent_idx] = True

            done_bonus = cfg.completion_bonus if is_done else 0.0

            shaped[agent_name] = (
                float(raw_rewards.get(agent_name, 0.0))
                + progress
                + step_pen
                + dead_pen
                + done_bonus
            )
            self._prev_distances[agent_idx] = curr_d
        return shaped

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
