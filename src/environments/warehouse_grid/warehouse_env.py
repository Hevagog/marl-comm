"""Multi-Robot Warehouse Environment.

A 2D gridworld warehouse environment for multi-agent reinforcement learning.

Features
-------------------
- Dynamic task assignment with throughput optimisation
- Agent attrition and communication disturbances
- Resource lifecycle: Pick → Treat → Deliver
— Config-level difficulty tuning + communication range
— Deadline-driven dynamic task queue
— Heterogeneous robots (speed / capacity / fragility)
— Spatially-varying communication interference zones
— Battery / charging cycle
— Structured fault-injection profiles
"""

from __future__ import annotations

from typing import Any, Literal
from collections.abc import Mapping

import numpy as np
import gymnasium
from gymnasium import spaces

from .config import WarehouseConfig
from .utils import (
    create_agent_state,
    get_agent_observation,
    compute_obs_dim,
    apply_agent_attrition,
    apply_movement,
    apply_interactions,
    apply_battery_logic,
    apply_rescue_completion,
    generate_new_tasks,
    expire_tasks,
    compute_rewards,
    update_task_priorities,
    EnvState,
    NUM_ACTIONS,
    create_warehouse_layout,
)
from .utils.types import CellType


RenderMode = Literal["human", "ascii", "rgb_array", None]


class MultiRobotWarehouseEnv:
    """Multi-Robot Warehouse Environment with PettingZoo parallel API.

    Parameters
    ----------
    config : WarehouseConfig or None
        Environment configuration. Uses defaults if None.
    render_mode : str or None
        "human" for live window, "rgb_array" for frame capture, "ascii" for text.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ascii"],
        "name": "warehouse_v0",
    }

    def __init__(
        self,
        config: WarehouseConfig | None = None,
        render_mode: RenderMode = None,
    ):
        if config is None:
            config = WarehouseConfig()
        self._config = config
        self._render_mode = render_mode

        # Agent metadata
        self._possible_agents = [f"agent_{i}" for i in range(config.max_agents)]
        self._agents: list[str] = [f"agent_{i}" for i in range(config.num_agents)]

        # Observation & state dimensions
        self._obs_dim = compute_obs_dim(config)

        H, W = config.grid_height, config.grid_width
        node_features = 6
        agent_features = 8
        self._state_dim = H * W * node_features + config.max_agents * agent_features

        # Spaces
        self._observation_spaces = {
            a: spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
            )
            for a in self._possible_agents
        }
        self._action_spaces = {
            a: spaces.Discrete(NUM_ACTIONS) for a in self._possible_agents
        }
        self._state_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._state_dim,), dtype=np.float32
        )

        # Internal state
        self._state: EnvState | None = None
        self._rng: np.random.Generator = np.random.default_rng()
        self._renderer = None

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render_mode: {render_mode}. "
                f"Valid modes: {self.metadata['render_modes']}"
            )

    @property
    def possible_agents(self) -> list[str]:
        return self._possible_agents

    @property
    def agents(self) -> list[str]:
        if self._state is None:
            return self._agents.copy()
        active_indices = np.where(self._state.agent.active)[0]
        return [f"agent_{i}" for i in active_indices.tolist()]

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        return len(self._possible_agents)

    @property
    def max_cycles(self) -> int:
        return self._config.max_cycles

    @property
    def observation_spaces(self) -> Mapping[str, gymnasium.Space]:
        return self._observation_spaces

    @property
    def action_spaces(self) -> Mapping[str, gymnasium.Space]:
        return self._action_spaces

    @property
    def state_spaces(self) -> Mapping[str, gymnasium.Space]:
        return {a: self._state_space for a in self._possible_agents}

    def observation_space(self, agent: str) -> gymnasium.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> gymnasium.Space:
        return self._action_spaces[agent]

    def state(self) -> np.ndarray:
        assert self._state is not None, "Environment must be reset first"
        H, W = self._config.grid_height, self._config.grid_width
        layout = self._state.grid.layout

        grid_features = []
        for r in range(H):
            for c in range(W):
                cell = layout[r, c]
                grid_features.extend(
                    [
                        float(cell == CellType.SHELF),
                        float(cell == CellType.TREATMENT),
                        float(cell == CellType.GOAL),
                        float(cell == CellType.CHARGER),
                        float(cell == CellType.REPAIR),
                        float(cell != CellType.WALL),
                    ]
                )

        agent_features = []
        for i in range(self._config.max_agents):
            stranded = not self._state.agent.active[i] and (
                self._state.agent.failed[i]
                or self._state.agent.burst_failed[i]
                or self._state.agent.battery_dead[i]
            )
            agent_features.extend(
                [
                    float(self._state.agent.active[i]),
                    float(stranded),
                    self._state.agent.positions[i, 0] / (H - 1) if H > 1 else 0.0,
                    self._state.agent.positions[i, 1] / (W - 1) if W > 1 else 0.0,
                    float(self._state.agent.carrying[i] > 0),
                    self._state.agent.resource_phase[i] / 3.0,
                    float(self._state.agent.locked[i]),
                    float(self._state.agent.rescue_target[i] >= 0),
                ]
            )

        return np.concatenate(
            [
                np.array(grid_features, dtype=np.float32),
                np.array(agent_features, dtype=np.float32),
            ]
        )

    def reset(
        self,
        seed: int | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        grid_state = create_warehouse_layout(self._config, rng=self._rng)
        agent_state = create_agent_state(self._config, grid_state.spawn_positions)

        self._state = EnvState(
            agent=agent_state,
            grid=grid_state,
            step_count=0,
            total_deliveries=0,
            task_queue=[],
            expired_tasks=0,
        )

        self._agents = [f"agent_{i}" for i in range(self._config.num_agents)]

        observations = self._get_observations()
        infos: dict[str, dict] = {
            a: self._build_info(int(a.split("_")[1])) for a in self._possible_agents
        }
        return observations, infos

    def step(
        self,
        actions: Mapping[str, int | np.integer],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        if self._state is None:
            raise RuntimeError("Environment must be reset before calling step()")

        # --- convert actions to array ---
        action_array = np.zeros(self._config.max_agents, dtype=np.int32)
        for agent, action in actions.items():
            idx = int(agent.split("_")[1])
            action_array[idx] = int(action)

        # --- 1. Agent attrition ---
        self._state, _failed = apply_agent_attrition(
            self._state, self._config, self._rng
        )

        # --- 2. Movement ---
        self._state, collision_mask = apply_movement(
            self._state, action_array, self._config
        )

        # --- 3. Battery management ---
        self._state = apply_battery_logic(self._state, self._config)

        # --- 4. Resolve rescue arrivals ---
        self._state, repair_rescue, charge_rescue = apply_rescue_completion(
            self._state, self._config
        )

        # --- 5. Interactions (pick / treat / deliver / start rescue) ---
        self._state, pick_success, treat_complete, delivery_success = (
            apply_interactions(self._state, action_array, self._config)
        )

        # --- 6. Dynamic task queue ---
        self._state = generate_new_tasks(self._state, self._config, self._rng)
        self._state, n_expired = expire_tasks(self._state, self._config)

        # --- 7. Rewards  ---
        rewards_array = compute_rewards(
            self._state,
            pick_success,
            treat_complete,
            delivery_success,
            repair_rescue,
            charge_rescue,
            collision_mask,
            self._config,
            expired_tasks=n_expired,
        )

        # --- 8. Task priorities ---
        self._state = update_task_priorities(self._state, self._config, self._rng)

        # --- 9. Increment step ---
        new_step = self._state.step_count + 1
        self._state = self._state._replace(step_count=new_step)

        # --- 10. Termination ---
        all_resources_depleted = not self._state.grid.shelf_resources.any()
        no_active_agents = not self._state.agent.active.any()
        terminated = all_resources_depleted or no_active_agents
        truncated = new_step >= self._config.max_cycles

        # --- 11. Observations & return dicts ---
        observations = self._get_observations()

        rewards: dict[str, float] = {}
        terminated_dict: dict[str, bool] = {}
        truncated_dict: dict[str, bool] = {}
        infos: dict[str, dict] = {}

        for agent in self._possible_agents:
            idx = int(agent.split("_")[1])
            rewards[agent] = float(rewards_array[idx])
            terminated_dict[agent] = terminated
            truncated_dict[agent] = truncated and not terminated
            infos[agent] = self._build_info(idx)

        # Update active agents list
        if terminated or truncated:
            self._agents = []
        else:
            self._agents = [
                f"agent_{i}"
                for i in range(self._config.max_agents)
                if self._state.agent.active[i]
            ]

        return observations, rewards, terminated_dict, truncated_dict, infos

    def _get_observations(self) -> dict[str, np.ndarray]:
        assert self._state is not None
        observations = {}
        for agent in self._possible_agents:
            idx = int(agent.split("_")[1])
            observations[agent] = get_agent_observation(
                idx, self._state, self._config, self._rng
            )
        return observations

    def _build_info(self, idx: int) -> dict:
        """Build per-agent info dict (extended with Layer metadata)."""
        assert self._state is not None
        info: dict = {
            "active": bool(self._state.agent.active[idx]),
            "carrying": int(self._state.agent.carrying[idx]),
            "phase": int(self._state.agent.resource_phase[idx]),
            "total_deliveries": int(self._state.total_deliveries),
            "stranded": bool(
                (not self._state.agent.active[idx])
                and (
                    self._state.agent.failed[idx]
                    or self._state.agent.burst_failed[idx]
                    or self._state.agent.battery_dead[idx]
                )
            ),
            "dragging": bool(self._state.agent.rescue_target[idx] >= 0),
            "rescue_target": int(self._state.agent.rescue_target[idx]),
            "being_dragged_by": int(self._state.agent.being_dragged_by[idx]),
            "burst_failed": bool(self._state.agent.burst_failed[idx]),
            "battery_dead": bool(self._state.agent.battery_dead[idx]),
            "failed": bool(self._state.agent.failed[idx]),
        }
        if self._config.enable_battery:
            info["battery"] = int(self._state.agent.battery[idx])
            info["charging"] = bool(self._state.agent.charging[idx])
        if self._config.enable_task_deadlines:
            info["pending_tasks"] = len(self._state.task_queue)
            info["expired_tasks"] = int(self._state.expired_tasks)
        if self._config.enable_heterogeneous:
            info["speed"] = int(self._state.agent.speed[idx])
            info["capacity"] = int(self._state.agent.capacity[idx])
            info["fragility"] = float(self._state.agent.fragility[idx])
        return info

    def render(self) -> np.ndarray | str | None:
        if self._state is None or self._render_mode is None:
            return None

        if self._render_mode == "ascii":
            from .utils.rendering import render_ascii

            output = render_ascii(self._state, self._config)
            print(output)
            return output

        elif self._render_mode in ("human", "rgb_array"):
            from .utils.rendering import WarehouseRenderer

            if self._renderer is None:
                self._renderer = WarehouseRenderer(
                    config=self._config,
                    render_mode=self._render_mode,
                )
            return self._renderer.render(self._state)

        return None

    @property
    def render_is_open(self) -> bool:
        if self._renderer is None:
            return True
        return self._renderer.is_open

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_warehouse_env(
    config: WarehouseConfig | None = None,
    render_mode: Literal["human", "rgb_array", "ascii"] | None = None,
) -> MultiRobotWarehouseEnv:
    """Create a Multi-Robot Warehouse environment.

    Parameters
    ----------
    config : WarehouseConfig or None
        Environment configuration. Uses defaults if None.
    render_mode : str or None
        "human" for live window, "rgb_array" for frame capture, "ascii" for text.

    Returns
    -------
    MultiRobotWarehouseEnv
    """
    if config is None:
        config = WarehouseConfig()
    return MultiRobotWarehouseEnv(config=config, render_mode=render_mode)
