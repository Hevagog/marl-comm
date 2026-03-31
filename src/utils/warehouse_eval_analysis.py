"""
warehouse_eval_analysis.py
===========================
Evaluation collector for the multi-robot warehouse environment.

Uses the per-agent info dict as the primary source for state data (reliable)
and decodes agent positions from the observation vector (obs[4]=row_norm,
obs[5]=col_norm) as a secondary source.

Usage
-----
from utils.warehouse_eval_analysis import WarehouseEvalCollector
from utils.warehouse_eval_visualizer import save_all_warehouse_figures

collector = WarehouseEvalCollector(
    grid_height=12, grid_width=16, num_agents=4, max_cycles=500
)
data = collector.collect(env, agent, n_episodes=30)
save_all_warehouse_figures(data, output_dir="eval_plots/warehouse")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .warehouse_eval_data import (
    AgentState,
    EpisodeData,
    EvalData,
    StepRecord,
)


# ─── Observation decoding helpers ─────────────────────────────────────────────
# Warehouse observation layout (obs_dim=227 with all layers enabled):
#   obs[0]  : carrying (binary, float)
#   obs[1]  : resource_phase / 3  (normalized 0–1)
#   obs[2]  : treatment_timer (normalized)
#   obs[3]  : locked (binary)
#   obs[4]  : row_position (normalized, 0–1 → actual row in [0, grid_height-1])
#   obs[5]  : col_position (normalized, 0–1 → actual col in [0, grid_width-1])
#   obs[6]  : is_rescuing (binary)
#   obs[7:15]: relative positions to treatment/goal/repair/charger (8 features)
#   obs[15:...]: local grid + nearby agent features
#   obs[-4] : speed_multiplier  (if enable_heterogeneous)
#   obs[-3] : capacity          (if enable_heterogeneous)
#   obs[-2] : fragility         (if enable_heterogeneous)
#   obs[-1] : battery_level / battery_capacity  (if enable_battery)


def _to_np(obs: Any) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).ravel()


def _decode_position(
    obs: np.ndarray,
    grid_height: int,
    grid_width: int,
) -> Tuple[int, int]:
    row = int(round(float(obs[4]) * max(grid_height - 1, 1)))
    col = int(round(float(obs[5]) * max(grid_width - 1, 1)))
    row = max(0, min(grid_height - 1, row))
    col = max(0, min(grid_width - 1, col))
    return row, col


def _build_agent_state(
    info_a: Dict,
    obs: np.ndarray,
    grid_height: int,
    grid_width: int,
    battery_capacity: float = 160.0,
) -> AgentState:
    """Build an AgentState from the environment info dict + observation."""
    pos = _decode_position(obs, grid_height, grid_width)

    # Info dict is the authoritative source for all discrete / categorical fields.
    # Fall back to obs-based decoding only if a key is absent (older env versions).
    active = bool(info_a.get("active", True))
    carrying = int(info_a.get("carrying", int(obs[0] > 0.5)))
    phase = int(info_a.get("phase", int(round(float(obs[1]) * 3))))
    total_deliveries = int(info_a.get("total_deliveries", 0))
    stranded = bool(info_a.get("stranded", False))
    dragging = bool(info_a.get("dragging", False))
    rescue_target = info_a.get("rescue_target")
    being_dragged_by = info_a.get("being_dragged_by")
    burst_failed = bool(info_a.get("burst_failed", False))
    battery_dead = bool(info_a.get("battery_dead", False))
    failed = bool(info_a.get("failed", False))
    charging = bool(info_a.get("charging", False))

    # Battery: prefer info dict, fall back to last obs element
    if "battery" in info_a:
        battery = float(info_a["battery"])
    else:
        battery = float(obs[-1]) * battery_capacity

    # Heterogeneous properties
    speed = float(info_a.get("speed", 1.0))
    capacity = int(info_a.get("capacity", 1))
    fragility = float(info_a.get("fragility", 1.0))

    return AgentState(
        active=active,
        carrying=carrying,
        phase=phase,
        battery=battery,
        speed=speed,
        capacity=capacity,
        fragility=fragility,
        stranded=stranded,
        charging=charging,
        dragging=dragging,
        rescue_target=rescue_target,
        being_dragged_by=being_dragged_by,
        burst_failed=burst_failed,
        battery_dead=battery_dead,
        failed=failed,
        total_deliveries=total_deliveries,
        position=pos,
    )


# ─── Main collector ───────────────────────────────────────────────────────────


class WarehouseEvalCollector:
    """
    Run evaluation episodes in the multi-robot warehouse and capture
    per-step state data for all agents.

    Parameters
    ----------
    grid_height      : warehouse grid rows (default 12)
    grid_width       : warehouse grid cols (default 16)
    num_agents       : number of active agents (default 4)
    max_cycles       : episode length cap (default 500)
    battery_capacity : full battery level for normalisation (default 160)
    """

    def __init__(
        self,
        grid_height: int = 12,
        grid_width: int = 16,
        num_agents: int = 4,
        max_cycles: int = 500,
        battery_capacity: float = 160.0,
    ) -> None:
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_agents = num_agents
        self.max_cycles = max_cycles
        self.battery_capacity = battery_capacity
        self._agents: List[str] = [f"agent_{i}" for i in range(num_agents)]

    def collect(
        self,
        env: Any,
        agent: Any,
        n_episodes: int = 20,
        max_steps_per_episode: Optional[int] = None,
    ) -> EvalData:
        """Run n_episodes evaluation episodes and return collected EvalData."""
        agent.set_running_mode("eval")
        max_steps = max_steps_per_episode or self.max_cycles

        data = EvalData(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            max_cycles=self.max_cycles,
            num_agents=self.num_agents,
            agents=list(self._agents),
        )

        for ep_idx in range(n_episodes):
            episode = EpisodeData(episode_idx=ep_idx, agents=list(self._agents))
            obs, info = env.reset()

            # Detect actual agent keys from the env (may differ from self._agents)
            actual_agents = list(obs.keys())

            for t in range(max_steps):
                actions, _, _ = agent.act(obs, timestep=t, timesteps=max_steps)
                actions_int = {
                    k: int(np.asarray(v).ravel()[0]) for k, v in actions.items()
                }

                next_obs, rewards, terminated, truncated, next_info = env.step(actions)

                # Build step record from the returned info dict + obs
                rewards_float = {
                    k: float(np.asarray(v).ravel()[0]) for k, v in rewards.items()
                }

                agent_states: Dict[str, AgentState] = {}
                for a in actual_agents:
                    a_info = next_info.get(a, {}) if next_info else {}
                    a_obs = _to_np(next_obs.get(a, obs.get(a, np.zeros(227))))
                    agent_states[a] = _build_agent_state(
                        a_info, a_obs, self.grid_height, self.grid_width, self.battery_capacity
                    )

                # Env-level task state: pick from any agent's info (they share it)
                first_info = next(
                    (next_info.get(a, {}) for a in actual_agents if next_info and a in next_info),
                    {},
                )
                pending_tasks = int(first_info.get("pending_tasks", 0))
                expired_cumulative = int(first_info.get("expired_tasks", 0))

                record = StepRecord(
                    step=t,
                    agent_states=agent_states,
                    actions=actions_int,
                    rewards=rewards_float,
                    pending_tasks=pending_tasks,
                    expired_tasks_cumulative=expired_cumulative,
                )
                episode.steps.append(record)

                done = any(
                    bool(np.asarray(v).ravel()[0])
                    for v in list(terminated.values()) + list(truncated.values())
                )
                if done:
                    episode.terminated = all(
                        bool(np.asarray(v).ravel()[0]) for v in terminated.values()
                    )
                    episode.truncated = any(
                        bool(np.asarray(v).ravel()[0]) for v in truncated.values()
                    )
                    break

                obs = next_obs

            # Update episode agent list from actual keys
            episode.agents = actual_agents
            data.episodes.append(episode)

            deliveries = episode.total_deliveries
            expired = episode.expired_tasks_total
            rescues = len(episode.rescue_event_steps())
            print(
                f"  Episode {ep_idx + 1:3d}/{n_episodes}  |  "
                f"Len: {episode.length:3d}  |  "
                f"Deliveries: {deliveries:3d}  |  "
                f"Expired: {expired:3d}  |  "
                f"Rescues: {rescues:3d}  |  "
                f"Reward: {sum(episode.total_rewards.values()):+7.1f}"
            )

        # Sync agents list from the collected data
        if data.episodes:
            data.agents = data.episodes[0].agents

        return data
