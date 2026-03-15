from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .blindspot_eval_data import (
    AGENT_A,
    AGENT_B,
    AGENTS,
    EpisodeData,
    EvalData,
    StepRecord,
)


def _denorm(val: float, grid_size: int) -> int:
    gs_norm = max(grid_size - 1, 1)
    return int(round(float(val) * gs_norm))


def _decode_positions_and_goal(
    obs_a: np.ndarray,
    grid_size: int,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    own_x = _denorm(obs_a[0], grid_size)
    own_y = _denorm(obs_a[1], grid_size)
    other_x = _denorm(obs_a[2], grid_size)
    other_y = _denorm(obs_a[3], grid_size)
    goal_x = _denorm(obs_a[4], grid_size)
    goal_y = _denorm(obs_a[5], grid_size)
    return (own_x, own_y), (other_x, other_y), (goal_x, goal_y)


def _decode_traps_from_obs(
    obs_a: np.ndarray,
    grid_size: int,
    num_traps: int,
) -> List[Tuple[int, int]]:
    gs_norm = max(grid_size - 1, 1)
    own_x = _denorm(obs_a[0], grid_size)
    own_y = _denorm(obs_a[1], grid_size)

    traps: List[Tuple[int, int]] = []
    trap_start = 8
    for idx in range(num_traps):
        base = trap_start + idx * 3
        rel_x = float(obs_a[base + 0])
        rel_y = float(obs_a[base + 1])
        present = float(obs_a[base + 2])
        if present <= 0.5:
            continue
        trap_x = int(round(own_x + rel_x * gs_norm))
        trap_y = int(round(own_y + rel_y * gs_norm))
        trap_x = max(0, min(grid_size - 1, trap_x))
        trap_y = max(0, min(grid_size - 1, trap_y))
        traps.append((trap_x, trap_y))
    return traps


def _decode_reached_flags(obs_a: np.ndarray, num_traps: int) -> Tuple[bool, bool]:
    reached_start = 8 + num_traps * 3
    own_reached = float(obs_a[reached_start]) > 0.5
    other_reached = float(obs_a[reached_start + 1]) > 0.5
    return own_reached, other_reached


class BlindSpotEvalCollector:
    """Collect evaluation rollouts for Blind-Spot Navigation."""

    def __init__(
        self,
        grid_size: int = 9,
        num_traps: int = 5,
        max_cycles: int = 100,
        use_communication: bool = False,
        num_message_tokens: int = 4,
    ) -> None:
        self.grid_size = grid_size
        self.num_traps = num_traps
        self.max_cycles = max_cycles
        self.use_communication = use_communication
        self.num_message_tokens = num_message_tokens if use_communication else 0

    def collect(
        self,
        env: Any,
        agent: Any,
        n_episodes: int = 20,
        max_steps_per_episode: Optional[int] = None,
    ) -> EvalData:
        agent.set_running_mode("eval")
        data = EvalData(
            grid_size=self.grid_size,
            max_cycles=self.max_cycles,
            use_communication=self.use_communication,
            num_message_tokens=self.num_message_tokens,
        )

        max_steps = max_steps_per_episode or self.max_cycles

        for ep_idx in range(n_episodes):
            episode = EpisodeData(episode_idx=ep_idx)
            obs, _ = env.reset()
            timesteps = max_steps

            obs_a = self._to_numpy(obs[AGENT_A])
            pos_a, pos_b, goal = _decode_positions_and_goal(obs_a, self.grid_size)
            traps = _decode_traps_from_obs(obs_a, self.grid_size, self.num_traps)
            episode.traps = traps
            episode.goal = goal

            for t in range(max_steps):
                actions, _, _ = agent.act(obs, timestep=t, timesteps=timesteps)
                actions_int = {
                    k: int(np.asarray(v).ravel()[0]) for k, v in actions.items()
                }

                movement_actions: Dict[str, int] = {}
                messages: Dict[str, Optional[int]] = {}
                for agent_name in AGENTS:
                    raw_action = actions_int[agent_name]
                    if self.use_communication:
                        msg = raw_action % self.num_message_tokens
                        movement = raw_action // self.num_message_tokens
                        movement = min(movement, 4)
                        movement_actions[agent_name] = movement
                        messages[agent_name] = msg
                    else:
                        movement_actions[agent_name] = raw_action
                        messages[agent_name] = None

                next_obs, rewards, terminated, truncated, _ = env.step(actions)

                next_obs_a = self._to_numpy(next_obs[AGENT_A])
                pos_a, pos_b, goal = _decode_positions_and_goal(
                    next_obs_a, self.grid_size
                )
                reached_a, reached_b = _decode_reached_flags(next_obs_a, self.num_traps)

                trap_set = set(episode.traps)
                trap_hits = {
                    AGENT_A: pos_a in trap_set,
                    AGENT_B: pos_b in trap_set,
                }
                dist_to_goal = {
                    AGENT_A: abs(pos_a[0] - goal[0]) + abs(pos_a[1] - goal[1]),
                    AGENT_B: abs(pos_b[0] - goal[0]) + abs(pos_b[1] - goal[1]),
                }

                rewards_float = {
                    k: float(np.asarray(v).ravel()[0]) for k, v in rewards.items()
                }

                episode.steps.append(
                    StepRecord(
                        step=t,
                        positions={AGENT_A: pos_a, AGENT_B: pos_b},
                        actions=movement_actions,
                        messages=messages,
                        rewards=rewards_float,
                        reached={AGENT_A: reached_a, AGENT_B: reached_b},
                        trap_hits=trap_hits,
                        dist_to_goal=dist_to_goal,
                    )
                )

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

            data.episodes.append(episode)
            print(
                f"  Episode {ep_idx + 1:3d}/{n_episodes}  |  "
                f"Len: {episode.length:3d}  |  "
                f"Success: {str(episode.success):5s}  |  "
                f"A reward: {episode.total_rewards[AGENT_A]:+6.1f}  "
                f"B reward: {episode.total_rewards[AGENT_B]:+6.1f}"
            )

        return data

    @staticmethod
    def _to_numpy(obs: Any) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32).ravel()
