"""
The collector works entirely from *observations* (no internal env state
required), making it compatible with any skrl-wrapped PettingZoo env.

Observation layout (per agent, 12-dim float32):
  [0-1]  own (x,y) / (grid_size-1)
  [2-3]  other agent (x,y) / (grid_size-1)
  [4-6]  red coin  (x_n, y_n, exists)
  [7-9]  blue coin (x_n, y_n, exists)
  [10]   color indicator (1.0=red/agent_0, 0.0=blue/agent_1)
  [11]   time remaining (1.0 -> 0.0)
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from .eval_data import (
    BLUE_AGENT,
    RED_AGENT,
    CoinEvent,
    EpisodeData,
    EvalData,
    StepRecord,
)


def _denorm(val: float, grid_size: int) -> int:
    """Convert a normalised coordinate back to an integer grid cell."""
    gs_norm = max(grid_size - 1, 1)
    return int(round(float(val) * gs_norm))


def _parse_obs(
    obs_red: np.ndarray,
    obs_blue: np.ndarray,
    grid_size: int,
) -> Tuple[
    Tuple[int, int],  # red agent pos
    Tuple[int, int],  # blue agent pos
    Optional[Tuple[int, int]],  # red coin pos (None = absent)
    Optional[Tuple[int, int]],  # blue coin pos
]:
    """Extract grid positions from a pair of raw observations.

    We use agent_0's (red) observation as the authoritative source because
    its 'own' fields always correspond to the red agent.
    """
    gs = grid_size

    # Red agent position (from agent_0's own-pos fields)
    red_x = _denorm(obs_red[0], gs)
    red_y = _denorm(obs_red[1], gs)

    # Blue agent position (from agent_0's other-pos fields)
    blue_x = _denorm(obs_red[2], gs)
    blue_y = _denorm(obs_red[3], gs)

    # Red coin
    red_coin: Optional[Tuple[int, int]] = None
    if float(obs_red[6]) > 0.5:
        rc_x = _denorm(obs_red[4], gs)
        rc_y = _denorm(obs_red[5], gs)
        red_coin = (rc_x, rc_y)

    # Blue coin
    blue_coin: Optional[Tuple[int, int]] = None
    if float(obs_red[9]) > 0.5:
        bc_x = _denorm(obs_red[7], gs)
        bc_y = _denorm(obs_red[8], gs)
        blue_coin = (bc_x, bc_y)

    return (red_x, red_y), (blue_x, blue_y), red_coin, blue_coin


def _detect_coin_events(
    step: int,
    prev_red_coin: Optional[Tuple[int, int]],
    prev_blue_coin: Optional[Tuple[int, int]],
    curr_red_coin: Optional[Tuple[int, int]],
    curr_blue_coin: Optional[Tuple[int, int]],
    red_pos: Tuple[int, int],
    blue_pos: Tuple[int, int],
) -> list[CoinEvent]:
    """Detect which coins were picked up this step by comparing positions.

    A coin was picked up iff its grid position changed OR its existence flag
    changed from True to False.  The picker is inferred from which agent's
    new position matches the *previous* coin position.
    """
    events: list[CoinEvent] = []

    def _check(
        coin_color: str,
        prev: Optional[Tuple[int, int]],
        curr: Optional[Tuple[int, int]],
    ) -> None:
        if prev is None:
            return  # coin didn't exist before
        if curr is not None and curr == prev:
            return  # coin didn't move → not collected

        # Coin was collected.  Infer picker from agent positions.
        # Agents move THEN collect, so agent is at prev_coin_pos in curr step.
        picker: Optional[str] = None
        if red_pos == prev:
            picker = RED_AGENT
        elif blue_pos == prev:
            picker = BLUE_AGENT

        if picker is None:
            # Rare: both agents on the coin (tie-break not visible from obs),
            # or floating-point mismatch.  Skip event.
            return

        is_steal = (coin_color == "red" and picker == BLUE_AGENT) or (
            coin_color == "blue" and picker == RED_AGENT
        )
        events.append(
            CoinEvent(
                step=step, coin_color=coin_color, picker=picker, is_steal=is_steal
            )
        )

    _check("red", prev_red_coin, curr_red_coin)
    _check("blue", prev_blue_coin, curr_blue_coin)
    return events


class EvalCollector:
    """Runs manual rollout loops and returns structured :class:`EvalData`.

    Parameters
    ----------
    grid_size : int
        Grid dimension used to decode normalised observations.
    pick_reward : float
        Reward for picking up any coin (used for metadata only).
    steal_penalty : float
        Penalty suffered when the opponent steals your coin (metadata only).
    """

    def __init__(
        self,
        grid_size: int = 7,
        pick_reward: float = 1.0,
        steal_penalty: float = -2.0,
    ) -> None:
        self.grid_size = grid_size
        self.pick_reward = pick_reward
        self.steal_penalty = steal_penalty

    def collect(
        self,
        env: Any,
        agent: Any,
        n_episodes: int = 20,
        max_steps_per_episode: int = 500,
    ) -> EvalData:
        """Run *n_episodes* episodes and return collected :class:`EvalData`.

        Parameters
        ----------
        env : skrl-wrapped PettingZoo env
            The wrapped environment (reset / step / render interface).
        agent : skrl MultiAgent
            The trained multi-agent instance (act interface).
        n_episodes : int
            Number of complete episodes to collect.
        max_steps_per_episode : int
            Safety cap to prevent infinite loops.

        Returns
        -------
        EvalData
        """
        agent.set_running_mode("eval")
        data = EvalData(
            grid_size=self.grid_size,
            pick_reward=self.pick_reward,
            steal_penalty=self.steal_penalty,
        )

        for ep_idx in range(n_episodes):
            episode = EpisodeData(episode_idx=ep_idx)
            obs, _ = env.reset()
            timesteps = max_steps_per_episode

            # Parse initial state
            obs_red_arr = self._to_numpy(obs[RED_AGENT])
            obs_blue_arr = self._to_numpy(obs[BLUE_AGENT])
            _, _, prev_red_coin, prev_blue_coin = _parse_obs(
                obs_red_arr, obs_blue_arr, self.grid_size
            )

            for t in range(max_steps_per_episode):
                obs_red_arr = self._to_numpy(obs[RED_AGENT])
                obs_blue_arr = self._to_numpy(obs[BLUE_AGENT])

                # Agent decision
                actions, _, _ = agent.act(obs, timestep=t, timesteps=timesteps)
                actions_int = {
                    k: int(np.asarray(v).ravel()[0]) for k, v in actions.items()
                }

                # Environment step
                next_obs, rewards, terminated, truncated, _ = env.step(actions)

                # Parse new state
                next_red_arr = self._to_numpy(next_obs[RED_AGENT])
                next_blue_arr = self._to_numpy(next_obs[BLUE_AGENT])
                red_pos, blue_pos, curr_red_coin, curr_blue_coin = _parse_obs(
                    next_red_arr, next_blue_arr, self.grid_size
                )

                rewards_float = {
                    k: float(np.asarray(v).ravel()[0]) for k, v in rewards.items()
                }

                # Detect coin pickup events
                events = _detect_coin_events(
                    step=t,
                    prev_red_coin=prev_red_coin,
                    prev_blue_coin=prev_blue_coin,
                    curr_red_coin=curr_red_coin,
                    curr_blue_coin=curr_blue_coin,
                    red_pos=red_pos,
                    blue_pos=blue_pos,
                )
                episode.events.extend(events)

                # Record step
                episode.steps.append(
                    StepRecord(
                        step=t,
                        red_pos=red_pos,
                        blue_pos=blue_pos,
                        red_coin_pos=curr_red_coin,
                        blue_coin_pos=curr_blue_coin,
                        actions=actions_int,
                        rewards=rewards_float,
                    )
                )

                prev_red_coin = curr_red_coin
                prev_blue_coin = curr_blue_coin
                obs = next_obs

                done = any(
                    bool(np.asarray(v).ravel()[0])
                    for v in list(terminated.values()) + list(truncated.values())
                )
                if done:
                    break

            data.episodes.append(episode)
            print(
                f"  Episode {ep_idx + 1:3d}/{n_episodes}  |  "
                f"Red reward: {episode.total_rewards[RED_AGENT]:+6.1f}  "
                f"Blue reward: {episode.total_rewards[BLUE_AGENT]:+6.1f}  |  "
                f"Steals Red/Blue: {episode.steals_by(RED_AGENT)}/{episode.steals_by(BLUE_AGENT)}"
            )

        return data

    @staticmethod
    def _to_numpy(obs: Any) -> np.ndarray:
        """Convert any observation tensor / array to a flat numpy float32 array."""
        arr = np.asarray(obs, dtype=np.float32).ravel()
        return arr
