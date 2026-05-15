"""
continuous_coord_eval_analysis.py
===================================
Evaluation collector for the Continuous Coordination (Rendezvous Pursuit) env.

Decodes all relevant state from observations since the env's info dict only
provides {"step": t}.  For typed mode, set ``type_dim`` to num_agent_types.

Usage
-----
from utils.continuous_coord_eval_analysis import ContinuousCoordEvalCollector
from utils.continuous_coord_eval_visualizer import save_all_cc_figures

collector = ContinuousCoordEvalCollector(
    num_agents=4, max_targets=3, max_cycles=200,
    collision_radius=0.03, capture_reward=10.0,
)
data = collector.collect(env, agent, n_episodes=30)
save_all_cc_figures(data, output_dir="eval_plots/continuous_coord")
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .continuous_coord_data import (
    AgentState,
    CaptureEvent,
    CollisionEvent,
    EpisodeData,
    EvalData,
    StepRecord,
    TargetState,
)


# ─── Obs-decoding helpers ──────────────────────────────────────────────────────


def _decode_agent_state(obs: np.ndarray) -> AgentState:
    """Extract own state from flat observation vector."""
    px = float(obs[0])
    py = float(obs[1])
    vx = float(obs[2])
    vy = float(obs[3])
    return AgentState(
        px=px, py=py, vx=vx, vy=vy, speed=float(np.sqrt(vx * vx + vy * vy))
    )


def _decode_targets(
    obs: np.ndarray,
    num_agents: int,
    max_targets: int,
    type_dim: int,
    deadline_avg: float,
) -> list[TargetState]:
    """
    Decode all target slots from one agent's obs.

    Target features (per slot, 4 + type_dim elements):
      [0] rel_dx  (target_x - agent_px)
      [1] rel_dy  (target_y - agent_py)
      [2] k_req_norm = k_req / num_agents
      [3] deadline / deadline_avg
    Zero-filled when inactive.
    """
    obs_tm_off = 6 + type_dim
    teammate_dim = (num_agents - 1) * (4 + type_dim)
    obs_tgt_off = obs_tm_off + teammate_dim
    tgt_slot = 4 + type_dim

    px = float(obs[0])
    py = float(obs[1])

    targets: list[TargetState] = []
    for t in range(max_targets):
        base = obs_tgt_off + t * tgt_slot
        k_norm = float(obs[base + 2])
        dl_norm = float(obs[base + 3])
        active = k_norm > 1e-6
        if active:
            abs_x = px + float(obs[base])
            abs_y = py + float(obs[base + 1])
        else:
            abs_x = abs_y = 0.0
        targets.append(
            TargetState(
                active=active,
                abs_x=abs_x,
                abs_y=abs_y,
                k_req=k_norm * num_agents,
                deadline_norm=dl_norm,
                deadline_steps=dl_norm * deadline_avg,
            )
        )
    return targets


def _detect_captures(
    prev_targets: list[TargetState],
    curr_targets: list[TargetState],
    rewards: dict[str, float],
    step: int,
    capture_reward: float,
) -> list[CaptureEvent]:
    """
    Detect captures: target slot transitions active→inactive AND mean
    per-agent reward is positive (capture) vs negative (expiry).

    Expiry gives each agent penalty_deadline/n (= -2.0/4 = -0.5 per agent).
    Capture gives each participant at least capture_reward/k (= 10/2 = 5.0).
    We classify by mean per-agent reward > 0 threshold.
    """
    events: list[CaptureEvent] = []
    n_agents = max(1, len(rewards))
    mean_rew = sum(rewards.values()) / n_agents

    for t_idx, (prev, curr) in enumerate(zip(prev_targets, curr_targets)):
        if not (prev.active and not curr.active):
            continue
        # Discriminate capture (positive reward) from expiry (negative reward)
        if mean_rew <= 0.0:
            continue  # target expired, not captured

        k_est = max(1, int(round(prev.k_req)))
        # Estimate sync bonus: excess above base capture reward per-agent
        base_per_agent = capture_reward / k_est
        sync_rew = max(0.0, mean_rew - base_per_agent)
        events.append(
            CaptureEvent(
                step=step,
                target_idx=t_idx,
                capture_reward=mean_rew * n_agents,
                sync_reward=sync_rew,
                total_agents_in_zone=k_est,
                k_required=k_est,
            )
        )
    return events


def _detect_collisions(
    states: dict[str, AgentState],
    step: int,
    collision_radius: float,
) -> list[CollisionEvent]:
    """Detect agent pairs within collision_radius."""
    events: list[CollisionEvent] = []
    agent_list = list(states.items())
    for i in range(len(agent_list)):
        a_name, a_state = agent_list[i]
        for j in range(i + 1, len(agent_list)):
            b_name, b_state = agent_list[j]
            dx = a_state.px - b_state.px
            dy = a_state.py - b_state.py
            d = float(np.sqrt(dx * dx + dy * dy))
            if d < collision_radius:
                events.append(
                    CollisionEvent(
                        step=step,
                        agent_a=a_name,
                        agent_b=b_name,
                        distance=d,
                    )
                )
    return events


# ─── Main collector ───────────────────────────────────────────────────────────


class ContinuousCoordEvalCollector:
    """
    Run evaluation episodes in the Continuous Coordination env and capture
    rich per-step state data.

    Parameters
    ----------
    num_agents      : number of agents
    max_targets     : max simultaneous targets
    max_cycles      : episode length cap
    capture_reward  : base capture reward (from env config, default 10.0)
    collision_radius: threshold for collision detection (from env config, 0.03)
    deadline_avg    : (target_deadline_min + target_deadline_max) / 2; used
                      to decode deadline from obs.  Default matches config defaults.
    type_dim        : 0 for non-typed; num_agent_types for typed mode.
    """

    def __init__(
        self,
        num_agents: int = 4,
        max_targets: int = 3,
        max_cycles: int = 200,
        capture_reward: float = 10.0,
        collision_radius: float = 0.03,
        deadline_avg: float = 55.0,  # (30 + 80) / 2
        type_dim: int = 0,
    ) -> None:
        self.num_agents = num_agents
        self.max_targets = max_targets
        self.max_cycles = max_cycles
        self.capture_reward = capture_reward
        self.collision_radius = collision_radius
        self.deadline_avg = deadline_avg
        self.type_dim = type_dim
        self._agents = [f"agent_{i}" for i in range(num_agents)]

    def collect(
        self,
        env: Any,
        agent: Any,
        n_episodes: int = 20,
        max_steps_per_episode: int | None = None,
    ) -> EvalData:
        """Run n_episodes and return EvalData."""
        agent.set_running_mode("eval")
        max_steps = max_steps_per_episode or self.max_cycles

        data = EvalData(
            num_agents=self.num_agents,
            max_targets=self.max_targets,
            max_cycles=self.max_cycles,
            agents=list(self._agents),
        )

        for ep_idx in range(n_episodes):
            episode = EpisodeData(
                episode_idx=ep_idx,
                num_agents=self.num_agents,
                max_targets=self.max_targets,
                max_cycles=self.max_cycles,
                agents=list(self._agents),
            )
            obs, _ = env.reset()
            actual_agents = list(obs.keys())

            # Decode initial target state from agent_0's obs
            ref_obs = np.asarray(obs[actual_agents[0]], dtype=np.float32).ravel()
            prev_targets = _decode_targets(
                ref_obs,
                self.num_agents,
                self.max_targets,
                self.type_dim,
                self.deadline_avg,
            )

            for t in range(max_steps):
                actions, _, _ = agent.act(obs, timestep=t, timesteps=max_steps)
                actions_int = {
                    k: int(np.asarray(v).ravel()[0]) for k, v in actions.items()
                }

                next_obs, rewards, terminated, truncated, _ = env.step(actions)

                rewards_float = {
                    k: float(np.asarray(v).ravel()[0]) for k, v in rewards.items()
                }

                # Decode agent states from their own observations
                agent_states: dict[str, AgentState] = {}
                for a in actual_agents:
                    a_obs = np.asarray(
                        next_obs.get(a, obs.get(a)), dtype=np.float32
                    ).ravel()
                    agent_states[a] = _decode_agent_state(a_obs)

                # Decode targets from agent_0's obs (authoritative view)
                ref_obs = np.asarray(
                    next_obs.get(actual_agents[0], obs.get(actual_agents[0])),
                    dtype=np.float32,
                ).ravel()
                curr_targets = _decode_targets(
                    ref_obs,
                    self.num_agents,
                    self.max_targets,
                    self.type_dim,
                    self.deadline_avg,
                )
                active_count = sum(1 for tgt in curr_targets if tgt.active)

                # Detect captures
                cap_events = _detect_captures(
                    prev_targets,
                    curr_targets,
                    rewards_float,
                    t,
                    self.capture_reward,
                )
                episode.capture_events.extend(cap_events)

                # Detect collisions
                col_events = _detect_collisions(agent_states, t, self.collision_radius)
                episode.collision_events.extend(col_events)

                record = StepRecord(
                    step=t,
                    agent_states=agent_states,
                    targets=curr_targets,
                    rewards=rewards_float,
                    actions=actions_int,
                    active_target_count=active_count,
                )
                episode.steps.append(record)
                prev_targets = curr_targets
                obs = next_obs

                done = any(
                    bool(np.asarray(v).ravel()[0])
                    for v in list(terminated.values()) + list(truncated.values())
                )
                if done:
                    break

            episode.agents = actual_agents
            data.episodes.append(episode)

            ep_ret = episode.episode_return
            caps = episode.total_captures
            cols = episode.total_collisions
            print(
                f"  Episode {ep_idx + 1:3d}/{n_episodes}  |  "
                f"Return: {ep_ret:+7.2f}  |  "
                f"Captures: {caps:3d}  |  "
                f"Collisions: {cols:3d}  |  "
                f"Len: {episode.length:3d}"
            )

        if data.episodes:
            data.agents = data.episodes[0].agents

        return data
