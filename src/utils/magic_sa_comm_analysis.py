from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .simple_adversary_eval_data import (
    ADVERSARY,
    AGENT_0,
    AGENT_1,
    ALL_AGENTS,
    GOOD_AGENTS,
)
from .magic_comm_analysis import _extract_comm_outputs


@dataclass
class SACommStepRecord:
    step: int

    positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    landmarks: List[Tuple[float, float]] = field(default_factory=list)
    dist_to_goal: Dict[str, float] = field(default_factory=dict)
    dist_to_spoof: Dict[str, float] = field(default_factory=dict)

    actions: Dict[str, int] = field(default_factory=dict)
    rewards: Dict[str, float] = field(default_factory=dict)

    adj_per_round: Optional[List[np.ndarray]] = None
    hard_adj: Optional[np.ndarray] = None
    messages: Optional[np.ndarray] = None
    agg_messages: Optional[np.ndarray] = None
    logits: Optional[np.ndarray] = None

    comm_density: float = 0.0


@dataclass
class SACommEpisodeData:
    episode_idx: int
    steps: List[SACommStepRecord] = field(default_factory=list)
    goal_landmark_idx: int = -1
    has_comm_data: bool = False

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def total_rewards(self) -> Dict[str, float]:
        out = {k: 0.0 for k in ALL_AGENTS}
        for s in self.steps:
            for k, r in s.rewards.items():
                out[k] += r
        return out


@dataclass
class MAGICSAAnalysisData:
    num_agents: int = 3
    num_comm_rounds: int = 2
    message_dim: int = 128
    episodes: List[SACommEpisodeData] = field(default_factory=list)
    has_comm_data: bool = False

    def all_steps(self) -> List[SACommStepRecord]:
        return [s for e in self.episodes for s in e.steps]

    def comm_steps(self) -> List[SACommStepRecord]:
        return [s for s in self.all_steps() if s.adj_per_round is not None]

    def adj_stack(self, round_idx: int = 0) -> np.ndarray:
        mats = [
            s.adj_per_round[round_idx]
            for s in self.comm_steps()
            if len(s.adj_per_round) > round_idx
        ]
        return (
            np.stack(mats, axis=0)
            if mats
            else np.empty((0, self.num_agents, self.num_agents))
        )


class MAGICSASCommCollector:
    """Collects MAGIC comm data specifically for simple_adversary (3 agents)."""

    def __init__(
        self,
        num_comm_rounds: int = 2,
        message_dim: int = 128,
    ) -> None:
        self.num_agents = 3
        self.num_comm_rounds = num_comm_rounds
        self.message_dim = message_dim

    def collect(
        self,
        env: Any,
        agent: Any,
        n_episodes: int = 20,
        max_steps_per_episode: int = 50,
    ) -> MAGICSAAnalysisData:
        agent.set_running_mode("eval")
        data = MAGICSAAnalysisData(
            num_agents=self.num_agents,
            num_comm_rounds=self.num_comm_rounds,
            message_dim=self.message_dim,
        )

        world = _find_world(env)
        comm_data_found = False

        for ep_idx in range(n_episodes):
            episode = SACommEpisodeData(episode_idx=ep_idx)
            obs, _ = env.reset()
            episode_comm_data = False

            goal_idx = -1
            if world:
                good_agent_obj = [a for a in world.agents if "adversary" not in a.name][
                    0
                ]
                goal_obj = good_agent_obj.goal_a
                for i, lm in enumerate(world.landmarks):
                    if lm is goal_obj:
                        goal_idx = i
                        break
            episode.goal_landmark_idx = goal_idx

            for t in range(max_steps_per_episode):
                actions, _, outputs_per_agent = agent.act(
                    obs, timestep=t, timesteps=max_steps_per_episode
                )

                # MAGIC outputs are grouped in the first agent's step output usually.
                merged_outputs = outputs_per_agent.get(ALL_AGENTS[0], {})
                if not merged_outputs and outputs_per_agent:
                    merged_outputs = next(iter(outputs_per_agent.values()))

                actions_int = {
                    k: int(np.asarray(v).ravel()[0]) for k, v in actions.items()
                }

                next_obs, rewards, terminated, truncated, _ = env.step(actions)
                rewards_float = {
                    k: float(np.asarray(v).ravel()[0]) for k, v in rewards.items()
                }

                adj_per_round, hard_adj, messages, agg_messages, logits = (
                    _extract_comm_outputs(
                        merged_outputs,
                        self.num_agents,
                        self.num_comm_rounds,
                        self.message_dim,
                    )
                )
                if adj_per_round is not None:
                    comm_data_found = True
                    episode_comm_data = True

                positions = {}
                landmarks = []
                if world:
                    for a in world.agents:
                        positions[a.name] = tuple(a.state.p_pos)
                    for lm in world.landmarks:
                        landmarks.append(tuple(lm.state.p_pos))
                else:
                    positions = {k: (0.0, 0.0) for k in ALL_AGENTS}
                    landmarks = [(0.0, 0.0), (0.0, 0.0)]

                dist_to_goal = {}
                dist_to_spoof = {}
                goal_pos = landmarks[goal_idx] if goal_idx != -1 else (0.0, 0.0)
                spoof_idx = (
                    1 - goal_idx if goal_idx in [0, 1] and len(landmarks) == 2 else -1
                )
                spoof_pos = landmarks[spoof_idx] if spoof_idx != -1 else (0.0, 0.0)

                for a_name in ALL_AGENTS:
                    p = np.array(positions[a_name])
                    dist_to_goal[a_name] = float(np.linalg.norm(p - np.array(goal_pos)))
                    if a_name in GOOD_AGENTS:
                        dist_to_spoof[a_name] = float(
                            np.linalg.norm(p - np.array(spoof_pos))
                        )

                comm_density = 0.0
                if adj_per_round:
                    comm_density = float(adj_per_round[-1].mean())

                record = SACommStepRecord(
                    step=t,
                    positions=positions,
                    landmarks=landmarks,
                    dist_to_goal=dist_to_goal,
                    dist_to_spoof=dist_to_spoof,
                    actions=actions_int,
                    rewards=rewards_float,
                    adj_per_round=adj_per_round,
                    hard_adj=hard_adj,
                    messages=messages,
                    agg_messages=agg_messages,
                    logits=logits,
                    comm_density=comm_density,
                )
                episode.steps.append(record)

                obs = next_obs
                done = any(
                    bool(np.asarray(v).ravel()[0])
                    for v in list(terminated.values()) + list(truncated.values())
                )
                if done:
                    break

            episode.has_comm_data = episode_comm_data
            data.episodes.append(episode)

        data.has_comm_data = comm_data_found
        return data


def compute_mean_adj(data: MAGICSAAnalysisData, round_idx: int = 0) -> np.ndarray:
    stack = data.adj_stack(round_idx)
    return (
        stack.mean(axis=0)
        if stack.size > 0
        else np.zeros((data.num_agents, data.num_agents))
    )


def compute_directional_asymmetry(
    data: MAGICSAAnalysisData,
    round_idx: int = 0,
) -> Dict[str, float]:
    stack = data.adj_stack(round_idx)
    if stack.size == 0:
        return {
            "adv_to_ag0": 0.0,
            "adv_to_ag1": 0.0,
            "ag0_to_adv": 0.0,
            "ag1_to_adv": 0.0,
            "ag0_to_ag1": 0.0,
            "ag1_to_ag0": 0.0,
            "self_adv": 0.0,
            "self_ag0": 0.0,
            "self_ag1": 0.0,
        }
    return {
        "adv_to_ag0": float(stack[:, 0, 1].mean()),
        "adv_to_ag1": float(stack[:, 0, 2].mean()),
        "ag0_to_adv": float(stack[:, 1, 0].mean()),
        "ag1_to_adv": float(stack[:, 2, 0].mean()),
        "ag0_to_ag1": float(stack[:, 1, 2].mean()),
        "ag1_to_ag0": float(stack[:, 2, 1].mean()),
        "self_adv": float(stack[:, 0, 0].mean()),
        "self_ag0": float(stack[:, 1, 1].mean()),
        "self_ag1": float(stack[:, 2, 2].mean()),
    }


def compute_role_comm_over_time(
    data: MAGICSAAnalysisData,
    round_idx: int = -1,
) -> Dict[str, np.ndarray]:
    r = round_idx if round_idx >= 0 else data.num_comm_rounds - 1
    max_len = max((e.length for e in data.episodes), default=0)
    if max_len == 0:
        return {
            "steps": np.array([], dtype=np.int32),
            "adv_out": np.array([], dtype=np.float32),
            "good_out": np.array([], dtype=np.float32),
            "cross_team": np.array([], dtype=np.float32),
        }

    adv_out = np.full((len(data.episodes), max_len), np.nan, dtype=np.float32)
    good_out = np.full((len(data.episodes), max_len), np.nan, dtype=np.float32)
    cross_team = np.full((len(data.episodes), max_len), np.nan, dtype=np.float32)

    for ep_idx, ep in enumerate(data.episodes):
        for s in ep.steps:
            if s.adj_per_round is None or len(s.adj_per_round) <= r:
                continue
            mat = s.adj_per_round[r]
            t = s.step
            if t >= max_len:
                continue
            adv_out[ep_idx, t] = float(mat[0, 1:].mean())
            good_out[ep_idx, t] = float(
                np.mean([mat[1, 0], mat[1, 2], mat[2, 0], mat[2, 1]])
            )
            cross_team[ep_idx, t] = float(
                np.mean([mat[0, 1], mat[0, 2], mat[1, 0], mat[2, 0]])
            )

    return {
        "steps": np.arange(max_len),
        "adv_out": np.nanmean(adv_out, axis=0),
        "good_out": np.nanmean(good_out, axis=0),
        "cross_team": np.nanmean(cross_team, axis=0),
    }


def compute_comm_by_goal_pressure(
    data: MAGICSAAnalysisData,
    n_bins: int = 8,
    round_idx: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = round_idx if round_idx >= 0 else data.num_comm_rounds - 1
    pressure = []
    density = []
    for ep in data.episodes:
        for s in ep.steps:
            if s.adj_per_round is None or len(s.adj_per_round) <= r:
                continue
            d_adv = s.dist_to_goal.get(ADVERSARY, 0.0)
            d_good = min(
                s.dist_to_goal.get(AGENT_0, np.inf),
                s.dist_to_goal.get(AGENT_1, np.inf),
            )
            pressure.append(float(d_adv - d_good))
            density.append(float(s.adj_per_round[r].mean()))

    if not pressure:
        return np.array([]), np.array([]), np.array([])

    x = np.asarray(pressure, dtype=np.float32)
    y = np.asarray(density, dtype=np.float32)
    bins = np.linspace(x.min(), x.max() + 1e-9, n_bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    means = np.full(n_bins, np.nan, dtype=np.float32)
    stds = np.full(n_bins, np.nan, dtype=np.float32)
    idx = np.digitize(x, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    for b in range(n_bins):
        vals = y[idx == b]
        if vals.size:
            means[b] = float(vals.mean())
            stds[b] = float(vals.std())
    return centres, means, stds


def compute_message_norms(data: MAGICSAAnalysisData) -> Dict[str, np.ndarray]:
    norms_adv = []
    norms_ag0 = []
    norms_ag1 = []
    for s in data.comm_steps():
        if s.messages is None:
            continue
        msg = np.asarray(s.messages)
        if msg.ndim != 2 or msg.shape[0] < 3:
            continue
        l2 = np.linalg.norm(msg, axis=1)
        norms_adv.append(l2[0])
        norms_ag0.append(l2[1])
        norms_ag1.append(l2[2])
    return {
        "adv": np.asarray(norms_adv, dtype=np.float32),
        "ag0": np.asarray(norms_ag0, dtype=np.float32),
        "ag1": np.asarray(norms_ag1, dtype=np.float32),
    }


def _find_world(env: Any) -> Any:
    visited = set()
    queue = [env]
    depth = 0
    max_depth = 8

    while queue and depth <= max_depth:
        next_queue = []
        for node in queue:
            if node is None:
                continue
            node_id = id(node)
            if node_id in visited:
                continue
            visited.add(node_id)

            world = getattr(node, "world", None)
            if (
                world is not None
                and hasattr(world, "agents")
                and hasattr(world, "landmarks")
            ):
                return world

            next_queue.extend(
                [
                    getattr(node, "unwrapped", None),
                    getattr(node, "env", None),
                    getattr(node, "_env", None),
                ]
            )
        queue = next_queue
        depth += 1

    return None
