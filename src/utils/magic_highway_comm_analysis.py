"""
magic_highway_comm_analysis.py
===============================
Data collection and statistical analysis for the MAGIC communication mechanism
on the Highway Intersection environment.

Architecture recap (MAGIC, Niu et al. AAMAS 2021)
--------------------------------------------------
 1. Scheduler  – GAT encoder → MLP → Gumbel-Softmax → directed adjacency G_t^(l)
 2. Message Processor – GAT over G_t aggregates messages z_i across num_comm_rounds
 3. Policy head – acts on [obs_enc || aggregated_msg]

The MAGIC policy net is expected to expose the following keys in its outputs dict
(returned from policy.act()):

    outputs["adj_matrices"]   – np/jax array, shape (num_comm_rounds, N, N) or (N, N)
                                 Soft Gumbel-Softmax weights BEFORE hard-threshold.
                                 Captures the probability that agent i sends to agent j.
    outputs["hard_adj"]       – binary adjacency after hard-threshold  (N, N), optional
    outputs["messages"]       – pre-aggregation message embeddings (N, message_dim)
    outputs["agg_messages"]   – post-aggregation (N, message_dim), optional
    outputs["net_output"]     – logits (N, num_actions)  — always present

If those keys are absent, the collector still runs but communication-specific plots
will be skipped and a warning is printed once.

Usage
-----
from utils.magic_highway_comm_analysis import MAGICHighwayCommCollector
from utils.magic_highway_comm_visualizer import save_all_magic_highway_figures

collector = MAGICHighwayCommCollector(
    num_agents=4, duration=13, num_comm_rounds=2, message_dim=64,
)
data = collector.collect(env, magic_agent, n_episodes=30)
save_all_magic_highway_figures(data, output_dir="eval_plots/magic", prefix="magic_highway_v0")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


@dataclass
class HighwayCommStepRecord:
    """All communication-relevant data captured at one environment step."""

    step: int

    # ---------- state context ------------------------------------------------
    positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    velocities: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    speeds: Dict[str, float] = field(default_factory=dict)
    inter_vehicle_distances: Dict[Tuple[str, str], float] = field(default_factory=dict)
    actions: Dict[str, int] = field(default_factory=dict)
    rewards: Dict[str, float] = field(default_factory=dict)
    crashed: Dict[str, bool] = field(default_factory=dict)
    arrived: Dict[str, bool] = field(default_factory=dict)

    # ---------- communication internals (may be None if unavailable) ---------
    # adj_per_round[r] → shape (N, N) soft adjacency for comm round r
    adj_per_round: Optional[List[np.ndarray]] = None
    # hard_adj → shape (N, N) binary adjacency (after threshold)
    hard_adj: Optional[np.ndarray] = None
    # messages[i] → shape (message_dim,) raw embedding for agent i
    messages: Optional[np.ndarray] = None  # (N, msg_dim)
    # agg_messages[i] → after GAT aggregation
    agg_messages: Optional[np.ndarray] = None  # (N, msg_dim)
    # policy logits per agent, shape (N, num_actions)
    logits: Optional[np.ndarray] = None

    # ---------- derived scalars (filled during collection) -------------------
    comm_density: float = 0.0  # fraction of directed edges that are active
    total_reward: float = 0.0


@dataclass
class HighwayCommEpisodeData:
    """Complete communication data for a single episode."""

    episode_idx: int
    agent_names: List[str] = field(default_factory=list)
    steps: List[HighwayCommStepRecord] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    has_comm_data: bool = False

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def num_agents(self) -> int:
        return len(self.agent_names)

    @property
    def total_rewards(self) -> Dict[str, float]:
        out: Dict[str, float] = {a: 0.0 for a in self.agent_names}
        for s in self.steps:
            for a in self.agent_names:
                out[a] += s.rewards.get(a, 0.0)
        return out

    @property
    def mean_comm_density(self) -> float:
        vals = [s.comm_density for s in self.steps if s.adj_per_round is not None]
        return float(np.mean(vals)) if vals else float("nan")

    def collision_steps(self) -> List[int]:
        """Steps where at least one agent crashed."""
        return [
            s.step
            for s in self.steps
            if any(s.crashed.get(a, False) for a in self.agent_names)
        ]

    def arrival_steps(self) -> List[int]:
        """Steps where at least one agent successfully arrived."""
        return [
            s.step
            for s in self.steps
            if any(s.arrived.get(a, False) for a in self.agent_names)
        ]


@dataclass
class MAGICHighwayAnalysisData:
    """Top-level container returned by MAGICHighwayCommCollector.collect()."""

    num_agents: int
    duration: int
    num_comm_rounds: int
    message_dim: int
    episodes: List[HighwayCommEpisodeData] = field(default_factory=list)
    has_comm_data: bool = False

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def agent_names(self) -> List[str]:
        if not self.episodes:
            return []
        return self.episodes[0].agent_names

    @property
    def mean_episode_length(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.length for e in self.episodes]))

    @property
    def collision_rate(self) -> float:
        """Fraction of episodes with at least one collision."""
        if not self.episodes:
            return 0.0
        with_collisions = sum(1 for e in self.episodes if len(e.collision_steps()) > 0)
        return with_collisions / len(self.episodes)

    @property
    def arrival_rate(self) -> float:
        """Fraction of episodes with at least one successful arrival."""
        if not self.episodes:
            return 0.0
        with_arrivals = sum(1 for e in self.episodes if len(e.arrival_steps()) > 0)
        return with_arrivals / len(self.episodes)

    def all_steps(self) -> List[HighwayCommStepRecord]:
        return [s for e in self.episodes for s in e.steps]

    def comm_steps(self) -> List[HighwayCommStepRecord]:
        """Steps that contain valid communication data."""
        return [s for s in self.all_steps() if s.adj_per_round is not None]

    def adj_stack(self, round_idx: int = 0) -> np.ndarray:
        """Stack adj matrices for the given communication round → (T, N, N)."""
        mats = [
            s.adj_per_round[round_idx]
            for s in self.comm_steps()
            if len(s.adj_per_round or []) > round_idx
        ]
        if not mats:
            return np.empty((0, self.num_agents, self.num_agents))
        return np.stack(mats, axis=0)

    def message_stack(self) -> np.ndarray:
        """Stack message vectors → (T, N, message_dim)."""
        msgs = [s.messages for s in self.comm_steps() if s.messages is not None]
        if not msgs:
            return np.empty((0, self.num_agents, self.message_dim))
        return np.stack(msgs, axis=0)

    def context_at_comm_steps(self) -> Dict[str, np.ndarray]:
        """
        Returns a dict of context feature arrays aligned with comm_steps().

        Features per agent:
            speed_<agent>: Speed of vehicle
            reward_<agent>: Reward at this step
            crashed_<agent>: Binary crashed flag
            arrived_<agent>: Binary arrived flag
            comm_density: Overall edge density in adjacency
            step_fraction: t / duration
        """
        csteps = self.comm_steps()
        if not csteps or not self.agent_names:
            return {}

        out: Dict[str, List] = {f"speed_{a}": [] for a in self.agent_names}
        out.update({f"reward_{a}": [] for a in self.agent_names})
        out.update({f"crashed_{a}": [] for a in self.agent_names})
        out.update({f"arrived_{a}": [] for a in self.agent_names})
        out["comm_density"] = []
        out["step_fraction"] = []

        for s in csteps:
            for a in self.agent_names:
                out[f"speed_{a}"].append(s.speeds.get(a, 0.0))
                out[f"reward_{a}"].append(s.rewards.get(a, 0.0))
                out[f"crashed_{a}"].append(1.0 if s.crashed.get(a, False) else 0.0)
                out[f"arrived_{a}"].append(1.0 if s.arrived.get(a, False) else 0.0)
            out["comm_density"].append(s.comm_density)
            out["step_fraction"].append(s.step / max(self.duration, 1))

        return {k: np.array(v, dtype=np.float32) for k, v in out.items()}


class MAGICHighwayCommCollector:
    """Collect MAGIC communication rollouts for highway intersection environment."""

    def __init__(
        self,
        num_agents: int = 4,
        duration: int = 13,
        num_comm_rounds: int = 2,
        message_dim: int = 64,
    ):
        """
        Initialize the collector.

        Args:
            num_agents: Number of controlled vehicles
            duration: Episode duration in policy steps
            num_comm_rounds: Number of GAT communication rounds
            message_dim: Dimension of message embeddings
        """
        self.num_agents = num_agents
        self.duration = duration
        self.num_comm_rounds = num_comm_rounds
        self.message_dim = message_dim
        self._warned_no_comm = False

    def collect(
        self,
        env: Any,
        agent: Any,
        n_episodes: int = 30,
        max_steps_per_episode: int = 100,
    ) -> MAGICHighwayAnalysisData:
        """
        Run evaluation episodes and collect communication data.

        Args:
            env: PettingZoo parallel environment (wrapped)
            agent: MAGIC multi-agent trained policy
            n_episodes: Number of evaluation episodes
            max_steps_per_episode: Maximum steps per episode

        Returns:
            MAGICHighwayAnalysisData with collected metrics
        """
        agent.set_running_mode("eval")

        agent_names = list(getattr(env, "possible_agents", []))
        if not agent_names:
            raise ValueError("Environment does not expose possible_agents")

        data = MAGICHighwayAnalysisData(
            num_agents=self.num_agents,
            duration=self.duration,
            num_comm_rounds=self.num_comm_rounds,
            message_dim=self.message_dim,
        )

        global_has_comm = False

        for ep_idx in range(n_episodes):
            obs, infos = env.reset()
            episode = HighwayCommEpisodeData(
                episode_idx=ep_idx,
                agent_names=agent_names,
            )

            for t in range(max_steps_per_episode):
                # Get actions from MAGIC policy
                actions, _, outputs = agent.act(
                    obs,
                    timestep=t,
                    timesteps=max_steps_per_episode,
                )

                # Extract communication data from outputs
                adj_mats = outputs.get("adj_matrices")
                hard_adj = outputs.get("hard_adj")
                messages = outputs.get("messages")
                agg_messages = outputs.get("agg_messages")
                logits = outputs.get("net_output")

                has_comm_this_step = adj_mats is not None
                if has_comm_this_step:
                    global_has_comm = True
                    episode.has_comm_data = True

                # Convert actions to integers
                actions_int = {
                    name: int(np.asarray(actions[name]).ravel()[0])
                    for name in agent_names
                }

                # Step environment
                next_obs, rewards, terminated, truncated, infos = env.step(actions)

                # Build step record
                step_record = HighwayCommStepRecord(step=t)

                # Extract state information
                for name in agent_names:
                    agent_obs = np.asarray(obs[name], dtype=np.float32).ravel()
                    reward = float(np.asarray(rewards[name]).ravel()[0])

                    # Decode position and velocity from observation
                    # Observation format: [presence, x, y, vx, vy] * vehicles_count
                    # First vehicle is ego (self)
                    ego_present = agent_obs[0]
                    if ego_present > 0.5:
                        ego_x = float(agent_obs[1])
                        ego_y = float(agent_obs[2])
                        ego_vx = float(agent_obs[3])
                        ego_vy = float(agent_obs[4])
                    else:
                        ego_x = ego_y = ego_vx = ego_vy = 0.0

                    speed = float(np.sqrt(ego_vx**2 + ego_vy**2))

                    # Extract crash/arrival status from info
                    info = infos.get(name, {})
                    crashed = info.get("crashed", False)
                    arrived = info.get("arrived", False)

                    step_record.positions[name] = (ego_x, ego_y)
                    step_record.velocities[name] = (ego_vx, ego_vy)
                    step_record.speeds[name] = speed
                    step_record.actions[name] = actions_int[name]
                    step_record.rewards[name] = reward
                    step_record.crashed[name] = crashed
                    step_record.arrived[name] = arrived
                    step_record.total_reward += reward

                # Compute inter-vehicle distances
                for i, name_i in enumerate(agent_names):
                    for j, name_j in enumerate(agent_names):
                        if i < j:
                            dist = _euclidean(
                                step_record.positions[name_i],
                                step_record.positions[name_j],
                            )
                            step_record.inter_vehicle_distances[(name_i, name_j)] = dist

                # Store communication data if available
                if has_comm_this_step:
                    adj_array = np.asarray(adj_mats, dtype=np.float32)
                    if adj_array.ndim == 2:
                        # Single round: (N, N) → [(N, N)]
                        step_record.adj_per_round = [adj_array]
                    elif adj_array.ndim == 3:
                        # Multi-round: (num_rounds, N, N)
                        step_record.adj_per_round = [
                            adj_array[r] for r in range(adj_array.shape[0])
                        ]
                    else:
                        step_record.adj_per_round = None

                    if hard_adj is not None:
                        step_record.hard_adj = np.asarray(hard_adj, dtype=np.float32)

                    if messages is not None:
                        step_record.messages = np.asarray(messages, dtype=np.float32)

                    if agg_messages is not None:
                        step_record.agg_messages = np.asarray(
                            agg_messages, dtype=np.float32
                        )

                    if logits is not None:
                        step_record.logits = np.asarray(logits, dtype=np.float32)

                    # Compute communication density (fraction of active edges)
                    if step_record.adj_per_round:
                        # Use first round adjacency
                        adj_first = step_record.adj_per_round[0]
                        # Exclude self-loops
                        mask = ~np.eye(adj_first.shape[0], dtype=bool)
                        active_edges = (adj_first[mask] > 0.5).sum()
                        total_edges = mask.sum()
                        step_record.comm_density = float(active_edges) / max(
                            total_edges, 1
                        )

                episode.steps.append(step_record)
                obs = next_obs

                # Check termination
                done = any(
                    bool(np.asarray(v).ravel()[0])
                    for v in list(terminated.values()) + list(truncated.values())
                )
                if done:
                    episode.terminated = any(
                        bool(np.asarray(v).ravel()[0]) for v in terminated.values()
                    )
                    episode.truncated = any(
                        bool(np.asarray(v).ravel()[0]) for v in truncated.values()
                    )
                    break

            data.episodes.append(episode)

            # Print episode summary
            collisions = len(episode.collision_steps())
            arrivals = len(episode.arrival_steps())
            total_reward = sum(episode.total_rewards.values())
            print(
                f"  Episode {ep_idx + 1:3d}/{n_episodes} | "
                f"Len: {episode.length:3d} | "
                f"Reward: {total_reward:+7.2f} | "
                f"Collisions: {collisions} | "
                f"Arrivals: {arrivals} | "
                f"Comm: {episode.mean_comm_density:.2f}"
            )

        data.has_comm_data = global_has_comm

        if not global_has_comm and not self._warned_no_comm:
            warnings.warn(
                "No communication data was captured. Make sure the MAGIC agent "
                "returns 'adj_matrices' and 'messages' in its act() outputs.",
                UserWarning,
            )
            self._warned_no_comm = True

        return data
