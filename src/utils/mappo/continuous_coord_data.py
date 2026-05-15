"""
continuous_coord_eval_data.py
==============================
Data structures for Continuous Coordination (Rendezvous Pursuit) evaluation.

Observation layout per agent (non-typed, n agents, mt targets):
  obs[0]  : own px  [0,1]
  obs[1]  : own py  [0,1]
  obs[2]  : own vx
  obs[3]  : own vy
  obs[4]  : agent_id_norm = i / (n-1)
  obs[5]  : time_left = 1 - step / max_cycles
  obs[6 + type_dim + s*4 + {0..3}] : teammate s relative (dx, dy, dvx, dvy)
  obs[6 + type_dim + (n-1)*4 + t*4 + {0..3}] : target t (rel_dx, rel_dy, k_norm, deadline_norm)
  (if typed: +type_dim offset applies throughout)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ACTION_NAMES = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→",
    4: "↖",
    5: "↗",
    6: "↙",
    7: "↘",
    8: "stay",
}


# ─── Per-step agent state ─────────────────────────────────────────────────────


@dataclass
class AgentState:
    px: float = 0.0  # absolute x position [0,1]
    py: float = 0.0  # absolute y position [0,1]
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0  # ||v||

    @property
    def position(self) -> tuple[float, float]:
        return (self.px, self.py)


@dataclass
class TargetState:
    """Decoded target state for one slot (active or inactive)."""

    active: bool = False
    abs_x: float = 0.0  # absolute x (decoded using observer pos)
    abs_y: float = 0.0  # absolute y
    k_req: float = 0.0  # k_req_norm * num_agents → raw k_req
    deadline_norm: float = 0.0
    deadline_steps: float = 0.0  # deadline_norm * deadline_avg → approx steps remaining


# ─── Events ──────────────────────────────────────────────────────────────────


@dataclass
class CaptureEvent:
    """A successful target capture event."""

    step: int
    target_idx: int
    capture_reward: float
    sync_reward: float
    total_agents_in_zone: int  # inferred from reward / (capture_reward_per_agent)
    k_required: int  # estimated from target k_req (pre-capture)


@dataclass
class CollisionEvent:
    """A pair of agents within collision radius at this step."""

    step: int
    agent_a: str
    agent_b: str
    distance: float


# ─── Step record ─────────────────────────────────────────────────────────────


@dataclass
class StepRecord:
    step: int
    agent_states: dict[str, AgentState] = field(default_factory=dict)
    targets: list[TargetState] = field(default_factory=list)  # length = max_targets
    rewards: dict[str, float] = field(default_factory=dict)
    actions: dict[str, int] = field(default_factory=dict)
    active_target_count: int = 0

    @property
    def total_reward(self) -> float:
        return sum(self.rewards.values())

    @property
    def mean_inter_agent_dist(self) -> float:
        states = list(self.agent_states.values())
        if len(states) < 2:
            return 0.0
        dists = []
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                dx = states[i].px - states[j].px
                dy = states[i].py - states[j].py
                dists.append(float(np.sqrt(dx * dx + dy * dy)))
        return float(np.mean(dists)) if dists else 0.0

    @property
    def min_inter_agent_dist(self) -> float:
        states = list(self.agent_states.values())
        if len(states) < 2:
            return 1.0
        min_d = 1.0
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                dx = states[i].px - states[j].px
                dy = states[i].py - states[j].py
                d = float(np.sqrt(dx * dx + dy * dy))
                if d < min_d:
                    min_d = d
        return min_d

    def coord_entropy(self) -> float:
        """Shannon entropy of the joint action distribution at this step."""
        actions = list(self.actions.values())
        if not actions:
            return 0.0
        counts = np.bincount(actions, minlength=9).astype(float)
        probs = counts / counts.sum()
        nonzero = probs[probs > 0]
        return float(-np.sum(nonzero * np.log2(nonzero)))


# ─── Episode data ─────────────────────────────────────────────────────────────


@dataclass
class EpisodeData:
    episode_idx: int
    steps: list[StepRecord] = field(default_factory=list)
    capture_events: list[CaptureEvent] = field(default_factory=list)
    collision_events: list[CollisionEvent] = field(default_factory=list)
    num_agents: int = 4
    max_targets: int = 3
    max_cycles: int = 200
    agents: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def total_rewards(self) -> dict[str, float]:
        out = {a: 0.0 for a in self.agents}
        for step in self.steps:
            for a, r in step.rewards.items():
                out[a] = out.get(a, 0.0) + r
        return out

    @property
    def episode_return(self) -> float:
        return sum(self.total_rewards.values()) / max(self.num_agents, 1)

    @property
    def total_captures(self) -> int:
        return len(self.capture_events)

    @property
    def total_collisions(self) -> int:
        return len(self.collision_events)

    @property
    def collision_rate(self) -> float:
        """Fraction of steps with at least one collision."""
        collision_steps = {e.step for e in self.collision_events}
        return len(collision_steps) / max(self.length, 1)

    @property
    def mean_sync_bonus(self) -> float:
        bonuses = [e.sync_reward for e in self.capture_events]
        return float(np.mean(bonuses)) if bonuses else 0.0

    def mean_inter_agent_dist_series(self) -> np.ndarray:
        return np.array([s.mean_inter_agent_dist for s in self.steps], dtype=np.float32)

    def min_inter_agent_dist_series(self) -> np.ndarray:
        return np.array([s.min_inter_agent_dist for s in self.steps], dtype=np.float32)

    def reward_series(self) -> np.ndarray:
        return np.array([s.total_reward for s in self.steps], dtype=np.float32)

    def cumulative_reward_series(self) -> np.ndarray:
        return np.cumsum(self.reward_series())

    def active_targets_series(self) -> np.ndarray:
        return np.array([s.active_target_count for s in self.steps], dtype=np.int32)

    def coord_entropy_series(self) -> np.ndarray:
        return np.array([s.coord_entropy() for s in self.steps], dtype=np.float32)

    def agent_positions(self, agent: str) -> list[tuple[float, float]]:
        return [
            s.agent_states[agent].position
            for s in self.steps
            if agent in s.agent_states
        ]

    def expirations(self) -> int:
        """Estimate expirations from large negative reward spikes."""
        pen_steps = 0
        for step in self.steps:
            if step.total_reward < -1.5:
                pen_steps += 1
        return pen_steps


# ─── Aggregated evaluation data ───────────────────────────────────────────────


@dataclass
class EvalData:
    episodes: list[EpisodeData] = field(default_factory=list)
    num_agents: int = 4
    max_targets: int = 3
    max_cycles: int = 200
    agents: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.episodes)

    def episode_returns(self) -> list[float]:
        return [ep.episode_return for ep in self.episodes]

    def mean_return(self) -> float:
        r = self.episode_returns()
        return float(np.mean(r)) if r else 0.0

    def std_return(self) -> float:
        r = self.episode_returns()
        return float(np.std(r)) if r else 0.0

    def mean_captures_per_episode(self) -> float:
        return float(np.mean([ep.total_captures for ep in self.episodes]))

    def mean_collisions_per_episode(self) -> float:
        return float(np.mean([ep.total_collisions for ep in self.episodes]))

    def mean_collision_rate(self) -> float:
        return float(np.mean([ep.collision_rate for ep in self.episodes]))

    def capture_steps(self) -> list[int]:
        return [e.step for ep in self.episodes for e in ep.capture_events]

    def sync_bonuses(self) -> list[float]:
        return [e.sync_reward for ep in self.episodes for e in ep.capture_events]

    def per_agent_returns(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {a: [] for a in self.agents}
        for ep in self.episodes:
            for a, r in ep.total_rewards.items():
                out.setdefault(a, []).append(r)
        return out

    def mean_cumulative_return_over_time(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mat = np.full((len(self.episodes), self.max_cycles), np.nan)
        for i, ep in enumerate(self.episodes):
            series = ep.cumulative_reward_series()
            n = min(len(series), self.max_cycles)
            mat[i, :n] = series[:n]
        t = np.arange(self.max_cycles)
        return t, np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)

    def mean_inter_agent_dist_over_time(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mat = np.full((len(self.episodes), self.max_cycles), np.nan)
        for i, ep in enumerate(self.episodes):
            series = ep.mean_inter_agent_dist_series()
            n = min(len(series), self.max_cycles)
            mat[i, :n] = series[:n]
        t = np.arange(self.max_cycles)
        return t, np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)

    def mean_active_targets_over_time(self) -> tuple[np.ndarray, np.ndarray]:
        mat = np.full((len(self.episodes), self.max_cycles), np.nan)
        for i, ep in enumerate(self.episodes):
            series = ep.active_targets_series().astype(float)
            n = min(len(series), self.max_cycles)
            mat[i, :n] = series[:n]
        t = np.arange(self.max_cycles)
        return t, np.nanmean(mat, axis=0)

    def position_heatmap(self, agent: str, grid_res: int = 20) -> np.ndarray:
        grid = np.zeros((grid_res, grid_res), dtype=np.float32)
        for ep in self.episodes:
            for pos in ep.agent_positions(agent):
                col = min(int(pos[0] * grid_res), grid_res - 1)
                row = min(int((1.0 - pos[1]) * grid_res), grid_res - 1)  # y-flip
                grid[row, col] += 1
        return grid

    def coord_entropy_over_time(self) -> tuple[np.ndarray, np.ndarray]:
        mat = np.full((len(self.episodes), self.max_cycles), np.nan)
        for i, ep in enumerate(self.episodes):
            series = ep.coord_entropy_series()
            n = min(len(series), self.max_cycles)
            mat[i, :n] = series[:n]
        t = np.arange(self.max_cycles)
        return t, np.nanmean(mat, axis=0)

    def action_distribution(self) -> np.ndarray:
        counts = np.zeros(9, dtype=float)
        for ep in self.episodes:
            for step in ep.steps:
                for a in step.actions.values():
                    counts[a] += 1
        total = counts.sum()
        return counts / total if total > 0 else counts
