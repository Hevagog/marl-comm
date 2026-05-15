"""
warehouse_eval_data.py
======================
Data structures for multi-robot warehouse evaluation.

The warehouse environment exposes a rich per-agent info dict on every step:
    info[agent_name] = {
        "active": bool,
        "carrying": int,            # items currently held
        "phase": int,               # 0=UNPICKED, 1=TO_TREATMENT, 2=TREATING, 3=TO_GOAL
        "total_deliveries": int,    # cumulative deliveries by this agent
        "stranded": bool,           # failed & awaiting rescue
        "dragging": bool,           # currently towing another agent
        "rescue_target": int | None,
        "being_dragged_by": int | None,
        "burst_failed": bool,
        "battery_dead": bool,
        "failed": bool,
        "battery": float,           # current battery level (if enabled)
        "charging": bool,
        "pending_tasks": int,       # task queue length (if deadlines enabled)
        "expired_tasks": int,       # cumulative expired tasks
        "speed": float,
        "capacity": int,
        "fragility": float,
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ─── Agent / action metadata ──────────────────────────────────────────────────

PHASE_NAMES = {
    0: "Searching",
    1: "To Treatment",
    2: "Treating",
    3: "To Goal",
}

ACTION_NAMES = {
    0: "STAY",
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
    5: "INTERACT",
}

# Heterogeneous speed groups (from config agent_speed_options=(1,1,2))
SPEED_FAST = 2
SPEED_SLOW = 1


# ─── Per-step agent state ─────────────────────────────────────────────────────


@dataclass
class AgentState:
    """Snapshot of a single agent at one environment step."""

    active: bool = True
    carrying: int = 0
    phase: int = 0
    battery: float = 160.0
    speed: float = 1.0
    capacity: int = 1
    fragility: float = 1.0
    stranded: bool = False
    charging: bool = False
    dragging: bool = False
    rescue_target: int | None = None
    being_dragged_by: int | None = None
    burst_failed: bool = False
    battery_dead: bool = False
    failed: bool = False
    total_deliveries: int = 0
    deliveries: int = 0  # per-agent cumulative deliveries (env.info["deliveries"])
    last_delivery_step: int = -1  # step at which this agent last delivered (-1 = never)
    on_rendezvous: bool = False
    position: tuple[int, int] = (0, 0)  # (row, col) decoded from obs

    @property
    def is_battery_critical(self) -> bool:
        return self.battery < 30.0

    @property
    def is_fast(self) -> bool:
        return self.speed >= SPEED_FAST

    @property
    def is_high_capacity(self) -> bool:
        return self.capacity >= 2


@dataclass
class StepRecord:
    """All per-agent data captured at one environment step."""

    step: int
    agent_states: dict[str, AgentState] = field(default_factory=dict)
    actions: dict[str, int] = field(default_factory=dict)
    rewards: dict[str, float] = field(default_factory=dict)

    # Environment-level fields (same for all agents)
    pending_tasks: int = 0
    expired_tasks_cumulative: int = 0

    @property
    def total_reward(self) -> float:
        return sum(self.rewards.values())

    @property
    def active_agents(self) -> list[str]:
        return [a for a, s in self.agent_states.items() if s.active]

    @property
    def battery_critical_agents(self) -> list[str]:
        return [a for a, s in self.agent_states.items() if s.is_battery_critical]

    @property
    def rescuing_agents(self) -> list[str]:
        return [a for a, s in self.agent_states.items() if s.dragging]

    @property
    def stranded_agents(self) -> list[str]:
        return [a for a, s in self.agent_states.items() if s.stranded]

    def phase_counts(self) -> dict[int, int]:
        counts = {p: 0 for p in PHASE_NAMES}
        for s in self.agent_states.values():
            if s.active:
                counts[s.phase] = counts.get(s.phase, 0) + 1
        return counts


# ─── Episode data ─────────────────────────────────────────────────────────────


@dataclass
class EpisodeData:
    """All step records for one evaluation episode."""

    episode_idx: int
    steps: list[StepRecord] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    agents: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def total_deliveries(self) -> int:
        """Total deliveries this episode.

        Prefers the env-global counter (`AgentState.total_deliveries`) which is
        replicated to every agent's info dict — taking the **max** across
        agents avoids the historical bug where summing replicated globals
        inflated the count by `num_agents`.  Falls back to summing per-agent
        `deliveries` when the global counter is absent (older runs).
        """
        if not self.steps:
            return 0
        last = self.steps[-1]
        if not last.agent_states:
            return 0
        global_max = max(s.total_deliveries for s in last.agent_states.values())
        if global_max > 0:
            return global_max
        return sum(s.deliveries for s in last.agent_states.values())

    @property
    def deliveries_by_agent(self) -> dict[str, int]:
        """Per-agent cumulative deliveries at the last step.

        Reads `AgentState.deliveries` (env.info["deliveries"]) which is now a
        true per-agent counter; falls back to 0 when not present.
        """
        if not self.steps:
            return {}
        last = self.steps[-1]
        return {a: s.deliveries for a, s in last.agent_states.items()}

    # ── Scenario-specific aggregates ──────────────────────────────────────

    def synchronized_delivery_events(self, window: int = 20) -> int:
        """Count deliveries that fell within `window` steps of another agent's
        delivery this episode.  Drives the team-sync bonus metric (Scenario 1).
        """
        delivery_steps_per_agent = {a: self.delivery_steps(a) for a in self.agents}
        sync = 0
        for a, my_steps in delivery_steps_per_agent.items():
            for t in my_steps:
                for b, other_steps in delivery_steps_per_agent.items():
                    if a == b:
                        continue
                    if any(abs(t - t2) <= window for t2 in other_steps):
                        sync += 1
                        break
        return sync

    def rendezvous_steps(self) -> int:
        """Steps where ≥1 agent was on a rendezvous cell (Scenario 2)."""
        return sum(
            1
            for s in self.steps
            if any(state.on_rendezvous for state in s.agent_states.values())
        )

    @property
    def expired_tasks_total(self) -> int:
        """Cumulative expired tasks by episode end."""
        if not self.steps:
            return 0
        return self.steps[-1].expired_tasks_cumulative

    @property
    def total_rewards(self) -> dict[str, float]:
        out: dict[str, float] = {a: 0.0 for a in self.agents}
        for step in self.steps:
            for a, r in step.rewards.items():
                out[a] = out.get(a, 0.0) + r
        return out

    def rescue_event_steps(self) -> list[int]:
        """Steps where at least one agent is performing a rescue."""
        return [s.step for s in self.steps if s.rescuing_agents]

    def stranded_event_steps(self) -> list[int]:
        """Steps where at least one agent is stranded."""
        return [s.step for s in self.steps if s.stranded_agents]

    def delivery_steps(self, agent: str) -> list[int]:
        """Steps where this agent's per-agent delivery count increased."""
        events = []
        prev = 0
        for step in self.steps:
            state = step.agent_states.get(agent)
            if state is not None:
                if state.deliveries > prev:
                    events.append(step.step)
                    prev = state.deliveries
        return events

    def battery_series(self, agent: str) -> np.ndarray:
        return np.array(
            [
                s.agent_states[agent].battery
                for s in self.steps
                if agent in s.agent_states
            ],
            dtype=np.float32,
        )

    def phase_series(self, agent: str) -> np.ndarray:
        return np.array(
            [
                s.agent_states[agent].phase
                for s in self.steps
                if agent in s.agent_states
            ],
            dtype=np.int32,
        )

    def cumulative_deliveries_series(self) -> np.ndarray:
        """Global cumulative deliveries per step.

        Uses max() across agents because total_deliveries is a global counter
        replicated to every agent's info dict — summing it inflates by num_agents.
        """
        return np.array(
            [
                max((s.total_deliveries for s in step.agent_states.values()), default=0)
                for step in self.steps
            ],
            dtype=np.float32,
        )


# ─── Aggregated evaluation data ───────────────────────────────────────────────


@dataclass
class EvalData:
    """Top-level container returned by WarehouseEvalCollector.collect()."""

    episodes: list[EpisodeData] = field(default_factory=list)
    grid_height: int = 12
    grid_width: int = 16
    max_cycles: int = 500
    num_agents: int = 4
    agents: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.episodes)

    def all_steps(self) -> list[StepRecord]:
        return [s for ep in self.episodes for s in ep.steps]

    # ── Reward aggregates ──────────────────────────────────────────────────────

    def mean_total_reward(self, agent: str) -> float:
        rewards = [ep.total_rewards.get(agent, 0.0) for ep in self.episodes]
        return float(np.mean(rewards)) if rewards else 0.0

    def mean_episode_reward(self) -> float:
        totals = [sum(ep.total_rewards.values()) for ep in self.episodes]
        return float(np.mean(totals)) if totals else 0.0

    # ── Delivery aggregates ───────────────────────────────────────────────────

    def mean_deliveries_per_episode(self) -> float:
        return float(np.mean([ep.total_deliveries for ep in self.episodes]))

    def mean_expired_per_episode(self) -> float:
        return float(np.mean([ep.expired_tasks_total for ep in self.episodes]))

    def deliveries_by_agent(self) -> dict[str, float]:
        """Mean deliveries per agent across episodes."""
        out: dict[str, list[float]] = {a: [] for a in self.agents}
        for ep in self.episodes:
            for a, d in ep.deliveries_by_agent.items():
                out.setdefault(a, []).append(float(d))
        return {a: float(np.mean(v)) if v else 0.0 for a, v in out.items()}

    # ── Delivery rate over time ───────────────────────────────────────────────

    def mean_cumulative_deliveries_over_time(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (time_steps, mean_deliveries, std_deliveries) aligned to max_cycles."""
        max_len = self.max_cycles
        mat = np.full((len(self.episodes), max_len), np.nan)
        for i, ep in enumerate(self.episodes):
            series = ep.cumulative_deliveries_series()
            n = min(len(series), max_len)
            mat[i, :n] = series[:n]
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        return np.arange(max_len), mean, std

    # ── Phase distribution ────────────────────────────────────────────────────

    def phase_fraction_over_time(self) -> np.ndarray:
        """Returns (max_cycles, 4) array: fraction of active agents in each phase per step."""
        max_len = self.max_cycles
        phase_mat = np.zeros((max_len, 4), dtype=np.float32)
        count_mat = np.zeros(max_len, dtype=np.float32)
        for ep in self.episodes:
            for step in ep.steps:
                t = step.step
                if t >= max_len:
                    continue
                active = [s for s in step.agent_states.values() if s.active]
                if not active:
                    continue
                for p in range(4):
                    phase_mat[t, p] += sum(1 for s in active if s.phase == p) / len(
                        active
                    )
                count_mat[t] += 1
        safe = np.where(count_mat > 0, count_mat, 1)
        return phase_mat / safe[:, None]

    # ── Spatial data ─────────────────────────────────────────────────────────

    def position_counts(self, agent: str) -> list[tuple[int, int]]:
        return [
            s.agent_states[agent].position
            for ep in self.episodes
            for s in ep.steps
            if agent in s.agent_states and s.agent_states[agent].active
        ]

    def position_heatmap(self, agent: str) -> np.ndarray:
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        for row, col in self.position_counts(agent):
            if 0 <= row < self.grid_height and 0 <= col < self.grid_width:
                grid[row, col] += 1
        return grid

    # ── Battery aggregates ────────────────────────────────────────────────────

    def mean_battery_over_time(self, agent: str) -> tuple[np.ndarray, np.ndarray]:
        """Returns (time_steps, mean_battery) for the given agent."""
        max_len = self.max_cycles
        mat = np.full((len(self.episodes), max_len), np.nan)
        for i, ep in enumerate(self.episodes):
            series = ep.battery_series(agent)
            n = min(len(series), max_len)
            mat[i, :n] = series[:n]
        return np.arange(max_len), np.nanmean(mat, axis=0)

    # ── Rescue / failure aggregates ───────────────────────────────────────────

    def mean_rescue_events_per_episode(self) -> float:
        return float(np.mean([len(ep.rescue_event_steps()) for ep in self.episodes]))

    def mean_episode_length(self) -> float:
        return float(np.mean([ep.length for ep in self.episodes]))

    # ── Role-based summary ────────────────────────────────────────────────────

    def role_deliveries(self) -> dict[str, float]:
        """Mean deliveries grouped by agent speed role (fast vs slow)."""
        fast_totals: list[float] = []
        slow_totals: list[float] = []
        for ep in self.episodes:
            for a, d in ep.deliveries_by_agent.items():
                if not ep.steps:
                    continue
                state = ep.steps[-1].agent_states.get(a)
                if state is None:
                    continue
                if state.is_fast:
                    fast_totals.append(float(d))
                else:
                    slow_totals.append(float(d))
        return {
            "fast (speed=2)": float(np.mean(fast_totals)) if fast_totals else 0.0,
            "slow (speed=1)": float(np.mean(slow_totals)) if slow_totals else 0.0,
        }
