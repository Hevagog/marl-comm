from __future__ import annotations

from dataclasses import dataclass, field

ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

RED_AGENT = "agent_0"
BLUE_AGENT = "agent_1"


@dataclass
class CoinEvent:
    """A single coin-pickup event within an episode."""

    step: int
    coin_color: str  # "red" | "blue"
    picker: str  # "agent_0" | "agent_1"
    is_steal: bool  # True iff picker's color != coin_color

    @property
    def victim(self) -> str | None:
        """Return the agent penalized by this steal, or None if not a steal."""
        if not self.is_steal:
            return None
        return BLUE_AGENT if self.picker == RED_AGENT else RED_AGENT


@dataclass
class StepRecord:
    """Per-step snapshot of the environment state."""

    step: int
    red_pos: tuple[int, int]
    blue_pos: tuple[int, int]
    red_coin_pos: tuple[int, int] | None  # None = doesn't exist this step
    blue_coin_pos: tuple[int, int] | None
    actions: dict[str, int]  # agent_name -> action_id
    rewards: dict[str, float]  # agent_name -> reward


@dataclass
class EpisodeData:
    """All data for a single evaluation episode."""

    episode_idx: int
    steps: list[StepRecord] = field(default_factory=list)
    events: list[CoinEvent] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def total_rewards(self) -> dict[str, float]:
        return {
            agent: sum(s.rewards.get(agent, 0.0) for s in self.steps)
            for agent in (RED_AGENT, BLUE_AGENT)
        }

    def steals_by(self, agent: str) -> int:
        return sum(1 for e in self.events if e.picker == agent and e.is_steal)

    def own_pickups_by(self, agent: str) -> int:
        return sum(1 for e in self.events if e.picker == agent and not e.is_steal)

    def total_pickups_by(self, agent: str) -> int:
        return sum(1 for e in self.events if e.picker == agent)

    def steals_suffered_by(self, agent: str) -> int:
        return sum(1 for e in self.events if e.victim == agent)

    def cooperation_rate_of(self, agent: str) -> float:
        """Fraction of pickups that were own-coin. NaN if no pickups."""
        total = self.total_pickups_by(agent)
        return self.own_pickups_by(agent) / total if total > 0 else float("nan")


@dataclass
class EvalData:
    """Aggregated data across all evaluation episodes."""

    episodes: list[EpisodeData] = field(default_factory=list)
    grid_size: int = 7
    pick_reward: float = 1.0
    steal_penalty: float = -2.0

    def __len__(self) -> int:
        return len(self.episodes)

    def mean_total_reward(self, agent: str) -> float:
        rewards = [ep.total_rewards[agent] for ep in self.episodes]
        return float(sum(rewards) / len(rewards)) if rewards else 0.0

    def overall_steal_rate(self, agent: str) -> float:
        total_pickups = sum(ep.total_pickups_by(agent) for ep in self.episodes)
        total_steals = sum(ep.steals_by(agent) for ep in self.episodes)
        return total_steals / total_pickups if total_pickups > 0 else 0.0

    def overall_cooperation_rate(self, agent: str) -> float:
        return 1.0 - self.overall_steal_rate(agent)

    def per_episode_rewards(self, agent: str) -> list[float]:
        return [ep.total_rewards[agent] for ep in self.episodes]

    def per_episode_steal_rates(self, agent: str) -> list[float]:
        rates = []
        for ep in self.episodes:
            total = ep.total_pickups_by(agent)
            rates.append(ep.steals_by(agent) / total if total > 0 else 0.0)
        return rates

    def position_counts(self, agent: str) -> list[tuple[int, int]]:
        """All grid positions visited by agent across all episodes."""
        positions = []
        for ep in self.episodes:
            for step in ep.steps:
                pos = step.red_pos if agent == RED_AGENT else step.blue_pos
                positions.append(pos)
        return positions

    def coin_events_at_step(self, agent: str) -> dict[int, list[CoinEvent]]:
        """Returns dict mapping step -> list of coin events by *agent*."""
        result: dict[int, list[CoinEvent]] = {}
        for ep in self.episodes:
            for ev in ep.events:
                if ev.picker == agent:
                    result.setdefault(ev.step, []).append(ev)
        return result
