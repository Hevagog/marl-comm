from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

AGENT_A = "agent_0"
AGENT_B = "agent_1"
AGENTS = (AGENT_A, AGENT_B)

MOVEMENT_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}


@dataclass
class StepRecord:
    """Per-step snapshot for Blind-Spot Navigation."""

    step: int
    positions: Dict[str, Tuple[int, int]]
    actions: Dict[str, int]
    messages: Dict[str, Optional[int]]
    rewards: Dict[str, float]
    reached: Dict[str, bool]
    trap_hits: Dict[str, bool]
    dist_to_goal: Dict[str, int]


@dataclass
class EpisodeData:
    """All data for a single evaluation episode."""

    episode_idx: int
    steps: List[StepRecord] = field(default_factory=list)
    traps: List[Tuple[int, int]] = field(default_factory=list)
    goal: Tuple[int, int] = (0, 0)
    terminated: bool = False
    truncated: bool = False

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def success(self) -> bool:
        return self.terminated and not self.truncated

    @property
    def total_rewards(self) -> Dict[str, float]:
        return {
            agent: sum(step.rewards.get(agent, 0.0) for step in self.steps)
            for agent in AGENTS
        }

    def trap_hits_by(self, agent: str) -> int:
        return sum(1 for step in self.steps if step.trap_hits.get(agent, False))

    def first_reach_step(self, agent: str) -> Optional[int]:
        for step in self.steps:
            if step.reached.get(agent, False):
                return step.step
        return None


@dataclass
class EvalData:
    """Aggregated evaluation data for Blind-Spot Navigation."""

    episodes: List[EpisodeData] = field(default_factory=list)
    grid_size: int = 9
    max_cycles: int = 100
    use_communication: bool = False
    num_message_tokens: int = 0

    def __len__(self) -> int:
        return len(self.episodes)

    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for ep in self.episodes if ep.success) / len(self.episodes)

    def mean_episode_length(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(ep.length for ep in self.episodes) / len(self.episodes)

    def mean_total_reward(self, agent: str) -> float:
        if not self.episodes:
            return 0.0
        return sum(ep.total_rewards[agent] for ep in self.episodes) / len(self.episodes)

    def total_trap_hits(self, agent: str) -> int:
        return sum(ep.trap_hits_by(agent) for ep in self.episodes)

    def position_counts(self, agent: str) -> List[Tuple[int, int]]:
        positions: List[Tuple[int, int]] = []
        for ep in self.episodes:
            for step in ep.steps:
                positions.append(step.positions[agent])
        return positions

    def message_counts(self, agent: str) -> List[int]:
        counts = [0 for _ in range(self.num_message_tokens)]
        if not self.use_communication:
            return counts
        for ep in self.episodes:
            for step in ep.steps:
                msg = step.messages.get(agent)
                if msg is None:
                    continue
                if 0 <= msg < self.num_message_tokens:
                    counts[msg] += 1
        return counts
