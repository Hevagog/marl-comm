from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


ADVERSARY = "adversary_0"
AGENT_0 = "agent_0"
AGENT_1 = "agent_1"
ALL_AGENTS = [ADVERSARY, AGENT_0, AGENT_1]
GOOD_AGENTS = [AGENT_0, AGENT_1]

# We use vectors for distance instead of discrete grid distances,
# since simple_adversary is a continuous environment.
Position = Tuple[float, float]

@dataclass
class StepRecord:
    step: int
    
    positions: Dict[str, Position]
    landmarks: List[Position]
    
    dist_to_goal: Dict[str, float]
    # For good agents, distance to the *bad* landmark (spoofing)
    dist_to_spoof: Dict[str, float]
    
    actions: Dict[str, int]
    rewards: Dict[str, float]


@dataclass
class EpisodeData:
    episode_idx: int
    steps: List[StepRecord] = field(default_factory=list)
    
    # Static environment info for the episode
    goal_landmark_idx: int = -1
    
    @property
    def total_rewards(self) -> Dict[str, float]:
        out = {k: 0.0 for k in ALL_AGENTS}
        for s in self.steps:
            for k, r in s.rewards.items():
                out[k] += r
        return out

    def get_agent_trajectories(self, agent_id: str) -> List[Position]:
        return [s.positions[agent_id] for s in self.steps]
    
    def get_avg_dist_to_goal(self, agent_id: str) -> float:
        if not self.steps:
            return 0.0
        return sum(s.dist_to_goal.get(agent_id, 0.0) for s in self.steps) / len(self.steps)


@dataclass
class EvalData:
    episodes: List[EpisodeData] = field(default_factory=list)
    
    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def avg_rewards(self) -> Dict[str, float]:
        if not self.episodes:
            return {a: 0.0 for a in ALL_AGENTS}
        sums = {a: 0.0 for a in ALL_AGENTS}
        for ep in self.episodes:
            tr = ep.total_rewards
            for a in ALL_AGENTS:
                sums[a] += tr[a]
        return {a: val / len(self.episodes) for a, val in sums.items()}
