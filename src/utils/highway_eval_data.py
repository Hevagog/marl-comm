"""
Highway Intersection Evaluation Data Structures
================================================

Data structures for collecting and analyzing evaluation metrics from the
multi-agent highway intersection environment.

The intersection environment features 4 controlled vehicles navigating an
intersection, with rewards for successful crossing, penalties for collisions,
and bonuses for maintaining target speeds.

Key metrics:
- Collision rates per vehicle
- Successful arrivals
- Speed maintenance
- Trajectory analysis
- Inter-vehicle distances (safety metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Action names for DiscreteMetaAction
ACTION_NAMES = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}


@dataclass
class VehicleStepRecord:
    """Per-step snapshot of a single vehicle's state."""

    step: int
    position: Tuple[float, float]  # (x, y) in world coordinates
    velocity: Tuple[float, float]  # (vx, vy)
    speed: float  # magnitude of velocity
    action: int  # 0-4: LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER
    reward: float
    crashed: bool
    arrived: bool


@dataclass
class IntersectionEpisodeData:
    """Complete data for a single evaluation episode."""

    episode_idx: int
    agent_names: List[str] = field(default_factory=list)
    vehicle_trajectories: Dict[str, List[VehicleStepRecord]] = field(
        default_factory=dict
    )
    total_reward: float = 0.0
    episode_length: int = 0
    terminated: bool = False
    truncated: bool = False

    @property
    def num_agents(self) -> int:
        return len(self.agent_names)

    def total_reward_for(self, agent: str) -> float:
        """Sum of rewards for a specific agent."""
        if agent not in self.vehicle_trajectories:
            return 0.0
        return sum(step.reward for step in self.vehicle_trajectories[agent])

    def crashed_agents(self) -> List[str]:
        """List of agents that crashed during the episode."""
        crashed = []
        for agent, traj in self.vehicle_trajectories.items():
            if any(step.crashed for step in traj):
                crashed.append(agent)
        return crashed

    def arrived_agents(self) -> List[str]:
        """List of agents that successfully crossed the intersection."""
        arrived = []
        for agent, traj in self.vehicle_trajectories.items():
            if any(step.arrived for step in traj):
                arrived.append(agent)
        return arrived

    def average_speed_for(self, agent: str) -> float:
        """Average speed maintained by an agent."""
        if agent not in self.vehicle_trajectories:
            return 0.0
        speeds = [step.speed for step in self.vehicle_trajectories[agent]]
        return sum(speeds) / len(speeds) if speeds else 0.0

    def action_distribution_for(self, agent: str) -> Dict[int, int]:
        """Count of each action taken by an agent."""
        if agent not in self.vehicle_trajectories:
            return {}
        counts = {i: 0 for i in range(5)}
        for step in self.vehicle_trajectories[agent]:
            counts[step.action] += 1
        return counts


@dataclass
class IntersectionEvalData:
    """Aggregated evaluation data across multiple episodes."""

    episodes: List[IntersectionEpisodeData] = field(default_factory=list)
    num_agents: int = 4
    duration: int = 13  # episode duration in steps
    collision_reward: float = -5.0
    arrived_reward: float = 1.0

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def agent_names(self) -> List[str]:
        if not self.episodes:
            return []
        return self.episodes[0].agent_names

    def mean_total_reward(self) -> float:
        """Mean total reward across all episodes."""
        if not self.episodes:
            return 0.0
        return sum(ep.total_reward for ep in self.episodes) / len(self.episodes)

    def mean_reward_per_agent(self, agent: str) -> float:
        """Mean reward for a specific agent across episodes."""
        rewards = [ep.total_reward_for(agent) for ep in self.episodes]
        return sum(rewards) / len(rewards) if rewards else 0.0

    def collision_rate(self) -> float:
        """Fraction of episodes with at least one collision."""
        if not self.episodes:
            return 0.0
        collisions = sum(1 for ep in self.episodes if len(ep.crashed_agents()) > 0)
        return collisions / len(self.episodes)

    def collision_rate_per_agent(self, agent: str) -> float:
        """Fraction of episodes where a specific agent crashed."""
        if not self.episodes:
            return 0.0
        crashed = sum(1 for ep in self.episodes if agent in ep.crashed_agents())
        return crashed / len(self.episodes)

    def arrival_rate(self) -> float:
        """Fraction of vehicles that successfully crossed."""
        if not self.episodes:
            return 0.0
        total_arrivals = sum(len(ep.arrived_agents()) for ep in self.episodes)
        total_vehicles = len(self.episodes) * self.num_agents
        return total_arrivals / total_vehicles if total_vehicles > 0 else 0.0

    def arrival_rate_per_agent(self, agent: str) -> float:
        """Fraction of episodes where a specific agent arrived."""
        if not self.episodes:
            return 0.0
        arrived = sum(1 for ep in self.episodes if agent in ep.arrived_agents())
        return arrived / len(self.episodes)

    def mean_episode_length(self) -> float:
        """Mean episode length in steps."""
        if not self.episodes:
            return 0.0
        return sum(ep.episode_length for ep in self.episodes) / len(self.episodes)

    def mean_speed_per_agent(self, agent: str) -> float:
        """Mean speed across all episodes for an agent."""
        speeds = [ep.average_speed_for(agent) for ep in self.episodes]
        return sum(speeds) / len(speeds) if speeds else 0.0

    def overall_action_distribution(self, agent: str) -> Dict[int, float]:
        """Overall action frequency distribution for an agent."""
        total_counts = {i: 0 for i in range(5)}
        for ep in self.episodes:
            dist = ep.action_distribution_for(agent)
            for action, count in dist.items():
                total_counts[action] += count

        total = sum(total_counts.values())
        if total == 0:
            return {i: 0.0 for i in range(5)}

        return {action: count / total for action, count in total_counts.items()}

    def per_episode_rewards(self) -> List[float]:
        """List of total rewards for each episode."""
        return [ep.total_reward for ep in self.episodes]

    def per_episode_collisions(self) -> List[int]:
        """List of collision counts for each episode."""
        return [len(ep.crashed_agents()) for ep in self.episodes]

    def per_episode_arrivals(self) -> List[int]:
        """List of arrival counts for each episode."""
        return [len(ep.arrived_agents()) for ep in self.episodes]
