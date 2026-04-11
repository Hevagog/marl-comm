from __future__ import annotations

from dataclasses import dataclass, field


ACTION_NAMES = {
    0: "DO_NOTHING",
    1: "MOVE_LEFT",
    2: "MOVE_FORWARD",
    3: "MOVE_RIGHT",
    4: "STOP_MOVING",
}


@dataclass
class FlatlandAgentStepRecord:
    step: int
    action: int
    reward: float
    active: bool
    done: bool
    reached_target: bool
    malfunction: int
    speed: float
    position: tuple[int, int] | None


@dataclass
class FlatlandEpisodeData:
    episode_idx: int
    agent_names: list[str] = field(default_factory=list)
    agent_steps: dict[str, list[FlatlandAgentStepRecord]] = field(default_factory=dict)
    total_reward: float = 0.0
    episode_length: int = 0
    completion_count: int = 0
    completion_ratio: float = 0.0
    terminated: bool = False
    truncated: bool = False

    def total_reward_for(self, agent: str) -> float:
        return sum(step.reward for step in self.agent_steps.get(agent, []))

    def completed_agents(self) -> list[str]:
        return [
            agent_name
            for agent_name, steps in self.agent_steps.items()
            if any(step.reached_target for step in steps)
        ]

    def action_distribution_for(self, agent: str) -> dict[int, int]:
        counts = {action: 0 for action in ACTION_NAMES}
        for step in self.agent_steps.get(agent, []):
            counts[step.action] += 1
        return counts


@dataclass
class FlatlandEvalData:
    episodes: list[FlatlandEpisodeData] = field(default_factory=list)
    num_agents: int = 0

    @property
    def agent_names(self) -> list[str]:
        if not self.episodes:
            return []
        return self.episodes[0].agent_names

    def mean_total_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(episode.total_reward for episode in self.episodes) / len(self.episodes)

    def mean_completion_ratio(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(episode.completion_ratio for episode in self.episodes) / len(self.episodes)

    def mean_episode_length(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(episode.episode_length for episode in self.episodes) / len(self.episodes)

    def per_episode_rewards(self) -> list[float]:
        return [episode.total_reward for episode in self.episodes]

    def per_episode_completion_ratios(self) -> list[float]:
        return [episode.completion_ratio for episode in self.episodes]

    def per_episode_lengths(self) -> list[int]:
        return [episode.episode_length for episode in self.episodes]

    def per_agent_completion_rates(self) -> dict[str, float]:
        if not self.episodes:
            return {}

        rates: dict[str, float] = {}
        for agent_name in self.agent_names:
            completed = sum(
                1 for episode in self.episodes if agent_name in episode.completed_agents()
            )
            rates[agent_name] = completed / len(self.episodes)
        return rates

    def completion_histogram(self) -> dict[str, int]:
        histogram = {agent_name: 0 for agent_name in self.agent_names}
        for episode in self.episodes:
            for agent_name in episode.completed_agents():
                histogram[agent_name] += 1
        return histogram

