from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class OvercookedEpisodeSummary:
    episode_idx: int
    steps: int = 0
    deliveries: int = 0
    sparse_return: float = 0.0
    returns: dict[str, float] = field(default_factory=dict)
    action_counts: dict[str, dict[int, int]] = field(default_factory=dict)

    @property
    def joint_return(self) -> float:
        return float(sum(self.returns.values()))


@dataclass
class OvercookedEvalData:
    episodes: list[OvercookedEpisodeSummary] = field(default_factory=list)
    agent_names: list[str] = field(default_factory=list)
    num_actions: int = 6

    def __len__(self) -> int:
        return len(self.episodes)


class OvercookedEvalCollector:
    """Collect rollout summaries for Overcooked training artifacts."""

    def collect(
        self,
        env: Any,
        agent: Any,
        n_episodes: int = 20,
        max_steps_per_episode: int = 200,
    ) -> OvercookedEvalData:
        agent.set_running_mode("eval")

        agent_names = list(getattr(env, "possible_agents", []))
        if not agent_names:
            raise ValueError(
                "Overcooked analyzer expected env.possible_agents to be set"
            )

        action_spaces = getattr(env, "action_spaces", {})
        num_actions = int(action_spaces[agent_names[0]].n) if action_spaces else 6

        data = OvercookedEvalData(agent_names=agent_names, num_actions=num_actions)

        for ep_idx in range(n_episodes):
            obs, _ = env.reset()
            episode = OvercookedEpisodeSummary(
                episode_idx=ep_idx,
                returns={name: 0.0 for name in agent_names},
                action_counts={
                    name: {action: 0 for action in range(num_actions)}
                    for name in agent_names
                },
            )

            for t in range(max_steps_per_episode):
                actions, _, _ = agent.act(
                    obs,
                    timestep=t,
                    timesteps=max_steps_per_episode,
                )

                actions_int = {
                    name: int(np.asarray(actions[name]).ravel()[0])
                    for name in agent_names
                }
                for name, action in actions_int.items():
                    episode.action_counts[name][action] += 1

                next_obs, rewards, terminated, truncated, infos = env.step(actions)

                rewards_float = {
                    name: float(np.asarray(rewards[name]).ravel()[0])
                    for name in agent_names
                }
                for name, reward in rewards_float.items():
                    episode.returns[name] += reward

                sparse_reward = self._extract_sparse_reward(infos, agent_names)
                episode.sparse_return += sparse_reward
                if sparse_reward > 0:
                    episode.deliveries += 1

                episode.steps = t + 1
                obs = next_obs

                done = any(
                    bool(np.asarray(v).ravel()[0])
                    for v in list(terminated.values()) + list(truncated.values())
                )
                if done:
                    break

            data.episodes.append(episode)

            per_agent_returns = " | ".join(
                f"{name}: {episode.returns[name]:+7.2f}" for name in agent_names
            )
            print(
                f"  Episode {ep_idx + 1:3d}/{n_episodes} | "
                f"Len: {episode.steps:3d} | "
                f"Deliveries: {episode.deliveries:2d} | "
                f"Sparse: {episode.sparse_return:+7.2f} | "
                f"{per_agent_returns}"
            )

        return data

    @staticmethod
    def _extract_sparse_reward(infos: dict[str, Any], agent_names: list[str]) -> float:
        for name in agent_names:
            info = infos.get(name, {})
            if "sparse_reward" in info:
                return float(np.asarray(info["sparse_reward"]).ravel()[0])
        return 0.0
