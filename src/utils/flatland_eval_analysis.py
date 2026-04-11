from __future__ import annotations

from typing import Any

import numpy as np

from .flatland_eval_data import (
    FlatlandAgentStepRecord,
    FlatlandEpisodeData,
    FlatlandEvalData,
)


class FlatlandEvalCollector:
    """Collect evaluation rollouts for Flatland environments."""

    def __init__(
        self,
        num_agents: int,
        max_steps_per_episode: int | None = None,
    ) -> None:
        self.num_agents = num_agents
        self.max_steps_per_episode = max_steps_per_episode

    def collect(
        self,
        env: Any,
        agent: Any,
        n_episodes: int = 10,
    ) -> FlatlandEvalData:
        agent.set_running_mode("eval")

        agent_names = list(getattr(env, "possible_agents", []))
        if not agent_names:
            raise ValueError("Environment does not expose possible_agents")

        default_max_steps = getattr(getattr(env, "unwrapped", env), "_max_episode_steps", 500)
        max_steps = int(self.max_steps_per_episode or default_max_steps)

        data = FlatlandEvalData(num_agents=self.num_agents)

        for episode_idx in range(n_episodes):
            obs, _ = env.reset()
            episode = FlatlandEpisodeData(
                episode_idx=episode_idx,
                agent_names=agent_names,
                agent_steps={agent_name: [] for agent_name in agent_names},
            )
            completed_agents: set[str] = set()

            for step in range(max_steps):
                actions, _, _ = agent.act(obs, timestep=step, timesteps=max_steps)
                action_dict = {
                    agent_name: int(np.asarray(actions[agent_name]).ravel()[0])
                    for agent_name in agent_names
                }

                next_obs, rewards, terminated, truncated, infos = env.step(actions)

                for agent_name in agent_names:
                    reward = float(np.asarray(rewards[agent_name]).ravel()[0])
                    info = infos.get(agent_name, {})
                    reached_target = bool(info.get("reached_target", False))
                    if reached_target:
                        completed_agents.add(agent_name)

                    record = FlatlandAgentStepRecord(
                        step=step,
                        action=action_dict[agent_name],
                        reward=reward,
                        active=bool(info.get("active", False)),
                        done=bool(info.get("done", False)),
                        reached_target=reached_target,
                        malfunction=int(info.get("malfunction", 0)),
                        speed=float(info.get("speed", 0.0)),
                        position=info.get("position"),
                    )
                    episode.agent_steps[agent_name].append(record)
                    episode.total_reward += reward

                episode.episode_length = step + 1
                episode.completion_count = len(completed_agents)
                episode.completion_ratio = len(completed_agents) / max(1, len(agent_names))

                done = any(
                    bool(np.asarray(value).ravel()[0])
                    for value in list(terminated.values()) + list(truncated.values())
                )
                if done:
                    episode.terminated = any(
                        bool(np.asarray(value).ravel()[0]) for value in terminated.values()
                    )
                    episode.truncated = any(
                        bool(np.asarray(value).ravel()[0]) for value in truncated.values()
                    )
                    break

                obs = next_obs

            data.episodes.append(episode)
            print(
                f"  Episode {episode_idx + 1:3d}/{n_episodes} | "
                f"Len: {episode.episode_length:3d} | "
                f"Reward: {episode.total_reward:+8.2f} | "
                f"Completion: {episode.completion_count}/{len(agent_names)}"
            )

        return data
