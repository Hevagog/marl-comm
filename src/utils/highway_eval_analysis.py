"""
Highway Intersection Evaluation Collector
==========================================

Collector class for running evaluation episodes and gathering metrics from
the multi-agent highway intersection environment.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .highway_eval_data import (
    IntersectionEpisodeData,
    IntersectionEvalData,
    VehicleStepRecord,
)


class HighwayIntersectionEvalCollector:
    """Collect evaluation rollouts for highway intersection environment."""

    def __init__(
        self,
        num_agents: int = 4,
        duration: int = 13,
        collision_reward: float = -5.0,
        arrived_reward: float = 1.0,
    ):
        """
        Initialize the collector.

        Args:
            num_agents: Number of controlled vehicles
            duration: Episode duration in policy steps
            collision_reward: Reward penalty for collision
            arrived_reward: Reward for successful crossing
        """
        self.num_agents = num_agents
        self.duration = duration
        self.collision_reward = collision_reward
        self.arrived_reward = arrived_reward

    def collect(
        self,
        env: Any,
        agent: Any,
        n_episodes: int = 20,
        max_steps_per_episode: int = 100,
    ) -> IntersectionEvalData:
        """
        Run evaluation episodes and collect data.

        Args:
            env: PettingZoo parallel environment (wrapped)
            agent: Multi-agent trained policy
            n_episodes: Number of evaluation episodes
            max_steps_per_episode: Maximum steps per episode

        Returns:
            IntersectionEvalData with collected metrics
        """
        agent.set_running_mode("eval")

        agent_names = list(getattr(env, "possible_agents", []))
        if not agent_names:
            raise ValueError("Environment does not expose possible_agents")

        data = IntersectionEvalData(
            num_agents=self.num_agents,
            duration=self.duration,
            collision_reward=self.collision_reward,
            arrived_reward=self.arrived_reward,
        )

        for ep_idx in range(n_episodes):
            obs, infos = env.reset()
            episode = IntersectionEpisodeData(
                episode_idx=ep_idx,
                agent_names=agent_names,
                vehicle_trajectories={name: [] for name in agent_names},
            )

            for t in range(max_steps_per_episode):
                # Get actions from trained policy
                actions, _, _ = agent.act(
                    obs,
                    timestep=t,
                    timesteps=max_steps_per_episode,
                )

                # Convert actions to integers
                actions_int = {
                    name: int(np.asarray(actions[name]).ravel()[0])
                    for name in agent_names
                }

                # Step environment
                next_obs, rewards, terminated, truncated, infos = env.step(actions)

                # Extract per-agent info
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

                    step_record = VehicleStepRecord(
                        step=t,
                        position=(ego_x, ego_y),
                        velocity=(ego_vx, ego_vy),
                        speed=speed,
                        action=actions_int[name],
                        reward=reward,
                        crashed=crashed,
                        arrived=arrived,
                    )
                    episode.vehicle_trajectories[name].append(step_record)
                    episode.total_reward += reward

                episode.episode_length = t + 1
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
            collisions = len(episode.crashed_agents())
            arrivals = len(episode.arrived_agents())
            print(
                f"  Episode {ep_idx + 1:3d}/{n_episodes} | "
                f"Len: {episode.episode_length:3d} | "
                f"Reward: {episode.total_reward:+7.2f} | "
                f"Collisions: {collisions} | "
                f"Arrivals: {arrivals}/{self.num_agents}"
            )

        return data
