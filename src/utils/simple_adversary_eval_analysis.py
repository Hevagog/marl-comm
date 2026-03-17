import numpy as np
from typing import Any

from .simple_adversary_eval_data import (
    ALL_AGENTS,
    GOOD_AGENTS,
    ADVERSARY,
    AGENT_0,
    AGENT_1,
    StepRecord,
    EpisodeData,
    EvalData,
)


class SimpleAdversaryEvalCollector:
    """
    Evaluator for the simple_adversary PettingZoo environment wrapper.
    Collects episodes and records positions, landmarks, distances, etc.
    """

    def collect(
        self,
        env: Any,
        agent: Any,
        n_episodes: int = 20,
        max_steps_per_episode: int = 50,
    ) -> EvalData:
        agent.set_running_mode("eval")
        data = EvalData()

        world = self._find_world(env)

        for ep_idx in range(n_episodes):
            episode = EpisodeData(episode_idx=ep_idx)
            obs, info = env.reset()

            # Determine goal landmark initially
            goal_idx = -1
            if world:
                # Agent 0 is adversary_0, Agent 1 is agent_0
                good_agent_obj = [a for a in world.agents if "adversary" not in a.name][
                    0
                ]
                goal_obj = good_agent_obj.goal_a

                # Find index in landmarks
                for i, lm in enumerate(world.landmarks):
                    if lm is goal_obj:
                        goal_idx = i
                        break
            episode.goal_landmark_idx = goal_idx

            for t in range(max_steps_per_episode):
                actions, _, _ = agent.act(
                    obs, timestep=t, timesteps=max_steps_per_episode
                )
                actions_int = {
                    k: int(np.asarray(v).ravel()[0]) for k, v in actions.items()
                }

                next_obs, rewards, terminated, truncated, _ = env.step(actions)
                rewards_float = {
                    k: float(np.asarray(v).ravel()[0]) for k, v in rewards.items()
                }

                # Extract state via world object
                positions = {}
                landmarks = []
                if world:
                    for a in world.agents:
                        positions[a.name] = tuple(a.state.p_pos)
                    for lm in world.landmarks:
                        landmarks.append(tuple(lm.state.p_pos))
                else:
                    # Fallback (empty data)
                    positions = {k: (0.0, 0.0) for k in ALL_AGENTS}
                    landmarks = [(0.0, 0.0), (0.0, 0.0)]

                dist_to_goal = {}
                dist_to_spoof = {}

                goal_pos = landmarks[goal_idx] if goal_idx != -1 else (0.0, 0.0)
                spoof_idx = (
                    1 - goal_idx if goal_idx in [0, 1] and len(landmarks) == 2 else -1
                )
                spoof_pos = landmarks[spoof_idx] if spoof_idx != -1 else (0.0, 0.0)

                for a_name in ALL_AGENTS:
                    p = np.array(positions[a_name])
                    dist_to_goal[a_name] = float(np.linalg.norm(p - np.array(goal_pos)))
                    if a_name in GOOD_AGENTS:
                        dist_to_spoof[a_name] = float(
                            np.linalg.norm(p - np.array(spoof_pos))
                        )

                step_record = StepRecord(
                    step=t,
                    positions=positions,
                    landmarks=landmarks,
                    dist_to_goal=dist_to_goal,
                    dist_to_spoof=dist_to_spoof,
                    actions=actions_int,
                    rewards=rewards_float,
                )
                episode.steps.append(step_record)

                obs = next_obs

                done = any(
                    bool(np.asarray(v).ravel()[0])
                    for v in list(terminated.values()) + list(truncated.values())
                )
                if done:
                    break

            data.episodes.append(episode)

            # Simple terminal output
            avg_adv_dist = episode.get_avg_dist_to_goal(ADVERSARY)
            avg_ag0_dist = episode.get_avg_dist_to_goal(AGENT_0)
            avg_ag1_dist = episode.get_avg_dist_to_goal(AGENT_1)
            total_rewards = episode.total_rewards

            print(
                f"  Episode {ep_idx + 1:3d}/{n_episodes}  |  "
                f"Len: {len(episode.steps):3d}  |  "
                f"Avg Goal Dists -- Adv: {avg_adv_dist:.2f}, Ag0: {avg_ag0_dist:.2f}, Ag1: {avg_ag1_dist:.2f}  |  "
                f"Returns -- Adv: {total_rewards[ADVERSARY]:+.2f}, "
                f"Ag0: {total_rewards[AGENT_0]:+.2f}, Ag1: {total_rewards[AGENT_1]:+.2f}"
            )

        return data

    @staticmethod
    def _find_world(env: Any) -> Any:
        visited = set()
        queue = [env]
        depth = 0
        max_depth = 8

        while queue and depth <= max_depth:
            next_queue = []
            for node in queue:
                if node is None:
                    continue
                node_id = id(node)
                if node_id in visited:
                    continue
                visited.add(node_id)

                world = getattr(node, "world", None)
                if (
                    world is not None
                    and hasattr(world, "agents")
                    and hasattr(world, "landmarks")
                ):
                    return world

                next_queue.extend(
                    [
                        getattr(node, "unwrapped", None),
                        getattr(node, "env", None),
                        getattr(node, "_env", None),
                    ]
                )
            queue = next_queue
            depth += 1

        return None
