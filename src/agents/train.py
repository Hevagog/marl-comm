from __future__ import annotations


from skrl.memories.jax import RandomMemory
from skrl.multi_agents.jax.mappo import MAPPO

from agents.mappo.models.generators import create_mappo_models
from agents.runner import BaseRunner

from .helper import build_mappo_cfg


class MAPPORunner(BaseRunner):
    def _build_agent(self) -> MAPPO:
        env = self._env
        cfg = self._cfg

        memory_size: int = cfg["memory"]["size"]
        memories: dict[str, RandomMemory] = {
            agent: RandomMemory(memory_size=memory_size, num_envs=1)
            for agent in env.possible_agents
        }

        models = create_mappo_models(
            possible_agents=env.possible_agents,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            shared_observation_spaces=env.state_spaces,
            cfg=cfg,
        )

        mappo_cfg = build_mappo_cfg(cfg)

        agent = MAPPO(
            possible_agents=env.possible_agents,
            models=models,  # type: ignore[arg-type]
            memories=memories,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            shared_observation_spaces=env.state_spaces,
            cfg=mappo_cfg,
        )
        return agent
