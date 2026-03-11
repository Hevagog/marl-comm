from __future__ import annotations

import copy

from skrl.memories.jax import RandomMemory
from skrl.multi_agents.jax.mappo import MAPPO_DEFAULT_CONFIG

from agents.mappo.categorical_mappo import CategoricalMAPPO
from agents.mappo.models.generators import create_mappo_models
from agents.runner import BaseRunner


class MAPPORunner(BaseRunner):
    def _build_agent(self) -> CategoricalMAPPO:
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

        mappo_cfg = copy.deepcopy(MAPPO_DEFAULT_CONFIG)

        project_mappo = cfg.get("mappo", {})
        for key, value in project_mappo.items():
            mappo_cfg[key] = value

        exp = cfg.get("experiment", {})
        mappo_cfg["experiment"]["directory"] = exp.get("directory", "")
        mappo_cfg["experiment"]["experiment_name"] = exp.get("name", "")
        mappo_cfg["experiment"]["write_interval"] = exp.get("write_interval", "auto")
        mappo_cfg["experiment"]["checkpoint_interval"] = exp.get(
            "checkpoint_interval", "auto"
        )
        mappo_cfg["experiment"]["store_separately"] = exp.get("store_separately", False)
        mappo_cfg["experiment"]["wandb"] = exp.get("wandb", True)
        mappo_cfg["experiment"]["wandb_kwargs"] = exp.get("wandb_kwargs", {})

        agent = CategoricalMAPPO(
            possible_agents=env.possible_agents,
            models=models,  # type: ignore[arg-type]
            memories=memories,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            shared_observation_spaces=env.state_spaces,
            cfg=mappo_cfg,
        )
        return agent
