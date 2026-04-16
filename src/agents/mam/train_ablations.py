from __future__ import annotations

import copy

from skrl.memories.jax import RandomMemory
from skrl.multi_agents.jax.mappo import MAPPO_DEFAULT_CONFIG

from agents.mam.mam_enc_only_mappo import MAMEncOnlyMAPPO
from agents.mam.models.generators_ablations import create_enc_only_models
from agents.mam.models.generators_hopfield import (
    create_hopfield_pooling_models,
    create_hopfield_layer_models,
    create_et_encoder_models,
)
from agents.runner import BaseRunner


def _build_mappo_cfg(cfg: dict, sync_fn) -> dict:
    """Shared helper: deep-copy MAPPO defaults, merge project cfg, sync sizes."""
    mappo_cfg = copy.deepcopy(MAPPO_DEFAULT_CONFIG)
    project_cfg = cfg.get("mam", {})
    for key, value in project_cfg.items():
        mappo_cfg[key] = value
    mappo_cfg = sync_fn(mappo_cfg)

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
    return mappo_cfg


def _build_memories(cfg: dict, possible_agents: list[str]) -> dict[str, RandomMemory]:
    memory_size: int = cfg["memory"]["size"]
    num_envs: int = cfg.get("env", {}).get("num_envs", 1)
    return {
        agent: RandomMemory(memory_size=memory_size, num_envs=num_envs)
        for agent in possible_agents
    }


class MAMEncOnlyRunner(BaseRunner):
    def _build_agent(self) -> MAMEncOnlyMAPPO:
        env = self._env
        cfg = self._cfg

        memories = _build_memories(cfg, env.possible_agents)
        models = create_enc_only_models(
            possible_agents=env.possible_agents,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            shared_observation_spaces=env.state_spaces,
            cfg=cfg,
        )
        mappo_cfg = _build_mappo_cfg(cfg, self._sync_preprocessor_sizes)

        agent = MAMEncOnlyMAPPO(
            possible_agents=env.possible_agents,
            models=models,
            memories=memories,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            shared_observation_spaces=env.state_spaces,
            cfg=mappo_cfg,
        )
        return agent


class MAMHopfieldPoolingRunner(BaseRunner):
    """Runner for BiMamba + HopfieldPooling encoder-only."""

    def _build_agent(self) -> MAMEncOnlyMAPPO:
        env = self._env
        cfg = self._cfg

        memories = _build_memories(cfg, env.possible_agents)
        models = create_hopfield_pooling_models(
            possible_agents=env.possible_agents,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            shared_observation_spaces=env.state_spaces,
            cfg=cfg,
        )
        mappo_cfg = _build_mappo_cfg(cfg, self._sync_preprocessor_sizes)

        return MAMEncOnlyMAPPO(
            possible_agents=env.possible_agents,
            models=models,
            memories=memories,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            shared_observation_spaces=env.state_spaces,
            cfg=mappo_cfg,
        )


class MAMHopfieldLayerRunner(BaseRunner):
    """Runner for BiMamba + HopfieldLayer prototype bank encoder-only."""

    def _build_agent(self) -> MAMEncOnlyMAPPO:
        env = self._env
        cfg = self._cfg

        memories = _build_memories(cfg, env.possible_agents)
        models = create_hopfield_layer_models(
            possible_agents=env.possible_agents,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            shared_observation_spaces=env.state_spaces,
            cfg=cfg,
        )
        mappo_cfg = _build_mappo_cfg(cfg, self._sync_preprocessor_sizes)

        return MAMEncOnlyMAPPO(
            possible_agents=env.possible_agents,
            models=models,
            memories=memories,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            shared_observation_spaces=env.state_spaces,
            cfg=mappo_cfg,
        )


class MAMETEncoderRunner(BaseRunner):
    """Runner for ET recurrent encoder-only (replaces BiMamba)."""

    def _build_agent(self) -> MAMEncOnlyMAPPO:
        env = self._env
        cfg = self._cfg

        memories = _build_memories(cfg, env.possible_agents)
        models = create_et_encoder_models(
            possible_agents=env.possible_agents,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            shared_observation_spaces=env.state_spaces,
            cfg=cfg,
        )
        mappo_cfg = _build_mappo_cfg(cfg, self._sync_preprocessor_sizes)

        return MAMEncOnlyMAPPO(
            possible_agents=env.possible_agents,
            models=models,
            memories=memories,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            shared_observation_spaces=env.state_spaces,
            cfg=mappo_cfg,
        )
