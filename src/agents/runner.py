from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skrl.trainers.jax import SequentialTrainer

from utils.warehouse_eval_analysis import WarehouseEvalCollector
from utils.warehouse_eval_visualizer import save_all_warehouse_figures

if TYPE_CHECKING:
    from skrl.multi_agents.jax import MultiAgent


class BaseRunner(ABC):
    def __init__(self, env: Any, cfg: dict[str, Any]) -> None:
        self._env = env
        self._cfg = cfg
        self._agent: MultiAgent = self._build_agent()

    @abstractmethod
    def _build_agent(self) -> Any:
        """Construct and return the configured multi-agent instance."""
        raise NotImplementedError

    def _sync_preprocessor_sizes(self, agent_cfg: dict[str, Any]) -> dict[str, Any]:
        """Align scaler sizes with the live environment spaces."""
        possible_agents = tuple(self._env.possible_agents)
        if not possible_agents:
            return agent_cfg

        def _flat_size(space: Any) -> int:
            shape = getattr(space, "shape", None)
            if shape is None:
                raise ValueError(f"Space {space!r} does not expose a shape")
            size = 1
            for dim in shape:
                size *= int(dim)
            return int(size)

        first_agent = possible_agents[0]
        obs_size = _flat_size(self._env.observation_spaces[first_agent])
        shared_obs_size = _flat_size(self._env.state_spaces[first_agent])

        state_kwargs = dict(agent_cfg.get("state_preprocessor_kwargs", {}))
        state_kwargs["size"] = obs_size
        agent_cfg["state_preprocessor_kwargs"] = state_kwargs

        shared_state_kwargs = dict(
            agent_cfg.get("shared_state_preprocessor_kwargs", {})
        )
        shared_state_kwargs["size"] = shared_obs_size
        agent_cfg["shared_state_preprocessor_kwargs"] = shared_state_kwargs
        return agent_cfg

    def train(self, resume_from: str | None = None) -> None:
        import re

        train_cfg = self._cfg["training"]
        total_timesteps: int = train_cfg["timesteps"]
        initial_timestep: int = 0

        if resume_from is not None:
            self._load_checkpoint(resume_from)
            stem = Path(resume_from).stem  # e.g. "agent_1000000"
            m = re.search(r"_(\d+)$", stem)
            if m:
                initial_timestep = int(m.group(1))
                print(f"Resuming from timestep {initial_timestep}")
            else:
                print(
                    f"Warning: could not parse timestep from '{stem}'; "
                    "starting counter from 0 but weights are loaded."
                )

        trainer_cfg = {
            "timesteps": total_timesteps,
            "headless": True,
            "disable_progressbar": False,
        }
        trainer = SequentialTrainer(
            env=self._env,
            agents=self._agent,  # type: ignore[arg-type]
            cfg=trainer_cfg,
        )
        trainer.initial_timestep = initial_timestep
        trainer.train()

    def eval(self, checkpoint_path: str | None = None) -> None:
        path = checkpoint_path or self._cfg["eval"]["checkpoint_path"]
        if path:
            self._load_checkpoint(path)

        eval_cfg = self._cfg["eval"]
        trainer_cfg = {
            "timesteps": eval_cfg["timesteps"],
            "headless": True,
            "disable_progressbar": False,
        }
        trainer = SequentialTrainer(
            env=self._env,
            agents=self._agent,  # type: ignore[arg-type]
            cfg=trainer_cfg,
        )
        trainer.eval()

    def record(self, checkpoint_path: str | None = None) -> None:
        import numpy as np
        import imageio  # type: ignore[import-untyped]

        path = checkpoint_path or self._cfg["record"]["checkpoint_path"]
        if path:
            self._load_checkpoint(path)

        record_cfg = self._cfg["record"]
        max_steps: int = record_cfg["timesteps"]
        fps: int = record_cfg["fps"]
        video_dir = Path(record_cfg["video_dir"])
        video_dir.mkdir(parents=True, exist_ok=True)
        exp_name: str = self._cfg["experiment"]["name"]
        out_path = video_dir / f"{exp_name}.mp4"

        self._agent.set_running_mode("eval")

        frames: list[np.ndarray] = []
        obs, _ = self._env.reset()
        frame = self._env.render()
        if frame is not None:
            frames.append(frame)

        for _ in range(max_steps):
            actions, _, _ = self._agent.act(obs, timestep=0, timesteps=max_steps)
            obs, _, terminated, truncated, _ = self._env.step(actions)

            frame = self._env.render()
            if frame is not None:
                frames.append(frame)

            done_values = list(truncated.values()) + list(terminated.values())
            if any(bool(v) for v in done_values):
                obs, _ = self._env.reset()
                frame = self._env.render()
                if frame is not None:
                    frames.append(frame)

        if frames:
            imageio.mimwrite(str(out_path), frames, fps=fps)
            print(f"Recording saved to {out_path}")
        else:
            print("Warning: no frames were captured (env.render() returned None).")

    def analyze(
        self,
        checkpoint_path: str | None = None,
        n_episodes: int = 5,
        output_dir: str = "eval_plots",
    ) -> None:
        path = checkpoint_path or self._cfg.get("eval", {}).get("checkpoint_path")
        if path:
            self._load_checkpoint(path)

        env_cfg = self._cfg.get("env", {})
        env_id = env_cfg.get("id", "coingame")
        agent_type = self._cfg.get("experiment", {}).get("agent_type", "mappo")

        if agent_type == "commformer":
            from utils.commformer_runner_integration import run_commformer_analysis

            run_commformer_analysis(
                self,
                checkpoint_path,
                n_episodes,
                output_dir=f"{output_dir}/commformer",
            )
            return

        if agent_type == "commformerhm":
            from utils.commformer_runner_integration import run_commformer_analysis

            run_commformer_analysis(
                self,
                checkpoint_path,
                n_episodes,
                output_dir=f"{output_dir}/commformerhm",
            )
            return

        if agent_type == "mamhm":
            from utils.mamhm_runner_integration import run_mamhm_analysis

            run_mamhm_analysis(
                self,
                checkpoint_path,
                n_episodes,
                output_dir=f"{output_dir}/mamhm",
            )
            return

        if env_id == "blindspot":
            from utils.blindspot_eval_analysis import BlindSpotEvalCollector
            from utils.blindspot_eval_visualizer import save_all_blindspot_figures

            if self._cfg.get("experiment", {}).get("agent_type") == "magic":
                from utils.magic_runner_integration import run_magic_analysis

                run_magic_analysis(
                    self, checkpoint_path, n_episodes, output_dir=f"{output_dir}/magic"
                )

            use_comm = env_cfg.get("use_communication", False)
            num_tokens = env_cfg.get("num_message_tokens", 4)

            collector = BlindSpotEvalCollector(
                grid_size=env_cfg.get("grid_size", 9),
                num_traps=env_cfg.get("num_traps", 5),
                max_cycles=env_cfg.get("max_cycles", 100),
                use_communication=use_comm,
                num_message_tokens=num_tokens,
            )

            data = collector.collect(
                env=self._env,
                agent=self._agent,
                n_episodes=n_episodes,
            )

            exp_name = self._cfg.get("experiment", {}).get("name", "experiment")
            save_all_blindspot_figures(data, output_dir=output_dir, prefix=exp_name)
        elif env_id == "simple_adversary":
            if self._cfg.get("experiment", {}).get("agent_type") == "magic":
                from utils.magic_runner_integration import run_magic_analysis

                run_magic_analysis(
                    self,
                    checkpoint_path,
                    n_episodes,
                    output_dir=f"{output_dir}/magic_sa",
                )

            from utils import SimpleAdversaryEvalCollector, save_all_sa_figures

            sa_collector = SimpleAdversaryEvalCollector()

            sa_data = sa_collector.collect(
                env=self._env,
                agent=self._agent,
                n_episodes=n_episodes,
            )

            exp_name = self._cfg.get("experiment", {}).get("name", "experiment")
            save_all_sa_figures(sa_data, output_dir=output_dir, prefix=exp_name)
        elif env_id == "overcooked":
            from utils.overcooked_eval_analysis import OvercookedEvalCollector
            from utils.overcooked_eval_visualizer import save_all_overcooked_figures

            collector = OvercookedEvalCollector()

            data = collector.collect(
                env=self._env,
                agent=self._agent,
                n_episodes=n_episodes,
                max_steps_per_episode=env_cfg.get(
                    "horizon", env_cfg.get("max_cycles", 200)
                ),
            )

            exp_name = self._cfg.get("experiment", {}).get("name", "experiment")
            save_all_overcooked_figures(data, output_dir=output_dir, prefix=exp_name)
        elif env_id == "intersection":
            from utils import HighwayIntersectionEvalCollector, save_all_highway_figures

            if self._cfg.get("experiment", {}).get("agent_type") == "magic":
                from utils import (
                    MAGICHighwayCommCollector,
                    save_all_magic_highway_figures,
                )

                collector = MAGICHighwayCommCollector(
                    num_agents=env_cfg.get("num_agents", 4),
                    duration=env_cfg.get("duration", 13),
                    num_comm_rounds=self._cfg.get("magic", {}).get(
                        "num_comm_rounds", 2
                    ),
                    message_dim=self._cfg.get("magic", {}).get("message_dim", 64),
                )
                data = collector.collect(
                    env=self._env, agent=self._agent, n_episodes=n_episodes
                )
                exp_name = self._cfg.get("experiment", {}).get("name", "experiment")
                save_all_magic_highway_figures(
                    data, output_dir=f"{output_dir}/magic", prefix=exp_name
                )

            collector = HighwayIntersectionEvalCollector(
                num_agents=env_cfg.get("num_agents", 4),
                duration=env_cfg.get("duration", 13),
                collision_reward=env_cfg.get("collision_reward", -5.0),
                arrived_reward=env_cfg.get("arrived_reward", 1.0),
            )
            data = collector.collect(
                env=self._env, agent=self._agent, n_episodes=n_episodes
            )
            exp_name = self._cfg.get("experiment", {}).get("name", "experiment")
            save_all_highway_figures(data, output_dir=output_dir, prefix=exp_name)
        elif env_id == "flatland":
            from utils.flatland_eval_analysis import FlatlandEvalCollector
            from utils.flatland_eval_visualizer import save_all_flatland_figures

            collector = FlatlandEvalCollector(
                num_agents=env_cfg.get("num_agents", len(self._env.possible_agents)),
                max_steps_per_episode=env_cfg.get("max_episode_steps"),
            )
            data = collector.collect(
                env=self._env, agent=self._agent, n_episodes=n_episodes
            )
            exp_name = self._cfg.get("experiment", {}).get("name", "experiment")
            save_all_flatland_figures(data, output_dir=output_dir, prefix=exp_name)
        elif env_id == "warehouse":
            if self._cfg.get("experiment", {}).get("agent_type") == "magic":
                from utils import (
                    MAGICWarehouseCommCollector,
                    save_all_magic_warehouse_figures,
                )

                collector = MAGICWarehouseCommCollector(
                    num_agents=env_cfg.get("num_agents", 4),
                    num_comm_rounds=self._cfg.get("magic", {}).get(
                        "num_comm_rounds", 2
                    ),
                    message_dim=self._cfg.get("magic", {}).get("message_dim", 64),
                    battery_capacity=env_cfg.get("battery_capacity", 160.0),
                )
                data = collector.collect(
                    env=self._env, agent=self._agent, n_episodes=n_episodes
                )
                exp_name = self._cfg.get("experiment", {}).get("name", "experiment")
                save_all_magic_warehouse_figures(
                    data, output_dir=f"{output_dir}/magic", prefix=exp_name
                )
            collector = WarehouseEvalCollector(
                num_agents=env_cfg.get("num_agents", 4),
                battery_capacity=env_cfg.get("battery_capacity", 160.0),
            )
            data = collector.collect(
                env=self._env, agent=self._agent, n_episodes=n_episodes
            )
            exp_name = self._cfg.get("experiment", {}).get("name", "experiment")
            save_all_warehouse_figures(data, output_dir=output_dir, prefix=exp_name)

        else:
            from utils import EvalCollector, save_all_figures

            collector = EvalCollector(
                grid_size=env_cfg.get("grid_size", 7),
                pick_reward=env_cfg.get("pick_reward", 1.0),
                steal_penalty=env_cfg.get("steal_penalty", -2.0),
            )

            data = collector.collect(
                env=self._env,
                agent=self._agent,
                n_episodes=n_episodes,
            )

            exp_name = self._cfg.get("experiment", {}).get("name", "experiment")
            save_all_figures(data, output_dir=output_dir, prefix=exp_name)

    def _load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint from *path*."""
        self._agent.load(path)
        print(f"Loaded checkpoint: {path}")
