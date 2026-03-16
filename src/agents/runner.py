from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skrl.trainers.jax import SequentialTrainer

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

    def train(self) -> None:
        """Run the full training loop."""
        train_cfg = self._cfg["training"]
        trainer_cfg = {
            "timesteps": train_cfg["timesteps"],
            "headless": True,
            "disable_progressbar": False,
        }
        trainer = SequentialTrainer(
            env=self._env,
            agents=self._agent,  # type: ignore[arg-type]
            cfg=trainer_cfg,
        )
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

        for _ in range(max_steps):
            actions, _, _ = self._agent.act(obs, timestep=0, timesteps=max_steps)
            obs, _, terminated, truncated, _ = self._env.step(actions)

            frame = self._env.render()
            if frame is not None:
                frames.append(frame)

            done_values = list(truncated.values()) + list(terminated.values())
            if any(bool(v) for v in done_values):
                obs, _ = self._env.reset()

        if frames:
            imageio.mimwrite(str(out_path), frames, fps=fps)
            print(f"Recording saved to {out_path}")
        else:
            print("Warning: no frames were captured (env.render() returned None).")

    def analyze(
        self,
        checkpoint_path: str | None = None,
        n_episodes: int = 30,
        output_dir: str = "eval_plots",
    ) -> None:
        path = checkpoint_path or self._cfg.get("eval", {}).get("checkpoint_path")
        if path:
            self._load_checkpoint(path)

        env_cfg = self._cfg.get("env", {})
        env_id = env_cfg.get("id", "coingame")

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
