from __future__ import annotations

import contextlib
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tqdm as _tqdm_mod
from skrl.trainers.jax import SequentialTrainer

from utils.warehouse_eval_analysis import WarehouseEvalCollector
from utils.warehouse_eval_visualizer import save_all_warehouse_figures

if TYPE_CHECKING:
    from skrl.multi_agents.jax import MultiAgent
    from environments.flatland import FlatlandConfig


class _CurriculumSequentialTrainer(SequentialTrainer):
    """SequentialTrainer variant for curriculum training.

    Separates two concerns that the stock trainer conflates:

    * **segment_end** — the timestep at which *this segment's loop* stops.
      Set to ``initial_timestep + segment_timesteps`` before each ``train()``
      call.  Controls ``range(initial_timestep, segment_end)`` in the loop.

    * **total_timesteps** — the full training budget passed to the agent's
      pre/post_interaction callbacks.  The agent uses this value for LR
      decay and entropy annealing.  Must equal the run's total timestep
      budget and must never change.

    Without this separation the stock trainer passes ``trainer.timesteps``
    (= ``segment_end``) to the agent, so the agent thinks its full budget
    is 2 000 steps and decays the LR to the floor within the first segment.
    """

    def __init__(self, total_timesteps: int, **kwargs: Any) -> None:
        self._total_timesteps: int = total_timesteps
        # segment_end is updated by _train_with_curriculum before each call.
        self.segment_end: int = total_timesteps
        super().__init__(**kwargs)

    def train(self) -> None:  # type: ignore[override]
        self.agents.set_running_mode("train")
        # Our env always has num_agents > 1 (Flatland).
        if self.env.num_agents > 1:
            self._curriculum_multi_agent_train()
        else:
            self.single_agent_train()

    def _curriculum_multi_agent_train(self) -> None:
        """multi_agent_train with segment_end loop bound and total_timesteps scheduling."""
        states, infos = self.env.reset()
        shared_states = self.env.state()

        for timestep in _tqdm_mod.tqdm(
            range(self.initial_timestep, self.segment_end),
            disable=self.disable_progressbar,
            file=sys.stdout,
        ):
            self.agents.pre_interaction(
                timestep=timestep, timesteps=self._total_timesteps
            )

            with contextlib.nullcontext():
                actions = self.agents.act(
                    states, timestep=timestep, timesteps=self._total_timesteps
                )[0]

                next_states, rewards, terminated, truncated, infos = self.env.step(
                    actions
                )
                shared_next_states = self.env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                if not self.headless:
                    self.env.render()

                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self._total_timesteps,
                )

            self.agents.post_interaction(
                timestep=timestep, timesteps=self._total_timesteps
            )

            if not self.env.agents:
                states, infos = self.env.reset()
                shared_states = self.env.state()
            else:
                states = next_states
                shared_states = shared_next_states


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

        # Dispatch to the curriculum path when enabled — it still honours
        # ``initial_timestep`` for resumption.
        curriculum_cfg = self._cfg.get("curriculum", {}) or {}
        if curriculum_cfg.get("enabled", False):
            self._train_with_curriculum(initial_timestep=initial_timestep)
            return

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

    # ------------------------------------------------------------------
    # Curriculum training
    # ------------------------------------------------------------------

    def _train_with_curriculum(self, initial_timestep: int = 0) -> None:
        """Segment-wise training with in-place stage advancement.

        The ``training.timesteps`` budget is consumed in fixed-size segments
        (``curriculum.segment_timesteps``).  Between segments we query the
        :class:`CurriculumScheduler` with the vectorised env's
        ``last_completion_ratio``; when the scheduler advances, we rebuild
        the underlying Flatland vec-env with the new stage config and
        re-wrap it for skrl.  The agent (policy + value + preprocessors)
        is **not** rebuilt, so network weights and running stats persist.

        This flow only works when the stages keep ``num_agents``/``tree_depth``
        fixed; the :class:`CurriculumScheduler` default stages enforce that.
        Changing ``num_agents`` would change the critic's input dim and
        invalidate the running scaler sizes, which we explicitly do not
        handle here.
        """
        from environments.flatland import (
            CurriculumScheduler,
            FlatlandConfig,
        )

        env_cfg = self._cfg.get("env", {})
        if env_cfg.get("id") != "flatland":
            raise RuntimeError(
                "Curriculum training is only wired up for the flatland env"
            )

        train_cfg = self._cfg["training"]
        total: int = int(train_cfg["timesteps"])
        curriculum_cfg = self._cfg.get("curriculum", {}) or {}
        segment: int = int(curriculum_cfg.get("segment_timesteps", 2_000))
        if segment <= 0:
            raise ValueError("curriculum.segment_timesteps must be positive")

        # Build a FlatlandConfig from the cfg['env'] dict, dropping keys that
        # don't belong on the dataclass.  This mirrors the factory logic in
        # ``cli._create_env_factory`` so the curriculum starts from exactly
        # the same base config the CLI would have produced.
        base_kwargs = {
            k: v for k, v in env_cfg.items() if k in FlatlandConfig.__dataclass_fields__
        }
        base_flatland_cfg = FlatlandConfig(**base_kwargs)

        stages = curriculum_cfg.get("stages")  # optional override
        scheduler = (
            CurriculumScheduler(base_flatland_cfg, stages=stages)
            if stages is not None
            else CurriculumScheduler(base_flatland_cfg)
        )

        # Apply stage-0 config immediately (the base cfg may be the terminal
        # stage, which is almost always *harder* than stage 0).
        self._env = self._rebuild_flatland_env(
            scheduler.config_for(scheduler.current)
        )
        self._rebind_agent_to_env(self._env)
        print(
            f"[curriculum] starting at stage 0/{scheduler.num_stages - 1}: "
            f"{scheduler.current}"
        )

        done = initial_timestep

        # Create the trainer ONCE.
        # _CurriculumSequentialTrainer separates two values that the stock
        # SequentialTrainer conflates:
        #   segment_end      — upper bound for this segment's loop
        #   _total_timesteps — budget passed to agent pre/post_interaction for
        #                      LR decay and entropy annealing
        # Without this split, each segment sets trainer.timesteps = done + seg,
        # so the agent decays its LR over 2 000 steps instead of 1 000 000,
        # reaching the min_lr floor by the end of segment 0.
        trainer = _CurriculumSequentialTrainer(
            total_timesteps=total,
            env=self._env,
            agents=self._agent,  # type: ignore[arg-type]
            cfg={
                "timesteps": total,
                "headless": True,
                "disable_progressbar": False,
                "close_environment_at_exit": False,
            },
        )

        while done < total:
            this_segment = min(segment, total - done)

            # Update per-segment fields — total_timesteps stays fixed.
            trainer.env = self._env
            trainer.initial_timestep = done
            trainer.segment_end = done + this_segment

            print(
                f"[curriculum] segment {done}–{done + this_segment} "
                f"(stage {scheduler.stage_index}/{scheduler.num_stages - 1})"
            )
            trainer.train()
            done += this_segment

            completion = self._env_completion_ratio()
            self._log_curriculum_metrics(done, scheduler.stage_index, completion)

            advanced = scheduler.maybe_advance(completion, this_segment)
            if advanced:
                new_cfg = scheduler.config_for(scheduler.current)
                print(
                    f"[curriculum] advancing to stage "
                    f"{scheduler.stage_index}/{scheduler.num_stages - 1} "
                    f"(ema_completion={scheduler.ema_completion:.2f}); "
                    f"rebuilding env: {new_cfg}"
                )
                self._env = self._rebuild_flatland_env(new_cfg)
                self._rebind_agent_to_env(self._env)

    def _env_completion_ratio(self) -> float:
        """Reach through the skrl wrapper for ``last_completion_ratio``."""
        unwrapped = getattr(self._env, "_unwrapped", self._env)
        return float(getattr(unwrapped, "last_completion_ratio", 0.0))

    def _log_curriculum_metrics(
        self, timestep: int, stage_index: int, completion: float
    ) -> None:
        """Push curriculum scalars to wandb (if active) and print to stdout."""
        print(
            f"[curriculum] t={timestep:>7d} | stage={stage_index} "
            f"| completion={completion:.3f}"
        )
        try:
            import wandb  # type: ignore[import-untyped]
            if wandb.run is not None:
                wandb.log(
                    {
                        "curriculum/stage": stage_index,
                        "curriculum/completion_ratio": completion,
                    },
                    step=timestep,
                    commit=False,
                )
        except ImportError:
            pass

    def _rebuild_flatland_env(self, flatland_cfg: FlatlandConfig):
        """Rebuild the vectorised Flatland env with a new stage config.

        We import here (not at module top) to keep ``BaseRunner`` importable
        in contexts where Flatland is not available (e.g. warehouse-only
        CI smoke tests).
        """
        from functools import partial

        from skrl.envs.wrappers.jax import wrap_env

        from environments import make_flatland_env, make_vectorized_env

        num_envs = int(self._cfg.get("env", {}).get("num_envs", 1))
        factory = partial(make_flatland_env, config=flatland_cfg, render_mode=None)
        raw = make_vectorized_env(env_fn=factory, num_envs=num_envs)
        return wrap_env(raw, wrapper="pettingzoo")

    def _rebind_agent_to_env(self, env) -> None:
        """Point an existing agent instance at a freshly-built env.

        skrl's ``MultiAgent`` keeps references to ``possible_agents`` and
        the vectorised env for sampling; when we swap the env we must also
        update those references so ``record_transition`` and ``act``
        read from the new instance.  The agent's weights / optimiser state
        are preserved.
        """
        agent = self._agent
        # Core env handles used by skrl's training loop.
        agent.possible_agents = list(env.possible_agents)  # type: ignore[attr-defined]
        if hasattr(agent, "_env"):
            agent._env = env  # type: ignore[attr-defined]
        # Some skrl builds store a ``cfg['environment_info']`` with a live
        # env ref; keep them consistent too.
        if hasattr(agent, "cfg") and isinstance(agent.cfg, dict):
            env_info = agent.cfg.get("environment_info")
            if isinstance(env_info, dict) and "env" in env_info:
                env_info["env"] = env

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

            is_done = all(
                terminated.get(a, False) or truncated.get(a, False)
                for a in terminated.keys()
            )
            if is_done:
                obs, _ = self._env.reset()
                frame = self._env.render()
                if frame is not None:
                    frames.append(frame)

        if frames:
            imageio.mimwrite(str(out_path), frames, fps=fps)
            print(f"Recording saved to {out_path}")
        else:
            print("Warning: no frames were captured (env.render() returned None).")

    def record_comm(self, checkpoint_path: str | None = None) -> None:
        """Record an episode video with a communication graph overlay (split-screen).

        Produces a side-by-side video: left panel = environment render,
        right panel = live communication graph (directed graph with soft edge
        weights for MAGIC, static graph + dynamic representations for CommFormer).

        Supported agent types: ``magic``, ``magic_hopfield``, ``commformer``,
        ``commformerhm``.  Falls back to plain ``record()`` for other types.
        """
        import numpy as np
        import imageio  # type: ignore[import-untyped]

        from utils.comm_graph_renderer import render_comm_graph_frame, make_split_frame

        path = checkpoint_path or self._cfg["record"]["checkpoint_path"]
        if path:
            self._load_checkpoint(path)

        record_cfg = self._cfg["record"]
        max_steps: int = record_cfg["timesteps"]
        fps: int = record_cfg["fps"]
        video_dir = Path(record_cfg["video_dir"])
        video_dir.mkdir(parents=True, exist_ok=True)
        exp_name: str = self._cfg["experiment"]["name"]
        out_path = video_dir / f"{exp_name}_comm.mp4"

        agent_type: str = self._cfg.get("experiment", {}).get("agent_type", "mappo")
        num_agents: int = len(self._env.possible_agents)
        agent_labels = [f"A{i}" for i in range(num_agents)]

        # Determine which output key holds adjacency data
        is_magic = agent_type in ("magic", "magic_hopfield")
        is_commformer = agent_type in ("commformer", "commformerhm")

        def _np(v):
            try:
                import jax
                return np.asarray(jax.device_get(v))
            except Exception:
                return np.asarray(v)

        def _extract_adj(outputs_per_agent):
            """Return (adj: np.ndarray (N,N), node_features: np.ndarray|None)."""
            uid0 = self._env.possible_agents[0]
            out0 = outputs_per_agent.get(uid0, {})

            adj = None
            node_features = None

            if "adj_matrices" in out0:
                raw = _np(out0["adj_matrices"])
                # MAGIC: (R, num_envs, N, N) — use last round [-1], first env [0]
                # CommFormer: (N, N) — static learned adjacency
                if raw.ndim == 4:
                    # shape[0]=R, shape[1]=num_envs  →  last round, first env
                    adj = raw[-1, 0]
                elif raw.ndim == 3:
                    adj = raw[-1]  # (R, N, N) → last round
                elif raw.ndim == 2:
                    adj = raw

            elif "hard_adj" in out0:
                raw = _np(out0["hard_adj"])
                adj = raw[0] if raw.ndim > 2 else raw

            # Node features: prefer agg_messages (MAGIC) or encoder_out (CF)
            if "agg_messages" in out0:
                raw = _np(out0["agg_messages"])
                node_features = raw[0] if raw.ndim == 3 else raw
            elif "messages" in out0:
                raw = _np(out0["messages"])
                node_features = raw[0] if raw.ndim == 3 else raw
            else:
                # CommFormer: collect per-agent encoder_out
                rows = []
                for uid in self._env.possible_agents:
                    ag_out = outputs_per_agent.get(uid, {})
                    if "encoder_out" in ag_out:
                        r = _np(ag_out["encoder_out"])
                        rows.append(r[0] if r.ndim == 2 else r)
                if len(rows) == num_agents:
                    node_features = np.stack(rows, axis=0)

            if adj is None:
                adj = np.zeros((num_agents, num_agents), dtype=np.float32)

            return adj, node_features

        battery_capacity: float = float(
            self._cfg.get("env", {}).get("battery_capacity", 0)
        )

        def _extract_agent_state(info: Any) -> tuple[list[float] | None, list[bool]]:
            """Pull battery fractions and dead flags from the step info dict."""
            has_battery = battery_capacity > 0
            levels: list[float] | None = [] if has_battery else None
            dead: list[bool] = []
            for uid in self._env.possible_agents:
                ag_info = info.get(uid, {}) if isinstance(info, dict) else {}
                if has_battery and levels is not None:
                    bat = ag_info.get("battery", battery_capacity)
                    levels.append(float(bat) / battery_capacity)
                dead.append(bool(ag_info.get("battery_dead", False)))
            return levels, dead

        self._agent.set_running_mode("eval")
        frames: list[np.ndarray] = []

        obs, _ = self._env.reset()
        env_frame = self._env.render()
        panel_h = env_frame.shape[0] if env_frame is not None else 480
        panel_w = panel_h  # square comm panel

        title = "MAGIC — Communication Graph" if is_magic else "CommFormer — Communication Graph"

        for step in range(max_steps):
            actions, _, outputs_per_agent = self._agent.act(
                obs, timestep=step, timesteps=max_steps
            )
            obs, _, terminated, truncated, info = self._env.step(actions)

            env_frame = self._env.render()
            if env_frame is None:
                env_frame = np.zeros((panel_h, panel_h, 3), dtype=np.uint8)
            else:
                panel_h = env_frame.shape[0]

            adj, node_features = _extract_adj(outputs_per_agent)
            battery_levels, agent_dead = _extract_agent_state(info)

            comm_frame = render_comm_graph_frame(
                adj=adj,
                title=title,
                step=step,
                agent_labels=agent_labels,
                node_features=node_features,
                battery_levels=battery_levels,
                agent_dead=agent_dead,
                width_px=panel_w,
                height_px=panel_h,
                is_dynamic=is_magic,
            )

            combined = make_split_frame(env_frame, comm_frame)
            frames.append(combined)

            is_done = all(
                terminated.get(a, False) or truncated.get(a, False)
                for a in terminated.keys()
            )
            if is_done:
                obs, _ = self._env.reset()
                env_frame = self._env.render()
                if env_frame is not None:
                    comm_frame = render_comm_graph_frame(
                        adj=np.zeros((num_agents, num_agents)),
                        title=title + " [reset]",
                        step=step,
                        agent_labels=agent_labels,
                        width_px=panel_w,
                        height_px=panel_h,
                        is_dynamic=is_magic,
                    )
                    frames.append(make_split_frame(env_frame, comm_frame))

        if frames:
            imageio.mimwrite(str(out_path), frames, fps=fps)
            print(f"Communication recording saved to {out_path}")
        else:
            print("Warning: no frames captured.")

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
        elif env_id == "continuous_coord":
            from utils.continuous_coord_eval_analysis import ContinuousCoordEvalCollector
            from utils.continuous_coord_eval_visualizer import save_all_cc_figures

            cfg_env = env_cfg
            deadline_min = cfg_env.get("target_deadline_min", 30)
            deadline_max = cfg_env.get("target_deadline_max", 80)
            deadline_avg = (deadline_min + deadline_max) / 2.0

            collector = ContinuousCoordEvalCollector(
                num_agents=cfg_env.get("num_agents", 4),
                max_targets=cfg_env.get("max_targets", 3),
                max_cycles=cfg_env.get("max_cycles", 200),
                capture_reward=cfg_env.get("reward_capture", 10.0),
                collision_radius=cfg_env.get("collision_radius", 0.03),
                deadline_avg=deadline_avg,
                type_dim=cfg_env.get("num_agent_types", 1)
                if cfg_env.get("num_agent_types", 1) > 1
                else 0,
            )
            data = collector.collect(
                env=self._env, agent=self._agent, n_episodes=n_episodes
            )
            exp_name = self._cfg.get("experiment", {}).get("name", "experiment")
            save_all_cc_figures(data, output_dir=output_dir, prefix=exp_name)

        elif env_id in ("warehouse", "warehouse-jax"):
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
                    data, output_dir=f"{output_dir}/{exp_name}/magic", prefix=exp_name
                )
            collector = WarehouseEvalCollector(
                num_agents=env_cfg.get("num_agents", 4),
                battery_capacity=env_cfg.get("battery_capacity", 160.0),
            )
            data = collector.collect(
                env=self._env, agent=self._agent, n_episodes=n_episodes
            )
            exp_name = self._cfg.get("experiment", {}).get("name", "experiment")
            save_all_warehouse_figures(data, output_dir=f"{output_dir}/{exp_name}", prefix=exp_name)

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
