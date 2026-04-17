"""Optuna hyperparameter search for the MAM agent.

Search space follows Daniel et al. 2024 (Table 7):
  PPO epochs:          {2, 5, 10}
  Mini-batches:        {1, 2, 4, 8}
  Clipping epsilon:    {0.05, 0.1, 0.2}
  Grad norm clip:      {0.5, 5, 10}
  Learning rate:       {1e-4, 2.5e-4, 5e-4, 1e-3}
  Embedding dim:       {32, 64, 128}
  Hidden state dim N:  {1, 2, 4, 8, 16, 32}
  Delta rank:          {1, 2, 4, 8, 16, 32, 64, 128}

Usage
-----
  # warehouse (full MAM encoder-decoder)
  python src/tune_mam.py --env warehouse \\
      --n-trials 40 --trial-timesteps 500000

  # continuous_coord (MAM enc-only)
  python src/tune_mam.py --env continuous_coord \\
      --n-trials 40 --trial-timesteps 500000

  # resume / shared storage
  python src/tune_mam.py --env warehouse \\
      --storage sqlite:///tune_mam.db --study-name mam_warehouse
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import optuna
from optuna.samplers import TPESampler
from skrl.envs.wrappers.jax import wrap_env
from skrl.resources.preprocessors.jax import RunningStandardScaler


_WAREHOUSE_BASE: dict[str, Any] = {
    "experiment": {
        "name": "mam_tune_warehouse",
        "agent_type": "mam",
        "directory": "runs/tune",
        "wandb": False,
        "wandb_kwargs": {},
        "write_interval": "auto",
        "checkpoint_interval": 0,
        "store_separately": False,
    },
    "env": {
        "id": "warehouse",
        "num_envs": 8,
        "grid_height": 10,
        "grid_width": 10,
        "num_agents": 4,
        "max_agents": 4,
        "num_shelves": 4,
        "resources_per_shelf": 6,
        "num_treatment_stations": 1,
        "num_goal_locations": 1,
        "treatment_duration": 3,
        "comm_noise_prob": 0.0,
        "vision_range": 2,
        "max_cycles": 200,
        "comm_range": 4,
        "randomize_layout": False,
        "enable_task_deadlines": False,
        "enable_heterogeneous": False,
        "enable_interference_zones": False,
        "enable_battery": False,
        "agent_failure_prob": 0.0,
    },
    "training": {
        "timesteps": 500_000,
        "seed": 42,
    },
    "eval": {
        "timesteps": 2_000,
        "checkpoint_path": None,
    },
    "record": {
        "timesteps": 500,
        "checkpoint_path": None,
        "video_dir": "recordings",
        "fps": 4,
    },
    "mam": {
        "rollouts": 128,
        "learning_epochs": 5,
        "mini_batches": 2,
        "discount_factor": 0.99,
        "lambda": 0.9,
        "learning_rate": 5e-4,
        "learning_rate_scheduler": None,
        "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay": True,
        "lr_decay_start_fraction": 0.05,
        "min_lr_fraction": 0.1,
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": 1},
        "shared_state_preprocessor": RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": 1},
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1},
        "random_timesteps": 0,
        "learning_starts": 0,
        "grad_norm_clip": 0.5,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "clip_predicted_values": False,
        "entropy_loss_scale": 0.01,
        "entropy_annealing": False,
        "entropy_loss_scale_start": 0.05,
        "entropy_loss_scale_end": 0.01,
        "value_loss_scale": 0.5,
        "kl_threshold": 0,
        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 20.0),
        "time_limit_bootstrap": True,
        "n_embd": 64,
        "n_block": 1,
        "d_state": 8,
        "d_conv": 4,
        "delta_rank": 8,
    },
    "policy": {"unnormalized_log_prob": True},
    "value": {"hidden_sizes": [128, 128]},
    "memory": {"size": 128},
}

_CONTINUOUS_COORD_BASE: dict[str, Any] = {
    "experiment": {
        "name": "mam_tune_continuous_coord",
        "agent_type": "mam_enc_only",
        "directory": "runs/tune",
        "wandb": False,
        "wandb_kwargs": {},
        "write_interval": "auto",
        "checkpoint_interval": 0,
        "store_separately": False,
    },
    "env": {
        "id": "continuous_coord",
        "num_envs": 16,
        "num_agents": 4,
        "max_cycles": 200,
        "max_targets": 3,
        "capture_radius": 0.08,
        "vision_range": 0.4,
        "collision_radius": 0.03,
        "target_arrival_rate": 0.15,
        "target_k_min": 2,
        "target_k_max": 3,
        "target_deadline_min": 30,
        "target_deadline_max": 80,
        "chain_event_prob": 0.2,
        "max_speed": 0.05,
    },
    "training": {
        "timesteps": 500_000,
        "seed": 42,
    },
    "eval": {
        "timesteps": 2_000,
        "checkpoint_path": None,
    },
    "record": {
        "timesteps": 500,
        "checkpoint_path": None,
        "video_dir": "recordings",
        "fps": 4,
    },
    "mam": {
        "rollouts": 1024,
        "learning_epochs": 5,
        "mini_batches": 2,
        "discount_factor": 0.99,
        "lambda": 0.9,
        "learning_rate": 5e-4,
        "learning_rate_scheduler": None,
        "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay": True,
        "lr_decay_start_fraction": 0.05,
        "min_lr_fraction": 0.1,
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": 1},
        "shared_state_preprocessor": RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": 1},
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1},
        "random_timesteps": 0,
        "learning_starts": 0,
        "grad_norm_clip": 0.5,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "clip_predicted_values": False,
        "entropy_loss_scale": 0.01,
        "entropy_annealing": False,
        "entropy_loss_scale_start": 0.05,
        "entropy_loss_scale_end": 0.01,
        "value_loss_scale": 0.5,
        "kl_threshold": 0,
        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 15.0),
        "time_limit_bootstrap": True,
        "n_embd": 64,
        "n_block": 1,
        "d_state": 8,
        "d_conv": 4,
        "delta_rank": 8,
    },
    "policy": {"unnormalized_log_prob": True},
    "value": {"hidden_sizes": [128, 128]},
    "memory": {"size": 1024},
}

_BASE_CONFIGS = {
    "warehouse": _WAREHOUSE_BASE,
    "continuous_coord": _CONTINUOUS_COORD_BASE,
}

# ---------------------------------------------------------------------------
# Hyperparameter search space (Table 7, Daniel et al. 2024)
# ---------------------------------------------------------------------------


def _sample_hyperparams(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_epochs": trial.suggest_categorical("learning_epochs", [2, 5, 10]),
        "mini_batches": trial.suggest_categorical("mini_batches", [1, 2, 4, 8]),
        "ratio_clip": trial.suggest_categorical("ratio_clip", [0.05, 0.1, 0.2]),
        "grad_norm_clip": trial.suggest_categorical("grad_norm_clip", [0.5, 5.0, 10.0]),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [1e-4, 2.5e-4, 5e-4, 1e-3]
        ),
        "n_embd": trial.suggest_categorical("n_embd", [32, 64, 128]),
        "d_state": trial.suggest_categorical("d_state", [1, 2, 4, 8, 16, 32]),
        "delta_rank": trial.suggest_categorical(
            "delta_rank", [1, 2, 4, 8, 16, 32, 64, 128]
        ),
    }


# ---------------------------------------------------------------------------
# Environment / runner helpers
# ---------------------------------------------------------------------------


def _make_env(env_id: str, env_cfg: dict, num_envs: int):
    """Create a wrapped vectorised environment."""
    from environments import make_vectorized_env

    match env_id:
        case "warehouse":
            from environments import make_warehouse_env, WarehouseConfig

            config_kwargs = {
                k: v
                for k, v in env_cfg.items()
                if k in WarehouseConfig.__dataclass_fields__
            }
            factory = partial(
                make_warehouse_env,
                config=WarehouseConfig(**config_kwargs),
                render_mode=None,
            )

        case "continuous_coord":
            from environments import ContinuousCoordConfig, make_continuous_coord_env

            config_kwargs = {
                k: v
                for k, v in env_cfg.items()
                if k in ContinuousCoordConfig.__dataclass_fields__
            }
            factory = partial(
                make_continuous_coord_env,
                config=ContinuousCoordConfig(**config_kwargs),
                render_mode=None,
            )

        case _:
            raise ValueError(f"Unsupported env: {env_id!r}")

    raw = make_vectorized_env(env_fn=factory, num_envs=num_envs)
    return wrap_env(raw, wrapper="pettingzoo")


def _get_runner_class(agent_type: str):
    if agent_type == "mam":
        from agents.mam.train import MAMRunner

        return MAMRunner
    if agent_type == "mam_enc_only":
        from agents.mam.train_ablations import MAMEncOnlyRunner

        return MAMEncOnlyRunner
    raise ValueError(f"Unknown agent_type: {agent_type!r}")


# ---------------------------------------------------------------------------
# Per-trial evaluation: run N short episodes, return mean total reward
# ---------------------------------------------------------------------------


def _eval_mean_return(runner, n_episodes: int = 5, max_steps: int = 500) -> float:
    """Run n_episodes with the trained agent; return mean episode return."""
    agent = runner._agent
    env = runner._env
    agent.set_running_mode("eval")

    returns: list[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        for step in range(max_steps):
            actions, _, _ = agent.act(obs, timestep=step, timesteps=max_steps)
            obs, rewards, terminated, truncated, _ = env.step(actions)
            n_agents = max(len(rewards), 1)
            ep_return += sum(rewards.values()) / n_agents
            done = all(
                terminated.get(a, False) or truncated.get(a, False) for a in terminated
            )
            if done:
                break
        returns.append(ep_return)

    return float(sum(returns) / len(returns))


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def _make_objective(env_id: str, trial_timesteps: int, n_eval_episodes: int):
    base_cfg = _BASE_CONFIGS[env_id]

    def objective(trial: optuna.Trial) -> float:
        hparams = _sample_hyperparams(trial)

        cfg = copy.deepcopy(base_cfg)
        cfg["training"]["timesteps"] = trial_timesteps

        mam_cfg = cfg["mam"]
        for key, value in hparams.items():
            mam_cfg[key] = value

        # rollouts must be divisible by mini_batches
        mam_cfg["rollouts"] = max(
            mam_cfg["mini_batches"],
            (mam_cfg["rollouts"] // mam_cfg["mini_batches"]) * mam_cfg["mini_batches"],
        )
        cfg["memory"]["size"] = mam_cfg["rollouts"]

        # value_clip should match ratio_clip
        mam_cfg["value_clip"] = mam_cfg["ratio_clip"]

        env_cfg = cfg["env"]
        num_envs: int = env_cfg["num_envs"]
        agent_type: str = cfg["experiment"]["agent_type"]

        try:
            env = _make_env(env_id, env_cfg, num_envs)
            RunnerClass = _get_runner_class(agent_type)
            runner = RunnerClass(env=env, cfg=cfg)
            runner.train()
            score = _eval_mean_return(runner, n_episodes=n_eval_episodes)
        except Exception as exc:
            print(f"[trial {trial.number}] failed: {exc}", flush=True)
            raise optuna.exceptions.TrialPruned() from exc
        finally:
            gc.collect()

        print(
            f"[trial {trial.number}] score={score:.4f} | "
            + " | ".join(f"{k}={v}" for k, v in hparams.items()),
            flush=True,
        )
        return score

    return objective


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------


def _save_results(study: optuna.Study, output_path: Path) -> None:
    trials_data = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": t.state.name,
            "duration_s": t.duration.total_seconds() if t.duration else None,
        }
        for t in study.trials
    ]
    best = study.best_trial
    results = {
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "all_trials": trials_data,
    }
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna TPE hyperparameter search for MAM agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env",
        required=True,
        choices=["warehouse", "continuous_coord"],
        help="Environment to tune on.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=40,
        help="Number of Optuna trials (paper uses 40).",
    )
    parser.add_argument(
        "--trial-timesteps",
        type=int,
        default=500_000,
        help="Training timesteps per trial.",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=5,
        help="Evaluation episodes after each trial.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name. Defaults to 'mam_<env>'.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. 'sqlite:///tune_mam.db'. "
        "If None, study is in-memory only.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON results. Defaults to 'tune_mam_<env>.json'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for TPE sampler.",
    )
    return parser.parse_args()


def main() -> None:
    # Ensure src/ is on the path when invoked as a script
    _src = Path(__file__).parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    args = _parse_args()

    study_name = args.study_name or f"mam_{args.env}"
    output_path = Path(args.output or f"tune_mam_{args.env}.json")

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=args.storage,
        load_if_exists=(args.storage is not None),
    )

    objective = _make_objective(
        env_id=args.env,
        trial_timesteps=args.trial_timesteps,
        n_eval_episodes=args.n_eval_episodes,
    )

    print(
        f"Starting MAM hyperparameter search\n"
        f"  env:              {args.env}\n"
        f"  n_trials:         {args.n_trials}\n"
        f"  trial_timesteps:  {args.trial_timesteps:,}\n"
        f"  n_eval_episodes:  {args.n_eval_episodes}\n"
        f"  study_name:       {study_name}\n"
        f"  storage:          {args.storage or 'in-memory'}\n",
        flush=True,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        catch=(Exception,),
    )

    print("\n=== Best trial ===")
    best = study.best_trial
    print(f"  value:  {best.value:.4f}")
    print(f"  params: {json.dumps(best.params, indent=4)}")

    _save_results(study, output_path)


if __name__ == "__main__":
    main()
