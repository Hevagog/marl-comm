# fmt: off
"""Phase 2 — MAPPO fine-tune transfer to standard reward.

Usage:
    python -m cli --config configs/magcomp/coingame_partialobs_mappo_transfer_finetune.py \
        --task train \
        --resume runs/magcomp_coingame_partialobs_mappo_altruistic/checkpoints/agent_5000000.pt

Memoryless control arm for the altruism-persistence experiment.  This isolates
ordinary policy-weight inertia after altruistic pretraining from the temporal
state and prototype-attractor mechanisms available to MAGIC-GRU / MAGIC-HSC.
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

_ENV = {
    "id": "coingame-partialobs", "num_envs": 16,
    "grid_size": 7, "max_cycles": 100,
    "vision_range": 2,
    "pick_reward": 1.0, "steal_penalty": -2.0,
    "social_welfare_alpha": 0.0,
}

CONFIG = {
    "experiment": {
        "name": "magcomp_coingame_partialobs_mappo_transfer_finetune", "agent_type": "mappo",
        "directory": "runs", "wandb": True,
        "wandb_kwargs": {"project": "marl-comm", "tags": ["magcomp", "coingame_partialobs", "mappo", "transfer", "full_finetune", "phase2", "symmetric"]},
        "write_interval": "auto", "checkpoint_interval": "auto", "store_separately": False,
    },
    "env": _ENV,
    "training": {"timesteps": 2_000_000, "seed": 42},
    "reset_value_preprocessor": True,
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 1_000, "checkpoint_path": None, "video_dir": "recordings", "fps": 4},
    "mappo": {
        "rollouts": 512, "learning_epochs": 10, "mini_batches": 4,
        "discount_factor": 0.99, "lambda": 0.95, "learning_rate": 3e-4,
        "learning_rate_scheduler": None, "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay": True, "lr_decay_start_fraction": 0.3, "min_lr_fraction": 0.1,
        "state_preprocessor":               RunningStandardScaler,
        "state_preprocessor_kwargs":        {"size": 13},
        "shared_state_preprocessor":        RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": 26},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},
        "random_timesteps": 0, "learning_starts": 0,
        "grad_norm_clip": 0.5, "ratio_clip": 0.2, "value_clip": 0.2,
        "clip_predicted_values": False,
        "entropy_loss_scale": 0.02, "entropy_annealing": True,
        "entropy_loss_scale_start": 0.05, "entropy_loss_scale_end": 0.02,
        "value_loss_scale": 1.0, "kl_threshold": 0.05, "kl_warmup_fraction": 0.1,
        "rewards_shaper": lambda rewards, *_: jnp.clip(rewards, -2.0, 1.0),
        "time_limit_bootstrap": True, "weight_decay": 1e-4,
    },
    "policy": {"hidden_sizes": [128, 128], "unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [128, 128]},
    "memory": {"size": 512},
}
