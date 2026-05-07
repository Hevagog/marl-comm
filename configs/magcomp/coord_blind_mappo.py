# fmt: off
"""CC Blind — MAPPO baseline.

vision_range=0.1, target_vision_range=0.2, typed agents (2×2). Slots zeroed
when out-of-range → MAPPO acts on incomplete obs. Lower bound. obs=44, state=46.
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

_ENV = {
    "id": "continuous_coord", "num_envs": 16,
    "num_agents": 4, "max_cycles": 200,
    "max_targets": 3, "capture_radius": 0.08,
    "vision_range": 0.1, "target_vision_range": 0.2,
    "collision_radius": 0.03, "target_arrival_rate": 0.15,
    "target_k_min": 2, "target_k_max": 3,
    "target_deadline_min": 40, "target_deadline_max": 100,
    "chain_event_prob": 0.1, "max_speed": 0.05,
    "num_agent_types": 2, "agent_types": [0, 0, 1, 1],
    "penalty_wrong_type": -0.5, "penalty_wrong_composition": -1.0,
}

CONFIG = {
    "experiment": {
        "name": "magcomp_cc_blind_mappo", "agent_type": "mappo",
        "directory": "runs", "wandb": True,
        "wandb_kwargs": {"project": "marl-comm", "tags": ["magcomp", "cc_blind", "mappo"]},
        "write_interval": 25_000, "checkpoint_interval": 200_000, "store_separately": False,
    },
    "env": _ENV,
    "training": {"timesteps": 2_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 2_000, "checkpoint_path": None, "video_dir": "recordings", "fps": 10},
    "mappo": {
        "rollouts": 4096, "learning_epochs": 10, "mini_batches": 2,
        "discount_factor": 0.99, "lambda": 0.95, "learning_rate": 3e-4,
        "learning_rate_scheduler": None, "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay": True, "lr_decay_start_fraction": 0.3, "min_lr_fraction": 0.1,
        "state_preprocessor":               RunningStandardScaler,
        "state_preprocessor_kwargs":        {"size": 44},
        "shared_state_preprocessor":        RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": 46},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},
        "random_timesteps": 0, "learning_starts": 0,
        "grad_norm_clip": 0.5, "ratio_clip": 0.2, "value_clip": 0.2,
        "clip_predicted_values": False,
        "entropy_loss_scale": 0.05, "entropy_annealing": True,
        "entropy_loss_scale_start": 0.05, "entropy_loss_scale_end": 0.005,
        "value_loss_scale": 1.0, "kl_threshold": 0.05, "kl_warmup_fraction": 0.3,
        "rewards_shaper": lambda rewards, *_: jnp.clip(rewards, -8.0, 15.0),
        "time_limit_bootstrap": True, "weight_decay": 1e-4,
    },
    "policy": {"hidden_sizes": [256, 128], "unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [256, 128]},
    "memory": {"size": 4096},
}
