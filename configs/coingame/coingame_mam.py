# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

_ENV = {
    "id": "coingame", "num_envs": 16,
    "grid_size": 5, "max_cycles": 50,
    "pick_reward": 1.0, "steal_penalty": -2.0,
}

CONFIG = {
    "experiment": {
        "name": "magcomp_coingame_mam", "agent_type": "mam",
        "directory": "runs", "wandb": True,
        "wandb_kwargs": {"project": "marl-comm", "tags": ["magcomp", "coingame", "mam", "diagnostic"]},
        "write_interval": 10_000, "checkpoint_interval": 500_000, "store_separately": False,
    },
    "env": _ENV,
    "training": {"timesteps": 1_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 1_000, "checkpoint_path": None, "video_dir": "recordings", "fps": 4},

    "mam": {
        "rollouts": 256, "learning_epochs": 10, "mini_batches": 2,
        "discount_factor": 0.99, "lambda": 0.95,

        "learning_rate": 1.5e-4,
        "learning_rate_scheduler": None, "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay": True, "lr_decay_start_fraction": 0.3, "min_lr_fraction": 0.1,

        "update_state_preprocessor_in_update": False,
        "update_shared_state_preprocessor_in_update": False,

        "state_preprocessor":               RunningStandardScaler,
        "state_preprocessor_kwargs":        {"size": 12},
        "shared_state_preprocessor":        RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": 24},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0, "learning_starts": 0,
        "grad_norm_clip": 0.5, "ratio_clip": 0.2, "value_clip": 0.2,
        "clip_predicted_values": False,

        "entropy_loss_scale": 0.02, "entropy_annealing": True,
        "entropy_loss_scale_start": 0.05, "entropy_loss_scale_end": 0.01,
        "value_loss_scale": 1.0,

        "kl_threshold": 0.03, "kl_warmup_fraction": 0.0,
        "ratio_max_threshold": 3.0,

        "rewards_shaper": lambda rewards, *_: jnp.clip(rewards, -2.0, 1.0),
        "time_limit_bootstrap": True,

        # MAM architecture (paper Table 4 scaled for 2 agents).
        "n_embd": 64, "n_block": 1,
        "d_state": 32, "d_conv": 4, "delta_rank": 32,
    },

    "policy": {"unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [128, 128]},
    "memory": {"size": 256},
}
