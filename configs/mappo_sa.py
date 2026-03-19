# fmt: off

from skrl.resources.preprocessors.jax import RunningStandardScaler  # noqa: E402

CONFIG = {
    "experiment": {
        "name":             "mappo_sa_v0",
        "agent_type":       "mappo",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mappo", "simple_adversary"],
        },
        "write_interval":      "auto",
        "checkpoint_interval": "auto",
        "store_separately":    False,
    },

    "env": {
        "id":            "simple_adversary",
        "num_envs":      16,           
    },
    "training": {
        "timesteps": 8_000_000,
        "seed":      42,
    },
    "eval": {
        "timesteps":       5_000,
        "checkpoint_path": None,
    },
    "record": {
        "timesteps":       1_000,
        "checkpoint_path": None,
        "video_dir":       "recordings",
        "fps":             4,
    },
    "mappo": {
        "rollouts":        2000,      # number of rollouts before updating
        "learning_epochs": 8,       # learning epochs per update
        "mini_batches":    4,       # mini-batches per learning epoch

        "discount_factor": 0.99,    # gamma
        "lambda":          0.95,    # TD(lambda) / GAE lambda

        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs": {
            "adversary_0": {"size": 8},
            "agent_0": {"size": 10},
            "agent_1": {"size": 10},
        },
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": 28},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        "entropy_loss_scale": 0.02,   # fallback when entropy_annealing is disabled
        "entropy_annealing": True,
        "entropy_loss_scale_start": 0.015,
        "entropy_loss_scale_end": 0.005,
        "debug_entropy_stats": False,
        "value_loss_scale":   1.0,    # 1.0 is correct for separate policy/value optimisers

        "kl_threshold": 0.05,
        "kl_warmup_fraction": 0.3,
        "debug_kl_stats": False,

        "rewards_shaper":       None,
        "time_limit_bootstrap": True,

        "weight_decay":       1e-4,
        "linear_lr_decay":    True,
        "lr_decay_start_fraction": 0.3,   # NEW — hold full LR for first 30% of training
        "min_lr_fraction":         0.1,
    },

    "policy": {
        "hidden_sizes":          [256, 256],
        "unnormalized_log_prob": True,
    },

    "value": {
        "hidden_sizes": [256, 256],
    },

    "memory": {
        "size": 2000,
    },
}
