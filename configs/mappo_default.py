# fmt: off
from skrl.resources.preprocessors.jax import RunningStandardScaler  # noqa: E402

CONFIG = {
    "experiment": {
        "name":             "mappo_coingame_v7",
        "agent_type":       "mappo",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mappo", "coingame"],
        },
        "write_interval":      "auto",
        "checkpoint_interval": "auto",
        "store_separately":    False,
    },

    "env": {
        "id":            "coingame",
        "grid_size":     5,
        "max_cycles":    50,
        "pick_reward":   1.0,
        "steal_penalty": -2.0,
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
        "rollouts":        2048,      # number of rollouts before updating
        "learning_epochs": 8,       # learning epochs per update
        "mini_batches":    4,       # mini-batches per learning epoch

        "discount_factor": 0.99,    # gamma
        "lambda":          0.95,    # TD(lambda) / GAE lambda

        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 12},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 24},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        "entropy_loss_scale": 0.005,   # prevents premature entropy collapse in 4-action space
        "value_loss_scale":   1.0,    # 1.0 is correct for separate policy/value optimisers

        "kl_threshold": 0.02,        # KL early stopping (Bug 4 fix) — safety valve against catastrophic policy change

        "rewards_shaper":       None,
        "time_limit_bootstrap": True,

        "weight_decay":       1e-4,
        "linear_lr_decay":    True,   # Bug 1 fix: linear LR decay from initial_lr to 0
    },

    "policy": {
        "hidden_sizes":          [128, 128],
        "unnormalized_log_prob": True,
    },

    "value": {
        "hidden_sizes": [128, 128],
    },

    "memory": {
        "size": 2048,
    },
}
