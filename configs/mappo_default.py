# fmt: off

CONFIG = {
    "experiment": {
        "name":             "mappo_coingame",
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
        "grid_size":     7,
        "max_cycles":    50,
        "pick_reward":   1.0,
        "steal_penalty": -2.0,
    },
    "training": {
        "timesteps": 2_000_000,
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
        "rollouts":        16,      # number of rollouts before updating
        "learning_epochs": 8,       # learning epochs per update
        "mini_batches":    2,       # mini-batches per learning epoch

        "discount_factor": 0.99,    # gamma
        "lambda":          0.95,    # TD(lambda) / GAE lambda

        "learning_rate":                  1e-3,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "state_preprocessor":                None,
        "state_preprocessor_kwargs":         {},
        "shared_state_preprocessor":         None,
        "shared_state_preprocessor_kwargs":  {},
        "value_preprocessor":               None,
        "value_preprocessor_kwargs":        {},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        "entropy_loss_scale": 0.01,
        "value_loss_scale":   0.5,

        "kl_threshold": 0,

        "rewards_shaper":       None,
        "time_limit_bootstrap": False,
    },

    "policy": {
        "hidden_sizes":          [64, 64],
        "unnormalized_log_prob": True,
    },

    "value": {
        "hidden_sizes": [64, 64],
    },

    "memory": {
        "size": 512,
    },
}
