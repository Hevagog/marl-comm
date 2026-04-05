# fmt: off
from skrl.resources.preprocessors.jax import RunningStandardScaler  # noqa: E402

CONFIG = {
    "experiment": {
        "name":             "commformer_blindspot_v1",
        "agent_type":       "commformer",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["commformer", "blindspot"],
        },
        "write_interval":      "auto",
        "checkpoint_interval": "auto",
        "store_separately":    False,
    },

    "env": {
        "id":            "blindspot",
        "num_envs":      1,
        "grid_size":     9,
        "max_cycles":    100,
        "num_traps":     5,
        "use_communication": False,  # Comm handled by CommFormer, not env tokens
        "random_goal":   True,
        "min_goal_start_distance": 4,
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

    # ---- CommFormer-specific PPO parameters ----
    "commformer": {
        "rollouts":        4096,
        "learning_epochs": 8,
        "mini_batches":    4,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        # obs_dim for BlindSpot (9×9, 2 agents, 5 traps): 8 + 5*3 + 2 = 25
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 25},
        # shared_obs_dim = 50 (both agents' obs concatenated)
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 50},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        "entropy_loss_scale": 0.02,
        "value_loss_scale":   1.0,

        "kl_threshold": 0.05,

        "rewards_shaper":       None,
        "time_limit_bootstrap": True,

        "weight_decay":       1e-4,

        # ---- CommFormer architecture hyperparameters ----
        # Modest capacity for 2-agent navigation task.
        "hidden_dim":  64,
        "num_blocks":  1,
        "num_heads":   1,
        "head_dim":    64,
        "mlp_dim":     128,
        "sparsity":    0.4,  # k=1 for N=2: at least 1 neighbor (full comm)
    },

    "policy": {
        "unnormalized_log_prob": True,
    },

    "value": {
        "hidden_sizes": [256, 256],
    },

    "memory": {
        "size": 4096,
    },
}
