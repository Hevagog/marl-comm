# fmt: off
from skrl.resources.preprocessors.jax import RunningStandardScaler  # noqa: E402

CONFIG = {
    "experiment": {
        "name":             "commformer_coingame_v1",
        "agent_type":       "commformer",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["commformer", "coingame"],
        },
        "write_interval":      "auto",
        "checkpoint_interval": "auto",
        "store_separately":    False,
    },

    "env": {
        "id":            "coingame",
        "num_envs":      16,           # parallel environments for faster training
        "grid_size":     7,
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

    # ---- CommFormer-specific PPO parameters ----
    # Merged into skrl's MAPPO_DEFAULT_CONFIG.
    # Also contains CommFormer architecture hyperparameters.
    "commformer": {
        # PPO parameters
        "rollouts":        3200,     # CommFormer Table 2: batch_size 3200
        "learning_epochs": 10,       # CommFormer Table 4
        "mini_batches":    1,        # CommFormer Table 2: num_mini_batch 1

        "discount_factor": 0.99,     # CommFormer Table 4: γ = 0.99
        "lambda":          0.95,

        "learning_rate":                  5e-4,   # CommFormer Table 2
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 11},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 22},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         10.0,    # CommFormer Table 2: max_grad_norm = 10
        "ratio_clip":             0.05,    # CommFormer Table 4: ppo_clip = 0.05
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        "entropy_loss_scale": 0.01,   # CommFormer Table 2: entropy_coef = 0.01
        "value_loss_scale":   0.5,

        "kl_threshold": 0,

        "rewards_shaper":       None,
        "time_limit_bootstrap": True,

        "weight_decay":       1e-4,

        # ---- CommFormer architecture hyperparameters ----
        # (Hu et al. 2024, §3.2 and Tables 2–4)
        "hidden_dim":    64,       # Table 2: hidden_layer_dim = 64
        "num_blocks":    1,        # Table 4: num_blocks = 1
        "num_heads":     1,        # Table 4: num_heads = 1
        "head_dim":      64,       # matched to hidden_dim
        "mlp_dim":       128,      # 2x hidden_dim (standard Transformer)
        "sparsity":      0.4,      # Table 1: S = 0.4 for all tasks
    },

    "policy": {
        "unnormalized_log_prob": True,
    },

    "value": {
        "hidden_sizes": [128, 128],
    },

    "memory": {
        "size": 3200,    # match rollout size
    },
}
