# fmt: off
from skrl.resources.preprocessors.jax import RunningStandardScaler  # noqa: E402

CONFIG = {
    "experiment": {
        "name":             "mam_coingame_v1",
        "agent_type":       "mam",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mam", "coingame"],
        },
        "write_interval":      "auto",
        "checkpoint_interval": "auto",
        "store_separately":    False,
    },

    "env": {
        "id":            "coingame",
        "num_envs":      16,
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

    # ---- MAM-specific PPO parameters ----
    # Merged into skrl's MAPPO_DEFAULT_CONFIG.
    # Also contains MAM Mamba architecture hyperparameters.
    "mam": {
        # PPO parameters (from paper Table 4 defaults)
        "rollouts":        128,       # rollout_length per paper default
        "learning_epochs": 5,         # ppo_epochs
        "mini_batches":    2,         # num_minibatches

        "discount_factor": 0.99,      # γ
        "lambda":          0.9,       # GAE λ (paper Table 4: gae_lambda=0.9)

        "learning_rate":                  2.5e-4, # middle of paper search range
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

        "grad_norm_clip":         5.0,    # max_grad_norm (paper search: {0.5, 5, 10})
        "ratio_clip":             0.2,    # clip_eps (paper search: {0.05, 0.1, 0.2})
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        "entropy_loss_scale": 0.01,       # ent_coef (paper Table 4)
        "value_loss_scale":   0.5,        # vf_coef (paper Table 4)

        "kl_threshold": 0,

        "rewards_shaper":       None,
        "time_limit_bootstrap": True,

        # ---- MAM Mamba architecture hyperparameters ----
        # (Daniel et al. 2024, Table 4 and network/mamba.yaml)
        "n_embd":      128,    # embedding dimension (model embedding dimension)
        "n_block":     1,      # number of Encoder/Decoder blocks
        "d_state":     32,     # latent state dim N (hidden state dimension)
        "d_conv":      4,      # 1D convolution kernel size
        "delta_rank":  128,    # Δ projection dimension
    },

    "policy": {
        "unnormalized_log_prob": True,
    },

    "value": {
        "hidden_sizes": [128, 128],
    },

    "memory": {
        "size": 128,   # match rollout size
    },
}
