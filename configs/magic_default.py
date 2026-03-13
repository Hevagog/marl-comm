# fmt: off
from skrl.resources.preprocessors.jax import RunningStandardScaler  # noqa: E402

CONFIG = {
    "experiment": {
        "name":             "magic_coingame_v3",
        "agent_type":       "magic",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["magic", "coingame"],
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

    "magic": {
        "rollouts":        2048,
        "learning_epochs": 8,
        "mini_batches":    4,

        "discount_factor": 0.99,
        "lambda":          0.95,

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

        "entropy_loss_scale": 0.005,   # match MAPPO: prevent premature entropy collapse
        "value_loss_scale":   1.0,    # match MAPPO: correct for separate optimisers

        "kl_threshold": 0.02,         # match MAPPO: safety valve against catastrophic policy change

        "rewards_shaper":       None,
        "time_limit_bootstrap": True,

        "weight_decay":       1e-4,
        "linear_lr_decay":    True,   # match MAPPO: linear LR decay from initial_lr to 0

        # ---- MAGIC communication hyperparameters ----
        # (Niu et al. 2021, §4: Scheduler + Message Processor)
        "message_dim":        64,     # dimension of message vectors
        "num_comm_rounds":    2,      # L: number of communication rounds
        "num_heads":          4,      # heads in Message Processor GAT
        "gumbel_temperature": 0.5,    # τ for Gumbel-Softmax in Scheduler
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
