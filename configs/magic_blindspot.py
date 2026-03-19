# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler  # noqa: E402

CONFIG = {
    "experiment": {
        "name":             "magic_blindspot_v2",
        "agent_type":       "magic",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["magic", "blindspot"],
        },
        "write_interval":      "auto",
        "checkpoint_interval": "auto",
        "store_separately":    False,
    },

    "env": {
        "id":            "blindspot",
        "num_envs":      16,           
        "grid_size":     9,
        "max_cycles":    100,
        "num_traps":     5,
        "use_communication": False, # Comm handled differentially by MAGIC, not env discrete messages
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
        "rollouts":        4096,      
        "learning_epochs": 8,
        "mini_batches":    4,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 25},
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

        "entropy_loss_scale":       0.02,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.05,
        "entropy_loss_scale_end":   0.01,
        "debug_entropy_stats":      False,
        "value_loss_scale":         1.0,

        "kl_threshold":       0.05,
        "kl_warmup_fraction": 0.3,
        "debug_kl_stats":     False,

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 10.0),

        "time_limit_bootstrap": True,

        "weight_decay":            1e-4,
        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.3,
        "min_lr_fraction":         0.1,

        "message_dim":        128,    # Important for coordinating on a 9x9 grid with multiple traps
        "num_comm_rounds":    2,      
        "num_heads":          4,      
        "gumbel_temperature": 0.5,    
    },

    "policy": {
        "hidden_sizes":          [256, 256],  
        "unnormalized_log_prob": True,
    },

    "value": {
        "hidden_sizes": [256, 256],           
    },

    "memory": {
        "size": 4096,                         
    },
}
