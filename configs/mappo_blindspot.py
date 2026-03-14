# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler  # noqa: E402

CONFIG = {
    "experiment": {
        "name":             "mappo_blindspot_v1",
        "agent_type":       "mappo",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mappo", "blindspot"],
        },
        "write_interval":      "auto",
        "checkpoint_interval": "auto",
        "store_separately":    False,
    },

    "env": {
        "id":            "blindspot",
        "grid_size":     9,
        "max_cycles":    100,
        "num_traps":     5,
        "use_communication": False,  # MAPPO baseline without explicit communication
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
        "rollouts":        4096,      # number of rollouts before updating
        "learning_epochs": 8,       # learning epochs per update
        "mini_batches":    4,       # mini-batches per learning epoch

        "discount_factor": 0.99,    # gamma
        "lambda":          0.95,    # TD(lambda) / GAE lambda

        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 25}, # 2 + 2 + 2 + 2 + 15 + 1 + 1 = 25
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 50}, # 25 * 2 = 50
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        "entropy_loss_scale": 0.02,
        "entropy_annealing": True,
        "entropy_loss_scale_start": 0.05,
        "entropy_loss_scale_end": 0.01,
        "debug_entropy_stats": False,
        "value_loss_scale":   1.0,

        "kl_threshold": 0.05,
        "kl_warmup_fraction": 0.3,
        "debug_kl_stats": False,

        # Clipping reward appropriately
        "rewards_shaper":       lambda rewards, *args: jnp.clip(rewards, -5.0, 10.0),
        "time_limit_bootstrap": True,

        "weight_decay":       1e-4,
        "linear_lr_decay":    True,
        "lr_decay_start_fraction": 0.3,
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
        "size": 4096,
    },
}
