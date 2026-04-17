# fmt: off
# SyncMixer on continuous_coord with 16 agents.
# Symmetric-attention token encoder scales to n=16 without positional bias (O(n²)=256).
# obs_dim = 6 + (16-1)*4 + 3*4 = 78
# state_dim = 16*4 + 3*4 = 76
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "syncmixer_continuous_coord_16ag_v1",
        "agent_type":       "syncmixer",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["syncmixer", "continuous_coord", "16_agents"],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 200_000,
        "store_separately":    False,
    },

    "env": {
        "id":                "continuous_coord",
        "num_envs":          16,
        "num_agents":        16,
        "max_cycles":        200,
        "max_targets":       3,
        "capture_radius":    0.08,
        "vision_range":      0.4,
        "collision_radius":  0.03,
        "target_arrival_rate":   0.15,
        "target_k_min":      2,
        "target_k_max":      3,
        "target_deadline_min":   30,
        "target_deadline_max":   80,
        "chain_event_prob":  0.2,
        "max_speed":         0.05,
    },

    "training": {
        "timesteps": 5_000_000,
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

    "syncmixer": {
        "rollouts":        4096,
        "learning_epochs": 5,
        "mini_batches":    8,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  2.5e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 78},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 76},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        "entropy_loss_scale":       0.03,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.05,
        "entropy_loss_scale_end":   0.01,
        "value_loss_scale":         1.0,

        "kl_threshold":       0.03,
        "kl_warmup_fraction": 0.3,

        "rewards_shaper":       lambda rewards, *args: jnp.clip(rewards, -5.0, 15.0),
        "time_limit_bootstrap": True,

        "weight_decay":            1e-4,
        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.3,
        "min_lr_fraction":         0.1,

        # ---- SyncMixer architecture ----
        # n_block=3, num_heads=8 give more representational capacity for 16-agent sequences
        "d_model":          128,
        "n_block":          3,
        "num_heads":        8,
        "hidden_mult":      4,
        "num_pool_queries": 4,
        "pool_dim":         32,
        "pool_beta":        2.0,
    },

    "policy": {
        "unnormalized_log_prob": True,
    },

    "value": {
        "hidden_sizes": [256, 128],
    },

    "memory": {
        "size": 4096,
    },
}
