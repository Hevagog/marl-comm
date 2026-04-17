# fmt: off
# MAPPO on typed continuous_coord.
#
# 4 agents, 2 types (2×type-0 "scouts", 2×type-1 "capturers").
# Each target requires a random composition of the two types (multinomial
# over k sampled from [k_min, k_max]).  Wrong-type agents in the capture
# zone receive per-step penalties; wrong-composition attempts are penalised
# on the full group.
#
# Observation dimensions (n=4, T=2, mt=3):
#   obs_dim   = 6 + T + (n-1)*(4+T) + mt*(4+T)
#             = 6 + 2 + 3*6 + 3*6 = 44
#   state_dim = n*(4+T) + mt*(4+T)
#             = 4*6 + 3*6 = 42
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "mappo_continuous_coord_typed_v1",
        "agent_type":       "mappo",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mappo", "continuous_coord", "typed"],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 200_000,
        "store_separately":    False,
    },

    "env": {
        "id":                    "continuous_coord",
        "num_envs":              16,
        "num_agents":            4,
        "max_cycles":            200,
        "max_targets":           3,
        "capture_radius":        0.08,
        "vision_range":          0.4,
        "collision_radius":      0.03,
        "target_arrival_rate":   0.15,
        "target_k_min":          2,
        "target_k_max":          3,
        "target_deadline_min":   30,
        "target_deadline_max":   80,
        "chain_event_prob":      0.2,
        "max_speed":             0.05,
        # Typed-agent settings
        "num_agent_types":       2,
        "agent_types":           [0, 0, 1, 1],   # 2 scouts (0), 2 capturers (1)
        "penalty_wrong_type":    -0.5,
        "penalty_wrong_composition": -1.0,
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

    "mappo": {
        "rollouts":        4096,
        "learning_epochs": 8,
        "mini_batches":    4,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 44},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 42},
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

        # Wider clip: typed env has extra per-step type penalties (~-0.5) on top
        # of existing deadline/collision penalties.
        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -8.0, 15.0),
        "time_limit_bootstrap": True,

        "weight_decay":            1e-4,
        "linear_lr_decay":         True,
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
