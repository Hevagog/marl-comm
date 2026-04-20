# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

# MAT continuous coordination v1
#
# Architecture: Multi-Agent Transformer (Wen et al. 2022, NeurIPS).
# Typed-agent mode: 4 agents, 2 types ([0,0,1,1]).
# Targets require a specific type composition; wrong-type penalties apply.
#
# Hyperparameter rationale vs. CommFormer continuous_coord v3
# -----------------------------------------------------------
# - learning_epochs 5 (vs. CF 3): same reasoning as warehouse v1.
# - entropy_loss_scale_start 0.01 (vs. CF 0.005): continuous_coord has
#   denser rewards than warehouse, so slightly higher entropy is safe
#   and beneficial for typed-role specialization in early training.
# - No comm_reg_scale (no α).
# - mlp_dim 512 (vs. CF 256): standard 2× FFN ratio.
CONFIG = {
    "experiment": {
        "name":             "mat_continuous_coord_v1",
        "agent_type":       "mat",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mat", "continuous_coord", "typed"],
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
        "num_agent_types":       2,
        "agent_types":           [0, 0, 1, 1],
        "penalty_wrong_type":    -0.5,
        "penalty_wrong_composition": -1.0,
    },

    "training": {
        "timesteps": 10_000_000,
        "seed":      42,
    },

    "eval": {
        "timesteps":       5_000,
        "checkpoint_path": None,
    },

    "record": {
        "timesteps":       2_000,
        "checkpoint_path": None,
        "video_dir":       "recordings",
        "fps":             10,
    },

    # ---- MAT-specific PPO parameters ----
    "mat": {
        "rollouts":        4096,
        "learning_epochs": 5,
        "mini_batches":    4,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.2,
        "min_lr_fraction":         0.1,

        # obs_dim = 6 + T + (n-1)*(4+T) + mt*(4+T) = 6+2+3*6+3*6 = 44 (n=4, T=2, mt=3)
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 44},
        # state_dim = n*(4+T) + mt*(4+T) = 4*6+3*6 = 42; +4 one-hot = 46
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 46},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        # Higher entropy start for typed-role specialization: early exploration
        # is critical to discover type-correct capture compositions.
        "entropy_loss_scale":       0.01,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.01,
        "entropy_loss_scale_end":   0.001,
        "debug_entropy_stats":      False,

        "value_loss_scale":   1.0,

        "kl_threshold":       0.05,
        "kl_warmup_fraction": 0.0,

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -8.0, 15.0),

        "time_limit_bootstrap": True,

        "weight_decay":       1e-4,

        # ---- MAT architecture hyperparameters ----
        "hidden_dim":  256,
        "num_blocks":  2,
        "num_heads":   4,
        "head_dim":    64,
        "mlp_dim":     512,
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
