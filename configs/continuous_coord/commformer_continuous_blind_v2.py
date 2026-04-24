# fmt: off
"""CommFormer continuous_coord blind — analog of warehouse v5 trust-region fix.

Diff vs commformer_continuous_blind.py (v1):
  * ratio_clip       0.05 → 0.05  (kept — already paper-faithful)
  * kl_threshold     0.05 → 0.02  (tighter early-stop, parity with warehouse v5)
  * lr_decay_start_fraction 0.2 → 0.2  (kept)
  * Adds `ratio_max_threshold: 2.0` safety valve so extreme ratios trip the
    same circuit-breaker the warehouse v5 uses.

Purpose: provide a head-to-head comparison config so a single sweep produces
matched ablations on both warehouse_grid and continuous_coord with identical
optimizer hygiene. See analysis.md §4 (Bug 2) and §5 for context.
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "commformer_continuous_coord_blind_v2",
        "agent_type":       "commformer",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["commformer", "continuous_coord", "blind", "v2_ratio_fix"],
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
        "vision_range":          0.1,
        "target_vision_range":   0.2,
        "collision_radius":      0.03,
        "target_arrival_rate":   0.15,
        "target_k_min":          2,
        "target_k_max":          3,
        "target_deadline_min":   40,
        "target_deadline_max":   100,
        "chain_event_prob":      0.1,
        "max_speed":             0.05,
        "num_agent_types":       2,
        "agent_types":           [0, 0, 1, 1],
        "penalty_wrong_type":    -0.5,
        "penalty_wrong_composition": -1.0,
    },

    "training": {"timesteps": 10_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 2_000, "checkpoint_path": None,
                 "video_dir": "recordings", "fps": 10},

    "commformer": {
        "rollouts":        4096,
        "learning_epochs": 10,
        "mini_batches":    4,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  5e-5,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.2,
        "min_lr_fraction":         0.1,

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 44},
        "update_state_preprocessor_in_update": False,
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 46},
        "update_shared_state_preprocessor_in_update": False,
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.05,    # paper-faithful
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        "entropy_loss_scale":       0.02,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.02,
        "entropy_loss_scale_end":   0.002,
        "debug_entropy_stats":      False,

        "value_loss_scale":   1.0,

        # Trust-region tightening (parity with warehouse v5).
        "kl_threshold":       0.02,    # was 0.05
        "kl_warmup_fraction": 0.0,
        "ratio_max_threshold": 2.0,    # NEW safety valve

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -8.0, 15.0),

        "time_limit_bootstrap": True,
        "weight_decay":       1e-4,
        "comm_reg_scale":     0.001,

        "hidden_dim":  64,
        "num_blocks":  1,
        "num_heads":   1,
        "head_dim":    64,
        "mlp_dim":     128,
        "sparsity":    0.5,
    },

    "policy": {"unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [128, 64]},
    "memory": {"size": 4096},
}
