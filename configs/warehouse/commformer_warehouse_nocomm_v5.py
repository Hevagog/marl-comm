# fmt: off
"""CommFormer warehouse no-comm — fixed trust region (v5).

Diff vs v4 (analysis.md §4 Bug 2):
  * ratio_clip   0.015 → 0.05  (paper-faithful, MuJoCo HP from MAT/CommFormer
                                Table 8 — Wen et al. 2022, Hu et al. 2024).
                                v4's 0.015 was over-aggressive: surrogate floor
                                triggered constantly while the *unclipped*
                                ratio drifted to 1.5–1.7, producing the slow-
                                ramp + grad spikes documented in analysis.md.
  * kl_threshold 0.05 → 0.02   tighter early-stop pairs naturally with the
                                tighter ratio clip and matches the CC blind
                                config (commformer_continuous_blind.py:125).
  * comm_reg_scale 0.001 unchanged — keeps adjacency density centered.
  * Architecture (hidden=128, blocks=2) kept identical to v4 so the only
    causal change visible in wandb is the trust-region settings.

Expected: steps-to-R≥180 drops from ~625 k (v4) to ≲200 k; grad_global_norm
plateaus near 0.5 instead of 1.0.
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "commformer_warehouse_nocomm_v5",
        "agent_type":       "commformer",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["commformer", "warehouse", "no_comm", "v5_ratio_fix"],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 200_000,
        "store_separately":    False,
    },

    "env": {
        "id":            "warehouse",
        "num_envs":      8,
        "grid_height":   12,
        "grid_width":    16,
        "num_agents":    4,
        "max_agents":    4,
        "num_shelves":   6,
        "resources_per_shelf": 4,
        "num_treatment_stations": 2,
        "num_goal_locations": 2,
        "treatment_duration": 5,
        "comm_noise_prob":    0.0,
        "vision_range":       2,
        "max_cycles":         500,
        "comm_range":         5,

        "no_comm":            True,
        "randomize_layout":   True,

        "enable_task_deadlines":  True,
        "task_arrival_rate":      0.2,
        "task_deadline_min":      80,
        "task_deadline_max":      200,
        "max_pending_tasks":      8,
        "penalty_task_expired":   -1.0,
        "reward_urgent_delivery": 5.0,

        "enable_heterogeneous":   True,
        "agent_speed_options":    (1, 1, 2),
        "agent_capacity_options": (1, 2, 1),
        "agent_fragility_options":(1.0, 0.5, 2.0),

        "enable_battery":             True,
        "battery_capacity":           200,
        "battery_drain_per_step":     1,
        "battery_drain_idle":         0,
        "battery_charge_rate":        8,
        "battery_critical_threshold": 25,
        "num_charging_stations":      2,
        "reward_rescue_repair":       12.0,
        "reward_rescue_charge":       12.0,
        "reward_rescue_proximity":    0.2,
        "agent_failure_prob":         0.001,

        "fault_profile": {
            "burst_attrition":     True,
            "burst_prob":          0.001,
            "correlated_failure":  False,
            "correlation_radius":  1,
            "load_dependent_comm": False,
            "base_packet_loss":    0.0,
            "congestion_factor":   0.0,
        },
    },

    "training": {"timesteps": 10_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 2_000, "checkpoint_path": None,
                 "video_dir": "recordings", "fps": 10},

    "commformer": {
        "rollouts":        1024,
        "learning_epochs": 3,
        "mini_batches":    4,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.2,
        "min_lr_fraction":         0.1,

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 190},
        "update_state_preprocessor_in_update": False,
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 1188},
        "update_shared_state_preprocessor_in_update": False,
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        # ↓↓↓ Bug fix from analysis.md §4 ↓↓↓
        "ratio_clip":             0.05,   # was 0.015 (over-aggressive)
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        "entropy_loss_scale":       0.005,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.005,
        "entropy_loss_scale_end":   0.0005,
        "debug_entropy_stats":      False,

        "value_loss_scale":   1.0,
        # Tighter KL stop pairs with the new ratio_clip — both bound update size.
        "kl_threshold":       0.02,   # was 0.05
        "kl_warmup_fraction": 0.0,
        "ratio_max_threshold":  2.0,

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 20.0),
        "time_limit_bootstrap": True,
        "weight_decay":       1e-4,
        "comm_reg_scale":     0.001,

        "hidden_dim":  128,
        "num_blocks":  2,
        "num_heads":   1,
        "head_dim":    64,
        "mlp_dim":     256,
        "sparsity":    0.5,
    },

    "policy": {"unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [128, 64]},
    "memory": {"size": 1024},
}
