# fmt: off
"""Scaled topology v2 — MAGIC + HopfieldSelfContext.

Changes vs magic_hopfield_warehouse_scaled_v1
(memory/warehouse_scaled_analysis_2026_05_04.md §5.3):

  * Code-level: HopfieldSelfContext gate_init −2.0 → 0.0 (sigmoid 0.12 →
    0.5). The −2 default was the "gate-starvation pathology" flagged in
    memory/hopfield_integration_strategy_2026_04_30.md §9 — prototypes
    contributed only 12% magnitude until they specialised, but they
    cannot specialise while suppressed. Module default already updated.

  * hopfield_beta  1.0 → 2.0  — Ramsauer et al. 2021 §3.1: at K=8 the
    threshold for winner-take-most retrieval is `β·||p||² > ln K ≈ 2.1`.
    β=2 is the smallest value that breaks uniform attention with unit
    LeCun-normal prototypes.

  * entropy schedule lowered to CommFormer levels (see analysis §5.2).

  * kl_threshold restored to 0.05 (was 0.02 defensively).
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "magic_hopfield_scaled_v2",
        "agent_type":       "magic_hopfield",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["magic_hopfield", "warehouse", "scaled12", "scenario5", "v2"],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 1_000_000,
        "store_separately":    False,
    },

    "env": {
        "id":            "warehouse",
        "num_envs":      1,
        "grid_height":   24,
        "grid_width":    32,
        "num_agents":    12,
        "max_agents":    12,
        "num_shelves":   18,
        "resources_per_shelf": 4,
        "num_treatment_stations": 4,
        "num_goal_locations": 4,
        "treatment_duration": 5,
        "comm_noise_prob":    0.0,
        "vision_range":       2,
        "max_cycles":         500,
        "comm_range":         8,

        "no_comm":          True,
        "randomize_layout": True,

        "enable_task_deadlines":  True,
        "task_arrival_rate":      0.5,
        "task_deadline_min":      80,
        "task_deadline_max":      200,
        "max_pending_tasks":      20,
        "penalty_task_expired":   -1.0,
        "reward_urgent_delivery": 5.0,

        "enable_heterogeneous":   True,
        "agent_speed_options":    (1, 1, 2),
        "agent_capacity_options": (1, 2, 1),
        "agent_fragility_options":(1.0, 0.5, 2.0),

        "enable_interference_zones":    True,
        "interference_base":            0.05,
        "interference_treatment_boost": 0.35,
        "interference_radius":          2,

        "enable_battery":             True,
        "battery_capacity":           250,
        "battery_drain_per_step":     1,
        "battery_drain_idle":         0,
        "battery_charge_rate":        8,
        "battery_critical_threshold": 30,
        "num_charging_stations":      4,
        "reward_rescue_repair":       12.0,
        "reward_rescue_charge":       12.0,

        "agent_failure_prob": 0.001,

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

    "training": {"timesteps": 2_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 2_000, "checkpoint_path": None,
                 "video_dir": "recordings", "fps": 10},

    "magic_hopfield": {
        "rollouts":        4096,
        "learning_epochs": 8,
        "mini_batches":    8,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  2e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 238},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 4704},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        "entropy_loss_scale":       0.005,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.005,
        "entropy_loss_scale_end":   0.0005,
        "value_loss_scale":         1.0,

        "kl_threshold":       0.05,
        "kl_warmup_fraction": 0.3,

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 20.0),
        "time_limit_bootstrap": True,

        "weight_decay":            1e-4,
        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.3,
        "min_lr_fraction":         0.1,

        "message_dim":        64,
        "num_comm_rounds":    3,
        "num_heads":          4,
        "gumbel_temperature":                 1.0,
        "gumbel_temperature_end":             0.5,
        "gumbel_temperature_anneal_fraction": 0.7,

        "hopfield_num_prototypes": 8,
        "hopfield_beta":           2.0,   # 1.0 → 2.0; Ramsauer §3.1 winner-take-most threshold

        "comm_reg_scale": 0.001,
    },

    "policy": {
        "hidden_sizes":          [256, 256],
        "unnormalized_log_prob": True,
    },

    "value":  {"hidden_sizes": [512, 512]},
    "memory": {"size": 4096},
}
