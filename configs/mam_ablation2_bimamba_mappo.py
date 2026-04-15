# fmt: off
# Ablation 2: MAPPO + BiMamba encoding (16-agent large warehouse)
# Same encoder-only architecture as Ablation 1 but at 16-agent scale.
# Tests whether bidirectional pooling alone (without AR decoding) helps
# at the scale where full MAM fails (IS ratio = 1.000).
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler  # noqa: E402

CONFIG = {
    "experiment": {
        "name":             "mam_enc_only_ablation2_bimamba_mappo_16ag",
        "agent_type":       "mam_enc_only",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mam_enc_only", "warehouse", "ablation2", "16agents", "bimamba_mappo"],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 200000,
        "store_separately":    False,
    },

    "env": {
        "id":            "warehouse",
        "num_envs":      8,
        "grid_height":   24,
        "grid_width":    32,
        "num_agents":    16,
        "max_agents":    16,
        "num_shelves":   12,
        "resources_per_shelf": 10,
        "num_treatment_stations": 4,
        "num_goal_locations": 2,
        "treatment_duration": 5,
        "comm_noise_prob":        0.0,
        "vision_range":           2,
        "max_cycles":             500,
        "comm_range":             5,

        "randomize_layout": True,

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

        "enable_interference_zones":    True,
        "interference_base":            0.05,
        "interference_treatment_boost": 0.15,
        "interference_radius":          2,

        "enable_battery":             True,
        "battery_capacity":           200,
        "battery_drain_per_step":     1,
        "battery_drain_idle":         0,
        "battery_charge_rate":        8,
        "battery_critical_threshold": 25,
        "num_charging_stations":      2,
        "reward_rescue_repair":       12.0,
        "reward_rescue_charge":       12.0,

        "agent_failure_prob": 0.0002,
        "fault_profile": {
            "burst_attrition":     True,
            "burst_prob":          0.0002,
            "correlated_failure":  False,
            "correlation_radius":  1,
            "load_dependent_comm": True,
            "base_packet_loss":    0.02,
            "congestion_factor":   0.05,
        },
    },

    "training": {
        "timesteps": 20_000_000,
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

    "mam": {
        "rollouts":        256,
        "learning_epochs": 10,
        "mini_batches":    2,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay":          True,
        "lr_decay_start_fraction":  0.03,
        "min_lr_fraction":          0.1,

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 190},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 1184},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        "entropy_loss_scale":       0.01,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.05,
        "entropy_loss_scale_end":   0.01,
        "value_loss_scale":   1.0,

        "kl_threshold": 0,

        "rewards_shaper":       lambda rewards, *args: jnp.clip(rewards, -5.0, 20.0),
        "time_limit_bootstrap": True,

        "n_embd":      128,
        "n_block":     1,
        # Encoder-only: no AR chain → d_state can stay at 32 (no exponential decay problem)
        "d_state":     32,
        "d_conv":      4,
        "delta_rank":  16,
    },

    "policy": {
        "unnormalized_log_prob": True,
    },

    "value": {
        "hidden_sizes": [256, 128],
    },

    "memory": {
        "size": 256,
    },
}
