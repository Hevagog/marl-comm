# fmt: off
import jax.numpy as jnp

from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "mappo_hopfield_warehouse_nocomm_v1",
        "agent_type":       "mappo_hopfield",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mappo_hopfield", "warehouse"],
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
        "comm_noise_prob":    0.0,  # irrelevant when no_comm=True
        "vision_range":       2,    # 5×5 local view — also the teammate gate in no_comm
        "max_cycles":         500,
        "comm_range":         5,    # used only when no_comm=False

        # KEY: local-only observations
        "no_comm":            True,

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

        "enable_interference_zones": False,  # no radio = no interference model

        "enable_battery":             True,
        "battery_capacity":           200,
        "battery_drain_per_step":     1,
        "battery_drain_idle":         0,
        "battery_charge_rate":        8,
        "battery_critical_threshold": 25,
        "num_charging_stations":      2,
        "reward_rescue_repair":       12.0,
        "reward_rescue_charge":       12.0,

        # Increase failure rate: more rescue events → stronger rescue gradient
        "reward_rescue_proximity":    0.2,
        "agent_failure_prob": 0.001,
        "fault_profile": {
            "burst_attrition":     True,
            "burst_prob":          0.001,
            "correlated_failure":  False,
            "correlation_radius":  1,
            "load_dependent_comm": False,  # no comm = no load-dependent comm noise
            "base_packet_loss":    0.0,
            "congestion_factor":   0.0,
        },
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

    # MAPPO-Hopfield inherits all standard MAPPO hyperparameters.
    # Key differences from baseline mappo_default:
    #   - rollouts=256 (single env, avoid high batch correlation per MAM findings)
    #   - learning_epochs=10, mini_batches=2 (offset shorter rollouts)
    #   - gradient clip 0.5 (same as MAM — Hopfield gate gradient can spike early)
    #   - weight_decay=1e-3 on prototype params (soft-forgetting for non-stationarity)
    "mappo": {
        "rollouts":        256,
        "learning_epochs": 10,
        "mini_batches":    2,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 190},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 1188},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":        0.5,
        "ratio_clip":            0.2,
        "value_clip":            0.2,
        "clip_predicted_values": False,

        "entropy_loss_scale":       0.02,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.05,
        "entropy_loss_scale_end":   0.01,

        "value_loss_scale":  1.0,
        "kl_threshold":      0.05,
        "kl_warmup_fraction": 0.3,

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 20.0),
        "time_limit_bootstrap": True,

        # Higher weight_decay on prototype params implements soft-forgetting.
        # Stale prototypes (no longer reinforced by current data distribution)
        # decay toward zero and are overwritten by gradients from new patterns.
        "weight_decay":            1e-3,
        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.3,
        "min_lr_fraction":         0.1,
    },

    "policy": {
        "hidden_sizes":          [256, 256],
        "unnormalized_log_prob": True,
        # Hopfield prototype bank
        "num_prototypes": 8,      # P=8 coordination modes (empirically sufficient)
        "hopfield_beta":  1.5,    # moderate sharpness; increase to 3.0 for sharper retrieval
        "gate_init":      -3.0,   # sigmoid(-3)≈5% initial contribution
    },

    "value": {
        "hidden_sizes": [256, 256],
        "num_prototypes": 8,
        "hopfield_beta":  1.5,
        "gate_init":      -4.0,   # sigmoid(-4)≈2%, more conservative for critic stability
    },

    "memory": {
        "size": 256,
    },
}
