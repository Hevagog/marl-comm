# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler


CONFIG = {
    "experiment": {
        "name":             "mappo_warehouse_v2",
        "agent_type":       "mappo",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mappo", "warehouse", "v2", "full-features"],
        },
        "write_interval":      "auto",
        "checkpoint_interval": 200_000,
        "store_separately":    False,
    },
    "env": {
        "id":                     "warehouse",
        "num_envs":               1,
        "grid_height":            12,
        "grid_width":             16,
        "num_agents":             4,
        "max_agents":             8,
        "num_shelves":            6,
        "resources_per_shelf":    4,
        "num_treatment_stations": 2,
        "num_goal_locations":     2,
        "treatment_duration":     5,
        "vision_range":           3,
        "max_cycles":             500,

        "comm_noise_prob":        0.08,   # keep noise, but reduce warehouse brittleness
        "comm_range":             5,      # same as MAGIC for fair comparison

        "enable_task_deadlines":  True,
        "task_arrival_rate":      0.6,
        "task_deadline_min":      40,
        "task_deadline_max":      100,
        "max_pending_tasks":      15,
        "penalty_task_expired":   -2.0,
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
        "battery_capacity":           160,
        "battery_drain_per_step":     1,
        "battery_drain_idle":         0,
        "battery_charge_rate":        8,
        "battery_critical_threshold": 25,
        "num_charging_stations":      2,
        "reward_rescue_repair":       12.0,
        "reward_rescue_charge":       12.0,

        "agent_failure_prob":     0.0002,
        "fault_profile": {
            "burst_attrition":     True,
            "burst_prob":          0.0005,
            "correlated_failure":  False,
            "correlation_radius":  1,
            "load_dependent_comm": True,
            "base_packet_loss":    0.05,
            "congestion_factor":   0.05,
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
        "state_preprocessor_kwargs":         {"size": 231},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 816},
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

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 20.0),
        "time_limit_bootstrap": True,
        "weight_decay":            1e-4,
        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.3,
        "min_lr_fraction":         0.1,
    },

    "policy": {
        "hidden_sizes":          [256, 256],
        "unnormalized_log_prob": True,
        "use_memory":            False,
    },
    "value": {
        "hidden_sizes": [256, 256],
        "use_memory":   False,
    },
    "memory": {
        "size": 4096,
    },
}
