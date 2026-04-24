# fmt: off
"""Scenario 3 — Re-enable comms, with interference & load-dependent loss.

Changes vs nocomm baseline:
  * no_comm = False  (peers' internal state visible over radio)
  * enable_interference_zones = True  (hot-spots near treatment stations)
  * load_dependent_comm = True with base_packet_loss=0.10,
    congestion_factor=0.05 — message loss grows with local agent density.

Hypothesis (analysis.md §5, item 3):
  With noisy peer state, MAPPO sees corrupted observations it cannot
  denoise.  CommFormer's graph attention and MAM's selective SSM can
  *average* / gate noisy peer channels (IC3Net regime, Singh 2019).
  Expected: MAPPO reward stays ~baseline, comm methods +20–40 reward.
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "mappo_warehouse_packet_loss_v1",
        "agent_type":       "mappo",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mappo", "warehouse", "noisy_comm", "scenario3"],
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
        "comm_noise_prob":    0.05,  # baseline radio noise
        "vision_range":       2,
        "max_cycles":         500,
        "comm_range":         5,

        "no_comm":            False,  # ← radio ON
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

        # ---- Scenario-3 knobs ----
        "enable_interference_zones":  True,
        "interference_base":          0.05,
        "interference_treatment_boost": 0.4,
        "interference_radius":        2,

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
            "load_dependent_comm": True,
            "base_packet_loss":    0.10,
            "congestion_factor":   0.05,
        },
    },

    "training": {"timesteps": 10_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 2_000, "checkpoint_path": None,
                 "video_dir": "recordings", "fps": 10},

    "mappo": {
        "rollouts":        256,
        "learning_epochs": 10,
        "mini_batches":    2,
        "discount_factor": 0.99,
        "lambda":          0.95,
        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.3,
        "min_lr_fraction":         0.1,
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 190},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 1184},
        "value_preprocessor":                RunningStandardScaler,
        "value_preprocessor_kwargs":         {"size": 1},
        "random_timesteps":  0,
        "learning_starts":   0,
        "grad_norm_clip":    0.5,
        "ratio_clip":        0.2,
        "value_clip":        0.2,
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
        "weight_decay":       1e-4,
    },

    "policy": {"hidden_sizes": [256, 128], "unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [256, 128]},
    "memory": {"size": 256},
}
