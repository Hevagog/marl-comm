# fmt: off
"""Scenario 5 — Scaled topology (12 agents, 24×32 grid).

Changes vs nocomm baseline:
  * grid_height/width 24×32, num_agents=12, max_agents=12
  * num_shelves = 18, num_treatment = 4, num_goals = 4
  * randomize_layout = True
  * num_envs reduced 8 → 4 to keep memory budget stable.

Hypothesis (analysis.md §5, item 5):
  MAPPO's monolithic critic input grows O(N²) with agent count, while
  CommFormer / MAM token-attention is O(N).  Wen 2022 (MAT) reports the
  cross-over emerging for SMAC maps with ≥8 units; Daniel 2024 (MAM)
  shows similar 12-agent results.  Expected: MAPPO sample-efficiency
  drops dramatically here, comm methods stay close to their 4-agent
  trajectory shape.
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "mappo_warehouse_scaled_v1",
        "agent_type":       "mappo",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mappo", "warehouse", "scaled12", "scenario5"],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 200_000,
        "store_separately":    False,
    },

    "env": {
        "id":            "warehouse",
        "num_envs":      4,
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

        "no_comm":            True,
        "randomize_layout":   True,

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

        "enable_battery":             True,
        "battery_capacity":           250,
        "battery_drain_per_step":     1,
        "battery_drain_idle":         0,
        "battery_charge_rate":        8,
        "battery_critical_threshold": 30,
        "num_charging_stations":      4,
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

    "training": {"timesteps": 20_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 2_000, "checkpoint_path": None,
                 "video_dir": "recordings", "fps": 10},

    "mappo": {
        "rollouts":        512,   # larger N → more rollout per update
        "learning_epochs": 8,
        "mini_batches":    4,
        "discount_factor": 0.99,
        "lambda":          0.95,
        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.3,
        "min_lr_fraction":         0.1,
        # Recompute obs / state dim:
        #   own(7) + relpos(8) + (5×5×6=150) + (12-1)×6=66 + 3 + 1 + 3 = 238
        #   shared = 24*32*6 + 12*8 = 4704
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 238},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 4704},
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

    "policy": {"hidden_sizes": [256, 256], "unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [256, 256]},
    "memory": {"size": 512},
}
