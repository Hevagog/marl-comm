# fmt: off
"""Scenario 2 — Tight peer visibility + rendezvous goal.

Changes vs nocomm baseline:
  * vision_range 2 → 1  (3×3 local view)
  * enable_rendezvous = True with reward_rendezvous = 20 per step both
    agents stand on the rendezvous cell.
  * Half the goal locations (1) so deliveries cluster.

Hypothesis (analysis.md §5, item 2):
  MAPPO with vision=1 has effectively no peer signal.  Comm methods can
  route a "I'm heading to rendezvous" bit through their attention graph
  (TarMAC / MAGIC regime: Das 2019, Niu 2021).  Expected gap: MAPPO
  collects ≤5 % of rendezvous bonus; CommFormer / MAM ≥40 %.
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "mappo_warehouse_rendezvous_v1",
        "agent_type":       "mappo",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mappo", "warehouse", "rendezvous", "scenario2"],
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
        "num_goal_locations": 1,
        "treatment_duration": 5,
        "comm_noise_prob":    0.0,
        "vision_range":       1,
        "max_cycles":         500,
        "comm_range":         5,

        "no_comm":            True,
        "randomize_layout":   False,

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

        # ---- Scenario-2 knobs ----
        "enable_rendezvous":    True,
        "num_rendezvous":       1,
        "rendezvous_min_agents":2,
        "reward_rendezvous":   20.0,

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
        # vision_range=1 → 3×3×6 = 54 local-grid features (vs 150 at vr=2).
        # Recompute obs dim:
        #   own_state(7) + relative_pos(8) + 54 + (4-1)*6 + 3(hetero)
        #   + 1(battery) + 3(task) = 94.  Shared state stays at 1184.
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 94},
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
        # Rendezvous can pay 20×N occupants — widen positive ceiling.
        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 60.0),
        "time_limit_bootstrap": True,
        "weight_decay":       1e-4,
    },

    "policy": {"hidden_sizes": [256, 128], "unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [256, 128]},
    "memory": {"size": 256},
}
