# fmt: off
"""Scaled topology v3 — base MAGIC.

v2 audit (2026-05-05): magic_v2 reached last10_mean +176 / peak max +543, but
eval shows zero deliveries and 58 expired tasks/episode — return is composed
almost entirely of rescue rewards (repair/charge 12.0 each) plus proximity
shaping. The pick→treat→deliver chain (~12-20 steps for 10+5 reward) is
strictly dominated on per-step yield by rescue (~3 steps for 12 reward),
and the v2 entropy schedule (0.005 → 0.0005) lets the policy commit to
the rescue mode before delivery ever pays out.

env config is *unchanged* (same warehouse settings, for back-compat with
older runs); fixes are agent-side:

  * entropy_loss_scale_start  0.005 → 0.02
    entropy_loss_scale_end    0.0005 → 0.005
      Schulman et al. 2017 §5 / Ahmed et al. 2019 §4.2: maintaining a
      higher entropy floor across the full 2 M is the standard remedy
      when shaped rewards (rescue) dominate the sparse target reward
      (delivery) and the policy converges to a local optimum. End >
      ln(2) ≈ 0.69 keeps multi-action repertoire alive.

  * rewards_shaper clip(-5, 20) → clip(-5, 15)
      Caps the rescue-burst tail (rescue 12 + proximity 0.2/step + charge
      bursts can spike per-step rewards above 15) without zeroing them;
      delivery + urgent (10 + 5 = 15) stays at full magnitude.
      Asymmetric clip is invariance-safe (Ng et al. 1999 §3) — modifies
      only the tail, not the relative reward ordering.

  * kl_warmup_fraction 0.3 → 0.1
      IS-fix is structural now; the long warmup was a v2 defensive
      remnant that delays specialisation past the rescue-trap point.
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "magic_warehouse_scaled_v3",
        "agent_type":       "magic",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["magic", "warehouse", "scaled12", "scenario5", "v3"],
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
        "num_goal_locations":   4,
        "treatment_duration":   5,
        "comm_noise_prob":      0.0,
        "vision_range":         2,
        "max_cycles":           500,
        "comm_range":           8,

        "no_comm":          True,
        "randomize_layout": True,

        "enable_task_deadlines":  True,
        "task_arrival_rate":      0.5,
        "task_deadline_min":      80,
        "task_deadline_max":      200,
        "max_pending_tasks":      20,
        "penalty_task_expired":   -1.0,
        "reward_urgent_delivery": 5.0,

        "enable_heterogeneous":    True,
        "agent_speed_options":     (1, 1, 2),
        "agent_capacity_options":  (1, 2, 1),
        "agent_fragility_options": (1.0, 0.5, 2.0),

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

    "training": {"timesteps": 2_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 2_000, "checkpoint_path": None,
                 "video_dir": "recordings", "fps": 10},

    "magic": {
        "rollouts":        4096,
        "learning_epochs": 8,
        "mini_batches":    8,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  2e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.3,
        "min_lr_fraction":         0.1,

        "state_preprocessor":               RunningStandardScaler,
        "state_preprocessor_kwargs":        {"size": 238},
        "shared_state_preprocessor":        RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": 4704},
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
        "entropy_loss_scale_start": 0.02,    # 0.005 → 0.02
        "entropy_loss_scale_end":   0.005,   # 0.0005 → 0.005 (keep H > ln2)
        "value_loss_scale":         1.0,

        "kl_threshold":       0.05,
        "kl_warmup_fraction": 0.1,           # 0.3 → 0.1

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 15.0),
        "time_limit_bootstrap": True,
        "weight_decay":       1e-4,

        "message_dim":     96,
        "num_comm_rounds": 3,
        "num_heads":       4,

        "gumbel_temperature":                 1.0,
        "gumbel_temperature_end":             0.5,
        "gumbel_temperature_anneal_fraction": 0.7,

        "comm_reg_scale": 0.002,

        "recurrent_type": None,
    },

    "policy": {
        "hidden_sizes":          [256, 256],
        "unnormalized_log_prob": True,
    },
    "value":  {"hidden_sizes": [512, 512]},
    "memory": {"size": 4096},
}
