# fmt: off
"""Scenario 4 — Heterogeneous hand-off pressure, MAM encoder-only ablation.

Same env as mappo_warehouse_handoff but using the BiMamba encoder-only
policy (MAMEncOnly).

Hypothesis (analysis.md §5 item 4):
  Alternating fast (speed=3, cap=1) and slow (speed=1, cap=3) agents must
  coordinate hand-offs: fast agents sprint between shelf and station, slow
  agents accumulate and batch-deliver.  Knowing peer's phase (carry_status)
  is critical.  no_comm hides this from MAPPO; BiMamba encoder's cross-agent
  context encodes it from the jointly-processed obs tokens.
  Expected: enc-only opens a 15–25 % reward gap over MAPPO (Overcooked-style
  hand-off, Carroll 2019).
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "mam_enc_only_handoff_v1",
        "agent_type":       "mam_enc_only",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mam_enc_only", "warehouse", "handoff", "scenario4"],
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
        "num_shelves":   8,
        "resources_per_shelf": 4,
        "num_treatment_stations": 2,
        "num_goal_locations": 1,
        "treatment_duration": 5,
        "comm_noise_prob":    0.0,
        "vision_range":       2,
        "max_cycles":         500,
        "comm_range":         5,

        "no_comm":            True,
        "randomize_layout":   True,

        "enable_task_deadlines":  True,
        "task_arrival_rate":      0.3,
        "task_deadline_min":      80,
        "task_deadline_max":      200,
        "max_pending_tasks":      10,
        "penalty_task_expired":   -1.0,
        "reward_urgent_delivery": 5.0,

        # ---- Scenario-4 knobs ----
        "enable_heterogeneous":   True,
        "agent_speed_options":    (1, 3, 1, 3),
        "agent_capacity_options": (3, 1, 3, 1),
        "agent_fragility_options":(0.5, 2.0, 0.5, 2.0),
        "step_penalty":           -0.005,

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

    "mam": {
        "rollouts":        256,
        "learning_epochs": 10,
        "mini_batches":    2,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  1.5e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay":          True,
        "lr_decay_start_fraction":  0.3,
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

        "kl_threshold":       0.05,
        "kl_warmup_fraction": 0.3,
        "weight_decay":       1e-4,

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 30.0),
        "time_limit_bootstrap": True,

        # 4-agent sizing
        "n_embd":      128,
        "n_block":     1,
        "d_state":     32,
        "d_conv":      4,
        "delta_rank":  16,
    },

    "policy": {"unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [256, 128]},
    "memory": {"size": 256},
}
