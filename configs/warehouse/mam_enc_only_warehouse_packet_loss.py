# fmt: off
"""Scenario 3 — Noisy communication channels, MAM encoder-only ablation.

Same env as mappo_warehouse_packet_loss but using the BiMamba encoder-only
policy (MAMEncOnly): no_comm=False with interference zones + load-dependent loss.

Hypothesis (analysis.md §5 item 3):
  With noisy peer state (base_packet_loss=0.10, interference near treatment
  stations), MAPPO's actor receives corrupted peer observations it cannot
  denoise.  The BiMamba's selective SSM can gate noisy channels — if agent j's
  observation is corrupt, Mamba can learn to down-weight that position's
  contribution via Δ-gating.  Expected: enc-only +15–30 reward above MAPPO.

Note: no_comm=False means obs_dim includes peer internals (battery, phase,
  active_flag) → obs_dim grows.  The shared state preprocessor size is unchanged
  (shared_state is always full), but the local obs preprocessor size must match
  the comm=True observation shape.  At no_comm=False with vision_range=2 and 4
  agents the obs_dim is 262 (see mam_warehouse.py: "obs_dim=262").
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "mam_enc_only_packet_loss_v1",
        "agent_type":       "mam_enc_only",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mam_enc_only", "warehouse", "noisy_comm", "scenario3"],
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
        "comm_noise_prob":    0.05,   # baseline radio noise
        "vision_range":       2,
        "max_cycles":         500,
        "comm_range":         5,

        "no_comm":            False,  # radio ON — full peer state visible (noisily)
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

        # no_comm=False: obs includes peer internals → obs_dim=262 (mam_warehouse.py)
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 262},
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

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 20.0),
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
