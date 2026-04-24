# fmt: off
"""Scenario 2 — Tight peer visibility + rendezvous goal, MAM encoder-only ablation.

Same env as mappo_warehouse_rendezvous but using the BiMamba encoder-only
policy (MAMEncOnly).

Hypothesis (analysis.md §5 item 2, §7.4):
  MAPPO at vision_range=1 is effectively blind to peers; its critic has global
  state but the actor cannot route peer information.  The BiMamba encoder in
  enc-only sees the concatenated team observations and can learn to attend to
  peer positions, broadcasting a "heading to rendezvous" signal through the
  SSM hidden state.
  Expected: enc-only collects 30–60 % of rendezvous bonus; MAPPO collects ≤5 %.
  This is the primary diagnostic scenario for communication value.

Note: obs_dim shrinks with vision_range=1 (3×3 = 9 cells × 6 features = 54
  vision features vs 150 at vr=2). Recomputed obs_dim ≈ 94 (see MAPPO config).
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "mam_enc_only_rendezvous_v1",
        "agent_type":       "mam_enc_only",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mam_enc_only", "warehouse", "rendezvous", "scenario2"],
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
        "vision_range":       1,    # 3×3 local view — peers visible only if adjacent
        "max_cycles":         500,
        "comm_range":         5,

        "no_comm":            True,
        "randomize_layout":   False,  # fixed layout for rendezvous cell reproducibility

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

        # vision_range=1 → obs_dim ≈ 94 (see mappo_warehouse_rendezvous.py comment)
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 94},
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

        # Rendezvous pays 20×N occupants; raise ceiling.
        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 60.0),
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
