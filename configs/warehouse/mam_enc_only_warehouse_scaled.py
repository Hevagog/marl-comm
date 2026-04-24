# fmt: off
"""Scenario 5 — Scaled topology (12 agents, 24×32 grid), MAM encoder-only ablation.

Same env as mappo_warehouse_scaled but using the BiMamba encoder-only
policy (MAMEncOnly).

Hypothesis (analysis.md §5 item 5):
  MAPPO's monolithic critic input grows O(N²) with agent count; BiMamba's
  SSM is O(N) in the agent sequence.  MAT/MAM papers report benefits emerging
  at N≥8.  Without the AR decoder, enc-only still processes all 12 agents in
  a single BiMamba forward pass, matching the O(N) scaling advantage.
  Expected: enc-only sample-efficiency stays closer to its 4-agent trajectory
  shape; MAPPO degrades more steeply with N.

Scaling adjustments:
  - d_state=64 (∝ sqrt(12) × paper-base; retains SSM memory across 12 agents)
  - delta_rank=32 (scaled proportionally)
  - n_embd=128 kept (obs_dim=238 → 128 is ~2:1 compression)
  - memory_size and rollouts doubled to account for longer episode setup time.
  - num_envs=4 (memory budget; same as MAPPO scaled config).
  - Larger value network [256,256] for the wider shared state (4704 dims).
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "mam_enc_only_scaled_v1",
        "agent_type":       "mam_enc_only",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mam_enc_only", "warehouse", "scaled12", "scenario5"],
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

    "mam": {
        "rollouts":        512,   # larger N → more rollout steps per update
        "learning_epochs": 8,
        "mini_batches":    4,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  1.5e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay":          True,
        "lr_decay_start_fraction":  0.3,
        "min_lr_fraction":          0.1,

        # 24×32 grid, 12 agents: obs_dim=238, shared_state=4704
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 238},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 4704},
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

        # 12-agent sizing: d_state ∝ sqrt(12) ≈ 3.5 × paper 4-agent base → 64
        "n_embd":      128,
        "n_block":     1,
        "d_state":     64,
        "d_conv":      4,
        "delta_rank":  32,
    },

    "policy": {"unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [256, 256]},  # wider for 4704-dim shared state
    "memory": {"size": 512},
}
