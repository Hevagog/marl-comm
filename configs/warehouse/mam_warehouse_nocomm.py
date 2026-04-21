# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler
CONFIG = {
    "experiment": {
        "name":             "mam_warehouse_nocomm_v1",
        "agent_type":       "mam",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mappo", "warehouse", "vrandomize_layout", "bigger", "scaled_ssm"],
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
        "timesteps":       1_000,
        "checkpoint_path": None,
        "video_dir":       "recordings",
        "fps":             4,
    },

    "mam": {
        "rollouts":        256,
        "learning_epochs": 10,
        "mini_batches":    2,

        "discount_factor": 0.99,   
        "lambda":          0.95,  # longer GAE horizon for multi-step lifecycle (Pick→Treat→Deliver spans 50–200 steps)

        "learning_rate":                  3e-4,   # scaled from 1.5e-4: match MAPPO baseline (analysis: LR too conservative)
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay":          True,
        "lr_decay_start_fraction":  0.03,   # start annealing at 3% (600k steps) not 5% (1M)
        "min_lr_fraction":          0.1,

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 262},   # 24x32 grid, 16 agents, vision_range=2 → verified via env
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 4736},  # 24*32*6 + 16*8 + 16 one-hot → verified via env
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.2,    # PPO clip ε
        "value_clip":             0.2,
        "clip_predicted_values":  False,  # match MAPPO and smoke config; True caused over-conservative critic updates

        "entropy_loss_scale":       0.01,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.05,   # reverted to 0.05: strong gradient (P1 fix) now overcomes entropy bonus
        "entropy_loss_scale_end":   0.01,
        "value_loss_scale":   1.0,   # scaled from 0.5: full critic gradient for faster GAE convergence

        "kl_threshold": 0,

        "rewards_shaper":       lambda rewards, *args: jnp.clip(rewards, -5.0, 20.0),  # clip extremes; None caused large negative advantages destabilizing training
        "time_limit_bootstrap": True,

        # (Daniel et al. 2024, Table 4) — scaled for 16 agents (paper tested 2–4 agents)
        # With 16-step AR decoder: d_state=32 insufficient to retain conditioning
        # across 15 prior agents; products of 15 Ā terms decay exponentially.
        # Rule of thumb: scale d_state ∝ sqrt(n_agents) from paper's 4-agent base.
        "n_embd":      128,    # embedding dimension; keep (obs_dim=262 → 128 is ~2:1 compression, reasonable)
        "n_block":     1,      # one Encoder + one Decoder block (paper default)

        "d_state":     64,     # scaled from 32: 2× for 4× more agents; maintains SSM memory across 16-step AR chain
        "d_conv":      4,      # 1D convolution kernel size (paper Table 4; sequence-length-independent)
        "delta_rank":  64,     # scaled from 16: match d_state order of magnitude for richer input-dependent transitions
    },

    "policy": {
        "unnormalized_log_prob": True,
    },

    "value": {
        "hidden_sizes": [256, 128],
    },

    "memory": {
        "size": 256,
    },
}
