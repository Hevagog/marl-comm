# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "commformer_warehouse_nocomm_v4",
        "agent_type":       "commformer",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["commformer", "warehouse", "no_comm"],
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
        "timesteps":       2_000,
        "checkpoint_path": None,
        "video_dir":       "recordings",
        "fps":             10,
    },

    # v3: Aligned with commformer_warehouse v13 training HPs.
    # v2 used rollouts=256, epochs=10, mini_batches=2, entropy=0.05,
    # kl_warmup=0.3, making it an invalid ablation — training dynamics
    # were 10× different from the comm-enabled baseline.
    "commformer": {
        "rollouts":        1024,  # v2: 256 → match v13
        "learning_epochs": 3,    # v2: 10 → match v13
        "mini_batches":    4,    # v2: 2  → match v13

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.2,  # v2: 0.3 → match v13
        "min_lr_fraction":         0.1,

        # obs_dim = 190 unchanged (same slots, zeroed where internal state was)
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 190},
        "update_state_preprocessor_in_update": False,
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 1188},
        "update_shared_state_preprocessor_in_update": False,
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.015,
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        # v3: match v13 entropy schedule (0.005 → 0.0005)
        # v2 used 0.05 start (10× too high), causing entropy dominance.
        "entropy_loss_scale":       0.005,  # v2: 0.02 → match v13
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.005,  # v2: 0.05 → match v13
        "entropy_loss_scale_end":   0.0005, # v2: 0.01 → match v13
        "debug_entropy_stats":      False,

        "value_loss_scale":   1.0,
        "kl_threshold":       0.05,
        "kl_warmup_fraction": 0.0,  # v2: 0.3 → match v13 (enable KL from step 1)
        "ratio_max_threshold":  2.0,

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 20.0),
        "time_limit_bootstrap": True,
        "weight_decay":       1e-4,
        "comm_reg_scale":     0.001,

        # Architecture: match v13 exactly for valid ablation.
        "hidden_dim":  128,
        "num_blocks":  2,
        "num_heads":   1,
        "head_dim":    64,
        "mlp_dim":     256,
        "sparsity":    0.5,
    },

    "policy": {
        "unnormalized_log_prob": True,
    },

    "value": {
        "hidden_sizes": [128, 64],
    },

    "memory": {
        "size": 1024,  # v2: 256 → match v13 rollouts
    },
}
