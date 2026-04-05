# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "magic_warehouse_v5",
        "agent_type":       "magic",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["magic", "warehouse", "rescue"],
        },
        "write_interval":      "auto",
        "checkpoint_interval": 200000,
        "store_separately":    False,
    },

    "env": {
        "id":            "warehouse",
        "num_envs":      1,
        "grid_height":   12,
        "grid_width":    16,
        "num_agents":    4,
        "max_agents":    4,   
        "num_shelves":   6,
        "resources_per_shelf": 4,
        "num_treatment_stations": 2,
        "num_goal_locations": 2,
        "treatment_duration": 5,
        "comm_noise_prob":        0.0,   # MAGIC handles communication
        "vision_range":           2,     # 5×5 patch — enough for navigation; comm_range=5 still exceeds it
        "max_cycles":             500,
        "comm_range":             5,

        "enable_task_deadlines":  True,
        "task_arrival_rate":      0.4,   # reduced: less queue pressure while comm learns
        "task_deadline_min":      50,    # more time per task (was 40)
        "task_deadline_max":      120,   # (was 100)
        "max_pending_tasks":      15,
        "penalty_task_expired":   -2.0,
        "reward_urgent_delivery": 5.0,

        "enable_heterogeneous":   True,
        "agent_speed_options":    (1, 1, 2),
        "agent_capacity_options": (1, 2, 1),
        "agent_fragility_options":(1.0, 0.5, 2.0),

        "enable_interference_zones":    True,
        "interference_base":            0.05,
        "interference_treatment_boost": 0.35,
        "interference_radius":          2,

        "enable_battery":             True,
        "battery_capacity":           160,
        "battery_drain_per_step":     1,
        "battery_drain_idle":         0,
        "battery_charge_rate":        8,
        "battery_critical_threshold": 25,
        "num_charging_stations":      2,
        "reward_rescue_repair":       12.0,
        "reward_rescue_charge":       12.0,

        "agent_failure_prob": 0.0002,
        "fault_profile": {
            "burst_attrition":     True,
            "burst_prob":          0.0005,
            "correlated_failure":  False,
            "correlation_radius":  1,
            "load_dependent_comm": True,
            "base_packet_loss":    0.02,
            "congestion_factor":   0.05,
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

    "magic": {
        "rollouts":        4096,
        "learning_epochs": 8,
        "mini_batches":    4,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        # obs_dim = 7+8+(5*5*6)+(3*6)+3+1+3 = 190  (vision=2, max_agents=4, battery+hetero+tasks)
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 190},
        # state_dim = 12*16*6 + 4*8 = 1152+32 = 1184; expanded with 4-agent one-hot = 1188
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 1188},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        "entropy_loss_scale":       0.02,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.05,
        "entropy_loss_scale_end":   0.015,  # higher floor encourages comm topology exploration
        "debug_entropy_stats":      False,
        "value_loss_scale":         1.0,

        "kl_threshold":       0.02,  # standard PPO; was 0.05 (too permissive)
        "kl_warmup_fraction": 0.3,
        "debug_kl_stats":     False,

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 20.0),

        "time_limit_bootstrap": True,

        "weight_decay":            1e-4,
        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.3,
        "min_lr_fraction":         0.1,

        # MAGIC-specific parameters
        "message_dim":        64,   # was 128; paper uses modest dims — better gradient flow for N=4
        "num_comm_rounds":    2,
        "num_heads":          2,    # was 4; 2 heads sufficient for N=4 agents (Velickovic et al. 2018)
        # Temperature starts warm (1.0) for better gradient flow through Gumbel-Softmax,
        # annealed to 0.5 over the first 70% of training (was 0.3/50% — too sharp/fast,
        # locking in the no-comm identity matrix before topology learning has a chance).
        "gumbel_temperature":                 1.0,
        "gumbel_temperature_end":             0.5,   # was 0.3; maintains meaningful gradient flow
        "gumbel_temperature_anneal_fraction": 0.7,   # was 0.5; slower schedule prevents early commit

        # Communication topology regularizer: penalizes edge-distribution collapse
        # (all-zero OR all-one adjacency) via binary entropy on mean off-diagonal density.
        # Weight 0.001 provides a gentle push without dominating the policy loss.
        # NOW IMPLEMENTED in _update_policy_fixed (categorical_mappo.py).
        "comm_reg_scale": 0.001,
    },

    "policy": {
        "hidden_sizes":          [256, 256],
        "unnormalized_log_prob": True,
    },

    "value": {
        "hidden_sizes": [256, 256],
    },

    "memory": {
        "size": 4096,
    },
}
