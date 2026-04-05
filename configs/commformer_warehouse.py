# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "commformer_warehouse_v2",
        "agent_type":       "commformer",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["commformer", "warehouse"],
        },
        "write_interval":      "auto",
        "checkpoint_interval": 200_000,
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
        "comm_noise_prob":        0.0,   # CommFormer handles comm
        "vision_range":           2,
        "max_cycles":             500,
        "comm_range":             5,

        "enable_task_deadlines":  True,
        "task_arrival_rate":      0.4,
        "task_deadline_min":      50,
        "task_deadline_max":      120,
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

    # ---- CommFormer-specific PPO parameters ----
    "commformer": {
        "rollouts":        2048,
        "learning_epochs": 8,
        "mini_batches":    4,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        # obs_dim = 7+8+(5*5*6)+(3*6)+3+1+3 = 190 (vision=2, max_agents=4)
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 190},
        # state_dim = 12*16*6 + 4*8 = 1184; expanded with 4-agent one-hot = 1188
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

        "entropy_loss_scale": 0.05,
        "value_loss_scale":   1.0,

        "kl_threshold": 0.05,

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 20.0),

        "time_limit_bootstrap": True,

        "weight_decay":       1e-4,

        # ---- CommFormer architecture hyperparameters ----
        # Larger capacity for 4-agent heterogeneous warehouse task.
        # sparsity=0.5 → k=2 for N=4: each agent attends to 2 others.
        # v2: upsized from 128→256 hidden / 256→512 mlp after observing no learning
        # in v1 (episodes dying at step ~10-100 due to insufficient exploration,
        # entropy_loss_scale bumped 0.02→0.05 to match MAPPO/MAGIC starting value).
        "hidden_dim":  256,
        "num_blocks":  2,
        "num_heads":   4,
        "head_dim":    64,
        "mlp_dim":     512,
        "sparsity":    0.5,
    },

    "policy": {
        "unnormalized_log_prob": True,
    },

    "value": {
        "hidden_sizes": [512, 512],
    },

    "memory": {
        "size": 2048,
    },
}
