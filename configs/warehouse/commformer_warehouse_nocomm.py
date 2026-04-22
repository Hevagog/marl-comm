# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

# CommFormer on warehouse with no-communication observation (local vision only).
#
# no_comm=True in env section:
#   - Other agents visible only within vision_range (not comm_range)
#   - Visible: position (dx,dy), is_carrying (shelf on robot), is_stranded (stopped)
#   - Hidden: battery, active flag (require radio to know)
#   - No comm noise (local vision has no packet loss)
#   - Obs dim stays 190 — same preprocessor size
#
# This is the UPPER BOUND for the no-comm communication benchmark:
#   CommFormer attends over all agents' local embeddings via its learned
#   communication graph (α), so it compensates for the missing radio obs
#   by routing relevant local features through model-level attention.
#   The α topology should converge to high-weight edges toward teammates
#   carrying/stranded (inferred from physically-observable features).
#
# rescue_proximity=0.2 + agent_failure_prob=0.001: same rescue tuning
#   as mappo_warehouse_nocomm — ensures rescue gradient exists for all agents.
CONFIG = {
    "experiment": {
        "name":             "commformer_warehouse_nocomm_v1",
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

    "commformer": {
        "rollouts":        1024,
        "learning_epochs": 3,
        "mini_batches":    4,

        "discount_factor": 0.99,
        "lambda":          0.95,

        # LR=3e-4 confirmed by c52g7m0z/tpgbpxwn/8g5sw7au: all reach 150+
        # by 200-275k.  My earlier 1e-4 "fix" (prcnkpyq) reached only +3 at
        # 275k — killing learning by over-constraining early exploration.
        # Root cause of ratio blowup was KL stop using MEAN KL (diluted by
        # 95% of transitions at ratio≈1), fixed by ratio_max_threshold below.
        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.2,
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
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        "entropy_loss_scale":       0.005,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.005,
        "entropy_loss_scale_end":   0.0005,
        "debug_entropy_stats":      False,

        "value_loss_scale":   1.0,
        "kl_threshold":       0.05,
        "kl_warmup_fraction": 0.0,
        # ratio_max_threshold: stop epoch loop when any mini-batch's
        # ratio_max_abs_dev exceeds this.  Catches per-transition drift
        # that mean-KL misses (c52g7m0z hit ratio_mad=14 with mean KL<0.05).
        # 2.0 allows ratio ∈ [0, 3.0] per step, preventing catastrophic
        # forgetting while preserving the large early-phase policy jumps
        # that drive the -30 → +150 transition.
        "ratio_max_threshold":  2.0,

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 20.0),
        "time_limit_bootstrap": True,
        "weight_decay":       1e-4,
        "comm_reg_scale":     0.001,

        # Matches continuous_coord middle ground: 128 hidden × 2 blocks ≈
        # MAPPO's [128, 64] MLP capacity via transformer attention.
        # Previous 256/2/4/64/256 combined with clip=0.2 blew ratio_mad to 10.
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
        "size": 1024,
    },
}
