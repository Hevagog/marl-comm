# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "commformer_warehouse_v13",
        "agent_type":       "commformer",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["commformer", "warehouse"],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 200_000,
        "store_separately":    False,
    },

    "env": {
        "id":            "warehouse",
        "num_envs":      8,    # NOTE: increase to 8–16 for production runs; paper uses 64
        "grid_height":   12,
        "grid_width":    16,
        "num_agents":    4,
        "max_agents":    4,
        "num_shelves":   6,
        "resources_per_shelf": 4,
        "num_treatment_stations": 2,
        "num_goal_locations": 2,
        "treatment_duration": 5,
        "comm_noise_prob":        0.0,   # MAM handles via BiMamba encoder
        "vision_range":           2,     # 5×5 patch
        "max_cycles":             500,
        "comm_range":             5,

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

        "enable_interference_zones":    True,
        "interference_base":            0.05,
        "interference_treatment_boost": 0.15,
        "interference_radius":          2,

        "enable_battery":             True,
        "battery_capacity":           200,
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
            "burst_prob":          0.0002,
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
        # v13: 256→1024 rollouts for 4× more data per update.
        # Parallel decoder is 4× faster at rollout (no AR scan), so net
        # compute per update is roughly unchanged.  4× larger buffer gives
        # substantially stronger policy-gradient signal relative to entropy,
        # breaking the entropy-dominance cycle observed in v12.
        "rollouts":        1024,
        "learning_epochs": 3,
        "mini_batches":    4,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        # v11: linear LR decay — match MAPPO to stabilise late-stage training.
        # Decay begins at 20% of total steps (earlier than MAPPO's 30%) because
        # CommFormer needs more aggressive stabilisation once the communication
        # graph starts to crystallise.
        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.2,
        "min_lr_fraction":         0.1,

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

        # v13: further entropy reduction.
        # v12 analysis: entropy still INCREASED (1.65→1.74) even at scale=0.02.
        # Root cause: policy gradient near-zero on sparse rewards → entropy
        # gradient dominates → policy driven toward uniform.
        # v13 fix: 0.005 start (4× lower than v12) so even weak policy gradients
        # dominate.  Parallel decoder also produces ratio=1.0 at update start,
        # allowing all 3×4=12 gradient steps to execute, strengthening signal.
        "entropy_loss_scale":       0.005,  # fallback if annealing is off
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.005,
        "entropy_loss_scale_end":   0.0005,
        "debug_entropy_stats":      False,

        "value_loss_scale":   1.0,

        # v13: relax KL threshold — parallel decoder starts ratio=1.0 so early
        # stopping is less likely to be triggered spuriously.  0.05 matches
        # typical MAPPO KL thresholds and allows 2-3 full epochs to execute
        # per rollout before the policy has drifted too far.
        "kl_threshold":       0.05,
        "kl_warmup_fraction": 0.0,

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 20.0),

        "time_limit_bootstrap": True,

        "weight_decay":       1e-4,

        # v11: communication topology regulariser (Bug-3 from memory).
        # Pushes adjacency density toward 0.5 (balanced) via binary-entropy
        # loss on off-diagonal elements.  Prevents α collapse to all-zero or
        # all-one, which would make communication trivially sparse or dense.
        "comm_reg_scale":     0.001,

        # ---- CommFormer architecture hyperparameters ----
        # Larger capacity for 4-agent heterogeneous warehouse task.
        # sparsity=0.5 → k=2 for N=4: each agent attends to 2 others.
        # v2: upsized from 128→256 hidden / 256→512 mlp after observing no learning
        # in v1.  v11: keeping 256 — the architecture was correct, the training
        # schedule was wrong (missing entropy annealing + LR decay).
        "hidden_dim":  256,
        "num_blocks":  2,
        "num_heads":   4,
        "head_dim":    64,
        "mlp_dim":     256,
        "sparsity":    0.5,
    },

    "policy": {
        "unnormalized_log_prob": True,
    },

    "value": {
        "hidden_sizes": [256, 128],
    },

    "memory": {
        "size": 1024,
    },
}
