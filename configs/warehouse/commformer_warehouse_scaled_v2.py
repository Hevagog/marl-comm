# fmt: off
"""Scaled topology v2 — CommFormer.

CommFormer-v1 was the only run that converged within 2 M steps
(last10_mean=+209, peak max=+514, entropy 0.23). v2 tests two
hypotheses raised in memory/warehouse_scaled_analysis_2026_05_04.md §5.6:

  * sparsity   0.5 → 0.3   — Hu et al. 2024 ICLR §5.1 / Table 4: best
    generalisation under ablation comes from the sparsest topology
    that still solves the task. k = round(0.3·12) = 4 of 11 peers.

  * num_blocks 2 → 3       — at N=12 the relation-aware attention has
    144 edges and previously only 2 blocks; one extra block doubles
    the graph diameter the model can chain (Velickovic et al. 2018).

  * num_heads  2 → 4       — heads ≥ log2(N²) ≈ log2(144) ≈ 7 → keep at
    4 as a budget compromise; doubles head capacity for heterogeneous
    role learning (fast/standard/heavy).

  * entropy_loss_scale_end  0.0005 → 0.001  — analysis §9 risk note:
    at 0.0005 with full 2 M annealing, entropy can drop below ln(2)≈0.69
    by 1 M, killing half the action repertoire. 0.001 is safer.
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "commformer_warehouse_scaled_v2",
        "agent_type":       "commformer",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["commformer", "warehouse", "scaled12", "scenario5", "v2"],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 200_000,
        "store_separately":    False,
    },

    "env": {
        "id":            "warehouse",
        "num_envs":      2,
        "grid_height":   24,
        "grid_width":    32,
        "num_agents":    12,
        "max_agents":    12,
        "num_shelves":   18,
        "resources_per_shelf": 4,
        "num_treatment_stations": 4,
        "num_goal_locations":   4,
        "treatment_duration":   5,
        "comm_noise_prob":      0.0,
        "vision_range":         2,
        "max_cycles":           500,
        "comm_range":           8,

        "no_comm":          True,
        "randomize_layout": True,

        "enable_task_deadlines":  True,
        "task_arrival_rate":      0.5,
        "task_deadline_min":      80,
        "task_deadline_max":      200,
        "max_pending_tasks":      20,
        "penalty_task_expired":   -1.0,
        "reward_urgent_delivery": 5.0,

        "enable_heterogeneous":    True,
        "agent_speed_options":     (1, 1, 2),
        "agent_capacity_options":  (1, 2, 1),
        "agent_fragility_options": (1.0, 0.5, 2.0),

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

    "training": {"timesteps": 2_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 2_000, "checkpoint_path": None,
                 "video_dir": "recordings", "fps": 10},

    "commformer": {
        "rollouts":        1024,
        "learning_epochs": 5,
        "mini_batches":    4,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.2,
        "min_lr_fraction":         0.1,

        "state_preprocessor":               RunningStandardScaler,
        "state_preprocessor_kwargs":        {"size": 238},
        "update_state_preprocessor_in_update": False,
        "shared_state_preprocessor":        RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": 4716},
        "update_shared_state_preprocessor_in_update": False,
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":        0.5,
        "ratio_clip":            0.2,
        "value_clip":            0.2,
        "clip_predicted_values": False,

        "entropy_loss_scale":       0.005,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.005,
        "entropy_loss_scale_end":   0.001,    # 0.0005 → 0.001 (entropy floor)
        "value_loss_scale":         1.0,

        "kl_threshold":       0.05,
        "kl_warmup_fraction": 0.0,
        "ratio_max_threshold": 2.0,

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 20.0),
        "time_limit_bootstrap": True,
        "weight_decay":       1e-4,

        "comm_reg_scale": 0.001,

        # N=12 CommFormer-v2 architecture
        "hidden_dim":  128,
        "num_blocks":  3,    # 2 → 3
        "num_heads":   4,    # 2 → 4
        "head_dim":    64,
        "mlp_dim":     256,
        "sparsity":    0.3,  # 0.5 → 0.3 (k=4 of 11 peers)
    },

    "policy": {"unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [512, 512]},
    "memory": {"size": 1024},
}
