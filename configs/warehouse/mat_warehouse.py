# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

# MAT warehouse v1
#
# Architecture: Multi-Agent Transformer (Wen et al. 2022, NeurIPS).
# Standard scaled dot-product self-attention over N agent observations —
# no CommGraph, no α, no k-hot STE.  GTrXL init on Wo/mlp2 projections
# (Parisotto et al. 2020) for stable early training.  The repo now uses
# a paper-faithful autoregressive decoder at rollout / teacher forcing in
# update, so PPO stability relies on matched action conditioning instead of
# the old parallel-decoder ratio=1.0 shortcut.
#
# Hyperparameter rationale vs. CommFormer v13
# -------------------------------------------
# - rollouts 1024: same as CommFormer v13.  MAT has no α to stabilise so
#   fewer rollouts than MAM (256) should suffice; 1024 balances data
#   diversity vs. update frequency.
# - learning_epochs 5 (vs. CF 3): MAT has 3.4× fewer params than CommFormer
#   (no CommGraph, no EdgeEmbedding) so it can tolerate more gradient steps
#   per rollout without ratio drift, because the network is less expressive
#   and the ratio stays closer to 1.0.
# - entropy_loss_scale_start 0.01: MAT attention is differentiable from
#   step 0 (no discrete STE bottleneck), so it can extract gradient signal
#   sooner and benefits from slightly higher initial exploration pressure.
# - comm_reg_scale: absent (no α to regularise).
# - hidden_dim 256, mlp_dim 512: increased FFN capacity vs. CommFormer
#   (mlp_dim 256) to match the wider standard-transformer FFN ratio (2× d).
CONFIG = {
    "experiment": {
        "name":             "mat_warehouse_v1",
        "agent_type":       "mat",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mat", "warehouse"],
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
        "comm_noise_prob":        0.0,
        "vision_range":           2,
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

    # ---- MAT-specific PPO parameters ----
    "mat": {
        "rollouts":        1024,
        "learning_epochs": 5,
        "mini_batches":    4,

        "discount_factor": 0.99,
        "lambda":          0.95,

        # 3e-4 matches CommFormer warehouse (dense reward env that needs
        # large early jumps).  ratio_max_threshold below caps per-step drift.
        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.2,
        "min_lr_fraction":         0.1,

        # obs_dim = 7+8+(5*5*6)+(3*6)+3+1+3 = 190 (vision=2, max_agents=4)
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 190},
        "update_state_preprocessor_in_update": False,
        # state_dim = 12*16*6 + 4*8 = 1184; +4 one-hot = 1188
        # WH-B02 note: runtime _sync_preprocessor_sizes corrects to 1184 if needed.
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

        # Slightly higher initial entropy than CommFormer v13 (0.005→0.01)
        # because MAT has no discrete STE bottleneck and can extract gradient
        # signal from step 0; the extra exploration helps shape attention.
        "entropy_loss_scale":       0.01,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.01,
        "entropy_loss_scale_end":   0.001,
        "debug_entropy_stats":      False,

        "value_loss_scale":   1.0,

        # With the paper-faithful AR decoder restored, KL=0.05 matches MAPPO
        # and bounds update drift without relying on the old parallel shortcut.
        "kl_threshold":       0.05,
        "kl_warmup_fraction": 0.0,
        "ratio_max_threshold":  2.0,

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 20.0),

        "time_limit_bootstrap": True,

        "weight_decay":       1e-4,

        # ---- MAT architecture hyperparameters ----
        # Middle-ground architecture: 128 × 2 blocks × 1 head ≈ MAPPO's
        # [128, 64] MLP capacity via transformer attention.  Previous
        # 256/2/4/64/512 combined with clip=0.2 amplified ratio drift
        # (same pattern as continuous_coord_blind 3ywolh5j run hit 4.1).
        # sparsity absent: MAT has full N×N attention, no CommGraph.
        "hidden_dim":  128,
        "num_blocks":  2,
        "num_heads":   1,
        "head_dim":    64,
        "mlp_dim":     256,
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
