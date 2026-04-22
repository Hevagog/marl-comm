# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "commformer_continuous_coord_blind_v1",
        "agent_type":       "commformer",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["commformer", "continuous_coord", "blind", "v2"],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 200_000,
        "store_separately":    False,
    },

   "env": {
        "id":                    "continuous_coord",
        "num_envs":              16,
        "num_agents":            4,
        "max_cycles":            200,
        "max_targets":           3,
        "capture_radius":        0.08,

        # Small ranges create the communication gap:
        "vision_range":          0.1,   # agents rarely see teammates directly
        "target_vision_range":   0.2,   # each agent sees only nearby targets

        "collision_radius":      0.03,
        "target_arrival_rate":   0.15,
        "target_k_min":          2,
        "target_k_max":          3,
        "target_deadline_min":   40,    # slightly more slack vs. fully-sighted
        "target_deadline_max":   100,
        "chain_event_prob":      0.1,
        "max_speed":             0.05,
        "num_agent_types":       2,
        "agent_types":           [0, 0, 1, 1],
        "penalty_wrong_type":    -0.5,
        "penalty_wrong_composition": -1.0,
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
    # Paper-faithful MuJoCo-style hyperparameters (CommFormer follows MAT's
    # Table 8: Wen et al. 2022 NeurIPS).  Previous run (onhblyxs, hidden=256,
    # blocks=2, heads=4, clip=0.2, LR=3e-4) plateaued at -12.6 with
    # ratio_max_abs_dev growing to 17.5 — classic transformer-in-RL
    # trust-region violation.  Downsizing to paper HP tightens the update.
    "commformer": {
        "rollouts":        4096,
        "learning_epochs": 10,       # Paper MuJoCo: 10
        "mini_batches":    4,        # Paper MuJoCo: 40; ours honours
                                     # buffer_size % num_agents invariant.

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  5e-5,  # Paper MuJoCo: 5e-5 (was 3e-4, 6× too high).
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        # v11: linear LR decay — match MAPPO to stabilise late-stage training.
        # Decay begins at 20% of total steps (earlier than MAPPO's 30%) because
        # CommFormer needs more aggressive stabilisation once the communication
        # graph starts to crystallise.
        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.2,
        "min_lr_fraction":         0.1,

        # obs_dim = 6 + T + (n-1)*(4+T) + mt*(4+T) = 6+2+3*6+3*6 = 44 (n=4,T=2,mt=3)
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 44},
        "update_state_preprocessor_in_update": False,
        # state_dim = n*(4+T) + mt*(4+T) = 4*6+3*6 = 42; expanded with 4-agent one-hot = 46
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 46},
        "update_shared_state_preprocessor_in_update": False,
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.05,  # Paper: 0.05 (was 0.2).  Tight trust region
                                         # required for transformer-in-RL stability.
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        # v2: lower starting entropy scale so policy gradient dominates from step 1.
        # v1 used 0.05 → entropy dominated policy gradient by 10-80× → entropy stuck at max.
        "entropy_loss_scale":       0.02,   # fallback if annealing is off
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.02,
        "entropy_loss_scale_end":   0.002,
        "debug_entropy_stats":      False,

        "value_loss_scale":   1.0,

        # v2: match MAPPO's kl_threshold=0.05 for continuous_coord (smaller obs space,
        # more stable gradients than warehouse).
        # v2: kl_warmup_fraction=0.0 — enable KL stopping from step 1.
        # v1's 0.05 warmup allowed ratio blow-up for the first 500k steps, destabilising
        # the policy before KL stop became active.
        "kl_threshold":       0.05,
        "kl_warmup_fraction": 0.0,

        # v2: match MAPPO typed rewards_shaper (wider lower bound for type penalties).
        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -8.0, 15.0),

        "time_limit_bootstrap": True,

        "weight_decay":       1e-4,

        # v11: communication topology regulariser (Bug-3 from memory).
        # Pushes adjacency density toward 0.5 (balanced) via binary-entropy
        # loss on off-diagonal elements.  Prevents α collapse to all-zero or
        # all-one, which would make communication trivially sparse or dense.
        "comm_reg_scale":     0.001,

        # Paper-MuJoCo architecture: hidden=64, 1 block, 1 head, head_dim=64.
        # Previous 256/2/4/64/256 oversized by 4× — amplified ratio drift
        # across deeper non-linearities per update step (onhblyxs ratio_mad → 17.5).
        # sparsity=0.5 → k=2 for N=4: each agent attends to 2 others.
        "hidden_dim":  64,
        "num_blocks":  1,
        "num_heads":   1,
        "head_dim":    64,
        "mlp_dim":     128,
        "sparsity":    0.5,
    },

    "policy": {
        "unnormalized_log_prob": True,
    },

    "value": {
        "hidden_sizes": [128, 64],
    },

    "memory": {
        "size": 4096,
    },
}
