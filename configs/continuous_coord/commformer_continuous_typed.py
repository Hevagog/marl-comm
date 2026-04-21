# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "commformer_continuous_coord_typed_v3",
        "agent_type":       "commformer",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["commformer", "continuous_coord", "typed"],
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
        "vision_range":          0.4,
        "collision_radius":      0.03,
        "target_arrival_rate":   0.15,
        "target_k_min":          2,
        "target_k_max":          3,
        "target_deadline_min":   30,
        "target_deadline_max":   80,
        "chain_event_prob":      0.2,
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
    "commformer": {
        "rollouts":        4096,
        # v2: reduced epochs 5→3, increased mini_batches 2→4.
        # 3 epochs × 4 mini-batches = 12 gradient steps (same compute as 5×2=10 but
        # more data coverage per epoch).  Fewer sequential steps on the same data
        # reduces ratio drift vs v1 which had ratio_max_abs_dev growing to 5+ by 350k.
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

        # obs_dim = 6 + T + (n-1)*(4+T) + mt*(4+T) = 6+2+3*6+3*6 = 44 (n=4,T=2,mt=3)
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 44},
        # state_dim = n*(4+T) + mt*(4+T) = 4*6+3*6 = 42; expanded with 4-agent one-hot = 46
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 46},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        # v3: match warehouse v13 entropy scale (0.005 start).
        # continuous_coord has denser rewards than warehouse but same issue
        # of entropy dominating early with random comm graph.
        "entropy_loss_scale":       0.005,  # fallback if annealing is off
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.005,
        "entropy_loss_scale_end":   0.0005,
        "debug_entropy_stats":      False,

        "value_loss_scale":   1.0,

        # v3: same KL settings as v2; parallel decoder keeps ratio near 1.0.
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
        "size": 4096,
    },
}
