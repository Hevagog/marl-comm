# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

# MAT on the "blind" continuous-coord benchmark.
# Same env as mappo_continuous_coord_blind.py — the communication gap config.
#
# MAT encoder stacks ALL agents' local observations → agent A's target slot
# (visible target T₁) flows through cross-attention to agent B's decoder →
# B learns to move toward T₁ even though it's outside B's target_vision_range.
# This is the regime where communication architectures should beat MAPPO.
CONFIG = {
    "experiment": {
        "name":             "mat_continuous_coord_blind_v1",
        "agent_type":       "mat",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mat", "continuous_coord", "blind"],
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
        "vision_range":          0.1,
        "target_vision_range":   0.2,
        "collision_radius":      0.03,
        "target_arrival_rate":   0.15,
        "target_k_min":          2,
        "target_k_max":          3,
        "target_deadline_min":   40,
        "target_deadline_max":   100,
        "chain_event_prob":      0.1,
        "max_speed":             0.05,
        "num_agent_types":       2,
        "agent_types":           [0, 0, 1, 1],
        "penalty_wrong_type":    -0.5,
        "penalty_wrong_composition": -1.0,
    },

    "training": {
        "timesteps": 5_000_000,
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

    "mat": {
        # Paper-faithful MuJoCo-style hyperparameters (Wen et al. 2022, Table 8).
        # Our continuous_coord_blind env is MuJoCo-analog: dense rewards, continuous
        # coordination, N=4 agents — so MuJoCo settings are the right baseline.
        #
        # Previous run (3ywolh5j, 2× hidden, 4× heads, clip=0.2, LR=3e-4) plateaued
        # at -12.5 with ratio_mad growing to 4.1 by 500k.  The trust-region violations
        # correlate with the oversized update per step.  Paper HP: clip=0.05, LR=5e-5,
        # smaller network — tighter trust region → ratio_mad should stay <0.5.
        "rollouts":        4096,
        "learning_epochs": 10,       # Paper MuJoCo: 10
        "mini_batches":    4,        # Paper MuJoCo: 40; ours is smaller but honours
                                     # buffer_size % num_agents == 0 invariant.

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  5e-5,  # Paper MuJoCo: 5e-5 (was 3e-4, 6× too high).
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.2,
        "min_lr_fraction":         0.1,

        # obs_dim = 44 (slots zero when outside range — same preprocessor size)
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 44},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 46},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.05,  # Paper: 0.05 (was 0.2).  Tight trust region
                                         # required for transformer-in-RL stability.
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        "entropy_loss_scale":       0.01,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.01,
        "entropy_loss_scale_end":   0.001,
        "debug_entropy_stats":      False,

        "value_loss_scale":   1.0,

        "kl_threshold":       0.05,
        "kl_warmup_fraction": 0.0,

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -8.0, 15.0),

        "time_limit_bootstrap": True,

        "weight_decay":       1e-4,

        # Paper-MuJoCo architecture: hidden=64, 1 block, 1 head, head_dim=64.
        # Our previous 128/2/4/32 oversized by ~4-8×, amplifying ratio drift
        # across the deeper non-linearities each update step.
        "hidden_dim":  64,
        "num_blocks":  1,
        "num_heads":   1,
        "head_dim":    64,
        "mlp_dim":     128,
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
