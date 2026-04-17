# fmt: off
# SyncMixer v2 on continuous_coord.
# Fix: align ALL PPO knobs with the MAPPO baseline that reaches 90+ reward.
#
# v1 diagnosis (wandb: pnfcny9s, 1.175M steps):
#   - Total reward flat: -15.8 → -14.4 (no phase transition observed)
#   - Clip fraction: 8–13% early, 6–9% late (4–6× higher than MAPPO 1.4%)
#   - Entropy never drops below 2.11 (MAPPO drops to ~1.49 at phase transition)
#
# Root cause: v1 config had 4 PPO knobs misaligned vs MAPPO:
#   learning_rate     2.5e-4 → 3e-4    (match MAPPO; 17% increase)
#   kl_threshold      0.03   → 0.05    (match MAPPO; allows larger policy steps
#                                        needed for the coordination phase transition)
#   learning_epochs   5      → 8       (match MAPPO; 60% more gradient updates/cycle)
#   mini_batches      8      → 4       (match MAPPO; 2× larger batches = lower
#                                        gradient variance per update)
#
# Theory: MAPPO's phase transition at ~475K steps requires a single large policy
# update (entropy 2.18→1.49). v1's kl_threshold=0.03 truncated this update before
# the coordination signal could propagate. With kl_threshold=0.05 (MAPPO level),
# the trust region is wide enough to allow the one-shot policy commit.
#
# Architecture unchanged: 2-block symmetric-attention encoder + HopfieldPooling
# (K=4, β=2.0, pool_dim=32) — head fan-in 256 stays within safe bounds per §11
# of Hopfield_Energy_Transformer_for_MAMEncoderOnly.md.
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "syncmixer_continuous_coord_v2",
        "agent_type":       "syncmixer",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["syncmixer", "continuous_coord", "v2"],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 200_000,
        "store_separately":    False,
    },

    "env": {
        "id":                "continuous_coord",
        "num_envs":          16,
        "num_agents":        4,
        "max_cycles":        200,
        "max_targets":       3,
        "capture_radius":    0.08,
        "vision_range":      0.4,
        "collision_radius":  0.03,
        "target_arrival_rate":   0.15,
        "target_k_min":      2,
        "target_k_max":      3,
        "target_deadline_min":   30,
        "target_deadline_max":   80,
        "chain_event_prob":  0.2,
        "max_speed":         0.05,
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
        "timesteps":       1_000,
        "checkpoint_path": None,
        "video_dir":       "recordings",
        "fps":             4,
    },

    "syncmixer": {
        # --- PPO knobs: now match MAPPO continuous_coord baseline ---
        "rollouts":        4096,
        "learning_epochs": 8,       # was 5; +60% gradient updates per cycle
        "mini_batches":    4,       # was 8; 2× larger batches → lower variance

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  3e-4,   # was 2.5e-4; match MAPPO
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 30},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 28},
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
        "entropy_loss_scale_end":   0.01,
        "value_loss_scale":         1.0,

        "kl_threshold":       0.05,     # was 0.03; match MAPPO; trust region wide
        "kl_warmup_fraction": 0.3,

        "rewards_shaper":       lambda rewards, *args: jnp.clip(rewards, -5.0, 15.0),
        "time_limit_bootstrap": True,

        "weight_decay":            1e-4,
        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.3,
        "min_lr_fraction":         0.1,

        # ---- SyncMixer architecture (unchanged from v1) ----
        "d_model":          128,
        "n_block":          2,
        "num_heads":        4,
        "hidden_mult":      4,
        "num_pool_queries": 4,
        "pool_dim":         32,
        "pool_beta":        2.0,
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
