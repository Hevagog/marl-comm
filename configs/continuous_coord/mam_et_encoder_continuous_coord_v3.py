# fmt: off
# MAM ET Encoder-Only v3 on continuous_coord.
# Fixes applied over v2 (wandb: vco9rmjc, 700K steps, zero learning):
#
# v2 diagnosis:
#   - Total reward flat: -16 → -14.3 over 700K steps (no phase transition)
#   - Entropy: 2.17–2.18 (near-uniform maximum = ln9 = 2.197) — never drops
#   - Value loss: INCREASING 0.28 → 0.40 (critic fitting noise, not signal)
#   - Clip fraction: ~4.2% (healthy externally, but policy never commits)
#
# Root causes:
#   1. Gradient chain through T=6 unrolled ET steps × α=1.0 produces unstable
#      Jacobian products. Even with grad_norm_clip=0.5, the internal gradient
#      magnitudes through 6 chained attention+Hopfield operations prevent the
#      policy from reliably ascending reward.
#   2. Learning rate 1.5e-4 is half of MAPPO's 3e-4 — too conservative for
#      the ET encoder's parameter count (O(n_embd²) attention matrices × T steps).
#
# Fixes (v3):
#   A. stop_grad_intermediate=True: apply jax.lax.stop_gradient after each ET
#      step except the last. The forward pass (energy minimization) runs fully;
#      only the backward pass is truncated to a single-step Jacobian. This is
#      the "one-step implicit differentiation" approach from DEQ models
#      (Bai et al. 2019, arXiv:1909.01377; Fung et al. 2022 arXiv:2305.13768).
#      The effective gradient becomes ∂x_T/∂θ ≈ ∂(x_{T-1} + α·f(x_{T-1}))/∂θ
#      (single Jacobian, bounded by construction if α·||∂f/∂x|| < 1).
#   B. learning_rate 1.5e-4 → 3e-4 (match MAPPO).
#
# Other PPO knobs unchanged from v2 (already match MAPPO):
#   kl_threshold=0.05, kl_warmup_fraction=0.3, learning_epochs=8, mini_batches=4
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "mam_et_encoder_continuous_coord_v3",
        "agent_type":       "mam_et_encoder",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mam_et_encoder", "continuous_coord", "v3", "stop_grad"],
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

    "mam": {
        "rollouts":        4096,
        "learning_epochs": 8,
        "mini_batches":    4,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  3e-4,   # was 1.5e-4; match MAPPO
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay":          True,
        "lr_decay_start_fraction":  0.05,
        "min_lr_fraction":          0.1,

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
        "value_loss_scale":   1.0,

        "kl_threshold":       0.05,
        "kl_warmup_fraction": 0.3,

        "rewards_shaper":       lambda rewards, *args: jnp.clip(rewards, -5.0, 15.0),
        "time_limit_bootstrap": True,

        # ---- ET Encoder architecture ----
        "n_embd":                  128,
        "num_heads":               4,
        "et_beta":                 1.0,
        "et_alpha":                1.0,
        "num_memories":            64,
        "num_et_steps":            6,
        "num_et_steps_eval":       12,
        "hn_activation":           "relu",
        # Key fix: stop_gradient after each non-final ET step.
        # Prevents gradient chain instability through T=6 unrolled Jacobians.
        "stop_grad_intermediate":  True,
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
