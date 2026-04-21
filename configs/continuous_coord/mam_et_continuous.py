# fmt: off
# §6.4: MAM ET Encoder-Only (Recurrent Energy Minimisation — replaces BiMamba)
# v2: α=1.0, T=6 (Hoover et al. 2023 §4.1 uses T∈[12,24] with α≈O(1);
# previous α=0.1,T=3 gave effective descent ~0.3× ∂E — flat-lined policy).
# PPO knobs aligned with MAPPO baseline (kl_threshold=0.05, entropy_scale=0.02)
# to match the MAPPO clip-frac=1.4 % regime rather than the 8 % MAM drift.
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler  # noqa: E402

CONFIG = {
    "experiment": {
        "name":             "mam_et_encoder_continuous_coord_v2",
        "agent_type":       "mam_et_encoder",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mam_et_encoder", "continuous_coord"],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 200000,
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
        "timesteps": 20_000_000,
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

        "learning_rate":                  1.5e-4,
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

        # ---- ET Encoder architecture (replaces BiMamba) ----
        "n_embd":           128,
        "num_heads":        4,      # multi-head energy self-attention
        "et_beta":          1.0,    # inverse temperature for attention softmax
        "et_alpha":         1.0,    # energy gradient step size (Hoover 2023 §4.1)
        "num_memories":     64,     # Hopfield memory patterns
        "num_et_steps":     6,      # energy minimization iterations (training)
        "num_et_steps_eval": 12,    # energy minimization iterations (eval — test-time compute)
        "hn_activation":    "relu", # Hopfield activation: "relu" (sparse) or "softmax"
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
