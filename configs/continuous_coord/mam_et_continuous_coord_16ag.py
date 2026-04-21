# fmt: off
# MAM ET Encoder-Only on continuous_coord, 16 agents.
# ET replaces BiMamba: symmetric O(n²=256) attention eliminates positional gradient bias.
# v2 hparams from §13.3: α=1.0, T_train=6, T_eval=12 (Hoover 2023 §4.1 regime).
# Hopfield memories scaled to 128 (from 64) to cover richer 16-agent coordination space.
# obs_dim = 6 + (16-1)*4 + 3*4 = 78
# state_dim = 16*4 + 3*4 = 76
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "mam_et_encoder_continuous_coord_16ag_v1",
        "agent_type":       "mam_et_encoder",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mam_et_encoder", "continuous_coord", "16_agents"],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 200_000,
        "store_separately":    False,
    },

    "env": {
        "id":                "continuous_coord",
        "num_envs":          16,
        "num_agents":        16,
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
        "mini_batches":    8,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  1.5e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay":          True,
        "lr_decay_start_fraction":  0.05,
        "min_lr_fraction":          0.1,

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 78},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 76},
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

        "kl_threshold":       0.05,
        "kl_warmup_fraction": 0.3,

        "rewards_shaper":       lambda rewards, *args: jnp.clip(rewards, -5.0, 15.0),
        "time_limit_bootstrap": True,

        # ---- ET Encoder architecture ----
        "n_embd":            128,
        "num_heads":         8,       # 8 heads for 16-agent sequence (1 head/2 agents)
        "et_beta":           1.0,
        "et_alpha":          1.0,     # Hoover 2023 §4.1 regime
        "num_memories":      128,     # scaled from 64 to match 16-agent coordination space
        "num_et_steps":      6,       # training iterations
        "num_et_steps_eval": 12,      # test-time compute (Proposal D from §6.4)
        "hn_activation":     "relu",  # sparse Hopfield retrieval
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
