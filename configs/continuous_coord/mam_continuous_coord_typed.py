# fmt: off
# MAM (BiMamba encoder) on typed continuous_coord.
#
# Same typed setup as mappo_continuous_coord_typed.py (4 agents, 2 types, 3 targets).
# BiMamba joint-observation encoder should give a structural advantage here:
# agents must not only rendezvous but also assemble the correct type composition,
# requiring richer inter-agent communication about roles.
#
# obs_dim = 44, state_dim = 42  (see mappo_continuous_coord_typed.py for derivation)
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "mam_continuous_coord_typed_v1",
        "agent_type":       "mam_enc_only",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mam_enc_only", "continuous_coord", "typed"],
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

        "learning_rate":                  1.5e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay":          True,
        "lr_decay_start_fraction":  0.05,
        "min_lr_fraction":          0.1,

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 44},
        "update_state_preprocessor_in_update": False,
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 42},
        "update_shared_state_preprocessor_in_update": False,
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        "entropy_loss_scale":       0.01,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.05,
        "entropy_loss_scale_end":   0.01,
        "value_loss_scale":         1.0,

        "kl_threshold": 0,

        "rewards_shaper":       lambda rewards, *args: jnp.clip(rewards, -8.0, 15.0),
        "time_limit_bootstrap": True,

        "n_embd":      128,
        "n_block":     1,
        "d_state":     32,
        "d_conv":      1,
        "delta_rank":  16,
    },

    "policy": {
        "unnormalized_log_prob": True,
    },
    "value": {
        "hidden_sizes": [256, 256],
    },
    "memory": {
        "size": 4096,
    },
}
