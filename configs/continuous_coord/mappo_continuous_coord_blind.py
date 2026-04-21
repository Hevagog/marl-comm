# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

# MAPPO on the "blind" continuous-coord benchmark.
#
# Key difference from mappo_continuous_coord.py:
#   target_vision_range=0.2  — each agent sees targets only within radius 0.2
#   vision_range=0.1         — agents rarely see teammates directly (map is [0,1]^2)
#
# Both range values are deliberately small so that:
#   - Each agent has PRIVATE information about nearby targets/teammates
#   - For k-of-n typed capture, an agent that sees a target cannot independently
#     attract the required teammate — it needs to communicate the location
#   - MAPPO receives the same obs_dim=44 tensor but the teammate/target slots
#     are zeroed when outside range → policy acts on incomplete info
#   - Serves as LOWER BOUND for the communication benchmark:
#     MAPPO should fail to coordinate k>=2 typed captures reliably
#
# Expected outcome:
#   MAPPO: captures only solo targets (k=1) or same-range k=2 by accident.
#   Communication agents (MAT/CommFormer/MAM): share target positions via
#   encoder attention → coordinate typed rendezvous → beat MAPPO.
CONFIG = {
    "experiment": {
        "name":             "mappo_continuous_coord_blind_v1",
        "agent_type":       "mappo",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mappo", "continuous_coord", "blind", "baseline"],
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

    "mappo": {
        "rollouts":        4096,
        "learning_epochs": 10,
        "mini_batches":    2,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.3,
        "min_lr_fraction":         0.1,

        # obs_dim = 44 (unchanged — slots zeroed when outside range)
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 44},
        # state_dim = n*(4+T) + mt*(4+T) = 4*6+3*6=42; +4 one-hot = 46
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

        "entropy_loss_scale":       0.05,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.05,
        "entropy_loss_scale_end":   0.005,
        "debug_entropy_stats":      False,

        "value_loss_scale":   1.0,

        "kl_threshold":       0.05,
        "kl_warmup_fraction": 0.3,

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -8.0, 15.0),

        "time_limit_bootstrap": True,

        "weight_decay":       1e-4,
    },

    "policy": {
        "unnormalized_log_prob": True,
        "hidden_sizes": [256, 128],
    },

    "value": {
        "hidden_sizes": [256, 128],
    },

    "memory": {
        "size": 4096,
    },
}
