# fmt: off
from flatland_basic import FLATLAND_BASIC_ENV, flatland_dims
from skrl.resources.preprocessors.jax import RunningStandardScaler

ENV = dict(FLATLAND_BASIC_ENV)
NUM_AGENTS = ENV["num_agents"]
TREE_DEPTH = ENV["tree_depth"]
OBS_DIM, STATE_DIM = flatland_dims(NUM_AGENTS, TREE_DEPTH)

CONFIG = {
    "experiment": {
        "name":             "mappo_flatland_v0",
        "agent_type":       "mappo",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mappo", "flatland"],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 250_000,
        "store_separately":    False,
    },
    "env": ENV,
    "training": {
        "timesteps": 8_000_000,
        "seed":      42,
    },
    "eval": {
        "timesteps":       2_000,
        "checkpoint_path": None,
    },
    "record": {
        "timesteps":       1_000,
        "checkpoint_path": None,
        "video_dir":       "recordings",
        "fps":             5,
    },
    "mappo": {
        "rollouts":        512,
        "learning_epochs": 8,
        "mini_batches":    4,
        "discount_factor": 0.99,
        "lambda":          0.95,
        "learning_rate":   3e-4,
        "state_preprocessor":               RunningStandardScaler,
        "state_preprocessor_kwargs":        {"size": OBS_DIM},
        "shared_state_preprocessor":        RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": STATE_DIM},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},
        "random_timesteps":       0,
        "learning_starts":        0,
        "grad_norm_clip":         0.5,
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  False,
        "entropy_loss_scale":     0.01,
        "value_loss_scale":       1.0,
        "kl_threshold":           0.0,
        "rewards_shaper":         None,
        "time_limit_bootstrap":   True,
        "linear_lr_decay":        True,
        "lr_decay_start_fraction":0.3,
        "min_lr_fraction":        0.1,
    },
    "policy": {
        "hidden_sizes":          [256, 256],
        "unnormalized_log_prob": True,
        "use_memory":            False,
    },
    "value": {
        "hidden_sizes": [256, 256],
        "use_memory":   False,
    },
    "memory": {
        "size": 512,
    },
}
