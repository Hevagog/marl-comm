# fmt: off
from configs.flatland_basic import FLATLAND_BASIC_ENV, flatland_dims
from skrl.resources.preprocessors.jax import RunningStandardScaler

ENV = dict(FLATLAND_BASIC_ENV)
NUM_AGENTS = ENV["num_agents"]
TREE_DEPTH = ENV["tree_depth"]
OBS_DIM, STATE_DIM = flatland_dims(NUM_AGENTS, TREE_DEPTH)

CONFIG = {
    "experiment": {
        "name":             "mam_flatland_v0",
        "agent_type":       "mam",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mam", "flatland"],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 250_000,
        "store_separately":    False,
    },
    "env": ENV,
    "training": {
        "timesteps": 12_000_000,
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
    "mam": {
        "rollouts":        256,
        "learning_epochs": 10,
        "mini_batches":    2,
        "discount_factor": 0.99,
        "lambda":          0.90,
        "learning_rate":   1.5e-4,
        "linear_lr_decay":          True,
        "lr_decay_start_fraction":  0.05,
        "min_lr_fraction":          0.1,
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
        "clip_predicted_values":  True,
        "entropy_loss_scale":     0.01,
        "entropy_annealing":      True,
        "entropy_loss_scale_start":0.03,
        "entropy_loss_scale_end": 0.01,
        "value_loss_scale":       0.5,
        "kl_threshold":           0.0,
        "rewards_shaper":         None,
        "time_limit_bootstrap":   True,
        "n_embd":                 128,
        "n_block":                1,
        "d_state":                32,
        "d_conv":                 4,
        "delta_rank":             16,
    },
    "policy": {
        "unnormalized_log_prob": True,
    },
    "value": {
        "hidden_sizes": [256, 128],
    },
    "memory": {
        "size": 256,
    },
}
