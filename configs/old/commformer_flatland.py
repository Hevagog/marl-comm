# fmt: off
from flatland_hard import FLATLAND_HARD_ENV, flatland_dims
from skrl.resources.preprocessors.jax import RunningStandardScaler

ENV = dict(FLATLAND_HARD_ENV)
NUM_AGENTS = ENV["num_agents"]
TREE_DEPTH = ENV["tree_depth"]
OBS_DIM, STATE_DIM = flatland_dims(NUM_AGENTS, TREE_DEPTH)

CONFIG = {
    "experiment": {
        "name":             "commformer_flatland_v0",
        "agent_type":       "commformer",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["commformer", "flatland"],
        },
        "write_interval":      "auto",
        "checkpoint_interval": 100,
        "store_separately":    False,
    },
    "env": ENV,
    "training": {
        "timesteps": 10_000_000,
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
    "commformer": {
        "rollouts":        256,
        "learning_epochs": 8,
        "mini_batches":    2,
        "discount_factor": 0.99,
        "lambda":          0.95,
        "learning_rate":   2e-4,
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
        "weight_decay":           1e-4,
        "hidden_dim":             256,
        "num_blocks":             2,
        "num_heads":              4,
        "head_dim":               64,
        "mlp_dim":                512,
        "sparsity":               0.3,
    },
    "policy": {
        "unnormalized_log_prob": True,
    },
    "value": {
        "hidden_sizes": [512, 256],
    },
    "memory": {
        "size": 256,
    },
}
