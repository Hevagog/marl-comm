# fmt: off
from skrl.resources.preprocessors.jax import RunningStandardScaler

TREE_DEPTH = 2
NUM_AGENTS = 5
TREE_FEATURE_DIM = 12
TREE_NODE_COUNT = sum(4 ** level for level in range(TREE_DEPTH + 1))
OBS_DIM = TREE_NODE_COUNT * TREE_FEATURE_DIM
STATE_DIM = NUM_AGENTS * (OBS_DIM + 7)

CONFIG = {
    "experiment": {
        "name":             "mappo_flatland_smoke_v0",
        "agent_type":       "mappo",
        "directory":        "runs",
        "wandb":            False,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mappo", "flatland", "smoke"],
        },
        "write_interval":      "auto",
        "checkpoint_interval": 20_000,
        "store_separately":    False,
    },
    "env": {
        "id":                      "flatland",
        "num_envs":                1,
        "width":                   25,
        "height":                  25,
        "num_agents":              NUM_AGENTS,
        "max_num_cities":          3,
        "max_rails_between_cities":2,
        "max_rail_pairs_in_city":  2,
        "grid_mode":               True,
        "tree_depth":              TREE_DEPTH,
        "prediction_depth":        20,
        "use_malfunctions":        False,
        "speed_ratio_map":         {1.0: 0.6, 0.5: 0.4},
        "max_episode_steps":       100,
    },
    "training": {
        "timesteps": 20_000,
        "seed":      42,
    },
    "eval": {
        "timesteps":       500,
        "checkpoint_path": None,
    },
    "record": {
        "timesteps":       250,
        "checkpoint_path": None,
        "video_dir":       "recordings",
        "fps":             4,
    },
    "mappo": {
        "rollouts":        128,
        "learning_epochs": 4,
        "mini_batches":    2,
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
        "size": 128,
    },
}
