# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler  # noqa: E402

CONFIG = {
    "experiment": {
        "name":             "magic_highway_v0",
        "agent_type":       "magic",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["magic", "highway", "intersection"],
        },
        "write_interval":      "auto",
        "checkpoint_interval": "auto",
        "store_separately":    False,
    },

    "env": {
        "id":                     "intersection",
        "num_agents":             4,
        "duration":               13,           # steps per episode
        "vehicles_count":         10,           # vehicles in each agent's observation
        "features":               ["presence", "x", "y", "vx", "vy"],
        "initial_vehicle_count":  10,
        "spawn_probability":      0.6,
        "collision_reward":       -5.0,
        "arrived_reward":         1.0,
        "high_speed_reward":      1.0,
        "reward_speed_range":     [7.0, 9.0],
        "normalize_reward":       True,
        "simulation_frequency":   15,
        "policy_frequency":       1,
    },

    "training": {
        "timesteps": 8_000_000,
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
        "fps":             5,
    },

    "magic": {
        "rollouts":        4096,      # number of rollouts before updating
        "learning_epochs": 8,         # learning epochs per update
        "mini_batches":    4,         # mini-batches per learning epoch

        "discount_factor": 0.99,      # gamma - standard for continuous control
        "lambda":          0.95,      # TD(lambda) / GAE lambda

        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        # Highway intersection: 10 vehicles * 5 features = 50 per agent
        # Shared state: 50 * 4 agents = 200
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 50},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 200},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        # Entropy annealing for exploration-exploitation tradeoff
        "entropy_loss_scale":       0.02,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.05,   # Higher initial exploration for traffic
        "entropy_loss_scale_end":   0.01,
        "debug_entropy_stats":      False,
        "value_loss_scale":         1.0,

        # KL divergence for policy stability
        "kl_threshold":       0.05,
        "kl_warmup_fraction": 0.3,
        "debug_kl_stats":     False,

        # Reward clipping: collision=-5, arrived=1, high_speed=1
        # Normalized rewards approximately in [-1, 1]
        "rewards_shaper":       lambda rewards, *args: jnp.clip(rewards, -5.0, 2.0),
        "time_limit_bootstrap": True,

        # Regularization and learning rate scheduling
        "weight_decay":            1e-4,
        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.3,
        "min_lr_fraction":         0.1,

        # MAGIC communication hyperparameters
        # Based on MAGIC paper (Niu et al., AAMAS 2021)
        "message_dim":        64,     # Message embedding dimension
        "num_comm_rounds":    2,      # Number of communication rounds (GAT layers)
        "num_heads":          4,      # Number of attention heads in GAT
        "gumbel_temperature": 0.5,    # Temperature for Gumbel-Softmax (scheduler)
    },

    "policy": {
        # Network architecture - deeper for continuous control task
        "hidden_sizes":          [256, 256],
        "unnormalized_log_prob": True,
    },

    "value": {
        "hidden_sizes": [256, 256],
    },

    "memory": {
        "size": 4096,
    },
}
