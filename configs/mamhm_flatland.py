# fmt: off
from flatland_hard import FLATLAND_HARD_ENV, flatland_dims
from skrl.resources.preprocessors.jax import RunningStandardScaler

ENV = dict(FLATLAND_HARD_ENV)
NUM_AGENTS = ENV["num_agents"]
TREE_DEPTH = ENV["tree_depth"]
OBS_DIM, STATE_DIM = flatland_dims(NUM_AGENTS, TREE_DEPTH)

CONFIG = {
    "experiment": {
        "name":             "mamhm_flatland_v0",
        "agent_type":       "mamhm",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mamhm", "flatland", "generic-hopfield"],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 250_000,
        "store_separately":    False,
    },
    "env": ENV,
    "training": {
        "timesteps": 14_000_000,
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
    "mamhm": {
        "rollouts":        256,
        "learning_epochs": 10,
        "mini_batches":    2,
        "discount_factor": 0.99,
        "lambda":          0.9,
        "learning_rate":                  1.5e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay":               True,
        "lr_decay_start_fraction":       0.05,
        "min_lr_fraction":               0.1,
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": OBS_DIM},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": STATE_DIM},
        "value_preprocessor":                RunningStandardScaler,
        "value_preprocessor_kwargs":         {"size": 1},
        "random_timesteps":      0,
        "learning_starts":       0,
        "grad_norm_clip":        0.5,
        "ratio_clip":            0.2,
        "value_clip":            0.2,
        "clip_predicted_values": True,
        "entropy_loss_scale":       0.01,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.03,
        "entropy_loss_scale_end":   0.01,
        "value_loss_scale":      0.5,
        "kl_threshold":          0,
        "rewards_shaper":        None,
        "time_limit_bootstrap":  True,
        "n_embd":      128,
        "n_block":     1,
        "d_state":     32,
        "d_conv":      4,
        "delta_rank":  16,
        # Flatland viability note:
        # current task/entity Hopfield modules are warehouse-shaped, so keep
        # only the generic post-decoder memory bank enabled here.
        "num_memories":              16,
        "memory_beta":               1.5,
        "memory_gamma":              0.25,
        "memory_gate_init":          -3.0,
        "memory_activation":         "softmax",
        "memory_use_pre_ln":         True,
        "diversity_loss_scale":      0.01,
        "use_task_hopfield":         False,
        "use_entity_hopfield":       False,
        "use_post_decoder_hopfield": True,
        "task_hopfield_num_heads":   4,
        "task_hopfield_beta":        2.0,
        "task_hopfield_gate_init":   -3.0,
        "use_structured_critic":     False,
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
