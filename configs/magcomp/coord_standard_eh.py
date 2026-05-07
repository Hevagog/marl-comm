# fmt: off
"""CC Standard — MAGIC Episodic Hopfield.

Mirrors magic_eh_continuous_coord_v1: T=8 buffer, β=2.0, gate=0.0.
Pattern-completion of last-known peer position when peer drops out of vision.
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

_ENV = {
    "id": "continuous_coord", "num_envs": 16,
    "num_agents": 4, "max_cycles": 200,
    "max_targets": 3, "capture_radius": 0.08,
    "vision_range": 0.4, "collision_radius": 0.03,
    "target_arrival_rate": 0.15,
    "target_k_min": 2, "target_k_max": 3,
    "target_deadline_min": 30, "target_deadline_max": 80,
    "chain_event_prob": 0.2, "max_speed": 0.05,
}

CONFIG = {
    "experiment": {
        "name": "magcomp_cc_std_eh", "agent_type": "magic",
        "directory": "runs", "wandb": True,
        "wandb_kwargs": {"project": "marl-comm", "tags": ["magcomp", "cc_std", "magic_eh"]},
        "write_interval": 25_000, "checkpoint_interval": 500_000, "store_separately": False,
    },
    "env": _ENV,
    "training": {"timesteps": 2_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 1_000, "checkpoint_path": None, "video_dir": "recordings", "fps": 4},
    "magic": {
        "rollouts": 4096, "learning_epochs": 8, "mini_batches": 4,
        "discount_factor": 0.99, "lambda": 0.95, "learning_rate": 3e-4,
        "learning_rate_scheduler": None, "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay": True, "lr_decay_start_fraction": 0.2, "min_lr_fraction": 0.1,
        "state_preprocessor":               RunningStandardScaler,
        "state_preprocessor_kwargs":        {"size": 30},
        "shared_state_preprocessor":        RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": 28},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},
        "random_timesteps": 0, "learning_starts": 0,
        "grad_norm_clip": 0.5, "ratio_clip": 0.2, "value_clip": 0.2,
        "clip_predicted_values": False,
        "entropy_loss_scale": 0.02, "entropy_annealing": True,
        "entropy_loss_scale_start": 0.05, "entropy_loss_scale_end": 0.015,
        "value_loss_scale": 1.0, "kl_threshold": 0.05, "kl_warmup_fraction": 0.1,
        "rewards_shaper": lambda rewards, *_: jnp.clip(rewards, -5.0, 15.0),
        "time_limit_bootstrap": True, "weight_decay": 1e-4,
        "message_dim": 32, "num_comm_rounds": 2, "num_heads": 2,
        "gumbel_temperature": 1.0, "gumbel_temperature_end": 0.5,
        "gumbel_temperature_anneal_fraction": 0.7,
        "comm_reg_scale": 0.001,
        "recurrent_type":        "hopfield",
        "recurrent_hidden_size": 32,
        "episodic_buffer_size":  8,
        "episodic_beta_init":    2.0,
        "episodic_gate_init":    0.0,
    },
    "policy": {"hidden_sizes": [128, 128], "unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [256, 256]},
    "memory": {"size": 4096},
}
