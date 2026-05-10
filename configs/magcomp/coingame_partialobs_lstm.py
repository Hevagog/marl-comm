# fmt: off
"""Coin Game Partial-Obs — MAGIC LSTM.

LSTM cell sits between the obs encoder and the message encoder, maintaining
a hidden carry that persists across timesteps (TBPTT(1)).

Under partial observability the LSTM has a concrete job: track the inferred
partner position across steps when the partner drifts out of the vision cone.
An agent that last saw the partner at (3,4) two steps ago can extrapolate
where they might be now.  The LSTM hidden state also tracks whether a steal
event occurred recently, which is useful for tit-for-tat-style retaliation.

recurrent_hidden_size=32 matches message_dim so there is no fan-in inflation
at the message encoder.
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

_ENV = {
    "id": "coingame-partialobs", "num_envs": 16,
    "grid_size": 7, "max_cycles": 100,
    "vision_range": 2,
    "pick_reward": 1.0, "steal_penalty": -2.0,
}

CONFIG = {
    "experiment": {
        "name": "magcomp_coingame_partialobs_lstm", "agent_type": "magic",
        "directory": "runs", "wandb": True,
        "wandb_kwargs": {"project": "marl-comm", "tags": ["magcomp", "coingame_partialobs", "magic_lstm"]},
        "write_interval": "auto", "checkpoint_interval": "auto", "store_separately": False,
    },
    "env": _ENV,
    "training": {"timesteps": 5_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 1_000, "checkpoint_path": None, "video_dir": "recordings", "fps": 4},
    "magic": {
        "rollouts": 2048, "learning_epochs": 8, "mini_batches": 4,
        "discount_factor": 0.99, "lambda": 0.95, "learning_rate": 3e-4,
        "learning_rate_scheduler": None, "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay": True, "lr_decay_start_fraction": 0.3, "min_lr_fraction": 0.1,
        "state_preprocessor":               RunningStandardScaler,
        "state_preprocessor_kwargs":        {"size": 13},
        "shared_state_preprocessor":        RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": 26},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},
        "random_timesteps": 0, "learning_starts": 0,
        "grad_norm_clip": 0.5, "ratio_clip": 0.2, "value_clip": 0.2,
        "clip_predicted_values": False,
        "entropy_loss_scale": 0.02, "entropy_annealing": True,
        "entropy_loss_scale_start": 0.10, "entropy_loss_scale_end": 0.02,
        "value_loss_scale": 1.0, "kl_threshold": 0.05, "kl_warmup_fraction": 0.3,
        "rewards_shaper": lambda rewards, *_: jnp.clip(rewards, -2.0, 1.0),
        "time_limit_bootstrap": True, "weight_decay": 1e-4,
        "message_dim": 32, "num_comm_rounds": 1, "num_heads": 1,
        "gumbel_temperature": 1.0, "gumbel_temperature_end": 0.5,
        "gumbel_temperature_anneal_fraction": 0.7,
        "comm_reg_scale": 0.001,
        "recurrent_type":        "lstm",
        "recurrent_hidden_size": 32,
    },
    "policy": {"hidden_sizes": [128, 128], "unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [128, 128]},
    "memory": {"size": 2048},
}
