# fmt: off
"""Coin Game Partial-Obs — MAM (Multi-Agent Mamba).

2 agents → AR chain length 2; minimal CrossMamba complexity.
obs_dim=13, state_dim=26.

Under partial observability MAM's BiMamba encoder must extract something
useful from a 13-dim vector where 6 of 11 feature dims (other agent xyz,
both coins xyz) may be zeroed out most of the time.  CrossMamba in the
decoder can still condition each agent's action on the partner's encoder
output, which is the "communication" channel in MAM.

Key hypothesis: does the bidirectional SSM learn to let the agent that sees
a coin flag it to the partner via the AR decoder channel?  If so, the
joint log-prob log π(a₁|enc(o), a₀) should show higher mutual information
between the flagging agent's encoder output and the partner's action
(go toward coin) than MAPPO.

Architecture scaled from v5b warehouse defaults:
  n_embd=64, n_block=1, d_state=32, d_conv=4, delta_rank=32
  (same as coingame_mam.py diagnostic — partial obs adds no architectural change)

Training budget: 2M steps (AR decoder is expensive; task is simpler than
warehouse even with partial obs — coin-only rewards, 2 agents, 4 actions).
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
        "name": "magcomp_coingame_partialobs_mam", "agent_type": "mam",
        "directory": "runs", "wandb": True,
        "wandb_kwargs": {"project": "marl-comm", "tags": ["magcomp", "coingame_partialobs", "mam"]},
        "write_interval": 10_000, "checkpoint_interval": 500_000, "store_separately": False,
    },
    "env": _ENV,
    "training": {"timesteps": 2_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 1_000, "checkpoint_path": None, "video_dir": "recordings", "fps": 4},
    "mam": {
        "rollouts": 256, "learning_epochs": 10, "mini_batches": 2,
        "discount_factor": 0.99, "lambda": 0.95,
        "learning_rate": 1.5e-4,
        "learning_rate_scheduler": None, "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay": True, "lr_decay_start_fraction": 0.3, "min_lr_fraction": 0.1,
        "update_state_preprocessor_in_update": False,
        "update_shared_state_preprocessor_in_update": False,
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
        "value_loss_scale": 1.0,
        "kl_threshold": 0.03, "kl_warmup_fraction": 0.0,
        "ratio_max_threshold": 3.0,
        "rewards_shaper": lambda rewards, *_: jnp.clip(rewards, -2.0, 1.0),
        "time_limit_bootstrap": True,
        "n_embd": 64, "n_block": 1,
        "d_state": 32, "d_conv": 4, "delta_rank": 32,
    },
    "policy": {"unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [128, 128]},
    "memory": {"size": 256},
}
