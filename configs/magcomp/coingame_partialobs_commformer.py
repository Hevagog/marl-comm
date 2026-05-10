# fmt: off
"""Coin Game Partial-Obs — CommFormer.

CommFormer's learned static graph (sparsity=0.5, N=2 → k=1) reduces to a
single directed edge per agent, which here determines *whether* A always
listens to B's messages.  The key question under partial observability:
does the learned edge weight correlate with whether agent A can typically
provide useful location information to B?

Under partial obs, CommFormer's relation-enhanced attention can encode
*what channel is open* (the edge) and *what's on that channel* (encoder
output) separately.  A cooperative equilibrium: A encodes red coin position
when it sees it; B's decoder conditions on that message to avoid wasting
steps searching.

ratio_clip=0.05: same tight clip as full-obs CommFormer to slow down early
graph learning while the observation embeddings stabilise.

obs_dim=13, state_dim=26.
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
        "name": "magcomp_coingame_partialobs_commformer", "agent_type": "commformer",
        "directory": "runs", "wandb": True,
        "wandb_kwargs": {"project": "marl-comm", "tags": ["magcomp", "coingame_partialobs", "commformer"]},
        "write_interval": "auto", "checkpoint_interval": "auto", "store_separately": False,
    },
    "env": _ENV,
    "training": {"timesteps": 5_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 1_000, "checkpoint_path": None, "video_dir": "recordings", "fps": 4},
    "commformer": {
        "rollouts": 2048, "learning_epochs": 8, "mini_batches": 4,
        "discount_factor": 0.99, "lambda": 0.95,
        "learning_rate": 3e-4,
        "learning_rate_scheduler": None, "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay": True, "lr_decay_start_fraction": 0.3, "min_lr_fraction": 0.1,
        "state_preprocessor":               RunningStandardScaler,
        "state_preprocessor_kwargs":        {"size": 13},
        "update_state_preprocessor_in_update": False,
        "shared_state_preprocessor":        RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": 26},
        "update_shared_state_preprocessor_in_update": False,
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},
        "random_timesteps": 0, "learning_starts": 0,
        "grad_norm_clip": 0.5, "ratio_clip": 0.05, "value_clip": 0.2,
        "clip_predicted_values": False,
        "entropy_loss_scale": 0.02, "entropy_annealing": True,
        "entropy_loss_scale_start": 0.10, "entropy_loss_scale_end": 0.02,
        "value_loss_scale": 1.0,
        "kl_threshold": 0.05, "kl_warmup_fraction": 0.0,
        "rewards_shaper": lambda rewards, *_: jnp.clip(rewards, -2.0, 1.0),
        "time_limit_bootstrap": True, "weight_decay": 1e-4,
        "comm_reg_scale": 0.001,
        "hidden_dim": 64, "num_blocks": 1, "num_heads": 1,
        "head_dim": 64, "mlp_dim": 128, "sparsity": 0.5,
    },
    "policy": {"unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [128, 128]},
    "memory": {"size": 2048},
}
