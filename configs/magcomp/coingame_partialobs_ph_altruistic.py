# fmt: off
"""Coin Game Partial-Obs — MAGIC HSC trained with altruistic reward mixing.

Phase 1 of the Hopfield attractor transfer experiment.

Social welfare mixing: r_i ← (1−α)·r_i + α·r_partner, α=0.7.
Closer to the partner's outcome (α>0.5) biases the agent toward a
cooperative equilibrium while still allowing self-interested gradients.

Literature basis:
  Matsumura et al. 2024 "Active Inference With Empathy Mechanism for
  Socially Behaved Artificial Agents" (Artificial Life, MIT Press) —
  α-weighted other-reward causes empathic policy gradient.

  Demekas et al. 2023 (arXiv:2306.15494): cooperative equilibria
  emerge under symmetric empathy; defection is stable under asymmetric.

After training, the checkpoint contains:
  - Policy / value weights
  - Hopfield prototype bank (K=16 attractors, shape (K,H)=(16,32))
  - Running StandardScaler stats (value_preprocessor must be reset on transfer)

The prototype bank encodes learned coordination attractors under the
altruistic reward surface.  The transfer experiment tests whether these
attractors survive reward-shift (Phase 2).
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

_ENV = {
    "id": "coingame-partialobs", "num_envs": 16,
    "grid_size": 7, "max_cycles": 100,
    "vision_range": 2,
    "pick_reward": 1.0, "steal_penalty": -2.0,
    "social_welfare_alpha": 0.7,  # Phase 1: empathy-weighted reward mixing
}

CONFIG = {
    "experiment": {
        "name": "magcomp_coingame_partialobs_ph_altruistic", "agent_type": "magic",
        "directory": "runs", "wandb": True,
        "wandb_kwargs": {"project": "marl-comm", "tags": ["magcomp", "coingame_partialobs", "magic_ph", "altruistic", "phase1"]},
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
        "recurrent_type":          "hopfield_state",
        "recurrent_hidden_size":   32,
        "hopfield_num_prototypes": 16,
        "hopfield_beta_init":      1.0,
        "hopfield_gate_init":      0.0,
    },
    "policy": {"hidden_sizes": [128, 128], "unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [128, 128]},
    "memory": {"size": 2048},
}
