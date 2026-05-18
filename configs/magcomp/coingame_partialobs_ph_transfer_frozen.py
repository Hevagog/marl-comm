# fmt: off
"""Phase 2 — HSC frozen-prototype transfer to standard (selfish) reward.

USAGE:
    python -m cli --config configs/magcomp/coingame_partialobs_ph_transfer_frozen.py \
        --task train \
        --resume runs/magcomp_coingame_partialobs_ph_altruistic/checkpoints/agent_5000000.pt

Transfer condition: frozen_proto (symmetric pairing).
  - Prototypes bank is frozen via stop_gradient (hopfield_freeze_prototypes=True).
  - Policy head, update gate, and value network adapt freely.
  - Both agents loaded from the same Phase 1 (altruistic) checkpoint.
  - Environment switches to standard reward (social_welfare_alpha=0.0).

Hypothesis: frozen prototype geometry preserves cooperative attractor basins
despite the reward signal no longer enforcing altruism.  The update gate can
still route the policy state toward cooperative attractors because they remain
the dominant fixed points in the prototype space.

Expected result: cooperation rate decays more slowly than full_finetune,
because policy optimization cannot reshape the attractor landscape — it can
only learn which attractors to select given the new reward.

Compared against:
  full_finetune — attractor geometry co-adapts with policy
  GRU full_finetune — no discrete attractors; inertia from continuous GRU state
  MAPPO fine_tune — no memory; pure reward-following

NOTE: Reset value_preprocessor stats before fine-tuning to avoid stale
altruistic-reward scale calibration.  The RunningStandardScaler is re-fitted
from the first few rollouts in the standard env.
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

_ENV = {
    "id": "coingame-partialobs", "num_envs": 16,
    "grid_size": 7, "max_cycles": 100,
    "vision_range": 2,
    "pick_reward": 1.0, "steal_penalty": -2.0,
    "social_welfare_alpha": 0.0,  # Phase 2: standard selfish env
}

CONFIG = {
    "experiment": {
        "name": "magcomp_coingame_partialobs_ph_transfer_frozen", "agent_type": "magic",
        "directory": "runs", "wandb": True,
        "wandb_kwargs": {"project": "marl-comm", "tags": ["magcomp", "coingame_partialobs", "magic_ph", "transfer", "frozen_proto", "phase2", "symmetric"]},
        "write_interval": "auto", "checkpoint_interval": "auto", "store_separately": False,
    },
    "env": _ENV,
    "training": {"timesteps": 2_000_000, "seed": 42},  # shorter: measure decay trajectory
    "reset_value_preprocessor": True,  # recalibrate scaler after loading Phase-1 ckpt
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
        "entropy_loss_scale_start": 0.05, "entropy_loss_scale_end": 0.02,  # lower: already trained
        "value_loss_scale": 1.0, "kl_threshold": 0.05, "kl_warmup_fraction": 0.1,
        "rewards_shaper": lambda rewards, *_: jnp.clip(rewards, -2.0, 1.0),
        "time_limit_bootstrap": True, "weight_decay": 0.0,  # no AdamW shrink on frozen protos
        "message_dim": 32, "num_comm_rounds": 1, "num_heads": 1,
        "gumbel_temperature": 0.5,              # start at end value: already converged
        "gumbel_temperature_end": 0.5,
        "gumbel_temperature_anneal_fraction": 1.0,
        "comm_reg_scale": 0.001,
        "recurrent_type":          "hopfield_state",
        "recurrent_hidden_size":   32,
        "hopfield_num_prototypes": 16,
        "hopfield_beta_init":      1.0,
        "hopfield_gate_init":      0.0,
        "hopfield_freeze_prototypes": True,     # THE KEY: freeze attractor bank
    },
    "policy": {"hidden_sizes": [128, 128], "unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [128, 128]},
    "memory": {"size": 2048},
}
