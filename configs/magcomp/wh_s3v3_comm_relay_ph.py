# fmt: off
"""S3-v3 Comm Relay — MAGIC Persistent Hopfield (PH).

no_comm=False: peer position/battery/carrying populated at comm_range=5.
MAGIC messages route peer vision patches. PH carry (K=16 prototype attractors)
provides compressed attractor dynamics for "last known infrastructure mode"
rather than EH's verbatim buffer — more robust under noisy comms because
prototype retrieval filters noise before the carry update.

Hyperparameters tuned from S3v3 Dense phase-transition observation:
Dense (FF, no recurrence) broke through at 0.72M with ratio_max_abs_dev≈25,
reward jumping −82 → +114.7. EH and LSTM plateaued at ≈−82 because their
learning_epochs=4 + grad_norm_clip=0.3 + kl_threshold=0.05 produced policy
updates too small to cross the cooperation phase transition.

PH's IS sensitivity is lower than LSTM/GRU because the carry update is
gate * softmax_retrieve(K, x) + (1−gate) * h_old — the new component depends
on current prototypes K (slow-drifting) and current obs, not on h_{t−1}
chained through time. Prototype drift per update batch is O(lr * grad_K) ≈
3e-4 * 0.5 * 16 steps ≈ 2.4e-3, small enough that IS drift stays bounded.
This allows more gradient steps per update than standard RNNs.

Changes vs original S3v3 PH config:
  kl_threshold:       0.02 → 0.05   (was terminating updates before large shifts)
  learning_epochs:    4    → 7      (between EH=4 and Dense=8; IS-safe for PH)
  grad_norm_clip:     0.3  → 0.5    (match Dense; attractor grads need headroom)
  ratio_max_threshold: 20.0 → 25.0  (Dense hit 24.87 on its phase transition)
  hopfield_beta_init: 0.5  → 1.0    (sharper prototype retrieval from start)
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

_ENV = {
    "id": "warehouse", "num_envs": 8,
    "grid_height": 12, "grid_width": 16,
    "num_agents": 4, "max_agents": 4,
    "num_shelves": 6, "resources_per_shelf": 4,
    "num_treatment_stations": 2, "num_goal_locations": 2,
    "treatment_duration": 5, "comm_noise_prob": 0.05,
    "vision_range": 2, "max_cycles": 500, "comm_range": 5,
    "no_comm": False, "randomize_layout": True,
    "hide_infra_obs": True,
    "enable_task_deadlines": True, "task_arrival_rate": 0.3,
    "task_deadline_min": 50, "task_deadline_max": 120,
    "max_pending_tasks": 15, "penalty_task_expired": -2.0,
    "reward_urgent_delivery": 5.0,
    "enable_heterogeneous": True,
    "agent_speed_options": (1, 1, 1, 1), "agent_capacity_options": (1, 1, 1, 1),
    "agent_fragility_options": (1.0, 1.0, 1.0, 1.0),
    "enable_interference_zones": True, "interference_base": 0.05,
    "interference_treatment_boost": 0.40, "interference_radius": 2,
    "enable_battery": True, "battery_capacity": 160,
    "battery_drain_per_step": 1, "battery_drain_idle": 0,
    "battery_charge_rate": 8, "battery_critical_threshold": 25,
    "num_charging_stations": 2,
    "reward_rescue_repair": 12.0, "reward_rescue_charge": 12.0,
    "agent_failure_prob": 0.0002,
    "fault_profile": {
        "burst_attrition": True, "burst_prob": 0.0005,
        "correlated_failure": False, "correlation_radius": 1,
        "load_dependent_comm": True,
        "base_packet_loss": 0.10, "congestion_factor": 0.05,
    },
}

CONFIG = {
    "experiment": {
        "name": "magcomp_wh_s3v3_ph", "agent_type": "magic",
        "directory": "runs", "wandb": True,
        "wandb_kwargs": {"project": "marl-comm", "tags": ["magcomp", "wh_s3v3", "magic_ph", "comm_relay"]},
        "write_interval": 25_000, "checkpoint_interval": 1_000_000, "store_separately": False,
    },
    "env": _ENV,
    "training": {"timesteps": 4_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 2_000, "checkpoint_path": None, "video_dir": "recordings", "fps": 10},
    "magic": {
        "rollouts": 4096, "learning_epochs": 7, "mini_batches": 4,
        "discount_factor": 0.99, "lambda": 0.95, "learning_rate": 3e-4,
        "learning_rate_scheduler": None, "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay": True, "lr_decay_start_fraction": 0.3, "min_lr_fraction": 0.1,
        "state_preprocessor":               RunningStandardScaler,
        "state_preprocessor_kwargs":        {"size": 190},
        "shared_state_preprocessor":        RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": 1188},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},
        "random_timesteps": 0, "learning_starts": 0,
        "grad_norm_clip": 0.5, "ratio_clip": 0.2, "value_clip": 0.2,
        "clip_predicted_values": False,
        "entropy_loss_scale": 0.025, "entropy_annealing": True,
        "entropy_loss_scale_start": 0.05, "entropy_loss_scale_end": 0.02,
        "value_loss_scale": 1.0, "kl_threshold": 0.05, "kl_warmup_fraction": 0.3,
        "ratio_max_threshold": 25.0,
        "rewards_shaper": lambda rewards, *_: jnp.clip(rewards, -5.0, 30.0),
        "time_limit_bootstrap": True, "weight_decay": 1e-4,
        "message_dim": 64, "num_comm_rounds": 2, "num_heads": 2,
        "gumbel_temperature": 1.0, "gumbel_temperature_end": 0.5,
        "gumbel_temperature_anneal_fraction": 0.7,
        "comm_reg_scale": 0.001,
        "recurrent_type":          "hopfield_state",
        "recurrent_hidden_size":   64,
        "hopfield_num_prototypes": 16,
        "hopfield_beta_init":      1.0,
        "hopfield_gate_init":      0.0,
    },
    "policy": {"hidden_sizes": [256, 256], "unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [256, 256]},
    "memory": {"size": 4096},
}
