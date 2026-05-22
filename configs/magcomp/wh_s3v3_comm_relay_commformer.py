# fmt: off
"""S3-v3 Comm Relay — CommFormer.

no_comm=False: peer position/battery/carrying populated at comm_range=5.
CommFormer's learned static graph α routes compressed obs embeddings
(including peer vision patches) over all agents unconditionally. With
no_comm=False the graph has richer input state to condition on than S3v2.

Static topology is a structural disadvantage with randomize_layout=True:
α is trained to be a single topology averaged over all layouts, which
works poorly for any individual layout. MAGIC's dynamic attention adapts
per step; CommFormer cannot. This is an architecture limitation, not a
hyperparameter one — but instability can be reduced.

Hyperparameters updated after v1 gradient explosion (grad_norm 0.29–0.46 at
clip=0.5, reward oscillating −95 to −113 since 0.4M). Static α matrix
receives high-variance gradients from episode-to-episode layout variation;
with rollouts=1024 and a 4-agent environment, each mini-batch contains only
128 samples per agent, making the gradient estimate noisy.

Changes vs v1:
  grad_norm_clip:  0.5 → 0.3  (reduce gradient variance; was at clip limit)
  learning_epochs: 2   → 1    (4 gradient steps total vs 8; prevent α drift)

obs_dim=190, shared_state_dim=1188.
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
        "name": "magcomp_wh_s3v3_commformer", "agent_type": "commformer",
        "directory": "runs", "wandb": True,
        "wandb_kwargs": {"project": "marl-comm", "tags": ["magcomp", "wh_s3v3", "commformer", "comm_relay"]},
        "write_interval": 25_000, "checkpoint_interval": 1_000_000, "store_separately": False,
    },
    "env": _ENV,
    "training": {"timesteps": 4_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 2_000, "checkpoint_path": None, "video_dir": "recordings", "fps": 10},
    "commformer": {
        "rollouts": 1024, "learning_epochs": 1, "mini_batches": 4,
        "discount_factor": 0.99, "lambda": 0.95,
        "learning_rate": 3e-4,
        "learning_rate_scheduler": None, "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay": True, "lr_decay_start_fraction": 0.5, "min_lr_fraction": 0.1,
        "state_preprocessor":               RunningStandardScaler,
        "state_preprocessor_kwargs":        {"size": 190},
        "update_state_preprocessor_in_update": False,
        "shared_state_preprocessor":        RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": 1188},
        "update_shared_state_preprocessor_in_update": False,
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},
        "random_timesteps": 0, "learning_starts": 0,
        "grad_norm_clip": 0.3, "ratio_clip": 0.2, "value_clip": 0.2,
        "clip_predicted_values": False,
        "entropy_loss_scale": 0.025, "entropy_annealing": True,
        "entropy_loss_scale_start": 0.05, "entropy_loss_scale_end": 0.02,
        "value_loss_scale": 1.0,
        "kl_threshold": 0.05, "kl_warmup_fraction": 0.2,
        "ratio_max_threshold": 5.0,
        "rewards_shaper": lambda rewards, *_: jnp.clip(rewards, -5.0, 30.0),
        "time_limit_bootstrap": True, "weight_decay": 1e-4,
        "comm_reg_scale": 0.001,
        "hidden_dim": 128, "num_blocks": 2, "num_heads": 1,
        "head_dim": 64, "mlp_dim": 256, "sparsity": 0.5,
    },
    "policy": {"unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [128, 64]},
    "memory": {"size": 1024},
}
