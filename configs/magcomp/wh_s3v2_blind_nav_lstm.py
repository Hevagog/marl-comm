# fmt: off
"""S3-v2 Blind Navigation — MAGIC LSTM.

Textbook regime where recurrence is essential (Foerster 2016 DIAL).  With
hide_infra_obs=True the LSTM cell must store "heading to last sighted
treatment/goal cell" in hidden state.  Both temporal smoothing (noisy comms)
and spatial memory (GPS removed) demand recurrence — LSTM should open a larger
gap over Dense than in the original S3 where the GPS made memory redundant.
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
    "no_comm": True, "randomize_layout": True,
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
        "name": "magcomp_wh_s3v2_lstm", "agent_type": "magic",
        "directory": "runs", "wandb": True,
        "wandb_kwargs": {"project": "marl-comm", "tags": ["magcomp", "wh_s3v2", "magic_lstm", "blind_nav"]},
        "write_interval": 25_000, "checkpoint_interval": 200_000, "store_separately": False,
    },
    "env": _ENV,
    "training": {"timesteps": 10_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 2_000, "checkpoint_path": None, "video_dir": "recordings", "fps": 10},
    "magic": {
        "rollouts": 4096, "learning_epochs": 8, "mini_batches": 4,
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
        "value_loss_scale": 1.0, "kl_threshold": 0.02, "kl_warmup_fraction": 0.3,
        "rewards_shaper": lambda rewards, *_: jnp.clip(rewards, -5.0, 30.0),
        "time_limit_bootstrap": True, "weight_decay": 1e-4,
        "message_dim": 64, "num_comm_rounds": 2, "num_heads": 2,
        "gumbel_temperature": 1.0, "gumbel_temperature_end": 0.5,
        "gumbel_temperature_anneal_fraction": 0.7,
        "comm_reg_scale": 0.001,
        "recurrent_type":        "lstm",
        "recurrent_hidden_size": 64,
    },
    "policy": {"hidden_sizes": [256, 256], "unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [256, 256]},
    "memory": {"size": 4096},
}
