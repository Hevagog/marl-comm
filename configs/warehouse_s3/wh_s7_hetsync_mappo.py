# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

_ENV = {
    "id": "warehouse", "num_envs": 2,
    "grid_height": 18, "grid_width": 24,
    "num_agents": 8, "max_agents": 8,
    "num_shelves": 12, "resources_per_shelf": 4,
    "num_treatment_stations": 3, "num_goal_locations": 3,
    "treatment_duration": 5, "comm_noise_prob": 0.0,
    "vision_range": 2, "max_cycles": 500, "comm_range": 6,
    "no_comm": True, "randomize_layout": True,
    "enable_task_deadlines": True, "task_arrival_rate": 0.35,
    "task_deadline_min": 80, "task_deadline_max": 200,
    "max_pending_tasks": 12, "penalty_task_expired": -1.0,
    "reward_urgent_delivery": 5.0,
    # --- wider heterogeneity: 6 profiles, speed fixed at 1 ---
    # capacity {1,2,3} = LIVE role axis; fragility = observation-only (inert
    # while attrition is disabled, since it only scales the failure threshold).
    "enable_heterogeneous": True,
    "agent_speed_options": (1, 1, 1, 1, 1, 1),
    "agent_capacity_options": (1, 2, 3, 1, 2, 3),
    "agent_fragility_options": (1.0, 0.5, 2.0, 1.0, 0.5, 2.0),
    # --- rescue / attrition sub-task DISABLED ---
    "enable_attrition": False,
    "agent_failure_prob": 0.0,
    "reward_rescue_repair": 0.0, "reward_rescue_charge": 0.0,
    "reward_rescue_proximity": 0.0,
    # battery kept enabled with zero drain → obs dim unchanged, never strands.
    "enable_battery": True, "battery_capacity": 200,
    "battery_drain_per_step": 0, "battery_drain_idle": 0,
    "battery_charge_rate": 8, "battery_critical_threshold": 25,
    "num_charging_stations": 3,
    # --- S1 team-synchronised delivery reward (the learnable objective) ---
    "enable_team_delivery_bonus": True, "team_delivery_window": 20,
    "team_delivery_bonus": 5.0, "team_delivery_min_partners": 1,
    "fault_profile": {
        "burst_attrition": False, "burst_prob": 0.0,
        "correlated_failure": False, "correlation_radius": 1,
        "load_dependent_comm": False, "base_packet_loss": 0.0, "congestion_factor": 0.0,
    },
}

CONFIG = {
    "experiment": {
        "name": "magcomp_wh_s7_mappo", "agent_type": "mappo",
        "directory": "runs", "wandb": True,
        "wandb_kwargs": {"project": "marl-comm", "tags": ["magcomp", "wh_s7", "mappo"]},
        "write_interval": 25_000, "checkpoint_interval": 1_000_000, "store_separately": False,
    },
    "env": _ENV,
    "training": {"timesteps": 2_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 2_000, "checkpoint_path": None, "video_dir": "recordings", "fps": 10},
    "mappo": {
        "rollouts": 4096, "learning_epochs": 10, "mini_batches": 2,
        "discount_factor": 0.99, "lambda": 0.95, "learning_rate": 3e-4,
        "learning_rate_scheduler": None, "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay": True, "lr_decay_start_fraction": 0.3, "min_lr_fraction": 0.1,
        "state_preprocessor":               RunningStandardScaler,
        "state_preprocessor_kwargs":        {"size": 214},
        "shared_state_preprocessor":        RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": 2656},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},
        "random_timesteps": 0, "learning_starts": 0,
        "grad_norm_clip": 0.5, "ratio_clip": 0.2, "value_clip": 0.2,
        "clip_predicted_values": False,
        "entropy_loss_scale": 0.02, "entropy_annealing": True,
        "entropy_loss_scale_start": 0.05, "entropy_loss_scale_end": 0.01,
        "value_loss_scale": 1.0, "kl_threshold": 0.05, "kl_warmup_fraction": 0.3,
        "rewards_shaper": lambda rewards, *_: jnp.clip(rewards, -5.0, 40.0),
        "time_limit_bootstrap": True, "weight_decay": 1e-4,
    },
    "policy": {"hidden_sizes": [256, 128], "unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [256, 128]},
    "memory": {"size": 4096},
}
