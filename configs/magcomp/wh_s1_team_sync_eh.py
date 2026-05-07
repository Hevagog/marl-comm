# fmt: off
"""S1 Team-sync — MAGIC Episodic Hopfield encoder (EH).

The EpisodicHopfieldEncoder replaces the LSTM/GRU cell with a rolling buffer
of the last T=16 obs_enc vectors plus a Hopfield-style β-scaled softmax
retrieval gate. Unlike LSTM/GRU which integrate state sequentially, the
Episodic Hopfield does content-addressable retrieval over recent observations:
the current obs_enc queries the buffer to retrieve the most similar past
embedding, weighted by β=2.0. The retrieved vector is gated into h_t.

For team-sync: if "I just delivered" produces a distinctive obs_enc pattern,
the buffer retrieves it reliably even several steps later, giving the policy
a persistent soft-memory of the last delivery event without relying on LSTM
gradient propagation. Carry width = T×H = 16×64 = 1024 (flat; side-buffer
plumbing unchanged from GRU/LSTM).
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

_ENV = {
    "id": "warehouse", "num_envs": 8,
    "grid_height": 12, "grid_width": 16,
    "num_agents": 4, "max_agents": 4,
    "num_shelves": 6, "resources_per_shelf": 4,
    "num_treatment_stations": 2, "num_goal_locations": 2,
    "treatment_duration": 5, "comm_noise_prob": 0.0,
    "vision_range": 2, "max_cycles": 500, "comm_range": 5,
    "no_comm": True, "randomize_layout": True,
    "enable_task_deadlines": True, "task_arrival_rate": 0.2,
    "task_deadline_min": 80, "task_deadline_max": 200,
    "max_pending_tasks": 8, "penalty_task_expired": -1.0,
    "reward_urgent_delivery": 5.0,
    "enable_heterogeneous": True,
    "agent_speed_options": (1, 1, 2), "agent_capacity_options": (1, 2, 1),
    "agent_fragility_options": (1.0, 0.5, 2.0),
    "enable_battery": True, "battery_capacity": 200,
    "battery_drain_per_step": 1, "battery_drain_idle": 0,
    "battery_charge_rate": 8, "battery_critical_threshold": 25,
    "num_charging_stations": 2,
    "reward_rescue_repair": 12.0, "reward_rescue_charge": 12.0,
    "reward_rescue_proximity": 0.2, "agent_failure_prob": 0.001,
    "enable_team_delivery_bonus": True, "team_delivery_window": 20,
    "team_delivery_bonus": 5.0, "team_delivery_min_partners": 1,
    "fault_profile": {
        "burst_attrition": True, "burst_prob": 0.001,
        "correlated_failure": False, "correlation_radius": 1,
        "load_dependent_comm": False, "base_packet_loss": 0.0, "congestion_factor": 0.0,
    },
}

CONFIG = {
    "experiment": {
        "name": "magcomp_wh_s1_eh", "agent_type": "magic",
        "directory": "runs", "wandb": True,
        "wandb_kwargs": {"project": "marl-comm", "tags": ["magcomp", "wh_s1", "magic_eh"]},
        "write_interval": 25_000, "checkpoint_interval": 200_000, "store_separately": False,
    },
    "env": _ENV,
    "training": {"timesteps": 2_000_000, "seed": 42},
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
        "entropy_loss_scale": 0.02, "entropy_annealing": True,
        "entropy_loss_scale_start": 0.05, "entropy_loss_scale_end": 0.015,
        "value_loss_scale": 1.0, "kl_threshold": 0.05, "kl_warmup_fraction": 0.3,
        "rewards_shaper": lambda rewards, *_: jnp.clip(rewards, -5.0, 40.0),
        "time_limit_bootstrap": True, "weight_decay": 1e-4,
        "message_dim": 64, "num_comm_rounds": 2, "num_heads": 2,
        "gumbel_temperature": 1.0, "gumbel_temperature_end": 0.5,
        "gumbel_temperature_anneal_fraction": 0.7,
        "comm_reg_scale": 0.001,
        "recurrent_type":        "hopfield",
        "recurrent_hidden_size": 64,
        "episodic_buffer_size":  16,   # T: rolling window of last 16 obs_enc vectors
        "episodic_beta_init":    2.0,  # Hopfield inverse temperature
        "episodic_gate_init":    0.0,  # gate starts open (not near-zero like CFHM)
    },
    "policy": {"hidden_sizes": [256, 256], "unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [256, 256]},
    "memory": {"size": 4096},
}
