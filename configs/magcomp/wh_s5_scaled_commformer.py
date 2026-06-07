# fmt: off
"""S5 Scaled — CommFormer (12 agents, 24×32).

CommFormer's O(N²) static adjacency matrix grows to 144 edges at N=12.
sparsity=0.4 → k=round(0.4*12)=5 outgoing links per agent, balancing
connectivity vs bandwidth. num_heads=4 provides capacity for heterogeneous
role learning across 3 agent archetypes. num_blocks=2 covers graph diameter.

Note: num_envs=1 due to memory pressure at N=12 on a single GPU.

obs_dim = 238  (vision_range=2, max_agents=12, 3-archetype one-hot)
shared_state_dim = 24*32*6 + 12*8 = 4704
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

_ENV = {
    "id": "warehouse", "num_envs": 1,
    "grid_height": 24, "grid_width": 32,
    "num_agents": 12, "max_agents": 12,
    "num_shelves": 18, "resources_per_shelf": 4,
    "num_treatment_stations": 4, "num_goal_locations": 4,
    "treatment_duration": 5, "comm_noise_prob": 0.0,
    "vision_range": 2, "max_cycles": 500, "comm_range": 8,
    "no_comm": True, "randomize_layout": True,
    "enable_task_deadlines": True, "task_arrival_rate": 0.5,
    "task_deadline_min": 80, "task_deadline_max": 200,
    "max_pending_tasks": 20, "penalty_task_expired": -1.0,
    "reward_urgent_delivery": 5.0,
    "enable_heterogeneous": True,
    "agent_speed_options": (1, 1, 1),
    "agent_capacity_options": (1, 2, 1),
    "agent_fragility_options": (1.0, 0.5, 2.0),
    "enable_battery": True, "battery_capacity": 250,
    "battery_drain_per_step": 1, "battery_drain_idle": 0,
    "battery_charge_rate": 8, "battery_critical_threshold": 30,
    "num_charging_stations": 4,
    "reward_rescue_repair": 8.0, "reward_rescue_charge": 8.0,
    # v3 incentive rebalance (2026-06-07): delivery>>rescue + dense pick/treat
    "reward_delivery": 20.0, "reward_pick": 1.0, "reward_treatment_complete": 2.0,
    "reward_rescue_proximity": 0.2, "penalty_collision": -0.1,
    "agent_failure_prob": 0.001,
    "fault_profile": {
        "burst_attrition": True, "burst_prob": 0.001,
        "correlated_failure": False, "correlation_radius": 1,
        "load_dependent_comm": False, "base_packet_loss": 0.0, "congestion_factor": 0.0,
    },
}

CONFIG = {
    "experiment": {
        "name": "magcomp_wh_s5_commformer_v3", "agent_type": "commformer",
        "directory": "runs", "wandb": True,
        "wandb_kwargs": {"project": "marl-comm", "tags": ["magcomp", "wh_s5", "v3", "commformer"]},
        "write_interval": 25_000, "checkpoint_interval": 2_500_000, "store_separately": False,
    },
    "env": _ENV,
    "training": {"timesteps": 5_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 2_000, "checkpoint_path": None, "video_dir": "recordings", "fps": 10},
    "commformer": {
        "rollouts": 1024, "learning_epochs": 3, "mini_batches": 4,
        "discount_factor": 0.99, "lambda": 0.95,
        "learning_rate": 2e-4,
        "learning_rate_scheduler": None, "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay": True, "lr_decay_start_fraction": 0.2, "min_lr_fraction": 0.1,
        "state_preprocessor":               RunningStandardScaler,
        "state_preprocessor_kwargs":        {"size": 238},
        "update_state_preprocessor_in_update": False,
        "shared_state_preprocessor":        RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": 4704},
        "update_shared_state_preprocessor_in_update": False,
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},
        "random_timesteps": 0, "learning_starts": 0,
        "grad_norm_clip": 0.5, "ratio_clip": 0.2, "value_clip": 0.2,
        "clip_predicted_values": False,
        "entropy_loss_scale": 0.005, "entropy_annealing": True,
        "entropy_loss_scale_start": 0.005, "entropy_loss_scale_end": 0.001,
        "value_loss_scale": 1.0,
        "kl_threshold": 0.05, "kl_warmup_fraction": 0.0,
        "ratio_max_threshold": 2.0,
        "rewards_shaper": lambda rewards, *_: jnp.clip(rewards, -5.0, 30.0),
        "time_limit_bootstrap": True, "weight_decay": 1e-4,
        "comm_reg_scale": 0.002,
        # 12-agent topology: sparsity=0.4→k=5 links; 4 heads for heterogeneous role separation
        "hidden_dim": 128, "num_blocks": 2, "num_heads": 4,
        "head_dim": 64, "mlp_dim": 256, "sparsity": 0.4,
    },
    "policy": {"unnormalized_log_prob": True},
    "value":  {"hidden_sizes": [256, 128]},
    "memory": {"size": 1024},
}
