# fmt: off
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler


CONFIG = {
    "experiment": {
        "name":             "mappo_warehouse_simple",
        "agent_type":       "mappo",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mappo", "warehouse" ],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 200_000,
        "store_separately":    False,
    },

    "env": {
        "id":            "warehouse",
        "num_envs":      8,    # NOTE: increase to 8–16 for production runs; paper uses 64
        "grid_height":   12,
        "grid_width":    16,
        "num_agents":    4,
        "max_agents":    4,
        "num_shelves":   6,
        "resources_per_shelf": 4,
        "num_treatment_stations": 2,
        "num_goal_locations": 2,
        "treatment_duration": 5,
        "comm_noise_prob":        0.0,   # MAM handles via BiMamba encoder
        "vision_range":           2,     # 5×5 patch
        "max_cycles":             500,
        "comm_range":             5,

        "enable_task_deadlines":  True,
        # v3→v4: arrival_rate 0.4→0.2. At λ=0.4, a random policy generates ~200 tasks/episode,
        # nearly all expire (deadline 50-120 steps), producing -160 expired-task penalty that
        # completely drowns delivery/pick signals. Halving to 0.2 keeps task pressure meaningful
        # but allows other reward components to contribute gradient signal.
        "task_arrival_rate":      0.2,
        # v3→v4: widened from (50, 120) → (80, 200).
        # Minimum viable task completion is pick(~5 steps) + treatment(5) + transit(~5) ≈ 15 steps.
        # A 50-step deadline means <35 steps of slack — essentially impossible to exploit during
        # early exploration. 80 gives a reasonable discovery window; 200 gives the policy room
        # to learn before deadlines become the primary feedback signal.
        "task_deadline_min":      80,
        "task_deadline_max":      200,
        # v3→v4: 15→8. Smaller queue reduces the per-step expiration count when the policy is
        # poor, preventing the penalty from becoming a constant negative offset that masks returns.
        "max_pending_tasks":      8,
        # v3→v4: -2.0→-1.0. The expired-task penalty is shared across all active agents
        # (score_utils.py:71-73). At -2.0 with 100 expired tasks/episode it generates ~-50
        # per agent — still penalising non-delivery but no longer making delivery rewards invisible.
        "penalty_task_expired":   -1.0,
        "reward_urgent_delivery": 5.0,

        "enable_heterogeneous":   True,
        "agent_speed_options":    (1, 1, 2),
        "agent_capacity_options": (1, 2, 1),
        "agent_fragility_options":(1.0, 0.5, 2.0),

        "enable_interference_zones":    True,
        "interference_base":            0.05,
        # v3→v4: treatment_boost 0.35→0.15. Treatment stations are where agents most need to
        # coordinate (locking, sequencing). Keeping noise high there hurts exactly the
        # communication the BiMamba encoder is meant to exploit.
        "interference_treatment_boost": 0.15,
        "interference_radius":          2,

        "enable_battery":             True,
        # v3→v4: capacity 160→200. With drain_per_step=1, an agent can now survive 200 steps
        # before forced-charging. Battery death is a secondary sub-task; giving more headroom
        # lets the policy learn pick→treat→deliver before also solving charge management.
        "battery_capacity":           200,
        "battery_drain_per_step":     1,
        "battery_drain_idle":         0,
        "battery_charge_rate":        8,
        "battery_critical_threshold": 25,
        "num_charging_stations":      2,
        "reward_rescue_repair":       12.0,
        "reward_rescue_charge":       12.0,

        "agent_failure_prob": 0.0002,
        "fault_profile": {
            "burst_attrition":     True,
            # v3→v4: burst_prob 0.0005→0.0002. Burst failures permanently deactivate an agent,
            # truncating its trajectory and adding rescue sub-tasks. Lowering frequency reduces
            # these hard resets during early learning without removing the mechanic.
            "burst_prob":          0.0002,
            "correlated_failure":  False,
            "correlation_radius":  1,
            "load_dependent_comm": True,
            "base_packet_loss":    0.02,
            "congestion_factor":   0.05,
        },
    },

    "training": {
        "timesteps": 20_000_000,
        "seed":      42,
    },
    "eval": {
        "timesteps":       1_000,
        "checkpoint_path": None,
    },
    "record": {
        "timesteps":       1_000,
        "checkpoint_path": None,
        "video_dir":       "recordings",
        "fps":             4,
    },

    "mappo": {
        "rollouts":        256,
        "learning_epochs": 10,
        "mini_batches":    2,
        "discount_factor": 0.99,
        "lambda":          0.95,
        "learning_rate":                  3e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 190},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 1184},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  False,

        "entropy_loss_scale":       0.02,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.05,
        "entropy_loss_scale_end":   0.01,
        "debug_entropy_stats":      False,

        "value_loss_scale":         1.0,
        "kl_threshold":       0.05,
        "kl_warmup_fraction": 0.3,
        "debug_kl_stats":     False,

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 20.0),
        "time_limit_bootstrap": True,
        "weight_decay":            1e-4,
        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.3,
        "min_lr_fraction":         0.1,
    },

    "policy": {
        "hidden_sizes":          [256, 128],
        "unnormalized_log_prob": True,
        "use_memory":            False,
    },
    "value": {
        "hidden_sizes": [256, 128],
        "use_memory":   False,
    },
    "memory": {
        "size": 256,
    },
}
