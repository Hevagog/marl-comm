# fmt: off
from skrl.resources.preprocessors.jax import RunningStandardScaler  # noqa: E402

CONFIG = {
    "experiment": {
        "name":             "mamhm_warehouse_v4",
        "agent_type":       "mamhm",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mamhm", "warehouse", "v4-gated-hopfield"],
        },
        "write_interval":      25000,
        "checkpoint_interval": 200000,
        "store_separately":    False,
    },

    "env": {
        # Same env config as mam_warehouse_v4 for fair comparison
        "id":            "warehouse",
        # v2→v3: reverted to 8 (same as MAM v4). The v2 increase to 16
        # combined with lr=2.5e-4 caused value function instability.
        "num_envs":      8,
        "grid_height":   12,
        "grid_width":    16,
        "num_agents":    4,
        "max_agents":    4,
        "num_shelves":   6,
        "resources_per_shelf": 4,
        "num_treatment_stations": 2,
        "num_goal_locations": 2,
        "treatment_duration": 5,
        "comm_noise_prob":        0.0,
        "vision_range":           2,
        "max_cycles":             500,
        "comm_range":             5,

        "enable_task_deadlines":  True,
        "task_arrival_rate":      0.2,
        "task_deadline_min":      80,
        "task_deadline_max":      200,
        "max_pending_tasks":      8,
        "penalty_task_expired":   -1.0,
        "reward_urgent_delivery": 5.0,

        "enable_heterogeneous":   True,
        "agent_speed_options":    (1, 1, 2),
        "agent_capacity_options": (1, 2, 1),
        "agent_fragility_options":(1.0, 0.5, 2.0),

        "enable_interference_zones":    True,
        "interference_base":            0.05,
        "interference_treatment_boost": 0.15,
        "interference_radius":          2,

        "enable_battery":             True,
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
        "timesteps":       5_000,
        "checkpoint_path": None,
    },

    "record": {
        "timesteps":       1_000,
        "checkpoint_path": None,
        "video_dir":       "recordings",
        "fps":             4,
    },

    # ---- MAMHM-specific PPO parameters ----
    "mamhm": {
        # v2→v3: reverted rollouts/mini_batches to match MAM v4.
        # v2's changes (128/4) combined with lr=2.5e-4 caused 2x more gradient
        # steps per rollout at a too-high lr, destabilizing value learning.
        "rollouts":        256,
        "learning_epochs": 10,
        "mini_batches":    2,

        "discount_factor": 0.99,
        "lambda":          0.9,

        # v2→v3: reverted to 1.5e-4 (same as MAM v4). The 2.5e-4 in v2
        # caused value loss to remain at ~0.08-0.12 (vs v1's stable 0.015),
        # producing noisy advantages that prevented policy convergence.
        "learning_rate":                  1.5e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "linear_lr_decay":          True,
        "lr_decay_start_fraction":  0.05,
        "min_lr_fraction":          0.1,

        # Preprocessors (same as MAM)
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
        "clip_predicted_values":  True,

        # v2→v3: reverted entropy_start to 0.03 (same as MAM v4).
        # The learnable gate makes the extra entropy buffer unnecessary.
        "entropy_loss_scale":       0.01,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.03,
        "entropy_loss_scale_end":   0.01,
        "value_loss_scale":   0.5,

        "kl_threshold": 0,

        "rewards_shaper":       None,
        "time_limit_bootstrap": True,

        # ---- MAM Mamba architecture hyperparameters (same as MAM v4) ----
        "n_embd":      128,
        "n_block":     1,
        "d_state":     32,
        "d_conv":      4,
        "delta_rank":  16,

        # ---- Hopfield Memory Bank hyperparameters ----
        # v2→v3: Major architectural fix — learnable gate + W_v projection.
        #
        # v2 regression root cause: gamma=0.15 + beta=4.0 on random-init
        # memory patterns injected ~15% noise into the decoder output,
        # preventing the MAM backbone from learning (reward stuck at -60).
        #
        # v3 fixes:
        # 1. Keep the gate small but not effectively frozen at init.
        #    The saved v3 eval plots show near-uniform memory usage
        #    (effective K≈63/64) and a gate that barely moved from init,
        #    so we slightly raise the initial contribution to improve signal.
        # 2. Separate W_v projection (per Ramsauer et al. Fig. 5) decouples
        #    key lookup direction from retrieved content.
        # 3. Post-memory LayerNorm stabilizes output scale.
        # Fewer slots encourage prototype reuse/specialisation on warehouse,
        # instead of spreading attention almost uniformly over 64 slots.
        "num_memories":         32,
        # Slightly sharper retrieval than v3 to move away from the global-
        # averaging regime seen in the saved attention-entropy plots.
        "memory_beta":          3.0,
        # Raise the max residual a bit so useful memories can matter once
        # learned, while remaining far below the ungated v2 regime.
        "memory_gamma":         0.15,
        # sigmoid(-2.2)≈0.10, so the initial effective contribution is about
        # 1.5% (0.10 * 0.15): still small, but no longer so tiny that the
        # memory path struggles to get gradient on warehouse.
        "memory_gate_init":     -2.2,
        "memory_activation":    "softmax",
        "memory_use_pre_ln":    True,
    },

    "policy": {
        "unnormalized_log_prob": True,
    },

    "value": {
        "hidden_sizes": [256, 128],
    },

    "memory": {
        # Must match rollout size.
        "size": 256,
    },
}
