# fmt: off
from skrl.resources.preprocessors.jax import RunningStandardScaler  # noqa: E402

CONFIG = {
    "experiment": {
        "name":             "mamhm_warehouse_v6",
        "agent_type":       "mamhm",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mamhm", "warehouse", "v6-adaptive-gate-diversity"],
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
        # v4→v5: Two architectural fixes for gradient starvation.
        #
        # v4 root cause: scalar gate barely opened (0.12, 1.2% contribution),
        # causing only 2/32 prototypes to activate (winner-take-all collapse).
        # The memory bank was functionally a learned constant bias.
        #
        # v5 fixes:
        # 1. Input-dependent gate: Dense(1)(h_ln) replaces scalar gate_logit.
        #    Gate now varies per agent per timestep with strong gradient flow.
        # 2. Prototype diversity loss: cosine similarity penalty directly
        #    drives prototype specialisation without relying on attenuated
        #    policy gradients through the gate.
        # 3. Softer beta (1.5 vs 3.0) prevents winner-take-all collapse.
        # 4. Fewer prototypes (16) with diversity loss → better specialisation.
        "num_memories":         16,
        # Softer retrieval — let diversity loss drive specialization rather
        # than forcing it via high temperature.
        "memory_beta":          1.5,
        # Max contribution scale. Effective initial noise = sigmoid(gate_init) * gamma.
        # v4 effective: sigmoid(-2.2) * 0.15 = 1.8%. Target ≤ 2% to not disrupt
        # MAM backbone bootstrap. At gate_init=-3.0: sigmoid(-3)*0.25 = 1.2%. ✓
        # NOTE: Do NOT set gamma=0.25 with gate_init=-2.0 — that gives 3.0% initial
        # noise, which delayed v5 breakout by ~125k steps vs v4.
        "memory_gamma":         0.25,
        # sigmoid(-3.0) ≈ 0.047, initial contribution = 0.047 * 0.25 = 1.2%.
        # Matches v4's effective starting noise while keeping the adaptive gate.
        # The gate kernel (zero-init) will differentiate once training stabilises.
        # (v5 used -2.0 → 3.0% initial noise → ~125k slower breakout than v4.)
        "memory_gate_init":     -3.0,
        "memory_activation":    "softmax",
        "memory_use_pre_ln":    True,
        # Prototype diversity regularization — penalises cosine similarity
        # between prototype pairs to prevent collapse.
        "diversity_loss_scale": 0.01,

        # ---- Upstream Hopfield pooling (Change 1) ----
        # Task Hopfield: pools task-queue features before BiMamba encoder.
        # This replaces the post-decoder bank as the primary memory mechanism.
        "use_task_hopfield":        True,
        "use_entity_hopfield":      False,
        "use_post_decoder_hopfield": False,    # legacy path, off by default
        "task_hopfield_num_heads":  4,
        "task_hopfield_beta":       2.0,
        "task_hopfield_gate_init":  -3.0,

        # ---- Structured critic (Change 2) ----
        "use_structured_critic":    True,
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
