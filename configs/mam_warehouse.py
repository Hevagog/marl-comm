# fmt: off
from skrl.resources.preprocessors.jax import RunningStandardScaler  # noqa: E402

CONFIG = {
    "experiment": {
        "name":             "mam_warehouse_v4",
        "agent_type":       "mam",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mam", "warehouse", "v4-hyperfix"],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 200000,
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
        # v3→v4: 8M→20M. MAM paper (Daniel et al. 2024) trains for 20M timesteps with 64
        # vectorised envs. With num_envs=1 our effective sample diversity is ~64× lower;
        # a minimum budget of 20M compensates partially for the single-env constraint.
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

    # ---- MAM-specific PPO parameters ----
    "mam": {
        # PPO parameters
        # v3→v4: rollouts 4096→256. Paper default is 128 steps/env. With num_envs=1,
        # a 4096-step rollout means all 4096 transitions come from a single correlated
        # trajectory, severely violating the i.i.d. assumption PPO relies on. 256 steps
        # gives faster updates (78k updates over 20M steps vs 1.9k with 4096) and far
        # less within-batch correlation while staying above the paper's per-env default.
        "rollouts":        256,
        # v3→v4: epochs 8→10. With shorter rollouts we extract less data per update cycle;
        # more SGD passes over each batch compensates. Paper search space includes 10.
        "learning_epochs": 10,
        # v3→v4: mini_batches 4→2. With rollout=256, four mini-batches would give
        # 64 samples/batch — too small for stable gradient estimates with Mamba's
        # sequential processing. Two mini-batches = 128/batch, matching the paper's
        # effective per-mini-batch size (128 * 64 envs / 64 = 128).
        "mini_batches":    2,

        "discount_factor": 0.99,      # γ
        # v3→v4: λ 0.95→0.9. Paper default (Table 4). The pick→treat→deliver chain spans
        # ≥15 steps; λ=0.95 propagates credit over 20+ steps on average, amplifying noise
        # from the dense task-expiration penalties. λ=0.9 reduces effective horizon (≈10
        # steps) and cuts variance while keeping enough look-ahead for sequential sub-tasks.
        "lambda":          0.9,

        "learning_rate":                  1.5e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        # v3→v4: lr_decay_start_fraction 0.2→0.05. With 20M steps, 0.2 delays decay until
        # 4M steps — too conservative. Starting decay at 1M steps (5%) keeps the warm
        # constant phase short and allows the LR to fall as the policy begins improving.
        "linear_lr_decay":          True,
        "lr_decay_start_fraction":  0.05,
        "min_lr_fraction":          0.1,   # decay to 10% of initial LR

        # obs_dim: 7+8+(5*5*6)+(3*6)+3+1+3 = 190 (vision=2, max_agents=4, battery+hetero+tasks)
        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 190},
        # state_dim: 12*16*6 + 4*8 = 1152+32 = 1184 (overridden by _sync_preprocessor_sizes)
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 1184},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        # v3→v4: grad_norm_clip 1.0→0.5. Paper search space lower bound is 0.5. Mamba's
        # selective-scan exp(Δ·A) terms can produce large gradient spikes; 0.5 provides a
        # tighter ceiling than 1.0 without being as restrictive as the 0.1 often used for
        # vanilla RNNs. This proved important in the v3 run where value loss spiked early.
        "grad_norm_clip":         0.5,
        "ratio_clip":             0.2,    # PPO clip ε
        "value_clip":             0.2,
        "clip_predicted_values":  True,

        # Entropy annealing: paper uses fixed 0.01 but warehouse needs broad exploration
        # early (many sub-tasks, sparse delivery signals). Keep annealing but tighten range
        # given that v3 stayed near-random for 1.68M steps — aggressive entropy didn't help.
        "entropy_loss_scale":       0.01,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.03,   # v3→v4: 0.05→0.03; slightly less aggressive
        "entropy_loss_scale_end":   0.01,
        "value_loss_scale":   0.5,

        "kl_threshold": 0,

        "rewards_shaper":       None,
        "time_limit_bootstrap": True,

        # ---- MAM Mamba architecture hyperparameters ----
        # (Daniel et al. 2024, Table 4)
        "n_embd":      128,    # embedding dimension
        "n_block":     1,      # one Encoder + one Decoder block (paper default)

        # d_state (N): paper ablates {1,2,4,8,16,32}. 32 was selected as the default.
        # Paper notes "small-to-modest values stabilise training; larger values risk
        # adding unnecessary noise." Keep at 32 — it is the paper's own default.
        "d_state":     32,
        "d_conv":      4,      # 1D convolution kernel size (paper Table 4)
        # delta_rank: paper ablates {1,...,128}. Mamba design rule: ceil(d_model/16).
        # For d_model=128: ceil(128/16) = 8. We use 16 for slightly more capacity.
        # Unchanged from v3 — this was already correct.
        "delta_rank":  16,
    },

    "policy": {
        "unnormalized_log_prob": True,
    },

    "value": {
        # Reduced from [256,256]→[256,128]. The critic sees the full global state (1188-dim)
        # which is already highly informative; a tapered MLP reduces overfitting to the
        # current (initially poor) policy's value targets.
        "hidden_sizes": [256, 128],
    },

    "memory": {
        # v3→v4: 4096→256. Must match rollout size for the replay buffer to hold
        # exactly one rollout worth of transitions without truncation or padding.
        "size": 256,
    },
}
