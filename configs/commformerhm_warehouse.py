# fmt: off
"""CommFormerHM warehouse config — v2: aligned with MAM v4 / MAMHM v5 PPO tuning.

v1 → v2 changelog (root cause: stock CommFormer hyperparameters):
  ── PPO ──
  • rollouts:       2048 → 256   (match MAM v4; 2048 single-env = massive correlation)
  • learning_epochs: 8 → 10      (compensate shorter rollouts with more SGD passes)
  • mini_batches:    4 → 2       (256/4=64 is too small; 256/2=128 matches MAM v4)
  • learning_rate:  3e-4 → 1.5e-4 (high lr + value_loss_scale=1.0 caused noisy advantages)
  • linear_lr_decay: NEW, True   (decay to 10% over training)
  • lambda:         0.95 → 0.9   (reduce credit horizon; 0.95 amplifies task-expiry noise)
  • entropy_loss_scale: 0.05 → 0.01 with annealing 0.03→0.01
  • value_loss_scale: 1.0 → 0.5  (bring policy loss back into balance)
  • clip_predicted_values: False → True
  • kl_threshold: 0.05 → 0      (disable KL early stopping; let PPO clip handle it)
  • rewards_shaper: clip → None  (reward clipping at [-5, 20] truncated large delivery signals)
  ── Environment ──
  • task_arrival_rate: 0.4 → 0.2 (v1's λ=0.4 generated ~200 tasks/ep, all expired → -400 penalty)
  • task_deadline_min: 50 → 80   (give agent time to discover pick→treat→deliver chain)
  • task_deadline_max: 120 → 200
  • max_pending_tasks: 15 → 8    (smaller queue = less expiry noise)
  • penalty_task_expired: -2.0 → -1.0
  • burst_prob: 0.0005 → 0.0002  (fewer hard resets during early learning)
  • battery_capacity: 160 → 200  (more headroom before battery death)
  • interference_treatment_boost: 0.35 → 0.15
  • num_envs: 1 → 8              (more sample diversity per rollout)
  ── Memory ──
  • size: 2048 → 256             (must match rollout size)
"""
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "commformerhm_warehouse_v4",
        "agent_type":       "commformerhm",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["commformerhm", "warehouse", "v4-jax-env"],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 200_000,
        "store_separately":    False,
    },

    "env": {
        # v3→v4: switched to warehouse-jax for ~2-4× throughput gain via
        # JIT-compiled pure-function steps. Render/analyze tasks auto-fall-back
        # to the vanilla numpy warehouse via cli.py.
        "id":            "warehouse-jax",
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
        "task_arrival_rate":      0.2,        # v1: 0.4 → too many expired tasks
        "task_deadline_min":      80,         # v1: 50 → too tight for early exploration
        "task_deadline_max":      200,        # v1: 120
        "max_pending_tasks":      8,          # v1: 15 → expiry noise drowned signals
        "penalty_task_expired":   -1.0,       # v1: -2.0
        "reward_urgent_delivery": 5.0,

        "enable_heterogeneous":   True,
        "agent_speed_options":    (1, 1, 2),
        "agent_capacity_options": (1, 2, 1),
        "agent_fragility_options":(1.0, 0.5, 2.0),

        "enable_interference_zones":    True,
        "interference_base":            0.05,
        "interference_treatment_boost": 0.15,  # v1: 0.35 → hurts coordination
        "interference_radius":          2,

        "enable_battery":             True,
        "battery_capacity":           200,     # v1: 160 → too tight
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
            "burst_prob":          0.0002,     # v1: 0.0005 → too many hard resets
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

    # ---- CommFormerHM PPO parameters (aligned with MAM v4 / MAMHM v5) ----
    "commformerhm": {
        "rollouts":        256,
        "learning_epochs": 10,
        "mini_batches":    2,

        "discount_factor": 0.99,
        "lambda":          0.9,

        "learning_rate":                  1.5e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        # Linear LR decay (matches MAM v4 / MAMHM v5)
        "linear_lr_decay":          True,
        "lr_decay_start_fraction":  0.05,
        "min_lr_fraction":          0.1,

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 190},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 1188},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        "random_timesteps": 0,
        "learning_starts":  0,

        "grad_norm_clip":         0.5,
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  True,

        # Entropy annealing (matches MAM v4 / MAMHM v5)
        "entropy_loss_scale":       0.01,
        "entropy_annealing":        True,
        "entropy_loss_scale_start": 0.03,
        "entropy_loss_scale_end":   0.01,
        "value_loss_scale":   0.5,

        "kl_threshold": 0,

        "rewards_shaper":       None,
        "time_limit_bootstrap": True,

        "weight_decay":       1e-4,
        # Mild adjacency-entropy regularization helps avoid graph collapse
        # while the communication topology is still learning.
        "comm_reg_scale":     0.001,

        # ---- CommFormer architecture ----
        "hidden_dim":  256,
        "num_blocks":  2,
        "num_heads":   4,
        "head_dim":    64,
        "mlp_dim":     512,
        "sparsity":    0.5,

        # The current warehouse observation exposes only a 3-scalar task
        # summary, not an explicit task set. Keep the CommFormerHM path
        # available in code, but disable it in the benchmark config until
        # task/entity tokens are surfaced more explicitly.
        "use_task_hopfield":        False,
        "task_hopfield_num_heads":  4,
        "task_hopfield_beta":       2.0,
        "task_hopfield_gate_init":  -3.0,

        "use_structured_critic":    True,
        "execution_mode": "ctde",
    },

    "policy": {
        "unnormalized_log_prob": True,
    },

    "value": {
        "hidden_sizes": [256, 128] ,
    },

    "memory": {
        "size": 256,
    },
}
