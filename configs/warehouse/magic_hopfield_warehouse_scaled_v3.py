# fmt: off
"""Scaled topology v3 — MAGIC + HopfieldSelfContext.

v2 audit (2026-05-05): magic_hopfield_v2 collapsed to last10_mean −138 / peak
max −27 (never reached positive territory) vs v1's marginal +12. Two confounds
relative to magic_v2 / magic_eh_v2:

  (a) Env disparity: magic_hopfield retains `enable_interference_zones=True`
      (treatment-radius 35 % stochastic-fail boost), depressing delivery
      yield further than the variants without interference.

  (b) Architectural over-correction at init: v2 set gate_init 0.0 (sigmoid
      0.5) AND β=2 simultaneously. With un-trained random LeCun-normal
      prototypes, the per-agent retrieved residual at init is a sharper-
      attended random mixture, perturbing messages at 50 % magnitude
      every forward pass. Compare EpisodicHopfield where retrieved=0 at
      t=0 (empty buffer) — that path self-bootstraps; HopfieldSelfContext
      doesn't.

v3 fixes (env config preserved for back-compat with older hopfield runs):

  * hopfield_gate_init  0.0 → −1.0    (sigmoid 0.27)
      Mid-point between v1's starvation gate (sigmoid 0.12) and v2's
      over-mix (0.5). Lets prototypes warm up with a smaller perturbation
      while still avoiding the gate-starvation failure mode flagged in
      memory/hopfield_integration_strategy_2026_04_30.md §9.
      (Plumbed end-to-end through magic_hopfield_mappo / generators /
      MAGICHopfieldPolicyNet / _CommBlockHopfieldSelf / HopfieldSelfContext.)

  * hopfield_beta       2.0 → 1.0
      Ramsauer 2021 §3.1 winner-take-most threshold β·||p||² > ln K
      assumes *trained* prototypes — at init with random patterns the
      sharper softmax produces noisier retrievals. β=1 is the safe
      default; if needed, future versions can make β learnable.

  * Same agent-side fixes as magic_v3 / magic_eh_v3:
        entropy_loss_scale  0.005 / 0.0005 → 0.02 / 0.005
        kl_warmup_fraction  0.3 → 0.1
        rewards_shaper      clip(-5, 20) → clip(-5, 15)
"""
import jax.numpy as jnp
from skrl.resources.preprocessors.jax import RunningStandardScaler

CONFIG = {
    "experiment": {
        "name":             "magic_hopfield_scaled_v3",
        "agent_type":       "magic_hopfield",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["magic_hopfield", "warehouse", "scaled12", "scenario5", "v3"],
        },
        "write_interval":      25_000,
        "checkpoint_interval": 1_000_000,
        "store_separately":    False,
    },

    "env": {
        "id":            "warehouse",
        "num_envs":      1,
        "grid_height":   24,
        "grid_width":    32,
        "num_agents":    12,
        "max_agents":    12,
        "num_shelves":   18,
        "resources_per_shelf": 4,
        "num_treatment_stations": 4,
        "num_goal_locations": 4,
        "treatment_duration": 5,
        "comm_noise_prob":    0.0,
        "vision_range":       2,
        "max_cycles":         500,
        "comm_range":         8,

        "no_comm":          True,
        "randomize_layout": True,

        "enable_task_deadlines":  True,
        "task_arrival_rate":      0.5,
        "task_deadline_min":      80,
        "task_deadline_max":      200,
        "max_pending_tasks":      20,
        "penalty_task_expired":   -1.0,
        "reward_urgent_delivery": 5.0,

        "enable_heterogeneous":   True,
        "agent_speed_options":    (1, 1, 2),
        "agent_capacity_options": (1, 2, 1),
        "agent_fragility_options":(1.0, 0.5, 2.0),

        "enable_interference_zones":    True,
        "interference_base":            0.05,
        "interference_treatment_boost": 0.35,
        "interference_radius":          2,

        "enable_battery":             True,
        "battery_capacity":           250,
        "battery_drain_per_step":     1,
        "battery_drain_idle":         0,
        "battery_charge_rate":        8,
        "battery_critical_threshold": 30,
        "num_charging_stations":      4,
        "reward_rescue_repair":       12.0,
        "reward_rescue_charge":       12.0,

        "agent_failure_prob": 0.001,

        "fault_profile": {
            "burst_attrition":     True,
            "burst_prob":          0.001,
            "correlated_failure":  False,
            "correlation_radius":  1,
            "load_dependent_comm": False,
            "base_packet_loss":    0.0,
            "congestion_factor":   0.0,
        },
    },

    "training": {"timesteps": 2_000_000, "seed": 42},
    "eval":     {"timesteps": 5_000, "checkpoint_path": None},
    "record":   {"timesteps": 2_000, "checkpoint_path": None,
                 "video_dir": "recordings", "fps": 10},

    "magic_hopfield": {
        "rollouts":        4096,
        "learning_epochs": 8,
        "mini_batches":    8,

        "discount_factor": 0.99,
        "lambda":          0.95,

        "learning_rate":                  2e-4,
        "learning_rate_scheduler":        None,
        "learning_rate_scheduler_kwargs": {},

        "state_preprocessor":                RunningStandardScaler,
        "state_preprocessor_kwargs":         {"size": 238},
        "shared_state_preprocessor":         RunningStandardScaler,
        "shared_state_preprocessor_kwargs":  {"size": 4704},
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
        "entropy_loss_scale_start": 0.02,    # 0.005 → 0.02
        "entropy_loss_scale_end":   0.005,   # 0.0005 → 0.005
        "value_loss_scale":         1.0,

        "kl_threshold":       0.05,
        "kl_warmup_fraction": 0.1,           # 0.3 → 0.1

        "rewards_shaper": lambda rewards, *args: jnp.clip(rewards, -5.0, 15.0),
        "time_limit_bootstrap": True,

        "weight_decay":            1e-4,
        "linear_lr_decay":         True,
        "lr_decay_start_fraction": 0.3,
        "min_lr_fraction":         0.1,

        "message_dim":        64,
        "num_comm_rounds":    3,
        "num_heads":          4,
        "gumbel_temperature":                 1.0,
        "gumbel_temperature_end":             0.5,
        "gumbel_temperature_anneal_fraction": 0.7,

        "hopfield_num_prototypes": 8,
        "hopfield_beta":           1.0,   # 2.0 → 1.0 (random-init safety)
        "hopfield_gate_init":      -1.0,  # 0.0 → -1.0 (gentle prototype warmup)

        "comm_reg_scale": 0.001,
    },

    "policy": {
        "hidden_sizes":          [256, 256],
        "unnormalized_log_prob": True,
    },

    "value":  {"hidden_sizes": [512, 512]},
    "memory": {"size": 4096},
}
