# fmt: off
from flatland_basic import FLATLAND_BASIC_ENV, flatland_dims
from skrl.resources.preprocessors.jax import RunningStandardScaler

# MAMHM = MAM + generic post-decoder Hopfield memory bank.
# This config inherits the full mam_flatland rescue recipe (dense shaping,
# 64-step rollouts, in-place curriculum, 0.05→0.02 entropy schedule, Huber
# value loss) and layers the Hopfield-specific knobs on top.
ENV = dict(FLATLAND_BASIC_ENV)
ENV.update({
    # 256 vec envs × 8 agents × 64-step rollouts ≈ ~2.4 GB on-device for
    # the MAPPO buffers.  MAMHM adds a memory bank per agent head but its
    # footprint is dominated by the same rollout buffers, so the same
    # num_envs/rollout budget applies.
    "num_envs":                16,

    # Dense reward shaping — consumed by FlatlandPettingZooEnv._shape_rewards.
    "use_shaped_reward":       True,
    "progress_coeff":          0.1,
    "step_penalty":            0.01,
    "deadlock_penalty":        1.0,
    "completion_bonus":        1.0,
    "progress_clip":           1.0,
})

NUM_AGENTS = ENV["num_agents"]
TREE_DEPTH = ENV["tree_depth"]
OBS_DIM, STATE_DIM = flatland_dims(NUM_AGENTS, TREE_DEPTH)

CONFIG = {
    "experiment": {
        "name":             "mamhm_flatland_v1_rescue",
        "agent_type":       "mamhm",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mamhm", "flatland", "rescue", "curriculum", "generic-hopfield"],
        },
        "write_interval":      800,
        "checkpoint_interval": 100_000,
        "store_separately":    False,
    },
    "env": ENV,
    "training": {
        "timesteps": 1_000_000,
        "seed":      42,
    },
    "eval": {
        "timesteps":       2_000,
        "checkpoint_path": None,
    },
    "record": {
        "timesteps":       1_000,
        "checkpoint_path": None,
        "video_dir":       "recordings",
        "fps":             5,
    },
    "mamhm": {
        # ---- Rollout + GAE ----
        # Rollout 25→64: ~6 effective GAE horizons with γλ=0.99*0.95.
        # Flatland's cost-only reward gives near-zero credit propagation
        # in the old 25-step window; Hopfield retrieval also benefits from
        # longer temporal context for key/value construction.
        "rollouts":        64,
        "learning_epochs": 8,
        "mini_batches":    4,
        "discount_factor": 0.99,
        "lambda":          0.95,

        # ---- Learning rate ----
        "learning_rate":                 1.5e-4,
        "learning_rate_scheduler":       None,
        "learning_rate_scheduler_kwargs": {},
        "linear_lr_decay":               True,
        "lr_decay_start_fraction":       0.05,
        "min_lr_fraction":               0.1,

        # ---- Preprocessors ----
        "state_preprocessor":               RunningStandardScaler,
        "state_preprocessor_kwargs":        {"size": OBS_DIM},
        "shared_state_preprocessor":        RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": STATE_DIM},
        "value_preprocessor":               RunningStandardScaler,
        "value_preprocessor_kwargs":        {"size": 1},

        # ---- PPO trust region ----
        "random_timesteps":       0,
        "learning_starts":        0,
        "grad_norm_clip":         0.5,     # Mamba + Hopfield gradients — keep tight
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  True,
        "kl_threshold":           0.0,
        "kl_warmup_fraction":     0.2,

        # ---- Entropy schedule ----
        # Matches mam rescue: start at 0.05, floor at 0.02 so off-dominant
        # actions stay alive through curriculum transitions.  Hopfield
        # retrieval is especially sensitive to early mode collapse because
        # a degenerate policy yields degenerate memory keys.
        "entropy_loss_scale":        0.05,
        "entropy_annealing":         True,
        "entropy_loss_scale_start":  0.05,
        "entropy_loss_scale_end":    0.02,

        # ---- Value stabilisation ----
        # Scale 0.5→1.0 + Huber δ=1.0.  Huber caps per-sample gradient at
        # |δ| so the first 1–2M transitions (worst reward variance under
        # curriculum stage 0) don't hand-grenade the critic.
        "value_loss_scale":       1.0,
        "use_huber_value_loss":   True,
        "huber_delta":            1.0,

        # ---- Misc ----
        "rewards_shaper":         None,    # shaping happens inside the env
        "time_limit_bootstrap":   True,

        # ---- MAM backbone (matches mam_flatland) ----
        "n_embd":                 128,
        "n_block":                1,
        "d_state":                32,
        "d_conv":                 4,
        "delta_rank":             16,

        # ---- Hopfield memory bank ----
        # Flatland viability note: the task/entity Hopfield modules are
        # warehouse-shaped, so only the generic post-decoder memory bank is
        # enabled here.  The bank is small (16 slots) to keep the diversity
        # regulariser meaningful at stage 0's narrow state distribution.
        "num_memories":              16,
        "memory_beta":               1.5,
        "memory_gamma":              0.25,
        "memory_gate_init":          -3.0,
        "memory_activation":         "softmax",
        "memory_use_pre_ln":         True,
        "diversity_loss_scale":      0.01,

        "use_task_hopfield":         False,
        "use_entity_hopfield":       False,
        "use_post_decoder_hopfield": True,
        "task_hopfield_num_heads":   4,
        "task_hopfield_beta":        2.0,
        "task_hopfield_gate_init":   -3.0,
        "use_structured_critic":     False,
    },
    "policy": {
        "unnormalized_log_prob": True,
    },
    "value": {
        "hidden_sizes": [256, 128],
    },
    "memory": {
        # Must match mamhm.rollouts — per-env ring-buffer depth.
        "size": 64,
    },
    "curriculum": {
        # In-place stage scheduler: keeps num_agents fixed and varies
        # grid size / city count / malfunctions.  Base stages are defined
        # in src/environments/flatland/curriculum.py.
        "enabled":            True,
        "segment_timesteps":  2_000,
        # "stages": ...   # optional: tuple[CurriculumStage, ...]
    },
}
