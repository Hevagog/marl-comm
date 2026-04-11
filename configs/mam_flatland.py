# fmt: off
from configs.flatland_basic import FLATLAND_BASIC_ENV, flatland_dims
from skrl.resources.preprocessors.jax import RunningStandardScaler

# Start from the shared Flatland base env config, then apply the
# rescue-run overrides: dense reward shaping, in-place curriculum, longer
# rollouts, aggressive entropy schedule, stable value learning.
ENV = dict(FLATLAND_BASIC_ENV)
ENV.update({
    # With 256 vec envs × 8 agents × 64-step rollouts we sit around
    # ~2.4 GB on the training accelerator for the MAPPO buffers, which is
    # the sweet spot for single-GPU runs.
    "num_envs":                256,

    # Dense reward shaping — turned ON for the rescue run.  These four
    # knobs are consumed by ``FlatlandPettingZooEnv._shape_rewards``.
    "use_shaped_reward":       True,
    "progress_coeff":          0.1,    # +0.1 * (d_{t-1} - d_t)
    "step_penalty":            0.01,   # constant time pressure
    # Deadlock penalty reduced 1.0→0.3: in stage-0 with a random policy all
    # 8 agents deadlock immediately every episode, so a 1.0 penalty floods
    # the critic with a large one-shot noise that overwhelms the 0.1-scale
    # progress signal.  0.3 keeps the penalty meaningful without dominating.
    "deadlock_penalty":        0.3,
    "completion_bonus":        1.0,    # terminal anchor on arrival
    "progress_clip":           1.0,    # cap per-step progress delta
})

NUM_AGENTS = ENV["num_agents"]
TREE_DEPTH = ENV["tree_depth"]
OBS_DIM, STATE_DIM = flatland_dims(NUM_AGENTS, TREE_DEPTH)

CONFIG = {
    "experiment": {
        "name":             "mam_flatland_v1_rescue",
        "agent_type":       "mam",
        "directory":        "runs",
        "wandb":            True,
        "wandb_kwargs": {
            "project": "marl-comm",
            "tags":    ["mam", "flatland", "rescue", "curriculum"],
        },
        "write_interval":      1_000,
        "checkpoint_interval": 25_000,
        "store_separately":    False,
    },
    "env": ENV,
    "training": {
        "timesteps": 500_000,
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
    "mam": {
        # ---- Rollout + GAE ----
        # Rollout 25→64: ~6 effective GAE horizons with γλ=0.99*0.95.
        # The old 25-step window gave the critic zero credit propagation
        # in a reward signal that is almost exclusively -1 per step.
        "rollouts":        64,
        "learning_epochs": 8,
        "mini_batches":    4,       # keep per-batch ~32k samples
        "discount_factor": 0.99,
        "lambda":          0.95,    # raised with the longer horizon

        # ---- Learning rate ----
        "learning_rate":            1.5e-4,
        "linear_lr_decay":          True,
        "lr_decay_start_fraction":  0.05,
        "min_lr_fraction":          0.1,

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
        "grad_norm_clip":         0.5,     # Mamba selective-scan gradients spike — keep tight
        "ratio_clip":             0.2,
        "value_clip":             0.2,
        "clip_predicted_values":  True,
        "kl_threshold":           0.0,
        "kl_warmup_fraction":     0.2,     # no KL early-stop for first 20% of training

        # ---- Entropy schedule (exploration pressure) ----
        # The old 0.03→0.01 schedule was too aggressive given that entropy
        # at log(5)=1.609 was collapsing by pure gradient-noise floor.
        # Start at 0.05 and floor at 0.02 to keep off-dominant actions
        # alive through the curriculum transitions.
        "entropy_loss_scale":        0.05,
        "entropy_annealing":         True,
        "entropy_loss_scale_start":  0.05,
        "entropy_loss_scale_end":    0.02,

        # ---- Value stabilisation ----
        # Scale 0.5→1.0 (MAPPO paper default) + Huber loss with δ=1.0.
        # Huber caps per-sample gradient at |δ| so the first 1–2M
        # transitions (worst reward variance under curriculum stage 0)
        # don't hand-grenade the critic.
        "value_loss_scale":       1.0,
        "use_huber_value_loss":   True,
        "huber_delta":            1.0,

        # ---- Misc ----
        "rewards_shaper":         None,    # shaping happens inside the env
        "time_limit_bootstrap":   True,

        # ---- MAM architecture (unchanged from baseline) ----
        "n_embd":                 128,
        "n_block":                1,
        "d_state":                32,
        "d_conv":                 4,
        "delta_rank":             16,
    },
    "policy": {
        "unnormalized_log_prob": True,
    },
    "value": {
        "hidden_sizes": [256, 128],
    },
    "memory": {
        # Must match mam.rollouts — this is the per-env ring-buffer depth.
        "size": 64,
    },
    "curriculum": {
        # In-place stage scheduler: keeps num_agents fixed and varies
        # grid size / city count / malfunctions.  Base stages are defined
        # in ``src/environments/flatland/curriculum.py``; override the
        # ``stages`` key here to customise.
        "enabled":            True,
        # 5 000-step segments: longer than 2 000 so each segment spans ~78
        # gradient updates (5000 / 64 rollout), giving the LR scheduler and
        # critic more time to converge before the curriculum check fires.
        "segment_timesteps":  5_000,
        # "stages": ...   # optional: tuple[CurriculumStage, ...]
    },
}
