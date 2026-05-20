# fmt: off
"""S3-v2 Blind Navigation — MAM sequence policy.

BiMamba encoder processes the agent sequence; CrossMamba conditions actions
on the observation sequence.  With hide_infra_obs=True the state space model
must encode "last sighted infrastructure direction" in its recurrent state —
an implicit memory demand that MAM's structured SSM should handle more
efficiently than MAPPO's stateless MLP.  No learned communication graph;
infrastructure sharing relies solely on the sequence ordering and implicit
state coupling through the shared reward.
"""
from configs.magcomp._mam import make_mam_config
from configs.magcomp.wh_s3v2_blind_nav_mappo import CONFIG as _BASE_CONFIG


CONFIG = make_mam_config(
    _BASE_CONFIG,
    name="magcomp_wh_s3v2_mam",
    tags=["magcomp", "wh_s3v2", "mam", "blind_nav"],
    learning_rate=1e-4,
    lr_decay_start_fraction=0.05,
    entropy_start=0.05,
    entropy_end=0.01,
    kl_threshold=0.05,
    kl_warmup_fraction=0.3,
    d_state=32,
    delta_rank=16,
    rollouts=1024,
    reward_clip=(-5.0, 30.0),
    sort_agents_by_type=False,
    type_cycle_len=4,
    learning_epochs=1,
)
# Comm agents use no_comm=True: advantage must come from learned sequence
# modelling, not free env-level radio features.  MAPPO base keeps no_comm=False
# as the upper-bound reference.
CONFIG["env"] = {**CONFIG["env"], "no_comm": True}
# Mamba exp(Δ·A) gradients spike without a tighter clip than the MAGIC default.
CONFIG["mam"]["grad_norm_clip"] = 0.3
