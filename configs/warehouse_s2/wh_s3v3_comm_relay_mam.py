# fmt: off
from configs.magcomp._mam import make_mam_config
from .wh_s3v3_comm_relay_mappo import CONFIG as _BASE_CONFIG


CONFIG = make_mam_config(
    _BASE_CONFIG,
    name="magcomp_wh_s3v3_mam",
    tags=["magcomp", "wh_s3v3", "mam", "comm_relay"],
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
# no_comm=False inherited from base; no override needed.
CONFIG["mam"]["grad_norm_clip"] = 0.3
