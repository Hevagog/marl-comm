# fmt: off
from configs.magcomp._mam import make_mam_config
from configs.magcomp.wh_s5_scaled_mappo import CONFIG as _BASE_CONFIG


CONFIG = make_mam_config(
    _BASE_CONFIG,
    name="magcomp_wh_s5_mam_v3",
    tags=["magcomp", "wh_s5", "v3", "mam"],
    learning_rate=1.5e-4,
    lr_decay_start_fraction=0.2,
    entropy_start=0.03,
    entropy_end=0.005,
    kl_threshold=0.03,
    kl_warmup_fraction=0.0,
    d_state=64,
    delta_rank=64,
    learning_epochs=10,
    mini_batches=2,
    rollouts=256,
    value_hidden_sizes=[512, 256],
    reward_clip=(-5.0, 30.0),
    sort_agents_by_type=True,
    type_cycle_len=3,
)
