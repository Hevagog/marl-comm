# fmt: off
from configs.magcomp._mam import make_mam_config
from .wh_s7_hetsync_mappo import CONFIG as _BASE_CONFIG


CONFIG = make_mam_config(
    _BASE_CONFIG,
    name="magcomp_wh_s7_mam",
    tags=["magcomp", "wh_s7", "mam"],
    lr_decay_start_fraction=0.03,
    d_state=64,
    delta_rank=64,
    rollouts=1024,
    learning_epochs=8,
    mini_batches=8,
    value_hidden_sizes=[256, 256],
    sort_agents_by_type=True,
    type_cycle_len=6,
)
