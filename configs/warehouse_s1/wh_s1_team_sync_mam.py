# fmt: off
from configs.magcomp._mam import make_mam_config
from .wh_s1_team_sync_mappo import CONFIG as _BASE_CONFIG


CONFIG = make_mam_config(
    _BASE_CONFIG,
    name="magcomp_wh_s1_mam",
    tags=["magcomp", "wh_s1", "mam"],
    lr_decay_start_fraction=0.03,
    d_state=32,
    delta_rank=32,
    rollouts=1024,
    learning_epochs=8,
    mini_batches=8,
    value_hidden_sizes=[256, 256],
)
