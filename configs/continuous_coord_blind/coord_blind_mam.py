# fmt: off
"""CC Blind — MAM sequence policy.

Same partial-observation typed scenario as coord_blind_mappo.py. This is the
cleanest MAGComp coordination gap for MAM because private target sightings can
flow through the joint sequence model.

The MAPPO baseline uses 4096-step rollouts. For MAM that creates PPO
minibatches with 16,384 agent-groups and XLA materializes multi-GiB Mamba
transpose temporaries on the first update. Keep num_envs=16 for environment
throughput, but reduce the update window and increase minibatches so each MAM
minibatch has 2,048 groups instead.
"""
from configs.magcomp._mam import make_mam_config
from configs.magcomp.coord_blind_mappo import CONFIG as _BASE_CONFIG


CONFIG = make_mam_config(
    _BASE_CONFIG,
    name="magcomp_cc_blind_mam",
    tags=["magcomp", "cc_blind", "mam"],
    learning_rate=1.5e-4,
    lr_decay_start_fraction=0.05,
    entropy_start=0.03,
    entropy_end=0.005,
    kl_threshold=0.05,
    kl_warmup_fraction=0.0,
    d_state=32,
    delta_rank=16,
    rollouts=1024,
    learning_epochs=8,
    mini_batches=8,
    value_hidden_sizes=[256, 256],
)
