"""SyncMixer MAPPO agent.

Identical to ``MAMEncOnlyMAPPO`` in that it stacks per-agent observations
before the shared policy and splits outputs back per agent; the encoder
and head differ (symmetric attention + Hopfield pooling instead of
BiMamba). Inherits the MAPPO training loop from ``CategoricalMAPPO``.
"""

from __future__ import annotations

from agents.mam.mam_enc_only_mappo import MAMEncOnlyMAPPO


class SyncMixerMAPPO(MAMEncOnlyMAPPO):
    pass
