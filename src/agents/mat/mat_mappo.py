"""MAT-enhanced MAPPO agent.

Thin subclass of CommFormerMAPPO that reuses the same group-batched act()
and interleaved-buffer shuffle logic.  MAT shares the same CTDE batching
requirements: all N agents' observations must be grouped together so the
transformer encoder-decoder sees a complete agent group.

The only behavioral difference from CommFormerMAPPO is that MATPolicyNet
does not produce ``adj_matrices`` in its output dict (no comm graph), so
the graph-key special-casing in CommFormerMAPPO is silently irrelevant.
All other output keys (``encoder_out``, ``stddev``, ``net_output``) are
handled identically by the parent's output-splitting loop.
"""

from __future__ import annotations

import numpy as np

from agents.commformer.commformer_mappo import CommFormerMAPPO


class MATAgent(CommFormerMAPPO):
    """MAT agent — CommFormerMAPPO without the communication graph.

    Overrides _shuffle_buffer_indices to add an explicit assertion that
    buffer_size is divisible by num_agents (fix for CC-B01 in memory/bugs.md).
    A buffer size that is not a multiple of N would silently truncate samples
    and produce incomplete agent groups in the mini-batch.
    """

    def _shuffle_buffer_indices(self, buffer_size: int) -> np.ndarray:
        n = len(self.possible_agents)
        assert buffer_size % n == 0, (
            f"MATAgent._shuffle_buffer_indices: buffer_size={buffer_size} is not "
            f"divisible by num_agents={n}.  Adjust rollouts or memory size so that "
            f"rollouts * num_envs is a multiple of {n}."
        )
        return super()._shuffle_buffer_indices(buffer_size)
