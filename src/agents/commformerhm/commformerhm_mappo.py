"""CommFormerHM-enhanced MAPPO agent.

Extends CommFormerMAPPO with the same act/shuffle logic.
The Task Hopfield memory augmentation is inside the policy network,
so the agent class is structurally identical to CommFormerMAPPO.
"""

from __future__ import annotations

from agents.commformer.commformer_mappo import CommFormerMAPPO


class CommFormerHMMAPPO(CommFormerMAPPO):
    """MAPPO agent with CommFormerHM's task-Hopfield-augmented policy.

    Inherits CommFormerMAPPO's act() method which stacks all agents'
    observations before calling the shared policy, and the grouped
    buffer shuffle for training.
    """

    ...
