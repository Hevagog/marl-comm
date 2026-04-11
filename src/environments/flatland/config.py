from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping


TREE_BRANCHING_FACTOR = 4
TREE_FEATURE_DIM = 12
FLATLAND_STATUS_DIM = 7


def flatland_tree_node_count(tree_depth: int) -> int:
    """Return the full 4-ary node count up to ``tree_depth`` inclusive."""
    return sum(TREE_BRANCHING_FACTOR**level for level in range(tree_depth + 1))


def flatland_observation_dim(tree_depth: int) -> int:
    return flatland_tree_node_count(tree_depth) * TREE_FEATURE_DIM


def flatland_state_dim(num_agents: int, tree_depth: int) -> int:
    per_agent_dim = flatland_observation_dim(tree_depth) + FLATLAND_STATUS_DIM
    return num_agents * per_agent_dim


@dataclass(slots=True)
class FlatlandConfig:
    width: int = 32
    height: int = 24
    num_agents: int = 10

    max_num_cities: int = 4
    max_rails_between_cities: int = 3
    max_rail_pairs_in_city: int = 2
    grid_mode: bool = True

    tree_depth: int = 2
    prediction_depth: int = 20

    use_malfunctions: bool = True
    malfunction_rate: float = 1 / 250.0
    malfunction_min_duration: int = 10
    malfunction_max_duration: int = 30

    # Heterogeneous speed profile used by Flatland's sparse line generator.
    speed_ratio_map: Mapping[float, float] = field(
        default_factory=lambda: {1.0: 0.5, 0.5: 0.25, 1 / 3: 0.25}
    )

    regenerate_rail_on_reset: bool = True
    regenerate_schedule_on_reset: bool = True
    remove_agents_at_target: bool = True
    max_episode_steps: int | None = None
    random_seed: int | None = None

    # Dense reward shaping (potential-based + step/deadlock/completion terms).
    # Flatland's native reward is almost entirely a per-step -1 bleed, so a
    # 25-step rollout sees no action-conditional signal.  These terms inject
    # local credit from the distance map and explicit terminal anchors.
    use_shaped_reward: bool = False
    progress_coeff: float = 0.1  # +coeff * (prev_d - curr_d) per agent per step
    step_penalty: float = 0.01  # constant time pressure for active agents
    deadlock_penalty: float = 1.0  # fired once when an agent first deadlocks
    completion_bonus: float = 1.0  # fired on terminal (reached target)
    progress_clip: float = 1.0  # absolute clip on per-step progress delta
