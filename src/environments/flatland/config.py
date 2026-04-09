from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


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
