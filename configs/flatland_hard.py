# fmt: off

TREE_DEPTH = 2
TREE_FEATURE_DIM = 12
STATUS_FEATURE_DIM = 7


def flatland_dims(num_agents: int, tree_depth: int = TREE_DEPTH) -> tuple[int, int]:
    node_count = sum(4 ** level for level in range(tree_depth + 1))
    obs_dim = node_count * TREE_FEATURE_DIM
    state_dim = num_agents * (obs_dim + STATUS_FEATURE_DIM)
    return obs_dim, state_dim


FLATLAND_HARD_ENV = {
    "id":                       "flatland",
    "num_envs":                 8,
    "width":                    48,
    "height":                   36,
    "num_agents":               16,
    "max_num_cities":           6,
    "max_rails_between_cities": 4,
    "max_rail_pairs_in_city":   2,
    "grid_mode":                True,
    "tree_depth":               TREE_DEPTH,
    "prediction_depth":         20,
    "use_malfunctions":         True,
    "malfunction_rate":         1 / 180.0,
    "malfunction_min_duration": 12,
    "malfunction_max_duration": 40,
    "speed_ratio_map":          {1.0: 0.4, 0.5: 0.35, 1 / 3: 0.25},
}
