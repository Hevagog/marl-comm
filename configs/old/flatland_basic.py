# fmt: off

TREE_DEPTH = 2
TREE_FEATURE_DIM = 12
STATUS_FEATURE_DIM = 7


def flatland_dims(num_agents: int, tree_depth: int = TREE_DEPTH) -> tuple[int, int]:
    node_count = sum(4 ** level for level in range(tree_depth + 1))
    obs_dim = node_count * TREE_FEATURE_DIM
    state_dim = num_agents * (obs_dim + STATUS_FEATURE_DIM)
    return obs_dim, state_dim


FLATLAND_BASIC_ENV = {
    "id":                       "flatland",
    "num_envs":                 8,
    "width":                    32,
    "height":                   24,
    "num_agents":               8,
    "max_num_cities":           4,
    "max_rails_between_cities": 3,
    "max_rail_pairs_in_city":   2,
    "grid_mode":                True,
    "tree_depth":               TREE_DEPTH,
    "prediction_depth":         20,
    "use_malfunctions":         True,
    "malfunction_rate":         1 / 350.0,
    "malfunction_min_duration": 8,
    "malfunction_max_duration": 25,
    "speed_ratio_map":          {1.0: 0.6, 0.5: 0.3, 1 / 3: 0.1},
}
