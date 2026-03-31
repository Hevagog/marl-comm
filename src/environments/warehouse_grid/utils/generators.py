import numpy as np

from ..config import WarehouseConfig
from .types import CellType, GridState


def _build_interference_map(
    config: WarehouseConfig,
    treatment_positions: np.ndarray,
) -> np.ndarray:
    """Build a spatial interference heat-map

    Higher values near treatment stations model RF congestion in busy
    areas.  Returns an (H, W) float32 array in [0, 1].
    """
    H, W = config.grid_height, config.grid_width
    imap = np.full((H, W), config.interference_base, dtype=np.float32)

    if not config.enable_interference_zones:
        return imap

    radius = config.interference_radius
    boost = config.interference_treatment_boost

    for pos in treatment_positions:
        r0, c0 = int(pos[0]), int(pos[1])
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < H and 0 <= nc < W:
                    dist = abs(dr) + abs(dc)  # Manhattan distance
                    if dist <= radius:
                        # Linear falloff from centre
                        strength = boost * (1.0 - dist / (radius + 1))
                        imap[nr, nc] = min(1.0, imap[nr, nc] + strength)

    return np.clip(imap, 0.0, 1.0)


def create_warehouse_layout(config: WarehouseConfig) -> GridState:
    """Create a warehouse grid layout.

    Layout structure
    ----------------
    - Row 1          : Spawn area
    - Rows 3-6       : Aisles with shelves (columns 3, 6, 9, 12)
    - Row H-3 left   : Treatment stations
    - Row H-3 right  : Goal (delivery) locations
    - Row H-4 centre : Charging stations (Layer 2.4, if enabled)
    - Top-right      : Repair centre for stranded robots
    """
    H, W = config.grid_height, config.grid_width
    layout = np.zeros((H, W), dtype=np.int32)

    # --- Walls on borders ---
    layout[0, :] = CellType.WALL
    layout[H - 1, :] = CellType.WALL
    layout[:, 0] = CellType.WALL
    layout[:, W - 1] = CellType.WALL

    # --- Shelves ---
    shelf_positions = []
    shelf_cols = [3, 6, 9, 12]
    shelf_rows = [3, 4, 5, 6]

    shelf_idx = 0
    for col in shelf_cols:
        if col < W - 1:
            for row in shelf_rows:
                if row < H - 3 and shelf_idx < config.num_shelves:
                    layout[row, col] = CellType.SHELF
                    shelf_positions.append([row, col])
                    shelf_idx += 1

    while len(shelf_positions) < config.num_shelves:
        shelf_positions.append([1, 1])
    shelf_positions = np.array(shelf_positions[: config.num_shelves], dtype=np.int32)

    # --- Treatment stations (bottom-left) ---
    treatment_positions = []
    for i in range(config.num_treatment_stations):
        row = H - 3
        col = 2 + i * 2
        if col < W // 2:
            layout[row, col] = CellType.TREATMENT
            treatment_positions.append([row, col])

    while len(treatment_positions) < config.num_treatment_stations:
        treatment_positions.append([H - 3, 2])
    treatment_positions = np.array(
        treatment_positions[: config.num_treatment_stations], dtype=np.int32
    )

    # --- Goal locations (bottom-right) ---
    goal_positions = []
    for i in range(config.num_goal_locations):
        row = H - 3
        col = W - 3 - i * 2
        if col > W // 2:
            layout[row, col] = CellType.GOAL
            goal_positions.append([row, col])

    while len(goal_positions) < config.num_goal_locations:
        goal_positions.append([H - 3, W - 3])
    goal_positions = np.array(
        goal_positions[: config.num_goal_locations], dtype=np.int32
    )

    # --- Spawn positions (top row) ---
    spawn_positions = []
    for i in range(config.max_agents):
        row = 1
        col = 2 + i
        if col < W - 1:
            layout[row, col] = CellType.SPAWN
            spawn_positions.append([row, col])

    while len(spawn_positions) < config.max_agents:
        spawn_positions.append([1, 2])
    spawn_positions = np.array(spawn_positions[: config.max_agents], dtype=np.int32)

    # --- Repair centre (top-right) ---
    repair_position = np.array([1, W - 2], dtype=np.int32)
    layout[repair_position[0], repair_position[1]] = CellType.REPAIR

    # --- Charging stations ---
    charger_positions = []
    if config.enable_battery and config.num_charging_stations > 0:
        # Place chargers in the middle row, spread across width
        charger_row = max(2, H - 4)
        mid_col = W // 2
        for i in range(config.num_charging_stations):
            offset = (i - config.num_charging_stations // 2) * 2
            col = mid_col + offset
            col = max(1, min(col, W - 2))
            if layout[charger_row, col] == CellType.EMPTY:
                layout[charger_row, col] = CellType.CHARGER
                charger_positions.append([charger_row, col])

    while len(charger_positions) < max(config.num_charging_stations, 1):
        # At least one dummy entry so the array is never empty
        charger_positions.append([max(2, H - 4), W // 2])
    charger_positions = np.array(
        charger_positions[: max(config.num_charging_stations, 1)], dtype=np.int32
    )

    # --- Shelf resources (all full) ---
    shelf_resources = np.ones(
        (config.num_shelves, config.resources_per_shelf), dtype=np.bool_
    )

    # --- Task priorities (higher near shelves with resources) ---
    task_priorities = np.zeros((H, W), dtype=np.float32)
    for i in range(len(shelf_positions)):
        if i < config.num_shelves:
            pos = shelf_positions[i]
            task_priorities[pos[0], pos[1]] = 1.0

    # --- Interference map  ---
    interference_map = _build_interference_map(config, treatment_positions)

    return GridState(
        layout=layout,
        shelf_resources=shelf_resources,
        shelf_positions=shelf_positions,
        treatment_positions=treatment_positions,
        goal_positions=goal_positions,
        spawn_positions=spawn_positions,
        task_priorities=task_priorities,
        interference_map=interference_map,
        charger_positions=charger_positions,
        repair_position=repair_position,
    )
