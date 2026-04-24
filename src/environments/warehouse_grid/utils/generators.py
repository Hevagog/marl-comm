import math
import warnings
from collections import deque

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


def create_warehouse_layout(
    config: WarehouseConfig,
    rng: np.random.Generator | None = None,
) -> GridState:
    """Build a warehouse grid.

    When ``config.randomize_layout`` is True and an ``rng`` is provided, the
    layout is sampled per call within fixed semantic bands (structural jitter).
    The deterministic legacy layout is used otherwise so existing reproducible
    runs are unaffected.

    Layout structure (deterministic mode)
    -------------------------------------
    - Row 1          : Spawn area
    - Rows 3-6       : Aisles with shelves (columns 3, 6, 9, 12)
    - Row H-3 left   : Treatment stations
    - Row H-3 right  : Goal (delivery) locations
    - Row H-4 centre : Charging stations (if enabled)
    - Top-right      : Repair centre for stranded robots
    """
    if config.randomize_layout and rng is not None:
        for _ in range(max(1, config.layout_shuffle_max_retries)):
            candidate = _create_jittered_layout(config, rng)
            if candidate is not None and _validate_layout_reachability(
                candidate, config
            ):
                return candidate
        warnings.warn(
            "randomize_layout=True but jitter sampler failed all retries; "
            "falling back to deterministic layout. Consider relaxing constraints "
            "or increasing grid_width/height.",
            RuntimeWarning,
            stacklevel=2,
        )

    return _create_deterministic_layout(config)


def _create_deterministic_layout(config: WarehouseConfig) -> GridState:
    """Original hardcoded layout. Preserves bit-for-bit legacy behavior."""
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

    # --- Rendezvous cells (Scenario 2) ---
    rendezvous_positions = _place_rendezvous_cells(config, layout, H, W)

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
        rendezvous_positions=rendezvous_positions,
    )


def _place_rendezvous_cells(
    config: WarehouseConfig,
    layout: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """Place rendezvous cells in the central interior.

    Returns an (k, 2) int32 array of placed positions (always at least one
    row, even when disabled, so the GridState shape is stable).
    """
    if not config.enable_rendezvous or config.num_rendezvous <= 0:
        return np.full((1, 2), -1, dtype=np.int32)

    placed: list[list[int]] = []
    mid_r, mid_c = H // 2, W // 2
    candidates = [
        (mid_r, mid_c),
        (mid_r, mid_c + 1),
        (mid_r + 1, mid_c),
        (mid_r - 1, mid_c),
    ]
    for r, c in candidates:
        if len(placed) >= config.num_rendezvous:
            break
        if 0 <= r < H and 0 <= c < W and layout[r, c] == CellType.EMPTY:
            layout[r, c] = CellType.RENDEZVOUS
            placed.append([r, c])
    if not placed:
        return np.full((1, 2), -1, dtype=np.int32)
    return np.array(placed, dtype=np.int32)


def _sample_with_min_spacing(
    rng: np.random.Generator,
    pool: list[int],
    n: int,
    min_spacing: int,
    max_attempts: int = 200,
) -> list[int] | None:
    """Sample ``n`` distinct ints from ``pool`` such that any two chosen
    values differ by at least ``min_spacing``. Returns a sorted list, or
    ``None`` if no valid sample was found within ``max_attempts``.
    """
    if n <= 0:
        return []
    pool = list(pool)
    if len(pool) < n:
        return None
    for _ in range(max_attempts):
        chosen = rng.choice(pool, size=n, replace=False)
        chosen_sorted = np.sort(chosen)
        if n == 1 or int(np.diff(chosen_sorted).min()) >= min_spacing:
            return [int(x) for x in chosen_sorted]
    return None


def _create_jittered_layout(
    config: WarehouseConfig,
    rng: np.random.Generator,
) -> GridState | None:
    """Sample a warehouse layout that preserves semantic bands but jitters
    exact column positions and the left/right side of treatment vs goal.

    Returns ``None`` if any sampling step exhausts its retry budget so the
    caller can try again.
    """
    H, W = config.grid_height, config.grid_width

    # Bands (rows are anchored, columns are sampled).
    spawn_row = 1
    repair_row = 1
    shelf_row_stop = max(4, H - 5)  # exclusive
    shelf_rows = list(range(3, shelf_row_stop))
    if not shelf_rows:
        return None
    charger_row = max(2, H - 4)
    service_row = H - 3

    # Service row must sit below the shelf band and not overlap chargers.
    if service_row <= shelf_rows[-1] or service_row >= H - 1:
        return None
    if charger_row in shelf_rows or charger_row == service_row:
        return None

    interior_cols = list(range(2, W - 2))
    if not interior_cols:
        return None

    layout = np.zeros((H, W), dtype=np.int32)
    layout[0, :] = CellType.WALL
    layout[H - 1, :] = CellType.WALL
    layout[:, 0] = CellType.WALL
    layout[:, W - 1] = CellType.WALL

    # --- Shelves ---------------------------------------------------------
    n_shelf_cols = max(1, math.ceil(config.num_shelves / len(shelf_rows)))
    shelf_cols = _sample_with_min_spacing(
        rng, interior_cols, n_shelf_cols, min_spacing=2
    )
    if shelf_cols is None:
        return None

    shelf_positions: list[list[int]] = []
    # Fill column-major so each chosen column forms a vertical aisle.
    for col in shelf_cols:
        for row in shelf_rows:
            if len(shelf_positions) >= config.num_shelves:
                break
            layout[row, col] = CellType.SHELF
            shelf_positions.append([row, col])
        if len(shelf_positions) >= config.num_shelves:
            break
    if len(shelf_positions) < config.num_shelves:
        return None
    shelf_positions_arr = np.array(shelf_positions, dtype=np.int32)

    # --- Service row split (treatment vs goal sides) ---------------------
    mid = W // 2
    left_pool = [c for c in interior_cols if c < mid]
    right_pool = [c for c in interior_cols if c >= mid]
    if not left_pool or not right_pool:
        return None
    if rng.random() < 0.5:
        treat_pool, goal_pool = left_pool, right_pool
    else:
        treat_pool, goal_pool = right_pool, left_pool

    treat_cols = _sample_with_min_spacing(
        rng, treat_pool, config.num_treatment_stations, min_spacing=1
    )
    goal_cols = _sample_with_min_spacing(
        rng, goal_pool, config.num_goal_locations, min_spacing=1
    )
    if treat_cols is None or goal_cols is None:
        return None

    treatment_positions = np.array(
        [[service_row, c] for c in treat_cols], dtype=np.int32
    )
    for pos in treatment_positions:
        layout[int(pos[0]), int(pos[1])] = CellType.TREATMENT

    goal_positions = np.array([[service_row, c] for c in goal_cols], dtype=np.int32)
    for pos in goal_positions:
        layout[int(pos[0]), int(pos[1])] = CellType.GOAL

    # --- Top corridor: spawns + repair ----------------------------------
    n_top = config.max_agents + 1  # +1 for repair
    top_cols = _sample_with_min_spacing(rng, interior_cols, n_top, min_spacing=1)
    if top_cols is None:
        return None
    # Pick a random one of the chosen columns to host the repair cell so
    # the repair location is not always the rightmost slot.
    repair_idx = int(rng.integers(0, n_top))
    repair_col = top_cols[repair_idx]
    spawn_cols = [c for i, c in enumerate(top_cols) if i != repair_idx]

    repair_position = np.array([repair_row, repair_col], dtype=np.int32)
    layout[repair_row, repair_col] = CellType.REPAIR

    spawn_positions: list[list[int]] = []
    for c in spawn_cols:
        layout[spawn_row, c] = CellType.SPAWN
        spawn_positions.append([spawn_row, c])
    spawn_positions_arr = np.array(spawn_positions, dtype=np.int32)

    # --- Charging stations ----------------------------------------------
    if config.enable_battery and config.num_charging_stations > 0:
        free_charger_cols = [
            c for c in interior_cols if layout[charger_row, c] == CellType.EMPTY
        ]
        chosen_chargers = _sample_with_min_spacing(
            rng,
            free_charger_cols,
            config.num_charging_stations,
            min_spacing=2,
        )
        if chosen_chargers is None:
            return None
        charger_positions_list = []
        for c in chosen_chargers:
            layout[charger_row, c] = CellType.CHARGER
            charger_positions_list.append([charger_row, c])
        charger_positions = np.array(charger_positions_list, dtype=np.int32)
    else:
        # Match deterministic mode: keep one dummy entry so downstream
        # code can index ``charger_positions[0]`` safely.
        charger_positions = np.array(
            [[charger_row, max(1, min(W // 2, W - 2))]], dtype=np.int32
        )

    # --- Shelf resources / priorities / interference (same as static) ---
    shelf_resources = np.ones(
        (config.num_shelves, config.resources_per_shelf), dtype=np.bool_
    )

    task_priorities = np.zeros((H, W), dtype=np.float32)
    for i in range(len(shelf_positions_arr)):
        if i < config.num_shelves:
            pos = shelf_positions_arr[i]
            task_priorities[int(pos[0]), int(pos[1])] = 1.0

    rendezvous_positions = _place_rendezvous_cells(config, layout, H, W)
    interference_map = _build_interference_map(config, treatment_positions)

    return GridState(
        layout=layout,
        shelf_resources=shelf_resources,
        shelf_positions=shelf_positions_arr,
        treatment_positions=treatment_positions,
        goal_positions=goal_positions,
        spawn_positions=spawn_positions_arr,
        task_priorities=task_priorities,
        interference_map=interference_map,
        charger_positions=charger_positions,
        repair_position=repair_position,
        rendezvous_positions=rendezvous_positions,
    )


def _validate_layout_reachability(
    grid_state: GridState,
    config: WarehouseConfig,
) -> bool:
    """BFS over non-WALL cells from every spawn. Confirms that every
    functional tile (shelves, treatment, goal, repair, and chargers when
    battery is enabled) is reachable from at least one spawn position.
    """
    layout = grid_state.layout
    H, W = layout.shape

    spawns = grid_state.spawn_positions
    if len(spawns) == 0:
        return False

    visited = np.zeros((H, W), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for pos in spawns:
        r, c = int(pos[0]), int(pos[1])
        if not visited[r, c]:
            visited[r, c] = True
            queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                if layout[nr, nc] != CellType.WALL:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    targets: list[tuple[int, int]] = []
    for arr in (
        grid_state.shelf_positions,
        grid_state.treatment_positions,
        grid_state.goal_positions,
    ):
        for pos in arr:
            targets.append((int(pos[0]), int(pos[1])))

    rp = grid_state.repair_position
    targets.append((int(rp[0]), int(rp[1])))

    if config.enable_battery and config.num_charging_stations > 0:
        for pos in grid_state.charger_positions:
            targets.append((int(pos[0]), int(pos[1])))

    return all(visited[r, c] for r, c in targets)
