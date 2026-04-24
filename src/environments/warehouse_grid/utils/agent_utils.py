import numpy as np

from ..config import WarehouseConfig
from .types import (
    Actions,
    CellType,
    AgentState,
    EnvState,
    ResourcePhase,
    TaskInfo,
    ACTION_DELTAS,
)


def _stranded_mask(state: EnvState) -> np.ndarray:
    """Return robots that are down and awaiting rescue."""
    return (~state.agent.active) & (
        state.agent.failed | state.agent.burst_failed | state.agent.battery_dead
    )


def create_agent_state(
    config: WarehouseConfig,
    spawn_positions: np.ndarray,
) -> AgentState:
    """Initialise agent state (supports heterogeneous & battery fields)."""
    n = config.num_agents
    m = config.max_agents

    positions = spawn_positions[:n].copy()
    if n < m:
        padding = np.zeros((m - n, 2), dtype=np.int32)
        positions = np.concatenate([positions, padding], axis=0)

    active = np.zeros(m, dtype=np.bool_)
    active[:n] = True

    # heterogeneous properties
    speed = np.ones(m, dtype=np.int32)
    capacity = np.ones(m, dtype=np.int32)
    fragility = np.ones(m, dtype=np.float32)

    if config.enable_heterogeneous:
        sp_opts = config.agent_speed_options
        ca_opts = config.agent_capacity_options
        fr_opts = config.agent_fragility_options
        for i in range(m):
            speed[i] = sp_opts[i % len(sp_opts)]
            capacity[i] = ca_opts[i % len(ca_opts)]
            fragility[i] = fr_opts[i % len(fr_opts)]

    # battery
    battery = np.full(m, config.battery_capacity, dtype=np.int32)
    if not config.enable_battery:
        battery[:] = 0  # sentinel: 0 means "disabled"

    return AgentState(
        positions=positions,
        active=active,
        carrying=np.zeros(m, dtype=np.int32),
        resource_phase=np.zeros(m, dtype=np.int32),
        treatment_timer=np.zeros(m, dtype=np.int32),
        locked=np.zeros(m, dtype=np.bool_),
        speed=speed,
        capacity=capacity,
        fragility=fragility,
        battery=battery,
        charging=np.zeros(m, dtype=np.bool_),
        battery_dead=np.zeros(m, dtype=np.bool_),
        rescue_target=np.full(m, -1, dtype=np.int32),
        being_dragged_by=np.full(m, -1, dtype=np.int32),
        burst_failed=np.zeros(m, dtype=np.bool_),
        failed=np.zeros(m, dtype=np.bool_),
        delivery_count=np.zeros(m, dtype=np.int32),
        last_delivery_step=np.full(m, -10_000, dtype=np.int32),
    )


def get_agent_observation(
    agent_idx: int,
    state: EnvState,
    config: WarehouseConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build observation vector for *agent_idx*."""
    H, W = config.grid_height, config.grid_width
    pos = state.agent.positions[agent_idx]
    row, col = int(pos[0]), int(pos[1])

    features: list[float] = []

    # ------ own state (7 features) ------
    features.extend(
        [
            float(state.agent.carrying[agent_idx] > 0),
            state.agent.resource_phase[agent_idx] / 3.0,
            state.agent.treatment_timer[agent_idx] / max(config.treatment_duration, 1),
            float(state.agent.locked[agent_idx]),
            row / (H - 1) if H > 1 else 0.0,
            col / (W - 1) if W > 1 else 0.0,
            float(state.agent.rescue_target[agent_idx] >= 0),
        ]
    )

    # ------ relative positions to task / support cells (8 features) ------
    treatment_pos = state.grid.treatment_positions[0]
    goal_pos = state.grid.goal_positions[0]
    repair_pos = state.grid.repair_position
    if config.enable_battery:
        charger_positions = state.grid.charger_positions
        charger_distances = np.abs(charger_positions - np.array([row, col])).sum(axis=1)
        charger_pos = charger_positions[int(np.argmin(charger_distances))]
    else:
        charger_pos = np.array([row, col], dtype=np.int32)
    features.extend(
        [
            (treatment_pos[0] - row) / H,
            (treatment_pos[1] - col) / W,
            (goal_pos[0] - row) / H,
            (goal_pos[1] - col) / W,
            (repair_pos[0] - row) / H,
            (repair_pos[1] - col) / W,
            (charger_pos[0] - row) / H,
            (charger_pos[1] - col) / W,
        ]
    )

    # ------ local grid view (6 features per cell) ------
    vr = config.vision_range
    for dr in range(-vr, vr + 1):
        for dc in range(-vr, vr + 1):
            nr, nc = row + dr, col + dc
            if 0 <= nr < H and 0 <= nc < W:
                cell = state.grid.layout[nr, nc]
                features.extend(
                    [
                        float(cell == CellType.SHELF),
                        float(cell == CellType.TREATMENT),
                        float(cell == CellType.GOAL),
                        float(cell == CellType.CHARGER),
                        float(cell == CellType.REPAIR),
                        float(cell != CellType.WALL),
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    stranded = _stranded_mask(state)

    # ------ other agents (5 or 6 features each) ------
    # no_comm=True: gate by vision_range, expose only physically-observable state.
    #   Visible: position (dx,dy), is_carrying (shelf on robot is visible), is_stranded
    #   (stopped robot is physically detectable).  Battery and active flag require radio.
    #   Obs-dim unchanged — same slots, zeroed where radio comm is required.
    # no_comm=False (default): gate by comm_range; full internal state shared via radio.
    _n_other = 6 if config.enable_battery else 5
    _no_comm = config.no_comm
    _gate_dist = config.vision_range if _no_comm else config.comm_range
    for i in range(config.max_agents):
        if i == agent_idx:
            continue
        if state.agent.active[i] or stranded[i]:
            other_pos = state.agent.positions[i]
            dist = abs(int(other_pos[0]) - row) + abs(int(other_pos[1]) - col)
            if _gate_dist is not None and dist > _gate_dist:
                features.extend([0.0] * _n_other)
                continue
            if _no_comm:
                # Physical vision only: position + visible state, no radio info
                features.extend(
                    [
                        (other_pos[0] - row) / H,
                        (other_pos[1] - col) / W,
                        float(state.agent.carrying[i] > 0),  # shelf visible on robot
                        0.0,  # active flag requires radio — zero in no-comm
                        float(stranded[i]),  # stopped robot is physically detectable
                    ]
                )
                if config.enable_battery:
                    features.append(0.0)  # battery requires radio — zero in no-comm
            else:
                features.extend(
                    [
                        (other_pos[0] - row) / H,
                        (other_pos[1] - col) / W,
                        float(state.agent.carrying[i] > 0),
                        float(state.agent.active[i]),
                        float(stranded[i]),
                    ]
                )
                if config.enable_battery:
                    features.append(
                        state.agent.battery[i] / max(config.battery_capacity, 1)
                    )
        else:
            features.extend([0.0] * _n_other)

    # ------  own heterogeneous properties (3 features) ------
    if config.enable_heterogeneous:
        features.extend(
            [
                state.agent.speed[agent_idx] / 2.0,
                state.agent.capacity[agent_idx] / 2.0,
                state.agent.fragility[agent_idx],
            ]
        )

    # ------  battery (1 feature) ------
    if config.enable_battery:
        features.append(
            state.agent.battery[agent_idx] / max(config.battery_capacity, 1)
        )

    # ------  task queue context (3 features) ------
    if config.enable_task_deadlines:
        features.append(len(state.task_queue) / max(config.max_pending_tasks, 1))
        if state.task_queue:
            most_urgent = max(
                state.task_queue,
                key=lambda t: t.priority / max(t.deadline - state.step_count, 1),
            )
            shelf_pos = state.grid.shelf_positions[most_urgent.shelf_idx]
            features.extend(
                [
                    (int(shelf_pos[0]) - row) / H,
                    (int(shelf_pos[1]) - col) / W,
                ]
            )
        else:
            features.extend([0.0, 0.0])

    # ------ assemble & apply noise ------
    obs = np.array(features, dtype=np.float32)

    # In no_comm mode obs is purely local vision — no radio channel to corrupt.
    if config.no_comm:
        return obs

    # Compute effective noise probability
    noise_prob = config.comm_noise_prob

    #  spatial interference
    if config.enable_interference_zones:
        noise_prob = float(state.grid.interference_map[row, col])

    #  load-dependent comm
    fp = config.fault_profile
    if fp.load_dependent_comm:
        # Count agents within vision range
        local_density = 0
        for i in range(config.max_agents):
            if i != agent_idx and state.agent.active[i]:
                d = abs(int(state.agent.positions[i, 0]) - row) + abs(
                    int(state.agent.positions[i, 1]) - col
                )
                if d <= config.vision_range:
                    local_density += 1
        noise_prob = fp.base_packet_loss + fp.congestion_factor * local_density

    noise_prob = min(noise_prob, 1.0)

    if rng.random() < noise_prob:
        noise = rng.normal(0, 0.1, size=obs.shape).astype(np.float32)
        obs = obs + noise

    return obs


def compute_obs_dim(config: WarehouseConfig) -> int:
    """Compute the observation dimension for an agent."""
    own_state = 7
    relative_pos = 8
    vr = config.vision_range
    view_size = 2 * vr + 1
    local_grid = view_size * view_size * 6
    other_agent_features = 6 if config.enable_battery else 5
    other_agents = (config.max_agents - 1) * other_agent_features
    dim = own_state + relative_pos + local_grid + other_agents
    if config.enable_heterogeneous:
        dim += 3
    if config.enable_battery:
        dim += 1
    if config.enable_task_deadlines:
        dim += 3  # pending_ratio + dx_urgent + dy_urgent
    return dim


def apply_agent_attrition(
    state: EnvState,
    config: WarehouseConfig,
    rng: np.random.Generator,
) -> tuple[EnvState, np.ndarray]:
    """Apply agent failures.

    Returns updated state and boolean mask of newly-failed agents.
    """
    m = config.max_agents
    fp = config.fault_profile

    # ---------- base uniform failures (modified by fragility) ----------
    thresholds = np.full(m, config.agent_failure_prob, dtype=np.float64)
    if config.enable_heterogeneous:
        thresholds = thresholds * state.agent.fragility

    failures = rng.random(m) < thresholds
    failures = failures & state.agent.active & ~state.agent.burst_failed

    # ---------- burst attrition ----------
    burst_new = np.zeros(m, dtype=np.bool_)
    if fp.burst_attrition:
        burst_trigger = rng.random(m) < fp.burst_prob
        burst_trigger = burst_trigger & state.agent.active & ~state.agent.burst_failed
        burst_new = burst_trigger
        failures = failures | burst_new

    # ---------- correlated failure ----------
    if fp.correlated_failure and failures.any():
        positions = state.agent.positions
        radius = fp.correlation_radius
        # For each newly-failed agent, fail neighbours
        seed_failures = np.where(failures)[0]
        for idx in seed_failures:
            r0, c0 = int(positions[idx, 0]), int(positions[idx, 1])
            for j in range(m):
                if j == idx or not state.agent.active[j] or failures[j]:
                    continue
                rj, cj = int(positions[j, 0]), int(positions[j, 1])
                if abs(rj - r0) + abs(cj - c0) <= radius:
                    failures[j] = True

    # ---------- apply failures ----------
    new_active = state.agent.active & ~failures
    new_carrying = state.agent.carrying.copy()
    new_carrying[failures] = 0
    new_burst = state.agent.burst_failed | burst_new
    # Mark non-burst failures (normal attrition + correlated cascade)
    new_failed = state.agent.failed | (failures & ~burst_new)
    new_rescue_target = state.agent.rescue_target.copy()
    new_being_dragged_by = state.agent.being_dragged_by.copy()

    for rescuer in np.where(failures)[0]:
        target = int(new_rescue_target[rescuer])
        if target >= 0 and new_being_dragged_by[target] == rescuer:
            new_being_dragged_by[target] = -1
        new_rescue_target[rescuer] = -1

    new_agent = state.agent._replace(
        active=new_active,
        carrying=new_carrying,
        rescue_target=new_rescue_target,
        being_dragged_by=new_being_dragged_by,
        burst_failed=new_burst,
        failed=new_failed,
    )

    return state._replace(agent=new_agent), failures


def apply_movement(
    state: EnvState,
    actions: np.ndarray,
    config: WarehouseConfig,
) -> tuple[EnvState, np.ndarray]:
    """Apply movement actions.  Supports multi-cell speed & battery drain."""
    H, W = config.grid_height, config.grid_width
    layout = state.grid.layout

    new_positions = state.agent.positions.copy()
    new_battery = state.agent.battery.copy()
    new_charging = state.agent.charging.copy()
    dragged_by = state.agent.being_dragged_by

    # Determine per-agent move distance
    speeds = state.agent.speed  # (m,)

    for i in range(config.max_agents):
        if not state.agent.active[i] or state.agent.locked[i]:
            continue

        #  if battery-enabled and charging, skip movement
        if config.enable_battery and state.agent.charging[i]:
            # Charge up
            new_battery[i] = min(
                config.battery_capacity,
                new_battery[i] + config.battery_charge_rate,
            )
            # Un-dock once full
            if new_battery[i] >= config.battery_capacity:
                new_charging[i] = False
            continue

        delta = ACTION_DELTAS[actions[i]]
        is_moving = not (delta[0] == 0 and delta[1] == 0)
        steps = int(speeds[i]) if is_moving else 0
        if state.agent.rescue_target[i] >= 0:
            steps = min(steps, 1)

        # Step cell-by-cell (handles walls along the path)
        r, c = int(new_positions[i, 0]), int(new_positions[i, 1])
        for _ in range(steps):
            nr, nc = r + int(delta[0]), c + int(delta[1])
            nr = max(0, min(nr, H - 1))
            nc = max(0, min(nc, W - 1))
            if layout[nr, nc] == CellType.WALL:
                break
            r, c = nr, nc

        new_positions[i] = [r, c]

        #  drain battery
        if config.enable_battery:
            drain = (
                config.battery_drain_per_step
                if is_moving
                else config.battery_drain_idle
            )
            new_battery[i] = max(0, new_battery[i] - drain)

    # ---------- collision detection ----------
    collision_mask = np.zeros(config.max_agents, dtype=np.bool_)
    for i in range(config.max_agents):
        for j in range(i + 1, config.max_agents):
            if (
                state.agent.active[i]
                and state.agent.active[j]
                and new_positions[i, 0] == new_positions[j, 0]
                and new_positions[i, 1] == new_positions[j, 1]
            ):
                collision_mask[i] = True
                collision_mask[j] = True

    # Revert colliders
    final_positions = np.where(
        collision_mask[:, None], state.agent.positions, new_positions
    )
    for target_idx in range(config.max_agents):
        rescuer = int(dragged_by[target_idx])
        if rescuer >= 0 and state.agent.active[rescuer]:
            final_positions[target_idx] = final_positions[rescuer]

    new_agent = state.agent._replace(
        positions=final_positions.astype(np.int32),
        battery=new_battery,
        charging=new_charging,
    )

    return state._replace(agent=new_agent), collision_mask


def apply_battery_logic(
    state: EnvState,
    config: WarehouseConfig,
) -> EnvState:
    """Force agents with critically low battery to seek a charger.

    If an agent is already on a charger cell, set ``charging=True``.
    If battery reaches 0, deactivate the agent (stranded).
    """
    if not config.enable_battery:
        return state

    new_active = state.agent.active.copy()
    new_charging = state.agent.charging.copy()
    new_locked = state.agent.locked.copy()
    new_carrying = state.agent.carrying.copy()
    new_battery_dead = state.agent.battery_dead.copy()
    new_rescue_target = state.agent.rescue_target.copy()
    new_being_dragged_by = state.agent.being_dragged_by.copy()
    battery = state.agent.battery

    layout = state.grid.layout

    for i in range(config.max_agents):
        if not state.agent.active[i]:
            continue

        r, c = int(state.agent.positions[i, 0]), int(state.agent.positions[i, 1])

        # Dock at charger if standing on one
        if layout[r, c] == CellType.CHARGER and battery[i] < config.battery_capacity:
            new_charging[i] = True

        # Battery dead → agent deactivated
        if battery[i] <= 0:
            new_active[i] = False
            new_charging[i] = False
            new_carrying[i] = 0
            new_battery_dead[i] = True
            target = int(new_rescue_target[i])
            if target >= 0 and new_being_dragged_by[target] == i:
                new_being_dragged_by[target] = -1
            new_rescue_target[i] = -1

    new_agent = state.agent._replace(
        active=new_active,
        charging=new_charging,
        locked=new_locked,
        carrying=new_carrying,
        battery_dead=new_battery_dead,
        rescue_target=new_rescue_target,
        being_dragged_by=new_being_dragged_by,
    )

    return state._replace(agent=new_agent)


def apply_rescue_completion(
    state: EnvState,
    config: WarehouseConfig,
) -> tuple[EnvState, np.ndarray, np.ndarray]:
    """Resolve rescues that reached repair or charging destinations."""
    repair_success = np.zeros(config.max_agents, dtype=np.bool_)
    charge_success = np.zeros(config.max_agents, dtype=np.bool_)

    new_positions = state.agent.positions.copy()
    new_active = state.agent.active.copy()
    new_carrying = state.agent.carrying.copy()
    new_phase = state.agent.resource_phase.copy()
    new_timer = state.agent.treatment_timer.copy()
    new_locked = state.agent.locked.copy()
    new_battery = state.agent.battery.copy()
    new_charging = state.agent.charging.copy()
    new_battery_dead = state.agent.battery_dead.copy()
    new_rescue_target = state.agent.rescue_target.copy()
    new_being_dragged_by = state.agent.being_dragged_by.copy()
    new_burst_failed = state.agent.burst_failed.copy()
    new_failed = state.agent.failed.copy()

    repair_row, repair_col = state.grid.repair_position
    layout = state.grid.layout

    for rescuer in range(config.max_agents):
        if not state.agent.active[rescuer]:
            continue

        target = int(state.agent.rescue_target[rescuer])
        if target < 0:
            continue

        row, col = state.agent.positions[rescuer]
        is_battery_rescue = bool(state.agent.battery_dead[target])
        on_destination = (
            layout[row, col] == CellType.CHARGER
            if is_battery_rescue
            else (int(row) == int(repair_row) and int(col) == int(repair_col))
        )
        if not on_destination:
            continue

        if is_battery_rescue:
            charge_success[rescuer] = True
            new_battery[target] = max(config.battery_charge_rate, 1)
            new_charging[target] = True
        else:
            repair_success[rescuer] = True
            if config.enable_battery:
                new_battery[target] = config.battery_capacity
            new_charging[target] = False

        new_positions[target] = [row, col]
        new_active[target] = True
        new_carrying[target] = 0
        new_phase[target] = ResourcePhase.UNPICKED
        new_timer[target] = 0
        new_locked[target] = False
        new_battery_dead[target] = False
        new_burst_failed[target] = False
        new_failed[target] = False
        new_being_dragged_by[target] = -1
        new_rescue_target[rescuer] = -1

    new_agent = state.agent._replace(
        positions=new_positions.astype(np.int32),
        active=new_active,
        carrying=new_carrying,
        resource_phase=new_phase,
        treatment_timer=new_timer,
        locked=new_locked,
        battery=new_battery,
        charging=new_charging,
        battery_dead=new_battery_dead,
        rescue_target=new_rescue_target,
        being_dragged_by=new_being_dragged_by,
        burst_failed=new_burst_failed,
        failed=new_failed,
    )

    return state._replace(agent=new_agent), repair_success, charge_success


def apply_interactions(
    state: EnvState,
    actions: np.ndarray,
    config: WarehouseConfig,
) -> tuple[EnvState, np.ndarray, np.ndarray, np.ndarray]:
    """Apply INTERACT actions.  Returns updated state and event masks."""
    layout = state.grid.layout

    pick_success = np.zeros(config.max_agents, dtype=np.bool_)
    treat_complete = np.zeros(config.max_agents, dtype=np.bool_)
    delivery_success = np.zeros(config.max_agents, dtype=np.bool_)

    new_carrying = state.agent.carrying.copy()
    new_phase = state.agent.resource_phase.copy()
    new_timer = state.agent.treatment_timer.copy()
    new_locked = state.agent.locked.copy()
    new_shelf_resources = state.grid.shelf_resources.copy()
    new_positions = state.agent.positions.copy()
    new_rescue_target = state.agent.rescue_target.copy()
    new_being_dragged_by = state.agent.being_dragged_by.copy()
    new_total_deliveries = state.total_deliveries
    new_delivery_count = state.agent.delivery_count.copy()
    new_last_delivery = state.agent.last_delivery_step.copy()

    interact_mask = (actions == Actions.INTERACT) & state.agent.active
    stranded = _stranded_mask(state)

    for agent_idx in range(config.max_agents):
        # --- tick treatment timer even without INTERACT ---
        if not interact_mask[agent_idx]:
            if new_locked[agent_idx] and new_phase[agent_idx] == ResourcePhase.TREATING:
                new_timer[agent_idx] = max(0, new_timer[agent_idx] - 1)
                if new_timer[agent_idx] <= 0:
                    new_phase[agent_idx] = ResourcePhase.IN_TRANSIT_TO_GOAL
                    new_locked[agent_idx] = False
                    treat_complete[agent_idx] = True
            continue

        pos = state.agent.positions[agent_idx]
        row, col = int(pos[0]), int(pos[1])
        cell_type = layout[row, col]

        cap = int(state.agent.capacity[agent_idx]) if config.enable_heterogeneous else 1

        if new_rescue_target[agent_idx] < 0 and not new_locked[agent_idx]:
            rescue_candidates = []
            for target_idx in range(config.max_agents):
                if target_idx == agent_idx:
                    continue
                if not stranded[target_idx]:
                    continue
                if new_being_dragged_by[target_idx] >= 0:
                    continue
                target_pos = state.agent.positions[target_idx]
                dist = abs(int(target_pos[0]) - row) + abs(int(target_pos[1]) - col)
                if dist <= 1:
                    rescue_candidates.append((dist, target_idx))

            if rescue_candidates:
                _, target_idx = min(rescue_candidates)
                new_rescue_target[agent_idx] = target_idx
                new_being_dragged_by[target_idx] = agent_idx
                new_positions[target_idx] = state.agent.positions[agent_idx]
                continue

        if new_rescue_target[agent_idx] >= 0:
            continue

        # PICK — at shelf, not at capacity
        if cell_type == CellType.SHELF and new_carrying[agent_idx] < cap:
            for shelf_idx in range(config.num_shelves):
                sp = state.grid.shelf_positions[shelf_idx]
                if int(sp[0]) == row and int(sp[1]) == col:
                    slots_to_pick = cap - new_carrying[agent_idx]
                    picked = 0
                    for _ in range(slots_to_pick):
                        if new_shelf_resources[shelf_idx].any():
                            first = int(np.argmax(new_shelf_resources[shelf_idx]))
                            new_shelf_resources[shelf_idx, first] = False
                            picked += 1
                    if picked > 0:
                        new_carrying[agent_idx] += picked
                        new_phase[agent_idx] = ResourcePhase.IN_TRANSIT_TO_TREATMENT
                        pick_success[agent_idx] = True
                    break

        # TREAT — at treatment, carrying, in-transit-to-treatment
        elif (
            cell_type == CellType.TREATMENT
            and new_carrying[agent_idx] > 0
            and new_phase[agent_idx] == ResourcePhase.IN_TRANSIT_TO_TREATMENT
            and not new_locked[agent_idx]
        ):
            new_phase[agent_idx] = ResourcePhase.TREATING
            new_timer[agent_idx] = config.treatment_duration
            new_locked[agent_idx] = True

        # DELIVER — at goal, carrying, in-transit-to-goal
        elif (
            cell_type == CellType.GOAL
            and new_carrying[agent_idx] > 0
            and new_phase[agent_idx] == ResourcePhase.IN_TRANSIT_TO_GOAL
        ):
            delivered_count = int(new_carrying[agent_idx])
            new_carrying[agent_idx] = 0
            new_phase[agent_idx] = ResourcePhase.UNPICKED
            delivery_success[agent_idx] = True
            new_total_deliveries += delivered_count
            new_delivery_count[agent_idx] += delivered_count
            new_last_delivery[agent_idx] = state.step_count

    new_agent = state.agent._replace(
        positions=new_positions.astype(np.int32),
        carrying=new_carrying,
        resource_phase=new_phase,
        treatment_timer=new_timer,
        locked=new_locked,
        rescue_target=new_rescue_target,
        being_dragged_by=new_being_dragged_by,
        delivery_count=new_delivery_count,
        last_delivery_step=new_last_delivery,
    )
    new_grid = state.grid._replace(shelf_resources=new_shelf_resources)
    new_state = state._replace(
        agent=new_agent,
        grid=new_grid,
        total_deliveries=new_total_deliveries,
    )

    return new_state, pick_success, treat_complete, delivery_success


def generate_new_tasks(
    state: EnvState,
    config: WarehouseConfig,
    rng: np.random.Generator,
) -> EnvState:
    """Spawn new deadline-driven tasks via a Poisson process."""
    if not config.enable_task_deadlines:
        return state

    queue: list = list(state.task_queue)

    # Number of new tasks this step ~ Poisson(λ)
    n_new = rng.poisson(config.task_arrival_rate)
    n_new = min(n_new, config.max_pending_tasks - len(queue))

    for _ in range(n_new):
        # Pick a random shelf that still has resources
        available_shelves = [
            s for s in range(config.num_shelves) if state.grid.shelf_resources[s].any()
        ]
        if not available_shelves:
            break
        shelf_idx = rng.choice(available_shelves)
        priority = int(rng.integers(1, config.task_priority_levels + 1))
        deadline = state.step_count + int(
            rng.integers(config.task_deadline_min, config.task_deadline_max + 1)
        )
        queue.append(
            TaskInfo(
                shelf_idx=shelf_idx,
                priority=priority,
                deadline=deadline,
                created_at=state.step_count,
            )
        )

    return state._replace(task_queue=queue)


def expire_tasks(
    state: EnvState,
    config: WarehouseConfig,
) -> tuple[EnvState, int]:
    """Remove tasks whose deadline has passed.  Returns count of expired."""
    if not config.enable_task_deadlines:
        return state, 0

    alive = []
    expired_count = 0
    for task in state.task_queue:
        if state.step_count > task.deadline:
            expired_count += 1
        else:
            alive.append(task)

    new_expired = state.expired_tasks + expired_count
    return state._replace(task_queue=alive, expired_tasks=new_expired), expired_count
