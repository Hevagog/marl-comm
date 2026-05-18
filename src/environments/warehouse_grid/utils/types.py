from enum import IntEnum
from typing import NamedTuple

import numpy as np


class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    SHELF = 2
    TREATMENT = 3
    GOAL = 4
    SPAWN = 5
    CHARGER = 6
    REPAIR = 7
    RENDEZVOUS = 8


class ResourcePhase(IntEnum):
    UNPICKED = 0
    IN_TRANSIT_TO_TREATMENT = 1
    TREATING = 2
    IN_TRANSIT_TO_GOAL = 3


class AgentState(NamedTuple):
    """Per-agent state information.

    Original fields are listed first; new fields added by each layer
    follow with a comment tag.

    Attributes
    ----------
    positions : (max_agents, 2) int32 — (row, col) coordinates.
    active : (max_agents,) bool — is the agent active?
    carrying : (max_agents,) int32 — number of resources currently held
    resource_phase : (max_agents,) int32 — current ResourcePhase.
    treatment_timer : (max_agents,) int32 — remaining treatment ticks.
    locked : (max_agents,) bool — frozen at treatment station?
    speed : (max_agents,) int32 — cells per move.
    capacity : (max_agents,) int32 — max items carried.
    fragility : (max_agents,) float32 — failure-prob multiplier.
    battery : (max_agents,) int32 — remaining battery.
    charging : (max_agents,) bool — currently docked at charger .
    battery_dead : (max_agents,) bool — stranded due to empty battery.
    rescue_target : (max_agents,) int32 — which stranded agent this robot is towing.
    being_dragged_by : (max_agents,) int32 — rescuer index for stranded robots.
    burst_failed : (max_agents,) bool — permanently burst-failed.
    failed : (max_agents,) bool — deactivated by normal attrition.
    """

    positions: np.ndarray
    active: np.ndarray
    carrying: np.ndarray
    resource_phase: np.ndarray
    treatment_timer: np.ndarray
    locked: np.ndarray

    speed: np.ndarray
    capacity: np.ndarray
    fragility: np.ndarray

    battery: np.ndarray
    charging: np.ndarray
    battery_dead: np.ndarray
    rescue_target: np.ndarray
    being_dragged_by: np.ndarray

    burst_failed: np.ndarray
    failed: np.ndarray

    delivery_count: np.ndarray
    last_delivery_step: np.ndarray


class TaskInfo(NamedTuple):
    """A single pending task in the dynamic task queue.

    Attributes
    ----------
    shelf_idx : int — which shelf the resource must be picked from.
    priority : int — 1 (low) … task_priority_levels (urgent).
    deadline : int — env step at which the task expires.
    created_at : int — env step when the task was created.
    """

    shelf_idx: int
    priority: int
    deadline: int
    created_at: int


class GridState(NamedTuple):
    """Grid / world state information.

    Attributes
    ----------
    layout : (H, W) int32 — CellType per cell.
    shelf_resources : (num_shelves, resources_per_shelf) bool — availability.
    shelf_positions : (num_shelves, 2) int32 — shelf coordinates.
    treatment_positions : (num_treatment, 2) int32.
    goal_positions : (num_goals, 2) int32.
    spawn_positions : (num_spawns, 2) int32.
    task_priorities : (H, W) float32 — dynamic priority heatmap.
    interference_map : (H, W) float32 — comm noise multiplier.
    charger_positions : (num_chargers, 2) int32.
    repair_position : (2,) int32.
    rendezvous_positions : (k, 2) int32 — [-1,-1] sentinel when disabled.
    rendezvous_triggered_at : (k,) int32 — step when edge-trigger last fired; init -(10**6).
    rendezvous_above_threshold : (k,) bool — whether cell was above min_agents last step.
    """

    layout: np.ndarray
    shelf_resources: np.ndarray
    shelf_positions: np.ndarray
    treatment_positions: np.ndarray
    goal_positions: np.ndarray
    spawn_positions: np.ndarray
    task_priorities: np.ndarray

    interference_map: np.ndarray

    charger_positions: np.ndarray
    repair_position: np.ndarray
    rendezvous_positions: np.ndarray
    rendezvous_triggered_at: np.ndarray
    rendezvous_above_threshold: np.ndarray


class EnvState(NamedTuple):
    """Complete environment state.

    Attributes
    ----------
    agent : AgentState.
    grid : GridState.
    step_count : int — current timestep.
    total_deliveries : int — cumulative successful deliveries.
    task_queue : list[TaskInfo] — pending dynamic tasks.
    expired_tasks : int — cumulative expired tasks.
    """

    agent: AgentState
    grid: GridState
    step_count: int
    total_deliveries: int

    task_queue: list[TaskInfo]
    expired_tasks: int


class Actions(IntEnum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    INTERACT = 5


NUM_ACTIONS = 6

# Movement deltas: mapping action index to (row, col) changes.
ACTION_DELTAS = np.array(
    [
        [0, 0],  # STAY
        [-1, 0],  # UP
        [1, 0],  # DOWN
        [0, -1],  # LEFT
        [0, 1],  # RIGHT
        [0, 0],  # INTERACT
    ],
    dtype=np.int32,
)
