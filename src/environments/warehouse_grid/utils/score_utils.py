import numpy as np

from ..config import WarehouseConfig
from .types import EnvState


def compute_congestion_penalty(
    state: EnvState,
    config: WarehouseConfig,
) -> np.ndarray:
    """Congestion penalty based on agents sharing the same grid column."""
    column_counts = np.zeros(config.grid_width, dtype=np.float32)
    for i in range(config.max_agents):
        if state.agent.active[i]:
            col = int(state.agent.positions[i, 1])
            column_counts[col] += 1.0

    penalties = np.zeros(config.max_agents, dtype=np.float32)
    for i in range(config.max_agents):
        col = int(state.agent.positions[i, 1])
        count = column_counts[col]
        penalties[i] = max(0, count - 1) * config.penalty_congestion

    return penalties


def compute_rewards(
    state: EnvState,
    pick_success: np.ndarray,
    treat_complete: np.ndarray,
    delivery_success: np.ndarray,
    repair_rescue_success: np.ndarray,
    charge_rescue_success: np.ndarray,
    collision_mask: np.ndarray,
    config: WarehouseConfig,
    expired_tasks: int = 0,
) -> np.ndarray:
    rewards = np.zeros(config.max_agents, dtype=np.float32)
    active_f = state.agent.active.astype(np.float32)

    # --- step penalty ---
    rewards += active_f * config.step_penalty

    # --- task rewards ---
    rewards += pick_success.astype(np.float32) * config.reward_pick
    rewards += treat_complete.astype(np.float32) * config.reward_treatment_complete

    # --- delivery: global bonus + individual ---
    num_deliveries = delivery_success.sum()
    num_active = state.agent.active.sum()
    if num_active > 0:
        global_bonus = num_deliveries * config.reward_delivery / num_active
        rewards += active_f * global_bonus

    rewards += delivery_success.astype(np.float32) * config.reward_delivery
    rewards += repair_rescue_success.astype(np.float32) * config.reward_rescue_repair
    rewards += charge_rescue_success.astype(np.float32) * config.reward_rescue_charge

    # --- urgent-delivery bonus ---
    # (Caller can provide per-agent priority info via infos; here we use a
    #  simple heuristic: agents that deliver while the task queue has urgent
    #  tasks get a bonus proportional to the highest pending priority.)
    if config.enable_task_deadlines and len(state.task_queue) > 0:
        max_priority = max(t.priority for t in state.task_queue)
        urgency_bonus = (
            max_priority / config.task_priority_levels
        ) * config.reward_urgent_delivery
        rewards += delivery_success.astype(np.float32) * urgency_bonus

    # --- team penalty for expired tasks ---
    if config.enable_task_deadlines and expired_tasks > 0 and num_active > 0:
        per_agent_penalty = (expired_tasks * config.penalty_task_expired) / num_active
        rewards += active_f * per_agent_penalty

    # --- congestion penalty ---
    rewards += compute_congestion_penalty(state, config)

    # --- collision penalty ---
    rewards += collision_mask.astype(np.float32) * config.penalty_collision

    # --- zero out inactive ---
    rewards *= active_f

    return rewards


def update_task_priorities(
    state: EnvState,
    config: WarehouseConfig,
    rng: np.random.Generator,
) -> EnvState:
    H, W = config.grid_height, config.grid_width

    new_priorities = state.grid.task_priorities * 0.95

    # Boost shelves with resources
    for i in range(config.num_shelves):
        pos = state.grid.shelf_positions[i]
        resource_count = state.grid.shelf_resources[i].sum()
        if resource_count > 0:
            priority_boost = resource_count / config.resources_per_shelf
            new_priorities[pos[0], pos[1]] += priority_boost * 0.1

    #  boost from approaching deadlines
    if config.enable_task_deadlines:
        for task in state.task_queue:
            remaining = max(1, task.deadline - state.step_count)
            urgency = task.priority / max(remaining, 1)
            sp = state.grid.shelf_positions[task.shelf_idx]
            new_priorities[sp[0], sp[1]] += urgency * 0.2

    # Random priority spikes
    spike_mask = rng.random((H, W)) < 0.01
    new_priorities += spike_mask.astype(np.float32) * 0.5

    new_priorities = np.clip(new_priorities, 0.0, 1.0)
    new_grid = state.grid._replace(task_priorities=new_priorities)

    return state._replace(grid=new_grid)
