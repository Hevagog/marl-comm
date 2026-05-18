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
) -> tuple[np.ndarray, EnvState]:
    rewards = np.zeros(config.max_agents, dtype=np.float32)
    active_f = state.agent.active.astype(np.float32)

    # --- step penalty ---
    rewards += active_f * config.step_penalty

    # --- task rewards ---
    rewards += pick_success.astype(np.float32) * config.reward_pick
    rewards += treat_complete.astype(np.float32) * config.reward_treatment_complete

    # --- delivery reward ---
    # Design choice: pure individual reward. The delivering agent receives
    # the full reward_delivery; non-delivering agents receive nothing for
    # this component. This avoids the prior bug (WH-B01) where both a global
    # team bonus AND an individual bonus were applied, effectively doubling
    # the nominal reward_delivery per successful delivery.
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
    num_active = state.agent.active.sum()
    if config.enable_task_deadlines and expired_tasks > 0 and num_active > 0:
        per_agent_penalty = (expired_tasks * config.penalty_task_expired) / num_active
        rewards += active_f * per_agent_penalty

    # --- congestion penalty ---
    rewards += compute_congestion_penalty(state, config)

    # --- collision penalty ---
    rewards += collision_mask.astype(np.float32) * config.penalty_collision

    # --- team-synchronized delivery bonus  ---
    # Pays the team_delivery_bonus to every agent that delivered within
    # the same `team_delivery_window` of steps as this delivery.  Designed
    # to make MAPPO's per-agent advantage estimator under-credit deliveries
    # that lack temporal cooperation, opening a measurable gap for methods
    # that share peer-phase information (CommFormer / MAM / Hopfield).
    if config.enable_team_delivery_bonus and delivery_success.any():
        window = max(1, config.team_delivery_window)
        last = state.agent.last_delivery_step
        active = state.agent.active
        for i in np.where(delivery_success)[0]:
            partners = 0
            for j in range(config.max_agents):
                if j == i or not active[j]:
                    continue
                if last[j] < 0:
                    continue
                if state.step_count - int(last[j]) <= window:
                    partners += 1
            if partners >= config.team_delivery_min_partners:
                rewards[i] += config.team_delivery_bonus
                # Also reward the partners — symmetric joint signal.
                for j in range(config.max_agents):
                    if j == i or not active[j]:
                        continue
                    if last[j] < 0:
                        continue
                    if state.step_count - int(last[j]) <= window:
                        rewards[j] += config.team_delivery_bonus

    # --- rendezvous occupancy bonus (Scenario 2) ---
    # Edge-trigger + cooldown: reward fires once on the LOW→HIGH transition
    # (occupants crosses rendezvous_min_agents threshold) and is suppressed
    # for rendezvous_cooldown steps afterward.  Prevents profitable camping.
    # Backed by Ng et al. (1999) potential-based shaping theory.
    if config.enable_rendezvous and state.grid.rendezvous_positions.size > 0:
        rendez = state.grid.rendezvous_positions
        new_triggered_at = state.grid.rendezvous_triggered_at.copy()
        new_above_threshold = state.grid.rendezvous_above_threshold.copy()
        for k in range(rendez.shape[0]):
            r, c = int(rendez[k, 0]), int(rendez[k, 1])
            if r < 0 or c < 0:
                continue
            occupants = []
            for i in range(config.max_agents):
                if not state.agent.active[i]:
                    continue
                if (
                    int(state.agent.positions[i, 0]) == r
                    and int(state.agent.positions[i, 1]) == c
                ):
                    occupants.append(i)
            currently_above = len(occupants) >= config.rendezvous_min_agents
            was_above = bool(state.grid.rendezvous_above_threshold[k])
            steps_since = state.step_count - int(state.grid.rendezvous_triggered_at[k])
            if currently_above and not was_above and steps_since > config.rendezvous_cooldown:
                for i in occupants:
                    rewards[i] += config.reward_rendezvous
                new_triggered_at[k] = state.step_count
            new_above_threshold[k] = currently_above
        new_grid = state.grid._replace(
            rendezvous_triggered_at=new_triggered_at,
            rendezvous_above_threshold=new_above_threshold,
        )
        state = state._replace(grid=new_grid)

    # --- rescue proximity shaping ---
    # Provides intermediate gradient for the rescue navigation sub-task.
    # Without this, agents must navigate 10-30 steps to a stranded teammate
    # before receiving any signal — the rescue gradient is essentially zero.
    # Reward = proximity_scale / (distance + 1) for each active agent near a
    # stranded teammate; capped so it never exceeds a single delivery reward.
    if config.reward_rescue_proximity > 0.0:
        from .agent_utils import _stranded_mask

        stranded = _stranded_mask(state)
        stranded_positions = [
            state.agent.positions[j] for j in range(config.max_agents) if stranded[j]
        ]
        if stranded_positions:
            scale = config.reward_rescue_proximity
            for i in range(config.max_agents):
                if not state.agent.active[i]:
                    continue
                # Skip agents already performing a rescue (they get the big reward)
                if state.agent.rescue_target[i] >= 0:
                    continue
                pi = state.agent.positions[i]
                for sp in stranded_positions:
                    dist = abs(int(sp[0]) - int(pi[0])) + abs(int(sp[1]) - int(pi[1]))
                    rewards[i] += scale / (dist + 1)

    # --- zero out inactive ---
    rewards *= active_f

    return rewards, state


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
