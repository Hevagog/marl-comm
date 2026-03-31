from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FaultProfile:
    # --- Burst failure ---
    # Robot works fine then suddenly enters permanent failure.
    # Models battery death, motor burnout.
    burst_attrition: bool = False
    burst_prob: float = 0.0  # per-step probability of entering burst-failure mode

    # --- Correlated failure ---
    # Multiple robots fail simultaneously when one fails.
    # Models network outage, power spike.
    correlated_failure: bool = False
    correlation_radius: int = 3  # robots within N cells of a failing robot also fail

    # --- Load-dependent communication ---
    # Message loss increases linearly with local agent density.
    # Models RF congestion under high robot density.
    load_dependent_comm: bool = False
    base_packet_loss: float = 0.05
    congestion_factor: float = 0.1  # packet_loss += factor * local_density


@dataclass
class WarehouseConfig:
    grid_height: int = 12
    grid_width: int = 16

    num_agents: int = 4
    max_agents: int = 8
    num_shelves: int = 6
    resources_per_shelf: int = 4
    num_treatment_stations: int = 2
    num_goal_locations: int = 2
    treatment_duration: int = 5

    # Anomaly probabilities
    agent_failure_prob: float = 0.001
    comm_noise_prob: float = 0.1

    # Observation settings
    vision_range: int = 3

    # Reward configuration
    reward_pick: float = 0.1
    reward_treatment_complete: float = 0.5
    reward_delivery: float = 10.0
    reward_rescue_repair: float = 12.0
    reward_rescue_charge: float = 12.0
    penalty_congestion: float = -0.05
    penalty_collision: float = -1.0
    step_penalty: float = -0.01

    # Episode configuration
    max_cycles: int = 500

    comm_range: Optional[int] = None  # max grid-cell distance for observing others

    enable_task_deadlines: bool = False
    task_arrival_rate: float = 0.5  # Poisson λ — expected new tasks per step
    task_deadline_min: int = 30  # minimum steps before a task expires
    task_deadline_max: int = 80  # maximum steps before a task expires
    max_pending_tasks: int = 20  # cap on simultaneous pending tasks
    penalty_task_expired: float = -2.0  # team penalty per expired task
    task_priority_levels: int = 3  # 1=low … 3=urgent
    reward_urgent_delivery: float = 5.0  # bonus multiplier for high-priority tasks

    enable_heterogeneous: bool = False
    # When enabled, agents are assigned profiles cyclically from these lists.
    # Length of each list defines the number of distinct robot types.
    agent_speed_options: tuple = (1, 1, 2)  # cells per move (assigned round-robin)
    agent_capacity_options: tuple = (1, 2, 1)  # max items carried simultaneously
    agent_fragility_options: tuple = (1.0, 0.5, 2.0)  # multiplier on failure prob

    enable_interference_zones: bool = False
    interference_base: float = 0.05  # baseline noise everywhere
    interference_treatment_boost: float = 0.4  # extra noise near treatment stations
    interference_radius: int = 2  # how far interference radiates from source

    enable_battery: bool = False
    battery_capacity: int = 100  # full charge value
    battery_drain_per_step: int = 1  # drain when moving
    battery_drain_idle: int = 0  # drain when stationary
    battery_charge_rate: int = 5  # charge gained per step at charger
    battery_critical_threshold: int = 15  # forced return to charger below this
    num_charging_stations: int = 2

    fault_profile: FaultProfile = field(default_factory=FaultProfile)

    def __post_init__(self):
        """Cast plain dicts into typed dataclasses."""
        if isinstance(self.fault_profile, dict):
            self.fault_profile = FaultProfile(**self.fault_profile)
