from dataclasses import dataclass, field


@dataclass(frozen=True, eq=True)
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


@dataclass(frozen=True, eq=True)
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

    # Per-episode layout randomization (structural jitter within fixed bands).
    randomize_layout: bool = False
    layout_shuffle_max_retries: int = 20

    # Anomaly probabilities
    agent_failure_prob: float = 0.001
    comm_noise_prob: float = 0.1

    # Master switch for the attrition/rescue sub-task.  When False, no agent can
    # become inactive/stranded: uniform + burst failures are skipped and battery
    # depletion can no longer deactivate an agent.
    enable_attrition: bool = True

    # Observation settings
    vision_range: int = 3

    # Reward configuration
    reward_pick: float = 0.1
    reward_treatment_complete: float = 0.5
    reward_delivery: float = 10.0
    reward_rescue_repair: float = 12.0
    reward_rescue_charge: float = 12.0
    # Small per-step reward for approaching a stranded teammate (0 = disabled).
    # Provides intermediate gradient for rescue navigation; without it, agents
    # must chain 10-30 steps blindly before receiving any rescue signal.
    reward_rescue_proximity: float = 0.0
    penalty_congestion: float = -0.05
    penalty_collision: float = -1.0
    step_penalty: float = -0.01

    # Episode configuration
    max_cycles: int = 500

    comm_range: int | None = None  # max grid-cell distance for observing others

    # ---- No-communication mode ----
    # When True, agents observe only local vision (no teammate internal state).
    # Visibility is gated by vision_range (not comm_range).
    # Within range: position + is_carrying + is_stranded visible (physically observable).
    # Not visible: battery, active flag, resource_phase, task state (all require radio).
    # Obs-dim is unchanged — same feature slots, zeroed where info requires comms.
    no_comm: bool = False

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

    # Team-synchronized delivery bonus
    # When True, every delivery emits a team bonus shared by any agent that
    # ALSO delivered within the last `team_delivery_window` steps.  Forces
    # temporal coordination of the Pick→Treat→Deliver chain across agents.
    enable_team_delivery_bonus: bool = False
    team_delivery_window: int = 20
    team_delivery_bonus: float = 5.0
    team_delivery_min_partners: int = 1  # need at least this many co-deliverers

    # Rendezvous cell
    # When True, env places `num_rendezvous` cells.  Standing on one with at
    # least `rendezvous_min_agents` total agents on it pays each occupant
    # `reward_rendezvous`.  Only paid once per visit (one-shot per occupancy).
    enable_rendezvous: bool = False
    num_rendezvous: int = 1
    rendezvous_min_agents: int = 2
    reward_rendezvous: float = 20.0
    rendezvous_cooldown: int = 50  # steps between re-trigger of same cell

    # Hide the 8-feature infrastructure GPS block (obs[7:15]).
    # When True, relative vectors to treatment/goal/repair/charger are zeroed.
    # Forces agents to discover infrastructure via exploration or peer communication.
    # obs_dim is unchanged — same feature slots, zeroed when radio info is hidden.
    hide_infra_obs: bool = False

    fault_profile: FaultProfile = field(default_factory=FaultProfile)

    def __post_init__(self):
        """Cast plain dicts into typed dataclasses."""
        if isinstance(self.fault_profile, dict):
            object.__setattr__(
                self, "fault_profile", FaultProfile(**self.fault_profile)
            )
