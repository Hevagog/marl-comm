from dataclasses import dataclass


@dataclass
class ContinuousCoordConfig:
    """Configuration for the Continuous Coordination (Rendezvous Pursuit) environment."""

    num_agents: int = 4
    """Number of agents."""

    max_cycles: int = 200
    """Maximum timesteps per episode."""

    max_targets: int = 3
    """Maximum simultaneous active targets."""

    capture_radius: float = 0.08
    """Radius within which agents count toward capturing a target."""

    vision_range: float = 0.4
    """Maximum distance at which an agent can observe teammates."""

    collision_radius: float = 0.03
    """Distance below which two agents receive a collision penalty."""

    target_arrival_rate: float = 0.15
    """Poisson arrival rate for new targets (per step)."""

    target_k_min: int = 2
    """Minimum agents required to capture a target."""

    target_k_max: int = 3
    """Maximum agents required to capture a target."""

    target_deadline_min: int = 30
    """Minimum deadline (steps) for a target before it expires."""

    target_deadline_max: int = 80
    """Maximum deadline (steps) for a target."""

    chain_event_prob: float = 0.2
    """Probability that a new target is a chain event (activates only after
    its parent target is captured)."""

    max_speed: float = 0.05
    """Maximum agent speed per step."""

    dt: float = 1.0
    """Simulation time step (kept at 1.0 for simplicity; speed controls magnitude)."""

    velocity_damping: float = 0.8
    """Damping factor on velocity each step (momentum)."""

    velocity_gain: float = 0.2
    """Gain factor for new action direction each step."""

    # --- reward structure ---
    reward_capture: float = 10.0
    """Reward per captured target (shared among participants)."""

    reward_synchrony_bonus: float = 5.0
    """Bonus for synchronized arrival (scaled by exp(-time_spread))."""

    penalty_deadline: float = -2.0
    """Penalty per expired target (shared across all agents)."""

    reward_proximity_shaping: float = 0.1
    """Per-step shaping reward for approaching an active target."""

    penalty_collision: float = -0.5
    """Penalty when two agents are within collision_radius."""

    # --- typed-agent mode (num_agent_types > 1 activates) ---
    num_agent_types: int = 1
    """Number of distinct agent types. 1 = homogeneous (disables all type logic)."""

    agent_types: list[int] | None = None
    """Per-agent type list of length num_agents. None = round-robin (agent_i → i % num_agent_types).
    Values must be in [0, num_agent_types). Design k_min/k_max so targets remain capturable
    given per-type agent counts."""

    penalty_wrong_type: float = -0.5
    """Per-step penalty for an agent whose type is NOT required by the target, while inside
    the capture zone. Only active when num_agent_types > 1."""

    penalty_wrong_composition: float = -1.0
    """Penalty applied to all agents in the capture zone when total count reaches k_req but
    the per-type composition requirement is unsatisfied. Only active when num_agent_types > 1."""

    # --- rendering ---
    cell_size: int = 600
    """Pixel width/height of the rendered window."""

    fps: int = 15
    """Target frames per second for render_mode='human'."""
