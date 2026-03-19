from dataclasses import dataclass, field


@dataclass
class IntersectionConfig:
    """Configuration for HighwayEnv Multi-Agent Intersection."""

    num_agents: int = 4
    """Number of controlled vehicles."""

    duration: int = 13
    """Episode duration in policy steps."""

    vehicles_count: int = 10
    """Number of vehicles in each agent's observation."""

    features: list[str] = field(
        default_factory=lambda: ["presence", "x", "y", "vx", "vy"]
    )
    """Kinematic features per observed vehicle."""

    initial_vehicle_count: int = 10
    """Number of non-controlled vehicles spawned initially."""

    spawn_probability: float = 0.6
    """Probability of spawning a new vehicle each step."""

    collision_reward: float = -5.0
    """Reward for collision."""

    arrived_reward: float = 1.0
    """Reward for successfully crossing the intersection."""

    high_speed_reward: float = 1.0
    """Reward for maintaining target speed."""

    reward_speed_range: list[float] = field(default_factory=lambda: [7.0, 9.0])
    """Speed range for speed reward interpolation."""

    normalize_reward: bool = True
    """Normalize reward to approximately [-1, 1]."""

    simulation_frequency: int = 15
    """Simulation frequency (Hz)."""

    policy_frequency: int = 1
    """Policy frequency (decisions per second)."""
