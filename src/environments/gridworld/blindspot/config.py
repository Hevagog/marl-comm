from dataclasses import dataclass


@dataclass
class BlindSpotConfig:
    """Configuration for the Blind-Spot Navigation environment."""

    grid_size: int = 7
    """Side length of the square grid."""

    max_cycles: int = 100
    """Maximum timesteps per episode."""

    num_traps: int = 5
    """Number of hidden traps placed each episode."""

    trap_penalty: float = -5.0
    """Reward penalty when an agent steps on a trap."""

    goal_reward: float = 10.0
    """Reward granted when an agent reaches the goal."""

    vision_range: int = 0
    """Agent B's vision radius (1 -> 3x3 window)."""

    step_penalty: float = -0.01
    """Small per-step cost to encourage efficiency."""

    use_distance_shaping: bool = False
    """Enable potential-based reward shaping (Ng et al. 1999) using
    Manhattan distance to goal.  Provides a reward gradient so that
    the preference prior C can learn even before the goal is reached."""

    distance_shaping_scale: float = 0.5
    """Scale factor for distance-based shaping reward.  The raw potential
    difference Φ(s') - Φ(s) = -(new_dist - old_dist) / max_dist lies in
    [-1/(2*(grid_size-1)), +1/(2*(grid_size-1))]; for grid_size=9 this is
    [-1/16, +1/16] ≈ ±0.0625 per step.  This multiplier controls how much
    the shaping reward contributes relative to the step penalty and
    goal/trap rewards.  With scale=0.5 and trap_penalty=-5.0, a two-step
    detour costs only ~0.08 in shaping loss — far less than a trap hit,
    so the shaping does not override trap-avoidance incentives."""

    fixed_trap_seed: int | None = None
    """When set, traps are placed using this fixed seed every episode,
    making the environment fully deterministic.  This allows the tabular
    transition model B and preference prior C to learn exact dynamics
    rather than averaging over random trap configurations.

    Rationale: with 16 hidden states and 81-choose-5 possible trap
    layouts, the agent cannot represent per-layout dynamics.  Fixing
    traps makes the environment learnable by tabular AIF while
    preserving the cooperative signaling challenge."""

    use_communication: bool = False
    """Enable discrete message channel between agents.  Each agent
    selects a composite action = movement * M + message_token.  The
    partner's last message is appended to the observation as a one-hot
    vector of dimension ``num_message_tokens``.

    Literature backing:
    - Friston & Frith (2015): communication IS action in AIF
    - MARL-CPC (Yoshida & Taniguchi 2025): discrete tokens for state inference
    - Maisto et al. (2023): emergent sensorimotor communication in AIF"""

    num_message_tokens: int = 4
    """Number of discrete message tokens available per step.  Each token
    is semantically ungrounded — meaning emerges from training.

    With 5 movement actions and M=4 tokens, composite action space has
    5*4 = 20 actions.  Tabular B grows from (5, 16, 16) = 1,280 cells
    to (20, 16, 16) = 5,120 cells — still manageable for 2M+ steps."""

    random_goal: bool = True
    """If True, sample a new goal position at every reset()."""

    min_goal_start_distance: int | None = None
    """Minimum Manhattan distance from BOTH spawn positions for sampled goals.

    If None, a dynamic default is used:
        max(2, grid_size // 2)

    This avoids trivially easy episodes where the goal spawns near starts.
    """
