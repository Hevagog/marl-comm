from dataclasses import dataclass


@dataclass
class CoinGameConfig:
    grid_size: int = 7
    """Side length of the square grid."""

    max_cycles: int = 50
    """Maximum number of timesteps per episode."""

    pick_reward: float = 1.0
    """Reward given to the agent that picks up any coin."""

    steal_penalty: float = -2.0
    """Penalty applied to the *other* agent when its colored coin is stolen."""
