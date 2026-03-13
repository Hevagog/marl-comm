"""Coin Game

Two agents (Red and Blue) move on a discrete grid collecting coins.  Each
agent earns a reward for picking up *any* coin, but stealing the opponent's
coin inflicts a penalty on the opponent.  This creates a social dilemma:
the individually rational strategy (grab everything) is collectively
harmful, making the game a useful testbed for cooperation, Theory-of-Mind,
and opponent modeling.

Rules
-----
- Red and Blue coins spawn randomly.  At most one coin of each color exists
  at a time.
- Picking up ANY coin gives **+pick_reward** to the picker.
- Picking up the OTHER agent's coin gives **steal_penalty** to the victim
  (the agent whose color matches the stolen coin).
- Picking up your OWN color coin has no side-effect on the other agent.
- When a coin is collected, a replacement of the same color spawns at a
  random empty cell.

Observation (per agent, float32, 11 dims)
-----------------------------------------
- Own position (x, y) normalized to [0, 1]              2
- Other agent's position (x, y) normalized to [0, 1]    2
- Red coin position (x, y, exists) normalized           3
- Blue coin position (x, y, exists) normalized          3
- Agent color indicator (1.0 = red/agent_0, 0.0 = blue) 1
- Time (step count normalized to [0, 1])                1
  Total                                                 12

Actions (Discrete 4)
--------------------
UP(0), DOWN(1), LEFT(2), RIGHT(3)
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Mapping, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from .config import CoinGameConfig

_ACTION_UP = 0
_ACTION_DOWN = 1
_ACTION_LEFT = 2
_ACTION_RIGHT = 3

_DIRECTION_DELTAS: dict[int, Tuple[int, int]] = {
    _ACTION_UP: (0, -1),
    _ACTION_DOWN: (0, 1),
    _ACTION_LEFT: (-1, 0),
    _ACTION_RIGHT: (1, 0),
}


class CoinGameEnv:
    """PettingZoo parallel-API Coin Game environment.

    Parameters
    ----------
    config : CoinGameConfig
        Environment configuration.
    render_mode : str or None
        Rendering mode: ``"human"`` (pygame window), ``"rgb_array"``
        (off-screen NumPy array), or ``None`` (headless, default).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "coin_game_v0"}

    def __init__(
        self,
        config: CoinGameConfig | None = None,
        render_mode: Literal["human", "rgb_array"] | None = None,
    ):
        if config is None:
            config = CoinGameConfig()
        self._config = config
        self._render_mode = render_mode

        self._num_players = 2
        self._agent_names = [f"agent_{i}" for i in range(self._num_players)]
        self._possible_agents = list(self._agent_names)

        self._num_actions = 4  # UP, DOWN, LEFT, RIGHT
        self._action_spaces = {
            a: spaces.Discrete(self._num_actions) for a in self._possible_agents
        }

        self._obs_dim = 12
        self._observation_spaces = {
            a: spaces.Box(low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)
            for a in self._possible_agents
        }

        self._state_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._obs_dim * self._num_players,),
            dtype=np.float32,
        )

        self._agents: list[str] = list(self._possible_agents)
        self._step_count: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

        self._agent_positions: dict[str, Tuple[int, int]] = {}
        self._red_coin_pos: Tuple[int, int] | None = None
        self._blue_coin_pos: Tuple[int, int] | None = None

        self._renderer = None

    @property
    def possible_agents(self) -> list[str]:
        return self._possible_agents

    @property
    def agents(self) -> list[str]:
        return list(self._agents)

    @property
    def num_agents(self) -> int:
        return len(self._agents)

    @property
    def max_num_agents(self) -> int:
        return self._num_players

    @property
    def observation_spaces(self) -> Mapping[str, gymnasium.Space]:
        return self._observation_spaces

    @property
    def action_spaces(self) -> Mapping[str, gymnasium.Space]:
        return self._action_spaces

    @property
    def state_spaces(self) -> Mapping[str, gymnasium.Space]:
        return {a: self._state_space for a in self._possible_agents}

    def observation_space(self, agent: str) -> gymnasium.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> gymnasium.Space:
        return self._action_spaces[agent]

    def state(self) -> np.ndarray:
        """Return global state (concatenation of all agent observations)."""
        obs = self._get_observations()
        return np.concatenate([obs[a] for a in self._possible_agents], axis=-1).astype(
            np.float32
        )

    def reset(
        self, seed: int | None = None, **kwargs: Any
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        """Reset the environment and return initial observations.

        Parameters
        ----------
        seed : int or None
            Optional random seed for reproducibility.

        Returns
        -------
        observations : dict[str, np.ndarray]
            Per-agent observation arrays.
        infos : dict[str, dict]
            Per-agent info dicts (empty on reset).
        """
        self._rng = np.random.default_rng(seed)
        self._agents = list(self._possible_agents)
        self._step_count = 0

        # Place agents at random non-overlapping positions
        occupied: set[Tuple[int, int]] = set()
        for agent in self._possible_agents:
            pos = self._random_empty_cell(occupied)
            self._agent_positions[agent] = pos
            occupied.add(pos)

        # Place coins at random positions (not on agents)
        self._red_coin_pos = self._random_empty_cell(occupied)
        occupied.add(self._red_coin_pos)
        self._blue_coin_pos = self._random_empty_cell(occupied)

        obs = self._get_observations()
        infos: Dict[str, dict] = {a: {} for a in self._agents}
        return obs, infos

    def step(self, actions: Mapping[str, int | np.integer]) -> Tuple[
        Dict[str, np.ndarray],  # observation
        Dict[str, float],  # rewards
        Dict[str, bool],  # terminated
        Dict[str, bool],  # truncated
        Dict[str, dict],  # infos
    ]:
        rewards: Dict[str, float] = {a: 0.0 for a in self._possible_agents}
        self._step_count += 1

        gs = self._config.grid_size

        # Move agents simultaneously
        for agent in self._possible_agents:
            action = int(actions[agent])
            dx, dy = _DIRECTION_DELTAS[action]
            old_x, old_y = self._agent_positions[agent]
            new_x = max(0, min(gs - 1, old_x + dx))
            new_y = max(0, min(gs - 1, old_y + dy))
            self._agent_positions[agent] = (new_x, new_y)

        # Resolve coin pickups
        #   Process agent_0 (Red) then agent_1 (Blue).  If both agents
        #   land on the same coin in the same step, only the first in
        #   processing order picks it up (deterministic tie-breaking).
        picked_red = False
        picked_blue = False

        for agent in self._possible_agents:
            pos = self._agent_positions[agent]

            # Check red coin
            if (
                not picked_red
                and self._red_coin_pos is not None
                and pos == self._red_coin_pos
            ):
                picked_red = True
                rewards[agent] += self._config.pick_reward
                # If the Blue agent (agent_1) picks up a RED coin,
                # the Red agent (agent_0) receives the steal penalty.
                if agent == self._possible_agents[1]:
                    rewards[self._possible_agents[0]] += self._config.steal_penalty
                # Respawn red coin
                self._red_coin_pos = self._spawn_coin()

            # Check blue coin
            if (
                not picked_blue
                and self._blue_coin_pos is not None
                and pos == self._blue_coin_pos
            ):
                picked_blue = True
                rewards[agent] += self._config.pick_reward
                # If the Red agent (agent_0) picks up a BLUE coin,
                # the Blue agent (agent_1) receives the steal penalty.
                if agent == self._possible_agents[0]:
                    rewards[self._possible_agents[1]] += self._config.steal_penalty
                # Respawn blue coin
                self._blue_coin_pos = self._spawn_coin()

        done = self._step_count >= self._config.max_cycles

        terminated: Dict[str, bool] = {a: False for a in self._possible_agents}
        truncated: Dict[str, bool] = {a: done for a in self._possible_agents}

        if done:
            self._agents = []

        obs = self._get_observations()
        infos: Dict[str, dict] = {
            a: {
                "step": self._step_count,
            }
            for a in self._possible_agents
        }

        return obs, rewards, terminated, truncated, infos

    def render(self) -> np.ndarray | None:
        """Render the current environment state.

        Returns
        -------
        np.ndarray or None
            An ``(H, W, 3)`` uint8 RGB array when ``render_mode == "rgb_array"``;
            ``None`` when ``render_mode == "human"`` (frame displayed in a
            pygame window) or when ``render_mode is None``.
        """
        if self._render_mode is None:
            return None

        if self._renderer is None:
            from .rendering import CoinGameRenderer

            self._renderer = CoinGameRenderer(
                grid_size=self._config.grid_size,
                render_mode=self._render_mode,
                cell_size=self._config.cell_size,
                fps=self._config.fps,
            )

        return self._renderer.render(
            agent_positions=dict(self._agent_positions),
            red_coin_pos=self._red_coin_pos,
            blue_coin_pos=self._blue_coin_pos,
            step=self._step_count,
            max_steps=self._config.max_cycles,
        )

    def close(self) -> None:
        """Shut down the renderer (if any) and release resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _random_empty_cell(self, occupied: set[Tuple[int, int]]) -> Tuple[int, int]:
        """Return a random grid cell that is not in occupied.

        Parameters
        ----------
        occupied : set of (x, y) tuples
            Positions that must be avoided.

        Returns
        -------
        (x, y) : tuple of int
            A randomly chosen empty position.

        Raises
        ------
        RuntimeError
            If the grid is fully occupied
        """
        gs = self._config.grid_size
        total_cells = gs * gs
        if len(occupied) >= total_cells:
            raise RuntimeError(
                "Cannot find an empty cell -- the grid is fully occupied."
            )

        while True:
            x = int(self._rng.integers(0, gs))
            y = int(self._rng.integers(0, gs))
            if (x, y) not in occupied:
                return (x, y)

    def _spawn_coin(self) -> Tuple[int, int]:
        """Spawn a coin at a random empty cell (not on agents or other coin).

        Returns
        -------
        (x, y) : tuple of int
            Position for the new coin.
        """
        occupied: set[Tuple[int, int]] = set()
        for agent in self._possible_agents:
            if agent in self._agent_positions:
                occupied.add(self._agent_positions[agent])
        if self._red_coin_pos is not None:
            occupied.add(self._red_coin_pos)
        if self._blue_coin_pos is not None:
            occupied.add(self._blue_coin_pos)
        return self._random_empty_cell(occupied)

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """
        Each observation is an 11-dimensional float32 vector:

        Index  Description
        -----  -----------
        0-1    Own position (x, y) normalized to [0, 1]
        2-3    Other agent's position (x, y) normalized to [0, 1]
        4-6    Red coin (x_norm, y_norm, exists)
        7-9    Blue coin (x_norm, y_norm, exists)
        10     Agent color indicator (1.0 for red/agent_0, 0.0 for blue/agent_1)
        11     Time (step count normalized to [0, 1])
        """
        gs_norm = max(self._config.grid_size - 1, 1)
        obs: Dict[str, np.ndarray] = {}

        for i, agent in enumerate(self._possible_agents):
            partner = self._possible_agents[1 - i]
            features = np.zeros(self._obs_dim, dtype=np.float32)

            # Own position
            own_x, own_y = self._agent_positions[agent]
            features[0] = own_x / gs_norm
            features[1] = own_y / gs_norm

            # Other agent position
            other_x, other_y = self._agent_positions[partner]
            features[2] = other_x / gs_norm
            features[3] = other_y / gs_norm

            # Red coin
            if self._red_coin_pos is not None:
                features[4] = self._red_coin_pos[0] / gs_norm
                features[5] = self._red_coin_pos[1] / gs_norm
                features[6] = 1.0
            # else: already zeros (position 0,0 with exists=0)

            # Blue coin
            if self._blue_coin_pos is not None:
                features[7] = self._blue_coin_pos[0] / gs_norm
                features[8] = self._blue_coin_pos[1] / gs_norm
                features[9] = 1.0
            # else: already zeros

            # Agent color indicator: 1.0 for red (agent_0), 0.0 for blue (agent_1)
            features[10] = 1.0 if i == 0 else 0.0

            features[11] = 1.0 - (self._step_count / self._config.max_cycles)

            obs[agent] = features

        return obs


def make_coin_game_env(
    config: CoinGameConfig | None = None,
    render_mode: Literal["human", "rgb_array"] | None = None,
) -> CoinGameEnv:
    """Create a Coin Game environment with PettingZoo parallel API.

    Parameters
    ----------
    config : CoinGameConfig or None
        Environment configuration.  Uses defaults if ``None``.
    render_mode : str or None
        Rendering mode.

    Returns
    -------
    CoinGameEnv
        PettingZoo-compatible parallel environment instance.
    """
    if config is None:
        config = CoinGameConfig()
    return CoinGameEnv(config=config, render_mode=render_mode)
