"""Coin Game with partial observability.

Each agent can only observe objects (other agent, coins) within
``vision_range`` Manhattan distance.  Coins outside that range are
treated as non-existent from the observer's perspective.

Observation (per agent, float32, 13 dims)
-----------------------------------------
  [0-1]    Own position (x, y) normalized to [0, 1]
  [2-4]    Other agent (x, y, visible)  — visible=0 when outside range
  [5-7]    Red coin   (x, y, in_range+exists)
  [8-10]   Blue coin  (x, y, in_range+exists)
  [11]     Agent color indicator (1.0 = red/agent_0, 0.0 = blue)
  [12]     Time remaining (1 - step/max_cycles)

Global state: concatenation of both agents' obs (26 dims).
Note: the critic therefore sees the *union* of what both agents observe,
not the ground-truth full state.  This matches the standard CTDE pattern
used throughout this codebase (see CoinGameEnv.state()).

Scientific motivation
---------------------
Leibo et al. (2017) "Multi-agent RL in Sequential Social Dilemmas"
showed that reducing vision range shifts equilibria toward defection
because agents cannot verify whether a partner is cooperating.
Here, communication-equipped agents (MAGIC, CommFormer, MAM) have a
genuine informational advantage: they can share coin locations and
partner positions that fall outside each other's vision cone.
The key dilemma:
  - Cooperative signal  → "coin at (x,y), not mine, go get it"
  - Exploitative signal → mislead partner while racing to steal

Reference: arxiv.org/abs/1702.03037 (Leibo et al. 2017)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from gymnasium import spaces

from .coingame import CoinGameEnv
from .config import CoinGameConfig


class CoinGamePartialObsEnv(CoinGameEnv):
    """CoinGame with Manhattan-distance-limited vision.

    Parameters
    ----------
    config : CoinGameConfig or None
        Must have ``vision_range`` set (default 2).  The base grid_size,
        max_cycles, pick_reward, and steal_penalty fields behave identically
        to the fully-observable env.
    render_mode : str or None
        Same as CoinGameEnv.
    """

    def __init__(
        self,
        config: CoinGameConfig | None = None,
        render_mode: Literal["human", "rgb_array"] | None = None,
    ):
        if config is None:
            config = CoinGameConfig()
        # Parent sets _obs_dim=12 and builds observation/state spaces.
        super().__init__(config=config, render_mode=render_mode)

        # Override obs dim: +1 for the other-agent visibility flag.
        self._obs_dim = 13
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

    # ------------------------------------------------------------------
    # Core observation logic
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict[str, np.ndarray]:  # type: ignore[override]
        gs_norm = max(self._config.grid_size - 1, 1)
        vr = self._config.vision_range
        obs: dict[str, np.ndarray] = {}

        for i, agent in enumerate(self._possible_agents):
            partner = self._possible_agents[1 - i]
            features = np.zeros(self._obs_dim, dtype=np.float32)

            own_x, own_y = self._agent_positions[agent]
            features[0] = own_x / gs_norm
            features[1] = own_y / gs_norm

            # Other agent — Manhattan distance gate
            other_x, other_y = self._agent_positions[partner]
            if abs(own_x - other_x) + abs(own_y - other_y) <= vr:
                features[2] = other_x / gs_norm
                features[3] = other_y / gs_norm
                features[4] = 1.0  # visible flag

            # Red coin — zero if outside range OR coin doesn't exist.
            # "exists=0" therefore conflates physical absence with out-of-range;
            # since coins respawn immediately, it effectively encodes only range.
            if self._red_coin_pos is not None:
                rx, ry = self._red_coin_pos
                if abs(own_x - rx) + abs(own_y - ry) <= vr:
                    features[5] = rx / gs_norm
                    features[6] = ry / gs_norm
                    features[7] = 1.0

            # Blue coin
            if self._blue_coin_pos is not None:
                bx, by = self._blue_coin_pos
                if abs(own_x - bx) + abs(own_y - by) <= vr:
                    features[8] = bx / gs_norm
                    features[9] = by / gs_norm
                    features[10] = 1.0

            features[11] = 1.0 if i == 0 else 0.0  # agent color
            features[12] = 1.0 - (self._step_count / self._config.max_cycles)

            obs[agent] = features

        return obs


def make_coin_game_partial_obs_env(
    config: CoinGameConfig | None = None,
    render_mode: Literal["human", "rgb_array"] | None = None,
) -> CoinGamePartialObsEnv:
    """Factory for CoinGamePartialObsEnv (PettingZoo parallel API)."""
    if config is None:
        config = CoinGameConfig()
    return CoinGamePartialObsEnv(config=config, render_mode=render_mode)
