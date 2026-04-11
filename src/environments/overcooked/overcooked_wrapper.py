"""Overcooked-AI PettingZoo-compatible parallel environment wrapper.

Two chefs must cooperate to pick up onions, cook them in a pot, plate the soup,
and deliver it to a serving location. No communication channel -- agents must
infer each other's intentions purely from observed behavior.

The wrapper converts the ``overcooked_ai_py`` ``OvercookedEnv`` into a
PettingZoo **parallel** API so it plugs directly into the existing training
infrastructure (MAPPO, AI, FAI, GNN).

Observation format
------------------
By default the wrapper uses the **lossless state encoding** from overcooked-ai
which produces a per-player tensor of shape ``(width, height, 26)`` (int64).
This is flattened to a 1-D float32 vector for compatibility with MLP-based
policies.  Set ``use_dense_obs=True`` to instead use a hand-crafted dense
feature vector (requires ``overcooked_ai_py >= 1.1``).

Layouts
-------
Common layouts for coordination study:

- ``cramped_room``  (5x4, tight space, forces close coordination)
- ``asymmetric_advantages`` (different sides have different ingredients)
- ``coordination_ring`` (circular layout, agents must pass each other)
- ``forced_coordination`` (one agent must hand ingredients to the other)
- ``counter_circuit`` (a loop with a counter in the middle)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from collections.abc import Mapping

import gymnasium
import numpy as np
from gymnasium import spaces

# overcooked-ai uses np.Inf which was removed in NumPy 2.0
if not hasattr(np, "Inf"):
    np.Inf = np.inf  # type: ignore[attr-defined]

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


@dataclass
class OvercookedConfig:
    """Configuration for Overcooked-AI environment."""

    layout_name: str = "cramped_room"
    """Layout to use. See overcooked_ai_py/data/layouts/ for options."""

    horizon: int = 200
    """Maximum steps per episode."""

    use_dense_obs: bool = False
    """Use dense feature vector instead of lossless grid encoding.
    If False (default), observations are the flattened lossless encoding
    of shape (width * height * 26,).  If True, uses a compact feature
    vector (requires featurize_fn or custom features)."""

    reward_shaping: bool = True
    """Enable shaped per-agent rewards from overcooked-ai (in addition
    to the shared sparse reward for successful deliveries)."""

    reward_shaping_factor: float = 1.0
    """Multiplier for shaped reward component."""


class OvercookedPettingZooEnv:
    """PettingZoo parallel-API wrapper for Overcooked-AI.

    Wraps ``overcooked_ai_py.mdp.overcooked_env.OvercookedEnv`` to provide
    the standard PettingZoo parallel interface: dict-keyed observations,
    actions, rewards, terminated, truncated, and infos.

    Parameters
    ----------
    config : OvercookedConfig
        Environment configuration.
    render_mode : str or None
        "human" for Pygame window, "rgb_array" for numpy frames, None for
        headless.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "overcooked_v0"}

    def __init__(
        self,
        config: OvercookedConfig | None = None,
        render_mode: Literal["human", "rgb_array"] | None = None,
    ):
        if config is None:
            config = OvercookedConfig()
        self._config = config
        self._render_mode = render_mode

        # Build the underlying overcooked env
        self._mdp = OvercookedGridworld.from_layout_name(config.layout_name)
        self._env = OvercookedEnv.from_mdp(self._mdp, horizon=config.horizon)

        # Agent metadata
        self._num_players = 2  # Overcooked is always 2-player
        self._agent_names = [f"agent_{i}" for i in range(self._num_players)]
        self._possible_agents = list(self._agent_names)

        # Action space: 6 discrete actions per agent
        # (North, South, East, West, Stay, Interact)
        self._num_actions = len(Action.ALL_ACTIONS)
        self._action_spaces = {
            a: spaces.Discrete(self._num_actions) for a in self._possible_agents
        }

        # Observation space
        if config.use_dense_obs:
            # Dense features: we build a compact vector from state
            # Position(2) + orientation(4 one-hot) + holding(3 one-hot: nothing/onion/dish/soup)
            # + partner_pos(2) + partner_orient(4) + partner_holding(3)
            # + pot_states(num_pots * 4) + closest distances(~8)
            # Approximate: 62 dims for cramped_room
            self._obs_dim = self._compute_dense_obs_dim()
            self._observation_spaces = {
                a: spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
                )
                for a in self._possible_agents
            }
        else:
            # Lossless encoding: (width, height, 26) flattened
            enc_shape = self._mdp.get_lossless_state_encoding_shape()
            self._obs_dim = int(np.prod(enc_shape))
            self._observation_spaces = {
                a: spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
                )
                for a in self._possible_agents
            }

        # Global state space (concatenation of both agents' observations)
        self._state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim * self._num_players,),
            dtype=np.float32,
        )

        # Internal state
        self._state = None
        self._agents = list(self._possible_agents)
        self._step_count = 0

    def _compute_dense_obs_dim(self) -> int:
        """Compute the dimensionality of the dense observation vector."""
        # We'll use: own_pos(2) + own_orient(4) + own_holding(4: none/onion/dish/soup)
        # + partner relative pos(2) + partner_orient(4) + partner_holding(4)
        # + per-pot features: (num_pots * 5: has_onion, num_onions/3, is_cooking, cook_progress, is_ready)
        # + nearest distances: to_onion(1) + to_dish(1) + to_pot(1) + to_serve(1)
        num_pots = len(self._mdp.get_pot_locations())
        return 2 + 4 + 4 + 2 + 4 + 4 + num_pots * 5 + 4

    # ------------------------------------------------------------------
    # PettingZoo parallel-env interface
    # ------------------------------------------------------------------

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
        if self._state is None:
            return np.zeros(self._state_space.shape, dtype=np.float32)
        obs_list = self._get_observations()
        return np.concatenate(
            [obs_list[a] for a in self._possible_agents], axis=-1
        ).astype(np.float32)

    def reset(
        self, seed: int | None = None, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Reset the environment."""
        self._env.reset()
        self._state = self._env.state
        self._agents = list(self._possible_agents)
        self._step_count = 0

        obs = self._get_observations()
        infos = {a: {} for a in self._agents}
        return obs, infos

    def step(
        self, actions: Mapping[str, int | np.integer]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Execute one timestep.

        Parameters
        ----------
        actions : dict
            Mapping of agent name -> discrete action index (0-5).

        Returns
        -------
        observations, rewards, terminated, truncated, infos
        """
        # Convert action indices to overcooked action tuples
        action_0 = int(actions[self._possible_agents[0]])
        action_1 = int(actions[self._possible_agents[1]])
        joint_action = (
            Action.INDEX_TO_ACTION[action_0],
            Action.INDEX_TO_ACTION[action_1],
        )

        # Step the underlying environment
        next_state_obj, sparse_reward, done, env_info = self._env.step(joint_action)
        self._state = self._env.state
        self._step_count += 1

        # Build per-agent rewards
        rewards = {}
        for i, agent in enumerate(self._possible_agents):
            r = float(sparse_reward)
            if self._config.reward_shaping and "shaped_r_by_agent" in env_info:
                shaped = env_info["shaped_r_by_agent"][i]
                if shaped is not None:
                    r += self._config.reward_shaping_factor * float(shaped)
            rewards[agent] = r

        # Observations
        obs = self._get_observations()

        # Terminated / truncated
        terminated = {a: bool(done) for a in self._possible_agents}
        truncated = {
            a: (self._step_count >= self._config.horizon and not done)
            for a in self._possible_agents
        }

        # Infos
        infos = {}
        for i, agent in enumerate(self._possible_agents):
            infos[agent] = {
                "sparse_reward": float(sparse_reward),
            }
            if "shaped_r_by_agent" in env_info:
                infos[agent]["shaped_reward"] = float(
                    env_info["shaped_r_by_agent"][i] or 0.0
                )

        if done or self._step_count >= self._config.horizon:
            self._agents = []

        return obs, rewards, terminated, truncated, infos

    def render(self) -> np.ndarray | None:
        """Render the current state."""
        if self._render_mode is None:
            return None
        if self._state is None:
            return None
        try:
            from overcooked_ai_py.visualization.state_visualizer import (
                StateVisualizer,
            )

            vis = StateVisualizer()
            surface = vis.render_state(self._state, grid=self._mdp.terrain_mtx)
            if self._render_mode == "rgb_array":
                import pygame

                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
                )
            elif self._render_mode == "human":
                import pygame

                if not hasattr(self, "_pygame_screen"):
                    pygame.init()
                    self._pygame_screen = pygame.display.set_mode(surface.get_size())
                    pygame.display.set_caption("Overcooked-AI")
                self._pygame_screen.blit(surface, (0, 0))
                pygame.display.flip()
                return None
        except ImportError:
            return None

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, "_pygame_screen"):
            import pygame

            pygame.quit()
            del self._pygame_screen

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict[str, np.ndarray]:
        """Convert current state to per-agent observations."""
        if self._state is None:
            return {
                a: np.zeros(self._obs_dim, dtype=np.float32)
                for a in self._possible_agents
            }

        if self._config.use_dense_obs:
            return self._get_dense_observations()
        else:
            return self._get_lossless_observations()

    def _get_lossless_observations(self) -> dict[str, np.ndarray]:
        """Lossless grid encoding, flattened to 1-D float32."""
        encodings = self._mdp.lossless_state_encoding(self._state)
        obs = {}
        for i, agent in enumerate(self._possible_agents):
            enc = np.asarray(encodings[i], dtype=np.float32).flatten()
            obs[agent] = enc
        return obs

    def _get_dense_observations(self) -> dict[str, np.ndarray]:
        """Compact hand-crafted feature vector per agent."""
        state = self._state
        players = state.players
        pot_locs = self._mdp.get_pot_locations()

        obs = {}
        for i, agent in enumerate(self._possible_agents):
            player = players[i]
            partner = players[1 - i]
            features = []

            # Own position (normalized to [0,1])
            w, h = self._mdp.width, self._mdp.height
            features.extend(
                [player.position[0] / max(w - 1, 1), player.position[1] / max(h - 1, 1)]
            )

            # Own orientation (one-hot of 4 directions)
            orient = [0.0] * 4
            orient_map = {(0, -1): 0, (0, 1): 1, (1, 0): 2, (-1, 0): 3}
            if player.orientation in orient_map:
                orient[orient_map[player.orientation]] = 1.0
            features.extend(orient)

            # Own held object (one-hot: nothing, onion, dish, soup)
            held = [0.0] * 4
            if player.held_object is None:
                held[0] = 1.0
            elif player.held_object.name == "onion":
                held[1] = 1.0
            elif player.held_object.name == "dish":
                held[2] = 1.0
            elif player.held_object.name == "soup":
                held[3] = 1.0
            features.extend(held)

            # Partner relative position
            features.extend(
                [
                    (partner.position[0] - player.position[0]) / max(w - 1, 1),
                    (partner.position[1] - player.position[1]) / max(h - 1, 1),
                ]
            )

            # Partner orientation (one-hot)
            p_orient = [0.0] * 4
            if partner.orientation in orient_map:
                p_orient[orient_map[partner.orientation]] = 1.0
            features.extend(p_orient)

            # Partner held object
            p_held = [0.0] * 4
            if partner.held_object is None:
                p_held[0] = 1.0
            elif partner.held_object.name == "onion":
                p_held[1] = 1.0
            elif partner.held_object.name == "dish":
                p_held[2] = 1.0
            elif partner.held_object.name == "soup":
                p_held[3] = 1.0
            features.extend(p_held)

            # Per-pot features
            for pot_loc in pot_locs:
                pot_state = state.objects.get(pot_loc, None)
                if pot_state is not None and pot_state.name == "soup":
                    num_onions = len(pot_state.ingredients)
                    is_cooking = pot_state.is_cooking
                    cook_time = pot_state.cook_time if is_cooking else 0
                    is_ready = pot_state.is_ready
                    features.extend(
                        [
                            1.0,  # has contents
                            num_onions / 3.0,  # normalized onion count
                            1.0 if is_cooking else 0.0,
                            min(cook_time / 20.0, 1.0),  # normalized cook progress
                            1.0 if is_ready else 0.0,
                        ]
                    )
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            # Nearest distances (Manhattan) to key locations
            pos = np.array(player.position)
            for locs_fn in [
                self._mdp.get_onion_dispenser_locations,
                self._mdp.get_dish_dispenser_locations,
                lambda: pot_locs,
                self._mdp.get_serving_locations,
            ]:
                locs = locs_fn()
                if locs:
                    dists = [abs(pos[0] - l[0]) + abs(pos[1] - l[1]) for l in locs]
                    features.append(min(dists) / max(w + h - 2, 1))
                else:
                    features.append(1.0)

            # Pad or truncate to expected dim
            feat_arr = np.array(features, dtype=np.float32)
            if len(feat_arr) < self._obs_dim:
                feat_arr = np.pad(feat_arr, (0, self._obs_dim - len(feat_arr)))
            elif len(feat_arr) > self._obs_dim:
                feat_arr = feat_arr[: self._obs_dim]

            obs[agent] = feat_arr

        return obs


def make_overcooked_env(
    config: OvercookedConfig | None = None,
    render_mode: Literal["human", "rgb_array"] | None = None,
) -> OvercookedPettingZooEnv:
    """Create an Overcooked-AI environment with PettingZoo parallel API.

    Args:
        config: Environment configuration. Uses defaults if None.
        render_mode: Rendering mode.

    Returns:
        PettingZoo-compatible parallel environment instance.
    """
    if config is None:
        config = OvercookedConfig()
    return OvercookedPettingZooEnv(config=config, render_mode=render_mode)
