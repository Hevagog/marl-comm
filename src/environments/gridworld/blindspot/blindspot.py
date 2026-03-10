"""Blind-Spot Navigation

Two agents navigate a 9x9 grid from the top-left area to a shared goal at
the bottom-right.  Hidden traps are scattered randomly each episode.

- **Agent A (agent_0)** has full vision: it sees every trap position.
- **Agent B (agent_1)** has restricted vision: it only sees traps within a
  3x3 window centred on itself.

Both agents share the same observation space so an encoder can be shared,
but the trap information in Agent B's observation is partially masked
(zeros for traps outside its vision range).

When ``use_communication=True``, each agent selects a **composite action**
encoding both movement and a discrete message token:

    composite_action = movement * num_message_tokens + message_token

The partner's last message is appended to the observation as a one-hot
vector.  This gives Agent A an explicit signalling channel to warn Agent B
about traps, while Agent B can acknowledge or request information.

Actions (without communication): UP(0), DOWN(1), LEFT(2), RIGHT(3), STAY(4)
Actions (with communication, M tokens): movement * M + token
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BlindSpotConfig:
    """Configuration for the Blind-Spot Navigation environment."""

    grid_size: int = 9
    """Side length of the square grid."""

    max_cycles: int = 100
    """Maximum timesteps per episode."""

    num_traps: int = 5
    """Number of hidden traps placed each episode."""

    trap_penalty: float = -5.0
    """Reward penalty when an agent steps on a trap."""

    goal_reward: float = 10.0
    """Reward granted when an agent reaches the goal."""

    vision_range: int = 1
    """Agent B's vision radius (1 -> 3x3 window)."""

    step_penalty: float = -0.01
    """Small per-step cost to encourage efficiency."""

    use_distance_shaping: bool = True
    """Enable potential-based reward shaping (Ng et al. 1999) using
    Manhattan distance to goal.  Provides a reward gradient so that
    the preference prior C can learn even before the goal is reached."""

    distance_shaping_scale: float = 0.5
    """Scale factor for distance-based shaping reward.  The raw potential
    difference is in [-1/(grid_size-1), +1/(grid_size-1)]; this multiplier
    controls how much the shaping reward contributes relative to the
    step penalty and goal/trap rewards."""

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


# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------

_ACTION_UP = 0
_ACTION_DOWN = 1
_ACTION_LEFT = 2
_ACTION_RIGHT = 3
_ACTION_STAY = 4

_DELTAS = {
    _ACTION_UP: (0, -1),
    _ACTION_DOWN: (0, 1),
    _ACTION_LEFT: (-1, 0),
    _ACTION_RIGHT: (1, 0),
    _ACTION_STAY: (0, 0),
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class BlindSpotEnv:
    """PettingZoo parallel-API environment for Blind-Spot Navigation.

    Parameters
    ----------
    config : BlindSpotConfig
        Environment configuration.
    render_mode : str or None
        ``"human"`` for live window, ``"rgb_array"`` for frame capture, or
        ``None`` for no rendering.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "blind_spot_v0"}

    def __init__(
        self,
        config: BlindSpotConfig | None = None,
        render_mode: Literal["human", "rgb_array"] | None = None,
    ):
        if config is None:
            config = BlindSpotConfig()
        self._config = config
        self._render_mode = render_mode

        # Agent metadata
        self._num_players = 2
        self._agent_names = [f"agent_{i}" for i in range(self._num_players)]
        self._possible_agents = list(self._agent_names)

        # Communication setup
        self._use_comm = config.use_communication
        self._num_msg_tokens = config.num_message_tokens if self._use_comm else 0
        self._num_movement_actions = 5

        # Action space: 5 movement actions, optionally * M message tokens
        if self._use_comm:
            self._num_actions = self._num_movement_actions * self._num_msg_tokens
        else:
            self._num_actions = self._num_movement_actions
        self._action_spaces = {
            a: spaces.Discrete(self._num_actions) for a in self._possible_agents
        }

        # Observation space
        #   own_pos(2) + partner_pos(2) + goal_pos(2) + goal_proximity(2)
        #   + trap_slots(num_traps * 3) + own_reached(1) + partner_reached(1)
        #   + [partner_message_onehot(M) if communication]
        #
        # goal_proximity = [1 - |gx-own_x|/gs, 1 - |gy-own_y|/gs] — per-axis
        # proximity to goal.  Values in [0, 1]: 1.0 at goal cell, 0.0 at max dist.
        # Added in Step 15 to fix encoder directional collapse.
        # Key property: goal observation has the HIGHEST magnitude of these 2 dims
        # (prox=[1,1] at goal) → ||mu_goal|| > ||mu_start|| → positive C gradient.
        # See ai-agent-implementation-steps.md Step 15 for full analysis.
        # Reference: Kaelbling, Littman & Cassandra (1998) §2 — goal-relative state
        # representation; Singh, Barto & Chentanez (2004) — subgoal-relative features.
        base_obs_dim = 2 + 2 + 2 + 2 + self._config.num_traps * 3 + 1 + 1
        self._obs_dim = base_obs_dim + (self._num_msg_tokens if self._use_comm else 0)
        self._observation_spaces = {
            a: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._obs_dim,),
                dtype=np.float32,
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

        # Internal state -- initialised properly in reset()
        self._agents: list[str] = list(self._possible_agents)
        self._step_count: int = 0
        self._positions: dict[str, tuple[int, int]] = {}
        self._traps: list[tuple[int, int]] = []
        self._goal: tuple[int, int] = (
            self._config.grid_size - 1,
            self._config.grid_size - 1,
        )
        self._reached_goal: dict[str, bool] = {}
        self._messages: dict[str, int] = {a: 0 for a in self._possible_agents}
        self._rng: np.random.Generator = np.random.default_rng()
        self._renderer = None

    # ------------------------------------------------------------------
    # PettingZoo parallel-env interface -- properties
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

    @property
    def num_movement_actions(self) -> int:
        """Number of movement-only actions (always 5: UP/DOWN/LEFT/RIGHT/STAY)."""
        return self._num_movement_actions

    @property
    def num_message_tokens(self) -> int:
        """Number of discrete message tokens (0 when communication is disabled)."""
        return self._num_msg_tokens

    def observation_space(self, agent: str) -> gymnasium.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> gymnasium.Space:
        return self._action_spaces[agent]

    # ------------------------------------------------------------------
    # Global state
    # ------------------------------------------------------------------

    def state(self) -> np.ndarray:
        """Return global state (concatenation of all agent observations)."""
        obs = self._get_observations()
        return np.concatenate([obs[a] for a in self._possible_agents], axis=-1).astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self, seed: int | None = None, **kwargs: Any
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        """Reset the environment and randomise trap positions.

        Parameters
        ----------
        seed : int or None
            Optional seed for the random number generator.

        Returns
        -------
        observations : dict[str, np.ndarray]
        infos : dict[str, dict]
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._agents = list(self._possible_agents)
        self._step_count = 0

        # Spawn positions
        self._positions = {
            self._possible_agents[0]: (0, 0),
            self._possible_agents[1]: (1, 0),
        }
        self._goal = (self._config.grid_size - 1, self._config.grid_size - 1)

        self._reached_goal = {a: False for a in self._possible_agents}
        self._messages = {a: 0 for a in self._possible_agents}

        # Place traps on random empty cells (not on spawns or goal)
        self._traps = self._place_traps()

        obs = self._get_observations()
        infos: Dict[str, dict] = {a: {} for a in self._agents}
        return obs, infos

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: Mapping[str, int | np.integer]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        """Execute one timestep.

        Parameters
        ----------
        actions : dict
            Mapping of agent name -> discrete action index.
            Without communication: 0-4 (movement only).
            With communication: 0 to (5*M - 1), encoding
            ``movement * M + message_token``.

        Returns
        -------
        observations, rewards, terminated, truncated, infos
        """
        self._step_count += 1

        rewards: Dict[str, float] = {a: 0.0 for a in self._possible_agents}

        for agent in self._possible_agents:
            # Agents that already reached the goal stay put
            if self._reached_goal[agent]:
                continue

            raw_action = int(actions[agent])

            # Decode composite action into movement + message
            if self._use_comm:
                M = self._num_msg_tokens
                movement = raw_action // M
                message = raw_action % M
                # Clamp movement to valid range
                movement = min(movement, self._num_movement_actions - 1)
                self._messages[agent] = message
            else:
                movement = raw_action

            dx, dy = _DELTAS.get(movement, (0, 0))
            old_x, old_y = self._positions[agent]
            new_x = max(0, min(self._config.grid_size - 1, old_x + dx))
            new_y = max(0, min(self._config.grid_size - 1, old_y + dy))
            self._positions[agent] = (new_x, new_y)

            # Step penalty
            rewards[agent] += self._config.step_penalty

            # Potential-based distance shaping (Ng et al. 1999):
            #   F(s, s') = Φ(s') - Φ(s)
            #   Φ(s) = -manhattan_distance(s, goal) / max_manhattan
            # Positive when moving closer to goal, negative when moving away.
            if self._config.use_distance_shaping:
                max_dist = (self._config.grid_size - 1) * 2
                gx, gy = self._goal
                old_dist = abs(old_x - gx) + abs(old_y - gy)
                new_dist = abs(new_x - gx) + abs(new_y - gy)
                old_phi = -old_dist / max_dist
                new_phi = -new_dist / max_dist
                rewards[agent] += self._config.distance_shaping_scale * (
                    new_phi - old_phi
                )

            # Trap check
            if (new_x, new_y) in self._traps:
                rewards[agent] += self._config.trap_penalty

            # Goal check
            if (new_x, new_y) == self._goal:
                self._reached_goal[agent] = True
                rewards[agent] += self._config.goal_reward

        # Determine termination
        all_reached = all(self._reached_goal[a] for a in self._possible_agents)
        time_up = self._step_count >= self._config.max_cycles

        terminated: Dict[str, bool] = {a: all_reached for a in self._possible_agents}
        truncated: Dict[str, bool] = {
            a: (time_up and not all_reached) for a in self._possible_agents
        }

        obs = self._get_observations()

        infos: Dict[str, dict] = {
            a: {
                "reached_goal": self._reached_goal[a],
                "step": self._step_count,
            }
            for a in self._possible_agents
        }

        if all_reached or time_up:
            self._agents = []

        return obs, rewards, terminated, truncated, infos

    # ------------------------------------------------------------------
    # Render / close
    # ------------------------------------------------------------------

    def render(self) -> np.ndarray | None:
        """Render the current state.

        Returns ``None`` in ``"human"`` mode (draws to screen),
        an (H, W, 3) uint8 array in ``"rgb_array"`` mode,
        or ``None`` when ``render_mode`` is unset.
        """
        if self._render_mode is None:
            return None

        if self._renderer is None:
            from .rendering import BlindSpotRenderer

            self._renderer = BlindSpotRenderer(
                grid_size=self._config.grid_size,
                render_mode=self._render_mode,
            )

        return self._renderer.render(
            positions=dict(self._positions),
            traps=list(self._traps),
            goal=self._goal,
            reached_goal=dict(self._reached_goal),
            step=self._step_count,
            max_steps=self._config.max_cycles,
            vision_range=self._config.vision_range,
            messages=dict(self._messages) if self._use_comm else None,
        )

    def close(self) -> None:
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def get_preference_observations(
        self,
        shaping_scale: float = 0.5,
        goal_bonus: float = 5.0,
    ) -> dict:
        """Return observations and distance-shaped preference values for C init.

        For each grid position, generates a synthetic full-vision observation
        (agent placed at that position, partner at origin, no traps nearby)
        paired with a Manhattan-distance-based preference value.

        Verified in Steps 1-11 (ai-agent-implementation-steps.md):
        distance-shaped C is essential for EFE gradient.  Flat C (zeros)
        produces zero pragmatic value, breaking the planning loop entirely.

        Reference:
            Ng, Russell & Harada (1999) "Policy Invariance Under Reward
            Transformations" — potential-based shaping justification.
            Da Costa et al. (2020), Section 2.3 — C as log-preference
            over observations, shaped by task potential.

        Args:
            shaping_scale: Scale factor for distance-based preferences.
            goal_bonus: Additional preference bonus for the goal state.

        Returns:
            Dictionary with:
                observations: (N, obs_dim) float32 array of observations
                preferences:  (N,) float32 array of preference values
        """
        gs = self._config.grid_size
        gx, gy = self._goal
        max_dist = (gs - 1) * 2

        observations = []
        preferences = []

        # Save current env state
        saved_positions = dict(self._positions)
        saved_reached = dict(self._reached_goal)
        saved_traps = list(self._traps)
        saved_messages = dict(self._messages)

        # Use fixed traps for consistent observations
        self._traps = saved_traps if saved_traps else []

        for x in range(gs):
            for y in range(gs):
                # Place agent_0 at (x, y), partner at (0, 0)
                self._positions[self._possible_agents[0]] = (x, y)
                self._positions[self._possible_agents[1]] = (0, 0)
                self._reached_goal = {a: False for a in self._possible_agents}

                # Get full-vision observation for agent_0
                obs = self._get_observations()
                observations.append(obs[self._possible_agents[0]])

                # Distance-shaped preference
                dist = abs(x - gx) + abs(y - gy)
                pref = -dist * shaping_scale / max(max_dist, 1)
                if (x, y) == self._goal:
                    pref += goal_bonus
                # Penalise trap positions
                if (x, y) in self._traps:
                    pref -= abs(self._config.trap_penalty) * shaping_scale
                preferences.append(pref)

        # Restore env state
        self._positions = saved_positions
        self._reached_goal = saved_reached
        self._traps = saved_traps
        self._messages = saved_messages

        return {
            "observations": np.stack(observations).astype(np.float32),
            "preferences": np.array(preferences, dtype=np.float32),
        }

    def collect_random_observations(
        self,
        num_episodes: int = 100,
        seed: int = 42,
    ) -> dict:
        """Collect diverse (obs, action, next_obs) tuples for encoder pretraining.

        Runs random-policy episodes, collecting transition data from both
        agents.  The resulting dataset is suitable for pretraining the
        encoder via the standard loss (SIGReg + prediction + variance floor).

        Verified in Steps 7-10: encoder pretraining on random exploration
        data resolves the bootstrap problem where random encoder →
        collapsed beliefs → no learning signal → encoder never improves.

        Reference:
            Çatal et al. (2020) "Learning perception and planning with
            deep active inference" — pretrain-then-couple approach.

        Args:
            num_episodes: Number of random exploration episodes.
            seed: Random seed for reproducibility.

        Returns:
            Dictionary with:
                observations: (N, obs_dim) float32 observations
                actions:      (N,) int32 actions taken
                next_observations: (N, obs_dim) float32 next observations
        """
        rng = np.random.default_rng(seed)
        obs_list, action_list, next_obs_list = [], [], []

        for ep in range(num_episodes):
            obs, _ = self.reset(seed=int(rng.integers(0, 2**31)))
            for _ in range(self._config.max_cycles):
                actions = {
                    a: int(rng.integers(0, self._num_actions))
                    for a in self._possible_agents
                }
                next_obs, _, terminated, truncated, _ = self.step(actions)

                for agent in self._possible_agents:
                    obs_list.append(obs[agent])
                    action_list.append(actions[agent])
                    next_obs_list.append(next_obs[agent])

                done = any(terminated.values()) or any(truncated.values())
                if done:
                    break
                obs = next_obs

        return {
            "observations": np.stack(obs_list).astype(np.float32),
            "actions": np.array(action_list, dtype=np.int32),
            "next_observations": np.stack(next_obs_list).astype(np.float32),
        }

    def _place_traps(self) -> list[tuple[int, int]]:
        """Place traps on empty cells.

        If ``fixed_trap_seed`` is set, uses a deterministic RNG seeded
        with that value — producing the same layout every episode.
        Otherwise, uses the environment's (potentially re-seeded) RNG.
        """
        occupied = set()
        for pos in self._positions.values():
            occupied.add(pos)
        occupied.add(self._goal)

        all_cells = [
            (x, y)
            for x in range(self._config.grid_size)
            for y in range(self._config.grid_size)
            if (x, y) not in occupied
        ]

        num_traps = min(self._config.num_traps, len(all_cells))

        if self._config.fixed_trap_seed is not None:
            trap_rng = np.random.default_rng(self._config.fixed_trap_seed)
            indices = trap_rng.choice(len(all_cells), size=num_traps, replace=False)
        else:
            indices = self._rng.choice(len(all_cells), size=num_traps, replace=False)

        return [all_cells[i] for i in indices]

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Build per-agent observation vectors.

        Both agents receive a float32 vector:
            own_pos (2) | partner_pos (2) | goal_pos (2) | goal_proximity (2) |
            trap_slots (num_traps * 3) | own_reached (1) | partner_reached (1)
            [| partner_message_onehot (M) if use_communication]

        Without communication: 25 dims (was 23 before Step 15).
        With communication (M=4): 29 dims.

        goal_proximity = [1 - |gx-x|/gs, 1 - |gy-y|/gs] — per-axis proximity
        to the goal cell.  Values in [0,1]: 1.0 at goal, 0.0 at farthest cell.
        Added in Step 15 to fix encoder directional collapse.  The goal cell
        gets the highest magnitude of these 2 dims, so after C ≈ mu_goal
        (softmax-weighted init), C·mu_goal > C·mu_start (positive gradient).

        Agent A (agent_0) sees all traps.  Agent B (agent_1) only sees traps
        within its vision range; out-of-range traps are masked to zeros.
        """
        gs = float(max(self._config.grid_size - 1, 1))  # for normalisation
        obs: Dict[str, np.ndarray] = {}

        for i, agent in enumerate(self._possible_agents):
            partner = self._possible_agents[1 - i]

            own_x, own_y = self._positions[agent]
            par_x, par_y = self._positions[partner]
            gx, gy = self._goal

            features: list[float] = []

            # Own position (normalised)
            features.append(own_x / gs)
            features.append(own_y / gs)

            # Partner position (normalised)
            features.append(par_x / gs)
            features.append(par_y / gs)

            # Goal position (normalised)
            features.append(gx / gs)
            features.append(gy / gs)

            # Goal proximity (normalised) — Step 15 positional fix
            # Range [0, 1]: 1.0 at goal cell, 0.0 at max-distance cell.
            # Per-axis: prox_x = 1 - |gx - own_x| / gs
            # Crucially: goal observation gets the HIGHEST magnitude of these
            # 2 dims → encoder output ||mu_goal|| > ||mu_start|| → C·mu_goal >
            # C·mu_start (positive C gradient) after C = softmax_init ≈ mu_goal.
            #
            # NOTE: goal_rel_disp = [(gx-x)/gs, (gy-y)/gs] was tried first but is
            # [1,1] at START and [0,0] at GOAL, inverting the magnitude ordering and
            # producing a NEGATIVE C gradient (see Step 15.6 in docs).
            features.append(1.0 - abs(gx - own_x) / gs)
            features.append(1.0 - abs(gy - own_y) / gs)

            # Trap slots
            is_full_vision = i == 0  # agent_0 has full vision
            vr = self._config.vision_range

            for t_idx in range(self._config.num_traps):
                if t_idx < len(self._traps):
                    tx, ty = self._traps[t_idx]
                    if is_full_vision:
                        # Agent A sees all traps
                        features.append(tx / gs)
                        features.append(ty / gs)
                        features.append(1.0)
                    else:
                        # Agent B only sees traps within vision range
                        if abs(tx - own_x) <= vr and abs(ty - own_y) <= vr:
                            features.append(tx / gs)
                            features.append(ty / gs)
                            features.append(1.0)
                        else:
                            features.append(0.0)
                            features.append(0.0)
                            features.append(0.0)
                else:
                    # Unused trap slot (fewer traps than slots)
                    features.append(0.0)
                    features.append(0.0)
                    features.append(0.0)

            # Reached-goal flags
            features.append(1.0 if self._reached_goal.get(agent, False) else 0.0)
            features.append(1.0 if self._reached_goal.get(partner, False) else 0.0)

            # Partner's last message (one-hot)
            if self._use_comm:
                msg_onehot = [0.0] * self._num_msg_tokens
                msg_onehot[self._messages[partner]] = 1.0
                features.extend(msg_onehot)

            obs[agent] = np.array(features, dtype=np.float32)

        return obs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_blind_spot_env(
    config: BlindSpotConfig | None = None,
    render_mode: Literal["human", "rgb_array"] | None = None,
) -> BlindSpotEnv:
    """Create a Blind-Spot Navigation environment with PettingZoo parallel API.

    Parameters
    ----------
    config : BlindSpotConfig or None
        Environment configuration.  Uses defaults if ``None``.
    render_mode : str or None
        ``"human"`` for live window, ``"rgb_array"`` for frame capture.

    Returns
    -------
    BlindSpotEnv
        PettingZoo-compatible parallel environment instance.
    """
    if config is None:
        config = BlindSpotConfig()
    return BlindSpotEnv(config=config, render_mode=render_mode)
