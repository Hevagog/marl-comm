"""Continuous Coordination Environment — "Rendezvous Pursuit"

N agents move in a continuous 2D unit square [0,1]².  Targets spawn
dynamically and each requires k-of-n agents to be simultaneously within
a capture radius.  Agents receive rewards for capturing targets, bonuses
for synchronized arrivals, and penalties for expired deadlines.

Performance
-----------
Optimised for maximum throughput on small agent counts (n=4–16).
All per-step allocations are eliminated; intermediate buffers are
pre-allocated in __init__ and reused via in-place updates.
NumPy function calls are minimised — the inner loop avoids
broadcasting for tiny arrays and uses direct element access.

PettingZoo parallel API
-----------------------
Implements: possible_agents, agents, observation_spaces, action_spaces,
state_spaces, reset(), step(), state(), render(), close().
"""

from __future__ import annotations

import math
from typing import Any, Literal
from collections.abc import Mapping

import gymnasium
import numpy as np
from gymnasium import spaces

from .config import ContinuousCoordConfig

_NUM_ACTIONS = 9

# Precomputed unit direction vectors for the 8 movement actions + stay.
_DIRS = np.array(
    [
        [0.0, 1.0],  # 0: up
        [0.0, -1.0],  # 1: down
        [-1.0, 0.0],  # 2: left
        [1.0, 0.0],  # 3: right
        [-1.0, 1.0],  # 4: up-left
        [1.0, 1.0],  # 5: up-right
        [-1.0, -1.0],  # 6: down-left
        [1.0, -1.0],  # 7: down-right
        [0.0, 0.0],  # 8: stay
    ],
    dtype=np.float32,
)
_norms = np.linalg.norm(_DIRS, axis=1, keepdims=True)
_norms[_norms == 0] = 1.0
_DIRS = _DIRS / _norms

# Flatten to (9, 2) float tuples for fast Python-level access.
_DIR_X = _DIRS[:, 0].tolist()
_DIR_Y = _DIRS[:, 1].tolist()

_TX, _TY = 0, 1
_TK = 2
_TDEADLINE = 3
_TACTIVE = 4
_TPARENT = 5
_TALIVE = 6
_TARRIVAL_MIN = 7
_TARRIVAL_MAX = 8
_TCOLS = 9


class ContinuousCoordEnv:
    """PettingZoo parallel-API Continuous Coordination environment.

    Parameters
    ----------
    config : ContinuousCoordConfig
        Environment configuration.
    render_mode : str or None
        ``"human"``, ``"rgb_array"``, or ``None`` (headless).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "continuous_coord_v0",
    }

    def __init__(
        self,
        config: ContinuousCoordConfig | None = None,
        render_mode: Literal["human", "rgb_array"] | None = None,
    ):
        if config is None:
            config = ContinuousCoordConfig()
        self._cfg = config
        self._render_mode = render_mode

        n = config.num_agents
        mt = config.max_targets

        # --- Cache config as plain Python scalars for hot-path access ---
        self._n = n
        self._mt = mt
        self._max_cycles = config.max_cycles
        self._cr2 = config.capture_radius * config.capture_radius
        self._vr2 = config.vision_range * config.vision_range
        self._coll_r2 = config.collision_radius * config.collision_radius
        self._vel_damp = config.velocity_damping
        self._vel_gain = config.velocity_gain * config.max_speed
        self._dt = config.dt
        self._arrival_rate = config.target_arrival_rate
        self._r_capture = config.reward_capture
        self._r_sync = config.reward_synchrony_bonus
        self._r_prox = config.reward_proximity_shaping
        self._p_deadline = config.penalty_deadline
        self._p_collision = config.penalty_collision
        self._chain_prob = config.chain_event_prob
        self._deadline_avg = max(
            (config.target_deadline_min + config.target_deadline_max) / 2.0, 1.0
        )
        self._id_norm = max(n - 1, 1)

        self._agent_names = [f"agent_{i}" for i in range(n)]
        self._possible_agents = list(self._agent_names)

        # --- observation / state dimensions ---
        self._teammate_dim = (n - 1) * 4
        self._target_dim = mt * 4
        self._obs_dim = 6 + self._teammate_dim + self._target_dim
        self._state_dim = n * 4 + mt * 4

        # Precomputed teammate-slot mapping: for agent i, which other agents
        # fill slots 0..(n-2)?  _tm_map[i] = list of agent indices != i.
        self._tm_map = [[j for j in range(n) if j != i] for i in range(n)]

        # --- spaces ---
        self._action_spaces = {
            a: spaces.Discrete(_NUM_ACTIONS) for a in self._possible_agents
        }
        self._observation_spaces = {
            a: spaces.Box(low=-1.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)
            for a in self._possible_agents
        }
        self._state_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._state_dim,), dtype=np.float32
        )

        # --- internal state: flat Python lists for zero-overhead access ---
        self._px: list[float] = [0.0] * n  # x-positions
        self._py: list[float] = [0.0] * n  # y-positions
        self._vx: list[float] = [0.0] * n  # x-velocities
        self._vy: list[float] = [0.0] * n  # y-velocities

        # numpy arrays for target data (accessed less frequently)
        self._targets = np.zeros((mt, _TCOLS), dtype=np.float32)
        self._arrival_steps = np.full((mt, n), -1, dtype=np.int32)

        # --- pre-allocated per-step buffers ---
        self._rewards: list[float] = [0.0] * n
        self._all_obs = np.zeros((n, self._obs_dim), dtype=np.float32)
        self._prev_dists: list[float] = [0.0] * n
        self._state_buf = np.zeros(self._state_dim, dtype=np.float32)

        # Pre-allocated agent-ID row for observations
        self._agent_ids = np.arange(n, dtype=np.float32) / self._id_norm

        self._agents: list[str] = list(self._possible_agents)
        self._step_count: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

        self._renderer = None
        self._clock = None

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
        return self._n

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
        """Global state vector (for centralized critic)."""
        n = self._n
        s = self._state_buf
        s[:] = 0.0
        px, py, vx, vy = self._px, self._py, self._vx, self._vy
        for i in range(n):
            s[i * 2] = px[i]
            s[i * 2 + 1] = py[i]
            s[n * 2 + i * 2] = vx[i]
            s[n * 2 + i * 2 + 1] = vy[i]
        offset = n * 4
        targets = self._targets
        da = self._deadline_avg
        for t in range(self._mt):
            if targets[t, _TALIVE] > 0.5:
                base = offset + t * 4
                s[base] = targets[t, _TX]
                s[base + 1] = targets[t, _TY]
                s[base + 2] = targets[t, _TK] / n
                s[base + 3] = targets[t, _TDEADLINE] / da
        return s

    def reset(
        self, seed: int | None = None, **kwargs: Any
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        self._rng = np.random.default_rng(seed)
        self._agents = list(self._possible_agents)
        self._step_count = 0

        rng = self._rng
        n = self._n
        for i in range(n):
            self._px[i] = float(rng.uniform(0.1, 0.9))
            self._py[i] = float(rng.uniform(0.1, 0.9))
            self._vx[i] = 0.0
            self._vy[i] = 0.0
        self._targets[:] = 0.0
        self._arrival_steps[:] = -1

        self._spawn_target()
        self._recompute_prev_dists()

        obs = self._build_obs()
        return obs, {a: {} for a in self._agents}

    def step(
        self, actions: Mapping[str, int | np.integer]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        n = self._n
        mt = self._mt
        self._step_count += 1
        step_count = self._step_count

        px = self._px
        py = self._py
        vx_l = self._vx
        vy_l = self._vy
        targets = self._targets
        arrival = self._arrival_steps

        rewards = self._rewards
        for i in range(n):
            rewards[i] = 0.0

        # ---- 1. Apply actions ----
        vd = self._vel_damp
        vg = self._vel_gain
        dt = self._dt
        agents = self._possible_agents
        for i in range(n):
            a_idx = int(actions[agents[i]])
            new_vx = vd * vx_l[i] + vg * _DIR_X[a_idx]
            new_vy = vd * vy_l[i] + vg * _DIR_Y[a_idx]
            vx_l[i] = new_vx
            vy_l[i] = new_vy
            nx = px[i] + new_vx * dt
            ny = py[i] + new_vy * dt
            # clip [0, 1]
            if nx < 0.0:
                nx = 0.0
            elif nx > 1.0:
                nx = 1.0
            if ny < 0.0:
                ny = 0.0
            elif ny > 1.0:
                ny = 1.0
            px[i] = nx
            py[i] = ny

        # ---- 2. Collision penalties ----
        coll_r2 = self._coll_r2
        p_coll = self._p_collision
        for i in range(n):
            px_i = px[i]
            py_i = py[i]
            for j in range(i + 1, n):
                ddx = px_i - px[j]
                ddy = py_i - py[j]
                if ddx * ddx + ddy * ddy < coll_r2:
                    rewards[i] += p_coll
                    rewards[j] += p_coll

        # ---- 3. Target dynamics ----
        cr2 = self._cr2
        p_deadline = self._p_deadline
        r_capture = self._r_capture
        r_sync = self._r_sync

        for t in range(mt):
            if targets[t, _TALIVE] < 0.5:
                continue
            if targets[t, _TACTIVE] < 0.5:
                continue

            # 3a. Decrement deadline
            targets[t, _TDEADLINE] -= 1.0

            # 3b. Expired?
            if targets[t, _TDEADLINE] <= 0.0:
                pen_share = p_deadline / n
                for _i in range(n):
                    rewards[_i] += pen_share
                self._deactivate_children(t)
                targets[t, _TALIVE] = 0.0
                arrival[t, :] = -1
                continue

            # 3c. Capture check
            tx = targets[t, _TX]
            ty = targets[t, _TY]
            k_req = int(targets[t, _TK])
            count = 0
            for i in range(n):
                ddx = px[i] - tx
                ddy = py[i] - ty
                if ddx * ddx + ddy * ddy < cr2:
                    count += 1
                    if arrival[t, i] < 0:
                        arrival[t, i] = step_count

            if count >= k_req:
                # Captured — distribute reward
                n_part = 0
                a_min = step_count
                a_max = 0
                for i in range(n):
                    ddx = px[i] - tx
                    ddy = py[i] - ty
                    if ddx * ddx + ddy * ddy < cr2:
                        n_part += 1
                        aval = arrival[t, i]
                        if aval < a_min:
                            a_min = aval
                        if aval > a_max:
                            a_max = aval
                        rewards[i] += r_capture / count

                # Synchrony bonus
                spread = float(a_max - a_min)
                bonus = r_sync * math.exp(-spread / max(k_req, 1)) / count
                for i in range(n):
                    ddx = px[i] - tx
                    ddy = py[i] - ty
                    if ddx * ddx + ddy * ddy < cr2:
                        rewards[i] += bonus

                self._activate_children(t)
                targets[t, _TALIVE] = 0.0
                arrival[t, :] = -1

        # ---- 4. Proximity shaping ----
        r_prox = self._r_prox
        prev_dists = self._prev_dists
        for i in range(n):
            px_i = px[i]
            py_i = py[i]
            min_d = 1e9
            for t in range(mt):
                if targets[t, _TALIVE] > 0.5 and targets[t, _TACTIVE] > 0.5:
                    ddx = targets[t, _TX] - px_i
                    ddy = targets[t, _TY] - py_i
                    d = math.sqrt(ddx * ddx + ddy * ddy)
                    if d < min_d:
                        min_d = d
            if min_d > 1e8:
                min_d = 0.0
            rewards[i] += (prev_dists[i] - min_d) * r_prox
            prev_dists[i] = min_d

        # ---- 5. Spawn new targets ----
        if self._rng.random() < self._arrival_rate:
            self._spawn_target()

        # ---- 6. Episode termination ----
        done = step_count >= self._max_cycles

        # --- Build output dicts ---
        reward_dict = {agents[i]: rewards[i] for i in range(n)}
        terminated = {a: False for a in agents}
        truncated = {a: done for a in agents}

        if done:
            self._agents = []

        obs = self._build_obs()
        infos = {a: {"step": step_count} for a in agents}

        return obs, reward_dict, terminated, truncated, infos

    def render(self) -> np.ndarray | None:
        if self._render_mode is None:
            return None

        size = self._cfg.cell_size
        frame = np.full((size, size, 3), 240, dtype=np.uint8)

        def to_px(x: float, y: float) -> tuple[int, int]:
            return int(x * size), int(size - y * size)

        for t_idx in range(self._mt):
            if self._targets[t_idx, _TALIVE] < 0.5:
                continue
            cx, cy = to_px(
                float(self._targets[t_idx, _TX]),
                float(self._targets[t_idx, _TY]),
            )
            r_px = max(2, int(self._cfg.capture_radius * size))
            active = self._targets[t_idx, _TACTIVE] > 0.5
            color = (0, 200, 0) if active else (180, 180, 180)
            yy, xx = np.ogrid[-r_px : r_px + 1, -r_px : r_px + 1]
            mask = xx * xx + yy * yy <= r_px * r_px
            y_min, y_max = max(0, cy - r_px), min(size, cy + r_px + 1)
            x_min, x_max = max(0, cx - r_px), min(size, cx + r_px + 1)
            mc = mask[
                y_min - (cy - r_px) : y_max - (cy - r_px),
                x_min - (cx - r_px) : x_max - (cx - r_px),
            ]
            frame[y_min:y_max, x_min:x_max][mc] = color

        agent_colors = [
            (220, 50, 50),
            (50, 50, 220),
            (50, 180, 50),
            (200, 150, 0),
            (150, 50, 200),
            (0, 180, 180),
            (200, 100, 50),
            (100, 100, 100),
        ]
        r_agent = max(3, int(0.015 * size))
        for i in range(self._n):
            cx, cy = to_px(self._px[i], self._py[i])
            color = agent_colors[i % len(agent_colors)]
            yy, xx = np.ogrid[-r_agent : r_agent + 1, -r_agent : r_agent + 1]
            mask = xx * xx + yy * yy <= r_agent * r_agent
            y_min, y_max = max(0, cy - r_agent), min(size, cy + r_agent + 1)
            x_min, x_max = max(0, cx - r_agent), min(size, cx + r_agent + 1)
            mc = mask[
                y_min - (cy - r_agent) : y_max - (cy - r_agent),
                x_min - (cx - r_agent) : x_max - (cx - r_agent),
            ]
            frame[y_min:y_max, x_min:x_max][mc] = color

        if self._render_mode == "human":
            try:
                import pygame

                if self._renderer is None:
                    pygame.init()
                    self._renderer = pygame.display.set_mode((size, size))
                    pygame.display.set_caption("Rendezvous Pursuit")
                    self._clock = pygame.time.Clock()
                surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                self._renderer.blit(surf, (0, 0))
                pygame.display.flip()
                self._clock.tick(self._cfg.fps)
            except ImportError:
                pass
            return None

        return frame

    def close(self) -> None:
        if self._renderer is not None:
            try:
                import pygame

                pygame.quit()
            except ImportError:
                pass
            self._renderer = None

    def _recompute_prev_dists(self) -> None:
        """Set _prev_dists from current positions/targets (for reset)."""
        n = self._n
        mt = self._mt
        targets = self._targets
        px_l, py_l = self._px, self._py
        for i in range(n):
            pxi = px_l[i]
            pyi = py_l[i]
            min_d = 1e9
            for t in range(mt):
                if targets[t, _TALIVE] > 0.5 and targets[t, _TACTIVE] > 0.5:
                    ddx = targets[t, _TX] - pxi
                    ddy = targets[t, _TY] - pyi
                    d = math.sqrt(ddx * ddx + ddy * ddy)
                    if d < min_d:
                        min_d = d
            self._prev_dists[i] = min_d if min_d < 1e8 else 0.0

    def _build_obs(self) -> dict[str, np.ndarray]:
        """Build per-agent observations.  Uses pre-allocated buffer."""
        n = self._n
        mt = self._mt
        vr2 = self._vr2
        px, py = self._px, self._py
        vx_l, vy_l = self._vx, self._vy
        targets = self._targets
        da = self._deadline_avg

        all_obs = self._all_obs
        all_obs[:] = 0.0

        # Own state
        time_left = 1.0 - self._step_count / self._max_cycles
        for i in range(n):
            all_obs[i, 0] = px[i]
            all_obs[i, 1] = py[i]
            all_obs[i, 2] = vx_l[i]
            all_obs[i, 3] = vy_l[i]
        all_obs[:, 4] = self._agent_ids
        all_obs[:, 5] = time_left

        # Teammates (relative, vision-gated)
        offset = 6
        tm_map = self._tm_map
        for i in range(n):
            pxi = px[i]
            pyi = py[i]
            for s, j in enumerate(tm_map[i]):
                ddx = px[j] - pxi
                ddy = py[j] - pyi
                if ddx * ddx + ddy * ddy <= vr2:
                    base = offset + s * 4
                    all_obs[i, base] = ddx
                    all_obs[i, base + 1] = ddy
                    all_obs[i, base + 2] = vx_l[j] - vx_l[i]
                    all_obs[i, base + 3] = vy_l[j] - vy_l[i]

        # Targets (relative)
        offset2 = 6 + self._teammate_dim
        for t in range(mt):
            if targets[t, _TALIVE] > 0.5 and targets[t, _TACTIVE] > 0.5:
                base = offset2 + t * 4
                tx = targets[t, _TX]
                ty = targets[t, _TY]
                tk = targets[t, _TK] / n
                tu = targets[t, _TDEADLINE] / da
                for i in range(n):
                    all_obs[i, base] = tx - px[i]
                    all_obs[i, base + 1] = ty - py[i]
                    all_obs[i, base + 2] = tk
                    all_obs[i, base + 3] = tu

        return {self._possible_agents[i]: all_obs[i] for i in range(n)}

    def _spawn_target(self) -> bool:
        cfg = self._cfg
        mt = self._mt
        targets = self._targets
        slot = -1
        for i in range(mt):
            if targets[i, _TALIVE] < 0.5:
                slot = i
                break
        if slot < 0:
            return False

        rng = self._rng
        px = float(rng.uniform(0.1, 0.9))
        py = float(rng.uniform(0.1, 0.9))
        k = int(rng.integers(cfg.target_k_min, cfg.target_k_max + 1))
        k = min(k, self._n)
        deadline = int(
            rng.integers(cfg.target_deadline_min, cfg.target_deadline_max + 1)
        )

        targets[slot, _TX] = px
        targets[slot, _TY] = py
        targets[slot, _TK] = float(k)
        targets[slot, _TDEADLINE] = float(deadline)
        targets[slot, _TALIVE] = 1.0
        targets[slot, _TARRIVAL_MIN] = 0.0
        targets[slot, _TARRIVAL_MAX] = 0.0
        self._arrival_steps[slot, :] = -1

        is_chain = rng.random() < self._chain_prob
        if is_chain:
            active_slots = [
                j
                for j in range(mt)
                if j != slot
                and targets[j, _TALIVE] > 0.5
                and targets[j, _TACTIVE] > 0.5
            ]
            if active_slots:
                parent = int(rng.choice(active_slots))
                targets[slot, _TACTIVE] = 0.0
                targets[slot, _TPARENT] = float(parent)
            else:
                targets[slot, _TACTIVE] = 1.0
                targets[slot, _TPARENT] = -1.0
        else:
            targets[slot, _TACTIVE] = 1.0
            targets[slot, _TPARENT] = -1.0

        return True

    def _activate_children(self, parent_idx: int) -> None:
        targets = self._targets
        for i in range(self._mt):
            if (
                targets[i, _TALIVE] > 0.5
                and targets[i, _TACTIVE] < 0.5
                and int(targets[i, _TPARENT]) == parent_idx
            ):
                targets[i, _TACTIVE] = 1.0
                targets[i, _TPARENT] = -1.0

    def _deactivate_children(self, parent_idx: int) -> None:
        targets = self._targets
        for i in range(self._mt):
            if (
                targets[i, _TALIVE] > 0.5
                and targets[i, _TACTIVE] < 0.5
                and int(targets[i, _TPARENT]) == parent_idx
            ):
                targets[i, _TALIVE] = 0.0
                self._arrival_steps[i, :] = -1


def make_continuous_coord_env(
    config: ContinuousCoordConfig | None = None,
    render_mode: Literal["human", "rgb_array"] | None = None,
) -> ContinuousCoordEnv:
    """Create a Continuous Coordination environment with PettingZoo parallel API."""
    if config is None:
        config = ContinuousCoordConfig()
    return ContinuousCoordEnv(config=config, render_mode=render_mode)
