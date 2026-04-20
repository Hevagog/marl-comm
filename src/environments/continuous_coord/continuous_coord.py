"""Continuous Coordination Environment — "Rendezvous Pursuit"

N agents move in a continuous 2D unit square [0,1]².  Targets spawn
dynamically and each requires k-of-n agents to be simultaneously within
a capture radius.  Agents receive rewards for capturing targets, bonuses
for synchronized arrivals, and penalties for expired deadlines.

Typed-agent mode (num_agent_types > 1)
---------------------------------------
Each agent is assigned a type id in [0, num_agent_types).  Each spawned
target stores a per-type required agent count.  Two additional penalties:
  * penalty_wrong_type       — per step, per agent of an unrequired type
                               inside the capture zone.
  * penalty_wrong_composition — all agents in zone when total count reaches
                               k_req but per-type composition is wrong.

Observation extensions (typed mode only):
  own:      +num_agent_types  (own type one-hot)
  teammate: +num_agent_types  (teammate type one-hot per slot)
  target:   +num_agent_types  (normalised required count per type per slot)

Performance
-----------
Optimised for maximum throughput on small agent counts (n=4–16).
All per-step allocations eliminated; buffers pre-allocated in __init__.

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

_DIRS = np.array(
    [
        [0.0, 1.0],   # 0: up
        [0.0, -1.0],  # 1: down
        [-1.0, 0.0],  # 2: left
        [1.0, 0.0],   # 3: right
        [-1.0, 1.0],  # 4: up-left
        [1.0, 1.0],   # 5: up-right
        [-1.0, -1.0], # 6: down-left
        [1.0, -1.0],  # 7: down-right
        [0.0, 0.0],   # 8: stay
    ],
    dtype=np.float32,
)
_norms = np.linalg.norm(_DIRS, axis=1, keepdims=True)
_norms[_norms == 0] = 1.0
_DIRS = _DIRS / _norms

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
_TTYPE_REQ = 9    # base column; cols 9..9+n_types-1 store per-type requirements
_TCOLS_BASE = 9   # target array columns when not typed


class ContinuousCoordEnv:
    """PettingZoo parallel-API Continuous Coordination environment."""

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

        self._n = n
        self._mt = mt
        self._max_cycles = config.max_cycles
        self._cr2 = config.capture_radius * config.capture_radius
        self._vr2 = config.vision_range * config.vision_range
        self._tvr2 = config.target_vision_range * config.target_vision_range
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

        # --- Typed-agent setup ---
        n_types = max(1, config.num_agent_types)
        self._n_types = n_types
        self._typed = n_types > 1

        if self._typed:
            if config.agent_types is not None:
                if len(config.agent_types) != n:
                    raise ValueError(
                        f"agent_types length {len(config.agent_types)} != num_agents {n}"
                    )
                raw = list(config.agent_types)
                for idx, t in enumerate(raw):
                    if t < 0 or t >= n_types:
                        raise ValueError(
                            f"agent_types[{idx}]={t} out of range [0, {n_types})"
                        )
            else:
                raw = [i % n_types for i in range(n)]

            self._agent_types_arr: list[int] = raw
            self._agent_type_counts: list[int] = [raw.count(t) for t in range(n_types)]

            # Pre-built (n × n_types) one-hot matrix — written vectorised in _build_obs
            self._agent_type_matrix = np.zeros((n, n_types), dtype=np.float32)
            for i, ti in enumerate(raw):
                self._agent_type_matrix[i, ti] = 1.0

            self._p_wrong_type = config.penalty_wrong_type
            self._p_wrong_comp = config.penalty_wrong_composition

            # Pre-allocated per-step buffers (avoid per-step allocation in hot loop)
            self._count_by_type: list[int] = [0] * n_types
            self._tgt_type_buf = np.zeros(n_types, dtype=np.float32)

            # Multinomial weights proportional to type populations
            tc = np.array(self._agent_type_counts, dtype=np.float64)
            self._type_sample_weights = tc / tc.sum()
        else:
            self._agent_types_arr = [0] * n
            self._agent_type_counts = [n]
            self._agent_type_matrix = np.zeros((n, 1), dtype=np.float32)
            self._p_wrong_type = 0.0
            self._p_wrong_comp = 0.0
            self._count_by_type = [0]
            self._tgt_type_buf = np.zeros(1, dtype=np.float32)
            self._type_sample_weights = np.array([1.0])

        # --- Observation / state dimensions ---
        type_dim = n_types if self._typed else 0
        self._type_dim = type_dim
        self._teammate_slot = 4 + type_dim
        self._target_slot = 4 + type_dim
        self._teammate_dim = (n - 1) * self._teammate_slot
        self._target_dim = mt * self._target_slot
        self._obs_dim = 6 + type_dim + self._teammate_dim + self._target_dim
        self._state_dim = n * (4 + type_dim) + mt * (4 + type_dim)

        # State layout offsets
        self._st_agent_type_off = n * 4
        self._st_target_off = n * 4 + n * type_dim
        self._st_target_type_off = self._st_target_off + mt * 4

        self._agent_names = [f"agent_{i}" for i in range(n)]
        self._possible_agents = list(self._agent_names)
        self._tm_map = [[j for j in range(n) if j != i] for i in range(n)]

        # --- Spaces ---
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

        # --- Internal state ---
        self._px: list[float] = [0.0] * n
        self._py: list[float] = [0.0] * n
        self._vx: list[float] = [0.0] * n
        self._vy: list[float] = [0.0] * n

        self._tcols = _TCOLS_BASE + (n_types if self._typed else 0)
        self._targets = np.zeros((mt, self._tcols), dtype=np.float32)
        self._arrival_steps = np.full((mt, n), -1, dtype=np.int32)

        self._rewards: list[float] = [0.0] * n
        self._all_obs = np.zeros((n, self._obs_dim), dtype=np.float32)
        self._prev_dists: list[float] = [0.0] * n
        self._state_buf = np.zeros(self._state_dim, dtype=np.float32)

        self._agent_ids = np.arange(n, dtype=np.float32) / self._id_norm

        self._agents: list[str] = list(self._possible_agents)
        self._step_count: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

        self._renderer = None
        self._clock = None

    # ---- Properties ----

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

        if self._typed:
            at_off = self._st_agent_type_off
            s[at_off : at_off + n * self._n_types] = self._agent_type_matrix.ravel()

        targets = self._targets
        da = self._deadline_avg
        tgt_off = self._st_target_off
        for t in range(self._mt):
            if targets[t, _TALIVE] > 0.5:
                base = tgt_off + t * 4
                s[base] = targets[t, _TX]
                s[base + 1] = targets[t, _TY]
                s[base + 2] = targets[t, _TK] / n
                s[base + 3] = targets[t, _TDEADLINE] / da

        if self._typed:
            ttr_off = self._st_target_type_off
            n_types = self._n_types
            for t in range(self._mt):
                if targets[t, _TALIVE] > 0.5:
                    base = ttr_off + t * n_types
                    for tp in range(n_types):
                        s[base + tp] = targets[t, _TTYPE_REQ + tp] / n

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
        typed = self._typed

        for t in range(mt):
            if targets[t, _TALIVE] < 0.5:
                continue
            if targets[t, _TACTIVE] < 0.5:
                continue

            targets[t, _TDEADLINE] -= 1.0

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
            do_capture = False

            if typed:
                ctb = self._count_by_type
                n_types = self._n_types
                for tp in range(n_types):
                    ctb[tp] = 0
                p_wt = self._p_wrong_type
                types_arr = self._agent_types_arr

                for i in range(n):
                    ddx = px[i] - tx
                    ddy = py[i] - ty
                    if ddx * ddx + ddy * ddy < cr2:
                        count += 1
                        ti = types_arr[i]
                        ctb[ti] += 1
                        if arrival[t, i] < 0:
                            arrival[t, i] = step_count
                        # Per-step wrong-type penalty: type not required by this target
                        if targets[t, _TTYPE_REQ + ti] < 0.5:
                            rewards[i] += p_wt

                # Composition check: every type requirement must be met
                comp_ok = True
                for tp in range(n_types):
                    if ctb[tp] < int(targets[t, _TTYPE_REQ + tp]):
                        comp_ok = False
                        break

                if comp_ok and count >= k_req:
                    do_capture = True
                elif count >= k_req:
                    # Enough total agents but wrong type composition
                    p_wc = self._p_wrong_comp
                    for i in range(n):
                        ddx = px[i] - tx
                        ddy = py[i] - ty
                        if ddx * ddx + ddy * ddy < cr2:
                            rewards[i] += p_wc
            else:
                for i in range(n):
                    ddx = px[i] - tx
                    ddy = py[i] - ty
                    if ddx * ddx + ddy * ddy < cr2:
                        count += 1
                        if arrival[t, i] < 0:
                            arrival[t, i] = step_count
                do_capture = count >= k_req

            if do_capture:
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

        # Typed mode: color by type (hue), dims slightly per agent within type
        _TYPE_COLORS = [
            (220, 50,  50),  # type 0: red
            (50,  50, 220),  # type 1: blue
            (50, 180,  50),  # type 2: green
            (200, 150,  0),  # type 3: yellow
            (150, 50, 200),  # type 4: purple
            (0,  180, 180),  # type 5: cyan
            (200, 100, 50),  # type 6: orange
            (100, 100, 100), # type 7: grey
        ]

        if self._typed:
            # Build per-type agent lists for rank-based dimming
            per_type: list[list[int]] = [[] for _ in range(self._n_types)]
            for i in range(self._n):
                per_type[self._agent_types_arr[i]].append(i)

            def _agent_color(i: int) -> tuple[int, int, int]:
                t = self._agent_types_arr[i]
                base = _TYPE_COLORS[t % len(_TYPE_COLORS)]
                rank = per_type[t].index(i)
                f = max(0.55, 1.0 - 0.15 * rank)
                return (int(base[0] * f), int(base[1] * f), int(base[2] * f))
        else:
            def _agent_color(i: int) -> tuple[int, int, int]:
                return _TYPE_COLORS[i % len(_TYPE_COLORS)]

        r_agent = max(3, int(0.015 * size))
        for i in range(self._n):
            cx, cy = to_px(self._px[i], self._py[i])
            color = _agent_color(i)
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
                if self._clock is not None:
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
        n = self._n
        mt = self._mt
        vr2 = self._vr2
        px, py = self._px, self._py
        vx_l, vy_l = self._vx, self._vy
        targets = self._targets
        da = self._deadline_avg
        typed = self._typed
        type_dim = self._type_dim
        n_types = self._n_types

        all_obs = self._all_obs
        all_obs[:] = 0.0

        # Own pos/vel, normalised id, time remaining
        time_left = 1.0 - self._step_count / self._max_cycles
        for i in range(n):
            all_obs[i, 0] = px[i]
            all_obs[i, 1] = py[i]
            all_obs[i, 2] = vx_l[i]
            all_obs[i, 3] = vy_l[i]
        all_obs[:, 4] = self._agent_ids
        all_obs[:, 5] = time_left

        # Own type one-hot (vectorised write)
        if typed:
            all_obs[:, 6 : 6 + n_types] = self._agent_type_matrix

        # Teammates (relative, vision-gated)
        obs_tm_off = 6 + type_dim
        tm_slot = self._teammate_slot
        tm_map = self._tm_map
        types_arr = self._agent_types_arr
        for i in range(n):
            pxi = px[i]
            pyi = py[i]
            for s, j in enumerate(tm_map[i]):
                ddx = px[j] - pxi
                ddy = py[j] - pyi
                if ddx * ddx + ddy * ddy <= vr2:
                    base = obs_tm_off + s * tm_slot
                    all_obs[i, base] = ddx
                    all_obs[i, base + 1] = ddy
                    all_obs[i, base + 2] = vx_l[j] - vx_l[i]
                    all_obs[i, base + 3] = vy_l[j] - vy_l[i]
                    if typed:
                        # One-hot: set single position, rest already 0
                        all_obs[i, base + 4 + types_arr[j]] = 1.0

        # Targets (relative position + type-requirement features, target-vision-gated)
        # When target_vision_range < sqrt(2) each agent only sees nearby targets;
        # this creates private information that communication architectures can share.
        obs_tgt_off = obs_tm_off + self._teammate_dim
        tgt_slot = self._target_slot
        tgt_type_buf = self._tgt_type_buf
        tvr2 = self._tvr2
        for t in range(mt):
            if targets[t, _TALIVE] > 0.5 and targets[t, _TACTIVE] > 0.5:
                base = obs_tgt_off + t * tgt_slot
                tx = targets[t, _TX]
                ty = targets[t, _TY]
                tk = targets[t, _TK] / n
                tu = targets[t, _TDEADLINE] / da
                if typed:
                    for tp in range(n_types):
                        tgt_type_buf[tp] = targets[t, _TTYPE_REQ + tp] / n
                for i in range(n):
                    ddx = tx - px[i]
                    ddy = ty - py[i]
                    if ddx * ddx + ddy * ddy <= tvr2:
                        all_obs[i, base] = ddx
                        all_obs[i, base + 1] = ddy
                        all_obs[i, base + 2] = tk
                        all_obs[i, base + 3] = tu
                        if typed:
                            all_obs[i, base + 4 : base + 4 + n_types] = tgt_type_buf

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
        tx = float(rng.uniform(0.1, 0.9))
        ty = float(rng.uniform(0.1, 0.9))
        deadline = int(rng.integers(cfg.target_deadline_min, cfg.target_deadline_max + 1))

        if self._typed:
            k = int(rng.integers(cfg.target_k_min, cfg.target_k_max + 1))
            k = min(k, self._n)
            # Distribute k across types; weights proportional to type populations
            type_reqs = rng.multinomial(k, self._type_sample_weights).tolist()
            # Clamp each type requirement to available agent count
            actual_k = 0
            for tp in range(self._n_types):
                type_reqs[tp] = min(type_reqs[tp], self._agent_type_counts[tp])
                actual_k += type_reqs[tp]
            # Ensure at least one agent is required
            if actual_k == 0:
                dominant = max(range(self._n_types), key=lambda t: self._agent_type_counts[t])
                type_reqs[dominant] = 1
                actual_k = 1
            targets[slot, _TK] = float(actual_k)
            for tp in range(self._n_types):
                targets[slot, _TTYPE_REQ + tp] = float(type_reqs[tp])
        else:
            k = int(rng.integers(cfg.target_k_min, cfg.target_k_max + 1))
            k = min(k, self._n)
            targets[slot, _TK] = float(k)

        targets[slot, _TX] = tx
        targets[slot, _TY] = ty
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
