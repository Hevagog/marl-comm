"""Rendering utilities for the warehouse environment.

Provides
--------
- ``render_ascii``       – rich terminal output showing every agent state.
- ``WarehouseRenderer``  – pixel-based renderer (rgb_array / pygame) with:
    phase-coloured rings, battery bars, carrying diamonds,
    treatment progress, charging glow/bolt, dead-agent markers,
    shelf resource dots, task-urgency glow, interference overlay,
    legend panel, HUD status bar, per-agent roster.
- ``VideoRecorder``      – captures rgb_array frames to MP4/GIF.
- ``record_episode``     – convenience function: reset, run, record, save.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

from ..config import WarehouseConfig
from .types import CellType, EnvState, ResourcePhase


_BG = (28, 28, 32)
_WALL = (60, 62, 68)
_SHELF = (160, 120, 50)
_SHELF_EMPTY = (90, 75, 40)
_TREATMENT = (50, 140, 190)
_GOAL = (50, 185, 80)
_SPAWN = (55, 55, 60)
_CHARGER = (210, 195, 45)
_CHARGER_GLOW = (255, 240, 100)
_REPAIR = (90, 205, 215)
_RESCUE = (240, 240, 255)

_AGENT_COLORS = [
    (220, 60, 60),
    (60, 120, 220),
    (220, 170, 40),
    (170, 60, 220),
    (50, 210, 120),
    (220, 120, 50),
    (50, 200, 210),
    (180, 180, 180),
]

_PHASE_COLORS = {
    ResourcePhase.UNPICKED: None,
    ResourcePhase.IN_TRANSIT_TO_TREATMENT: (80, 160, 240),
    ResourcePhase.TREATING: (50, 220, 220),
    ResourcePhase.IN_TRANSIT_TO_GOAL: (80, 240, 100),
}

_BAT_HIGH = (60, 200, 80)
_BAT_MED = (230, 190, 40)
_BAT_LOW = (230, 60, 50)
_BAT_BG = (40, 40, 44)
_BAT_DEAD = (100, 40, 40)
_FAILED = (160, 60, 60)
_DEAD = (140, 40, 40)

_HUD_TEXT = (220, 220, 220)
_HUD_ACCENT = (100, 180, 255)
_URG_LOW = (80, 120, 60)
_URG_MED = (200, 180, 40)
_URG_HIGH = (230, 70, 40)

CELL_PX = 48
HUD_HEIGHT = 90
LEGEND_WIDTH = 220


def _fill_rect(img, r0, c0, h, w, color, alpha=1.0):
    r1 = min(r0 + h, img.shape[0])
    c1 = min(c0 + w, img.shape[1])
    r0 = max(r0, 0)
    c0 = max(c0, 0)
    if r1 <= r0 or c1 <= c0:
        return
    if alpha >= 1.0:
        img[r0:r1, c0:c1] = color
    else:
        region = img[r0:r1, c0:c1].astype(np.float32)
        overlay = np.array(color, dtype=np.float32)
        img[r0:r1, c0:c1] = (region * (1 - alpha) + overlay * alpha).astype(np.uint8)


def _draw_circle(img, cy, cx, radius, color, filled=True, thickness=2):
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    dsq = xx**2 + yy**2
    mask = (
        dsq <= radius**2
        if filled
        else (dsq <= radius**2) & (dsq >= (radius - thickness) ** 2)
    )
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if mask[dy + radius, dx + radius]:
                py, px_ = cy + dy, cx + dx
                if 0 <= py < img.shape[0] and 0 <= px_ < img.shape[1]:
                    img[py, px_] = color


def _draw_x(img, cy, cx, size, color, thickness=2):
    for d in range(-size, size + 1):
        for t in range(-thickness // 2, thickness // 2 + 1):
            for py, px_ in [(cy + d, cx + d + t), (cy + d, cx - d + t)]:
                if 0 <= py < img.shape[0] and 0 <= px_ < img.shape[1]:
                    img[py, px_] = color


def _draw_diamond(img, cy, cx, size, color):
    for dy in range(-size, size + 1):
        w = size - abs(dy)
        for dx in range(-w, w + 1):
            py, px_ = cy + dy, cx + dx
            if 0 <= py < img.shape[0] and 0 <= px_ < img.shape[1]:
                img[py, px_] = color


def _draw_bolt(img, cy, cx, size, color):
    pts = [
        (-size, 1),
        (-size, 2),
        (-size + 1, 0),
        (-size + 1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (size, -1),
        (size, -2),
    ]
    for dy, dx in pts:
        py, px_ = cy + dy, cx + dx
        if 0 <= py < img.shape[0] and 0 <= px_ < img.shape[1]:
            img[py, px_] = color


def _draw_empty_battery(img, cy, cx, color):
    for dx in range(-5, 6):
        for dy in [-4, 4]:
            py, px_ = cy + dy, cx + dx
            if 0 <= py < img.shape[0] and 0 <= px_ < img.shape[1]:
                img[py, px_] = color
    for dy in range(-4, 5):
        for dx in [-5, 5]:
            py, px_ = cy + dy, cx + dx
            if 0 <= py < img.shape[0] and 0 <= px_ < img.shape[1]:
                img[py, px_] = color
    for dy in range(-2, 3):
        px_ = cx + 6
        py = cy + dy
        if 0 <= py < img.shape[0] and 0 <= px_ < img.shape[1]:
            img[py, px_] = color


def _draw_arrow_down(img, cy, cx, size, color):
    for dy in range(-size, size + 1):
        py, px_ = cy + dy, cx
        if 0 <= py < img.shape[0] and 0 <= px_ < img.shape[1]:
            img[py, px_] = color
    for d in range(1, size):
        for px_ in [cx - d, cx + d]:
            py = cy + size - d
            if 0 <= py < img.shape[0] and 0 <= px_ < img.shape[1]:
                img[py, px_] = color


# ===================================================================== #
#  Tiny 5x3 bitmap font
# ===================================================================== #

_FONT = {
    "0": [0b111, 0b101, 0b101, 0b101, 0b111],
    "1": [0b010, 0b110, 0b010, 0b010, 0b111],
    "2": [0b111, 0b001, 0b111, 0b100, 0b111],
    "3": [0b111, 0b001, 0b111, 0b001, 0b111],
    "4": [0b101, 0b101, 0b111, 0b001, 0b001],
    "5": [0b111, 0b100, 0b111, 0b001, 0b111],
    "6": [0b111, 0b100, 0b111, 0b101, 0b111],
    "7": [0b111, 0b001, 0b010, 0b010, 0b010],
    "8": [0b111, 0b101, 0b111, 0b101, 0b111],
    "9": [0b111, 0b101, 0b111, 0b001, 0b111],
    "A": [0b010, 0b101, 0b111, 0b101, 0b101],
    "B": [0b110, 0b101, 0b110, 0b101, 0b110],
    "C": [0b111, 0b100, 0b100, 0b100, 0b111],
    "D": [0b110, 0b101, 0b101, 0b101, 0b110],
    "E": [0b111, 0b100, 0b111, 0b100, 0b111],
    "F": [0b111, 0b100, 0b111, 0b100, 0b100],
    "G": [0b111, 0b100, 0b101, 0b101, 0b111],
    "H": [0b101, 0b101, 0b111, 0b101, 0b101],
    "I": [0b111, 0b010, 0b010, 0b010, 0b111],
    "K": [0b101, 0b110, 0b100, 0b110, 0b101],
    "L": [0b100, 0b100, 0b100, 0b100, 0b111],
    "M": [0b101, 0b111, 0b111, 0b101, 0b101],
    "N": [0b101, 0b111, 0b111, 0b101, 0b101],
    "O": [0b111, 0b101, 0b101, 0b101, 0b111],
    "P": [0b111, 0b101, 0b111, 0b100, 0b100],
    "R": [0b110, 0b101, 0b110, 0b101, 0b101],
    "S": [0b111, 0b100, 0b111, 0b001, 0b111],
    "T": [0b111, 0b010, 0b010, 0b010, 0b010],
    "U": [0b101, 0b101, 0b101, 0b101, 0b111],
    "V": [0b101, 0b101, 0b101, 0b101, 0b010],
    "W": [0b101, 0b101, 0b111, 0b111, 0b101],
    "X": [0b101, 0b101, 0b010, 0b101, 0b101],
    "Y": [0b101, 0b101, 0b010, 0b010, 0b010],
    "/": [0b001, 0b001, 0b010, 0b100, 0b100],
    ":": [0b000, 0b010, 0b000, 0b010, 0b000],
    " ": [0b000, 0b000, 0b000, 0b000, 0b000],
    "-": [0b000, 0b000, 0b111, 0b000, 0b000],
    ".": [0b000, 0b000, 0b000, 0b000, 0b010],
    "%": [0b101, 0b001, 0b010, 0b100, 0b101],
    "#": [0b101, 0b111, 0b101, 0b111, 0b101],
    "!": [0b010, 0b010, 0b010, 0b000, 0b010],
    ">": [0b100, 0b010, 0b001, 0b010, 0b100],
    "<": [0b001, 0b010, 0b100, 0b010, 0b001],
}


def _put_char(img, r, c, char, color, scale=1):
    bitmap = _FONT.get(char.upper(), _FONT.get(char, _FONT[" "]))
    for row_i, row_bits in enumerate(bitmap):
        for col_i in range(3):
            if row_bits & (1 << (2 - col_i)):
                for sy in range(scale):
                    for sx in range(scale):
                        py, px_ = r + row_i * scale + sy, c + col_i * scale + sx
                        if 0 <= py < img.shape[0] and 0 <= px_ < img.shape[1]:
                            img[py, px_] = color


def _put_text(img, r, c, text, color, scale=1):
    for i, ch in enumerate(text):
        _put_char(img, r, c + i * 4 * scale, ch, color, scale=scale)


_CELL_CHARS = {
    CellType.EMPTY: ".",
    CellType.WALL: "#",
    CellType.SHELF: "S",
    CellType.TREATMENT: "T",
    CellType.GOAL: "G",
    CellType.SPAWN: " ",
    CellType.CHARGER: "C",
    CellType.REPAIR: "R",
}
_AGENT_CHARS = "0123456789ABCDEF"
_PHASE_NAMES = {
    ResourcePhase.UNPICKED: "IDLE",
    ResourcePhase.IN_TRANSIT_TO_TREATMENT: "->TREAT",
    ResourcePhase.TREATING: "TREATING",
    ResourcePhase.IN_TRANSIT_TO_GOAL: "->GOAL",
}


def render_ascii(state: EnvState, config: WarehouseConfig) -> str:
    H, W = config.grid_height, config.grid_width
    grid = [[" "] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            grid[r][c] = _CELL_CHARS.get(CellType(state.grid.layout[r, c]), "?")

    for i in range(config.max_agents):
        r, c = int(state.agent.positions[i, 0]), int(state.agent.positions[i, 1])
        if state.agent.burst_failed[i]:
            grid[r][c] = "X"
        elif state.agent.battery_dead[i]:
            grid[r][c] = "B"
        elif state.agent.failed[i]:
            grid[r][c] = "F"
        elif not state.agent.active[i]:
            continue
        elif state.agent.rescue_target[i] >= 0:
            grid[r][c] = "&"
        elif state.agent.charging[i]:
            grid[r][c] = "Z"
        elif state.agent.locked[i]:
            grid[r][c] = "~"
        elif state.agent.carrying[i] > 0:
            grid[r][c] = _AGENT_CHARS[i % len(_AGENT_CHARS)].lower()
        else:
            grid[r][c] = _AGENT_CHARS[i % len(_AGENT_CHARS)]

    header = f"Step {state.step_count:>4d}  Del {state.total_deliveries:>3d}  Active {int(state.agent.active.sum()):>2d}/{config.max_agents}"
    if config.enable_task_deadlines:
        header += f"  Tasks {len(state.task_queue):>2d}  Expired {state.expired_tasks}"

    lines = [header, "+" + "-" * W + "+"]
    for r in range(H):
        lines.append("|" + "".join(grid[r]) + "|")
    lines.append("+" + "-" * W + "+")

    for i in range(config.max_agents):
        if (
            not state.agent.active[i]
            and not state.agent.burst_failed[i]
            and not state.agent.battery_dead[i]
            and not state.agent.failed[i]
        ):
            continue
        s = f"  Agent {i}: "
        if state.agent.burst_failed[i]:
            s += "BURST-DEAD"
        elif state.agent.battery_dead[i]:
            s += "BATTERY-DEAD"
        elif state.agent.failed[i]:
            s += "ATTRITION-FAIL"
        elif state.agent.rescue_target[i] >= 0:
            target = int(state.agent.rescue_target[i])
            target_kind = (
                "TO-CHARGE" if state.agent.battery_dead[target] else "TO-REPAIR"
            )
            s += f"DRAGGING agent={target} {target_kind}"
        elif state.agent.charging[i]:
            bat = int(state.agent.battery[i]) if config.enable_battery else "?"
            s += f"CHARGING bat={bat}/{config.battery_capacity}"
        else:
            phase = ResourcePhase(state.agent.resource_phase[i])
            carry = int(state.agent.carrying[i])
            s += f"carry={carry} phase={_PHASE_NAMES[phase]}"
            if state.agent.locked[i]:
                s += f" TREATING({state.agent.treatment_timer[i]}t left)"
            if config.enable_battery:
                pct = int(
                    100 * state.agent.battery[i] / max(config.battery_capacity, 1)
                )
                s += f" bat={pct}%"
            if config.enable_heterogeneous:
                s += f" spd={int(state.agent.speed[i])} cap={int(state.agent.capacity[i])}"
        lines.append(s)

    if config.enable_task_deadlines and len(state.task_queue) > 0:
        lines.append("  --- Pending Tasks ---")
        for t in state.task_queue[:5]:
            remaining = max(0, t.deadline - state.step_count)
            lines.append(
                f"    Shelf {t.shelf_idx} prio={t.priority} deadline in {remaining}t"
            )
        if len(state.task_queue) > 5:
            lines.append(f"    ... +{len(state.task_queue) - 5} more")

    return "\n".join(lines)


class WarehouseRenderer:
    def __init__(self, config: WarehouseConfig, render_mode: str = "rgb_array"):
        self._config = config
        self._render_mode = render_mode
        self._px = CELL_PX
        self._grid_w = config.grid_width * self._px
        self._grid_h = config.grid_height * self._px
        self._legend_w = LEGEND_WIDTH
        self._hud_h = HUD_HEIGHT
        self._width = self._grid_w + self._legend_w
        self._height = self._grid_h + self._hud_h
        self._screen = None
        self._clock = None
        self._is_open = True
        if render_mode == "human":
            try:
                import pygame

                pygame.init()
                self._screen = pygame.display.set_mode((self._width, self._height))
                pygame.display.set_caption("Multi-Robot Warehouse")
                self._clock = pygame.time.Clock()
            except ImportError:
                raise ImportError("pygame required for render_mode='human'")

    @property
    def is_open(self):
        return self._is_open

    def render(self, state):
        img = self._render_rgb(state)
        if self._render_mode == "human":
            self._blit_pygame(img)
            return None
        return img

    def _render_rgb(self, state):
        cfg = self._config
        px = self._px
        H, W = cfg.grid_height, cfg.grid_width
        img = np.full((self._height, self._width, 3), _BG[0], dtype=np.uint8)
        img[:, :, 1] = _BG[1]
        img[:, :, 2] = _BG[2]
        layout = state.grid.layout

        # shelf urgency map
        shelf_urgency = {}
        if cfg.enable_task_deadlines:
            for task in state.task_queue:
                remaining = max(1, task.deadline - state.step_count)
                urg = min(1.0, task.priority / max(remaining / 10.0, 1.0))
                shelf_urgency[task.shelf_idx] = max(
                    shelf_urgency.get(task.shelf_idx, 0.0), urg
                )

        # ---- cells ----
        for r in range(H):
            for c in range(W):
                r0, c0 = r * px, c * px
                cell = CellType(layout[r, c])

                if cell == CellType.WALL:
                    _fill_rect(img, r0, c0, px, px, _WALL)
                elif cell == CellType.SHELF:
                    has_res, res_count, si_found = False, 0, -1
                    for si in range(cfg.num_shelves):
                        sp = state.grid.shelf_positions[si]
                        if int(sp[0]) == r and int(sp[1]) == c:
                            has_res = bool(state.grid.shelf_resources[si].any())
                            res_count = int(state.grid.shelf_resources[si].sum())
                            si_found = si
                            break
                    _fill_rect(
                        img,
                        r0 + 2,
                        c0 + 2,
                        px - 4,
                        px - 4,
                        _SHELF if has_res else _SHELF_EMPTY,
                    )
                    urg = shelf_urgency.get(si_found, 0.0)
                    if urg > 0.1:
                        uc = (
                            _URG_LOW
                            if urg < 0.4
                            else _URG_MED
                            if urg < 0.7
                            else _URG_HIGH
                        )
                        for e in [
                            (r0, c0, 3, px),
                            (r0 + px - 3, c0, 3, px),
                            (r0, c0, px, 3),
                            (r0, c0 + px - 3, px, 3),
                        ]:
                            _fill_rect(img, *e, uc, alpha=0.6)
                        _put_text(img, r0 + 3, c0 + 3, "!", uc, scale=2)
                    _put_text(img, r0 + 4, c0 + px - 16, "S", (255, 240, 180), scale=2)
                    if has_res:
                        for ri in range(min(res_count, cfg.resources_per_shelf)):
                            _fill_rect(
                                img, r0 + px - 9, c0 + 6 + ri * 8, 4, 4, (255, 255, 200)
                            )
                    _put_text(
                        img,
                        r0 + px - 9,
                        c0 + px - 10,
                        str(res_count),
                        (255, 240, 160),
                        scale=1,
                    )

                elif cell == CellType.TREATMENT:
                    _fill_rect(img, r0 + 1, c0 + 1, px - 2, px - 2, _TREATMENT)
                    _put_text(img, r0 + 4, c0 + 4, "T", (200, 240, 255), scale=2)
                    mid_r, mid_c = r0 + px // 2 + 4, c0 + px // 2 + 4
                    _fill_rect(img, mid_r - 1, mid_c - 6, 3, 12, (200, 240, 255))
                    _fill_rect(img, mid_r - 6, mid_c - 1, 12, 3, (200, 240, 255))

                elif cell == CellType.GOAL:
                    _fill_rect(img, r0 + 1, c0 + 1, px - 2, px - 2, _GOAL)
                    _put_text(img, r0 + 4, c0 + 4, "G", (200, 255, 200), scale=2)
                    _draw_arrow_down(
                        img, r0 + px // 2 + 5, c0 + px // 2 + 6, 5, (200, 255, 200)
                    )

                elif cell == CellType.SPAWN:
                    _fill_rect(img, r0, c0, px, px, _SPAWN)

                elif cell == CellType.CHARGER:
                    _fill_rect(img, r0 + 1, c0 + 1, px - 2, px - 2, (60, 58, 30))
                    _draw_bolt(img, r0 + px // 2, c0 + px // 2, 6, _CHARGER)
                    for e in [
                        (r0, c0, 2, px),
                        (r0 + px - 2, c0, 2, px),
                        (r0, c0, px, 2),
                        (r0, c0 + px - 2, px, 2),
                    ]:
                        _fill_rect(img, *e, _CHARGER_GLOW)

                elif cell == CellType.REPAIR:
                    _fill_rect(img, r0 + 1, c0 + 1, px - 2, px - 2, (28, 70, 78))
                    _put_text(img, r0 + 4, c0 + 4, "R", _REPAIR, scale=2)
                    mid_r, mid_c = r0 + px // 2, c0 + px // 2
                    _fill_rect(img, mid_r - 1, mid_c - 9, 3, 18, _REPAIR)
                    _fill_rect(img, mid_r - 9, mid_c - 1, 18, 3, _REPAIR)

                _fill_rect(img, r0, c0, 1, px, (38, 38, 42))
                _fill_rect(img, r0, c0, px, 1, (38, 38, 42))

        # ---- interference overlay ----
        if cfg.enable_interference_zones:
            imap = state.grid.interference_map
            for r in range(H):
                for c in range(W):
                    if imap[r, c] > cfg.interference_base + 0.05:
                        intensity = min(1.0, (imap[r, c] - cfg.interference_base) * 2)
                        _fill_rect(
                            img,
                            r * px,
                            c * px,
                            px,
                            px,
                            (180, 40, 40),
                            alpha=intensity * 0.15,
                        )

        # ---- agents ----
        for i in range(cfg.max_agents):
            r0 = int(state.agent.positions[i, 0]) * px
            c0 = int(state.agent.positions[i, 1]) * px
            cy, cx = r0 + px // 2, c0 + px // 2
            acol = _AGENT_COLORS[i % len(_AGENT_COLORS)]
            radius = px // 3

            # --- dead agents at last known position ---
            if state.agent.burst_failed[i]:
                _draw_x(img, cy, cx, px // 4, _DEAD, thickness=3)
                _put_text(img, r0 + 2, c0 + 2, str(i), _DEAD, scale=1)
                _put_text(img, r0 + px - 8, c0 + 2, "BURST", _DEAD, scale=1)
                continue
            if state.agent.battery_dead[i]:
                dim = tuple(max(0, v // 3) for v in acol)
                _draw_circle(img, cy, cx, radius, dim, filled=True)
                _draw_circle(img, cy, cx, radius, _BAT_DEAD, filled=False, thickness=2)
                _draw_empty_battery(img, cy, cx, _BAT_LOW)
                if state.agent.being_dragged_by[i] >= 0:
                    _draw_circle(
                        img, cy, cx, radius + 4, _RESCUE, filled=False, thickness=2
                    )
                _put_text(img, r0 + 2, c0 + 2, str(i), _BAT_LOW, scale=1)
                _put_text(img, r0 + px - 8, c0 + 2, "BAT0", _BAT_LOW, scale=1)
                continue
            if state.agent.failed[i] and not state.agent.active[i]:
                _draw_circle(img, cy, cx, radius // 2, _FAILED, filled=True)
                _draw_x(img, cy, cx, radius // 3, (200, 80, 80), thickness=1)
                if state.agent.being_dragged_by[i] >= 0:
                    _draw_circle(
                        img, cy, cx, radius + 4, _RESCUE, filled=False, thickness=2
                    )
                _put_text(img, r0 + 2, c0 + 2, str(i), _FAILED, scale=1)
                _put_text(img, r0 + px - 8, c0 + 2, "FAIL", _FAILED, scale=1)
                continue
            if not state.agent.active[i]:
                continue

            # --- active agents ---
            phase = ResourcePhase(state.agent.resource_phase[i])

            # phase ring
            ring_c = _PHASE_COLORS.get(phase)
            if ring_c is not None:
                _draw_circle(img, cy, cx, radius + 3, ring_c, filled=False, thickness=3)

            # treatment progress bar
            if state.agent.locked[i] and phase == ResourcePhase.TREATING:
                timer = state.agent.treatment_timer[i]
                progress = 1.0 - timer / max(cfg.treatment_duration, 1)
                bar_total = px - 8
                bar_fill = int(bar_total * progress)
                _fill_rect(img, r0 + px - 7, c0 + 4, 5, bar_total, (40, 40, 44))
                _fill_rect(
                    img, r0 + px - 7, c0 + 4, 5, max(bar_fill, 1), (50, 220, 220)
                )
                _put_text(
                    img,
                    r0 + px - 7,
                    c0 + px - 16,
                    f"{int(progress * 100)}%",
                    (50, 220, 220),
                    scale=1,
                )

            # charging glow
            if state.agent.charging[i]:
                _draw_circle(
                    img, cy, cx, radius + 5, _CHARGER_GLOW, filled=False, thickness=2
                )
                _fill_rect(
                    img,
                    cy - radius,
                    cx - radius,
                    radius * 2,
                    radius * 2,
                    (70, 65, 25),
                    alpha=0.3,
                )

            # body
            _draw_circle(img, cy, cx, radius, acol, filled=True)

            # carrying diamond
            if state.agent.carrying[i] > 0:
                _draw_diamond(img, cy, cx, 5, (255, 255, 220))
                cn = int(state.agent.carrying[i])
                if cn > 1:
                    _put_text(img, cy - 3, cx + 6, str(cn), (255, 255, 220), scale=1)

            if state.agent.rescue_target[i] >= 0:
                _draw_circle(
                    img, cy, cx, radius + 6, _RESCUE, filled=False, thickness=2
                )
                _put_text(
                    img,
                    r0 + px - 10,
                    c0 + 2,
                    f"T{int(state.agent.rescue_target[i])}",
                    _RESCUE,
                    scale=1,
                )

            # charging bolt
            if state.agent.charging[i]:
                _draw_bolt(img, cy, cx, 4, _CHARGER_GLOW)

            # agent ID
            _put_text(img, cy - 3, cx - 2, str(i), (255, 255, 255), scale=1)

            # battery bar
            if cfg.enable_battery:
                bf = state.agent.battery[i] / max(cfg.battery_capacity, 1)
                bw = px - 12
                bx = c0 + 6
                by = r0 + 2
                _fill_rect(img, by, bx, 4, bw, _BAT_BG)
                fill = max(1, int(bw * bf))
                bc = _BAT_HIGH if bf > 0.5 else _BAT_MED if bf > 0.2 else _BAT_LOW
                _fill_rect(img, by, bx, 4, fill, bc)

            # speed dots
            if cfg.enable_heterogeneous:
                for s in range(int(state.agent.speed[i])):
                    _fill_rect(img, r0 + px - 3, c0 + 4 + s * 5, 2, 3, (180, 200, 255))

        self._draw_legend(img, state)
        self._draw_hud(img, state)
        return img

    def _draw_hud(self, img, state):
        cfg = self._config
        hy = self._grid_h
        _fill_rect(img, hy, 0, self._hud_h, self._width, (20, 20, 24))
        _fill_rect(img, hy, 0, 2, self._width, (60, 60, 70))
        y, x, sc = hy + 8, 10, 2
        _put_text(img, y, x, f"STEP:{state.step_count}", _HUD_TEXT, scale=sc)
        x += 14 * 4 * sc
        _put_text(img, y, x, f"DEL:{state.total_deliveries}", _HUD_ACCENT, scale=sc)
        x += 10 * 4 * sc
        na = int(state.agent.active.sum())
        _put_text(img, y, x, f"ACTIVE:{na}/{cfg.max_agents}", _HUD_TEXT, scale=sc)

        y2, x2 = hy + 8 + 14 * sc, 10
        if cfg.enable_task_deadlines:
            _put_text(
                img, y2, x2, f"TASKS:{len(state.task_queue)}", (255, 200, 80), scale=sc
            )
            x2 += 12 * 4 * sc
            _put_text(
                img, y2, x2, f"EXPIRED:{state.expired_tasks}", (230, 80, 80), scale=sc
            )
            x2 += 16 * 4 * sc
        deaths = []
        nb = int(state.agent.burst_failed.sum())
        nbd = int(state.agent.battery_dead.sum())
        nf = int((state.agent.failed & ~state.agent.active).sum())
        if nb:
            deaths.append(f"BURST:{nb}")
        if nbd:
            deaths.append(f"BAT:{nbd}")
        if nf:
            deaths.append(f"FAIL:{nf}")
        if deaths:
            _put_text(img, y2, x2, " ".join(deaths), _DEAD, scale=sc)
            x2 += (len(" ".join(deaths)) + 2) * 4 * sc
        towing = int((state.agent.rescue_target >= 0).sum())
        stranded = int(
            (
                (~state.agent.active)
                & (
                    state.agent.failed
                    | state.agent.burst_failed
                    | state.agent.battery_dead
                )
            ).sum()
        )
        if towing or stranded:
            _put_text(
                img,
                y2,
                x2,
                f"TOW:{towing} DOWN:{stranded}",
                _RESCUE,
                scale=sc,
            )

        y3 = y2 + 14 * sc
        total_res = int(state.grid.shelf_resources.sum())
        total_max = cfg.num_shelves * cfg.resources_per_shelf
        carrying = int(state.agent.carrying.sum())
        _put_text(
            img,
            y3,
            10,
            f"RES:{total_res}/{total_max} CARRY:{carrying}",
            (160, 160, 170),
            scale=max(1, sc - 1),
        )

    def _draw_legend(self, img, state):
        cfg = self._config
        lx = self._grid_w + 8
        _fill_rect(img, 0, self._grid_w, self._grid_h, self._legend_w, (24, 24, 28))
        _fill_rect(img, 0, self._grid_w, self._grid_h, 2, (50, 50, 58))
        y, sc, gap = 8, 2, 16
        _put_text(img, y, lx, "LEGEND", (180, 180, 200), scale=sc)
        y += gap + 4

        for color, label in [
            (_SHELF, "S SHELF"),
            (_TREATMENT, "T TREAT"),
            (_GOAL, "G GOAL"),
            (_CHARGER, "C CHARGE"),
            (_REPAIR, "R REPAIR"),
            (_WALL, "# WALL"),
        ]:
            _fill_rect(img, y, lx, 8, 8, color)
            _put_text(img, y, lx + 12, label, (170, 170, 180), scale=1)
            y += gap - 2

        y += 4
        _put_text(img, y, lx, "AGENTS", (180, 180, 200), scale=sc)
        y += gap + 2
        _draw_circle(img, y + 4, lx + 4, 4, _AGENT_COLORS[0], filled=True)
        _put_text(img, y, lx + 14, "IDLE", (170, 170, 180), scale=1)
        y += gap - 2
        _draw_circle(img, y + 4, lx + 4, 4, _AGENT_COLORS[1], filled=True)
        _draw_diamond(img, y + 4, lx + 4, 2, (255, 255, 220))
        _put_text(img, y, lx + 14, "CARRYING", (170, 170, 180), scale=1)
        y += gap - 2
        _draw_circle(img, y + 4, lx + 4, 4, _AGENT_COLORS[2], filled=True)
        _draw_circle(img, y + 4, lx + 4, 7, _RESCUE, filled=False, thickness=2)
        _put_text(img, y, lx + 14, "TOWING", _RESCUE, scale=1)
        y += gap - 2
        for phase, label in [
            (ResourcePhase.IN_TRANSIT_TO_TREATMENT, "> TREAT"),
            (ResourcePhase.TREATING, "TREATING"),
            (ResourcePhase.IN_TRANSIT_TO_GOAL, "> GOAL"),
        ]:
            _draw_circle(
                img, y + 4, lx + 4, 6, _PHASE_COLORS[phase], filled=False, thickness=2
            )
            _put_text(img, y, lx + 14, label, _PHASE_COLORS[phase], scale=1)
            y += gap - 2

        y += 2
        _put_text(img, y, lx, "DEATHS", (180, 180, 200), scale=sc)
        y += gap + 2
        _draw_x(img, y + 4, lx + 4, 3, _DEAD, thickness=2)
        _put_text(img, y, lx + 14, "BURST", _DEAD, scale=1)
        y += gap - 2
        _draw_empty_battery(img, y + 3, lx + 4, _BAT_LOW)
        _put_text(img, y, lx + 18, "BAT DEAD", _BAT_LOW, scale=1)
        y += gap - 2
        _draw_circle(img, y + 4, lx + 4, 3, _FAILED, filled=True)
        _put_text(img, y, lx + 14, "ATTRITION", _FAILED, scale=1)
        y += gap - 2

        if cfg.enable_battery:
            y += 4
            _put_text(img, y, lx, "BATTERY", (180, 180, 200), scale=sc)
            y += gap + 2
            for bw, bc, bl in [
                (20, _BAT_HIGH, "HIGH"),
                (12, _BAT_MED, "MED"),
                (5, _BAT_LOW, "LOW"),
            ]:
                _fill_rect(img, y, lx, 4, bw, bc)
                _put_text(img, y, lx + 24, bl, bc, scale=1)
                y += gap - 4

        if cfg.enable_task_deadlines:
            y += 4
            _put_text(img, y, lx, "URGENCY", (180, 180, 200), scale=sc)
            y += gap + 2
            for uc, ul in [(_URG_LOW, "LOW"), (_URG_MED, "MED"), (_URG_HIGH, "HIGH")]:
                _fill_rect(img, y, lx, 6, 6, uc)
                _put_text(img, y, lx + 10, ul, uc, scale=1)
                y += gap - 4

        if cfg.enable_interference_zones:
            y += 4
            _fill_rect(img, y, lx, 8, 8, (180, 40, 40), alpha=0.5)
            _put_text(img, y, lx + 14, "RF NOISE", (180, 80, 80), scale=1)
            y += gap

        y += 4
        _put_text(img, y, lx, "ROSTER", (180, 180, 200), scale=sc)
        y += gap + 2
        for i in range(cfg.max_agents):
            if y + 12 > self._grid_h - 10:
                break
            col = _AGENT_COLORS[i % len(_AGENT_COLORS)]
            if state.agent.burst_failed[i]:
                label, col = f"{i}:BURST", _DEAD
            elif state.agent.battery_dead[i]:
                label, col = f"{i}:BAT0", _BAT_LOW
            elif state.agent.failed[i] and not state.agent.active[i]:
                label, col = f"{i}:FAIL", _FAILED
            elif not state.agent.active[i]:
                label, col = f"{i}:OFF", (60, 60, 60)
            elif state.agent.rescue_target[i] >= 0:
                label, col = f"{i}:TOW{int(state.agent.rescue_target[i])}", _RESCUE
            elif state.agent.charging[i]:
                label, col = f"{i}:CHG", _CHARGER
            elif state.agent.locked[i]:
                timer = int(state.agent.treatment_timer[i])
                label = f"{i}:TX{timer}"
                col = _PHASE_COLORS[ResourcePhase.TREATING]
            elif state.agent.carrying[i] > 0:
                phase = ResourcePhase(state.agent.resource_phase[i])
                cn = int(state.agent.carrying[i])
                ph = {
                    ResourcePhase.IN_TRANSIT_TO_TREATMENT: ">T",
                    ResourcePhase.TREATING: "TX",
                    ResourcePhase.IN_TRANSIT_TO_GOAL: ">G",
                }.get(phase, "?")
                label = f"{i}:x{cn}{ph}"
            else:
                label = f"{i}:IDLE"
            _fill_rect(img, y, lx, 7, 7, col)
            _put_text(img, y, lx + 10, label, col, scale=1)
            if cfg.enable_battery and state.agent.active[i]:
                bf = state.agent.battery[i] / max(cfg.battery_capacity, 1)
                bw = int(20 * bf)
                bc = _BAT_HIGH if bf > 0.5 else _BAT_MED if bf > 0.2 else _BAT_LOW
                _fill_rect(img, y + 8, lx, 2, 20, _BAT_BG)
                _fill_rect(img, y + 8, lx, 2, max(bw, 1), bc)
            y += gap - 2

    def _blit_pygame(self, rgb):
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._is_open = False
                return
        surface = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
        self._screen.blit(surface, (0, 0))
        pygame.display.flip()
        self._clock.tick(30)

    def close(self):
        if self._screen is not None:
            import pygame

            pygame.quit()
        self._is_open = False


class VideoRecorder:
    """Captures rgb_array frames and writes them to MP4 or GIF."""

    def __init__(self, video_dir: str = "recordings", fps: int = 10):
        self._video_dir = video_dir
        self._fps = fps
        self._frames: List[np.ndarray] = []
        self._episode_name = "episode"

    def start(self, episode_name: str = "episode"):
        os.makedirs(self._video_dir, exist_ok=True)
        self._episode_name = episode_name
        self._frames = []

    def capture(self, frame: np.ndarray):
        if frame is not None:
            self._frames.append(frame.copy())

    @property
    def frame_count(self):
        return len(self._frames)

    def finish(self, fmt: str = "mp4") -> str:
        if not self._frames:
            raise RuntimeError("No frames captured")
        path = os.path.join(self._video_dir, f"{self._episode_name}.{fmt}")
        try:
            import imageio.v3 as iio

            if fmt == "gif":
                iio.imwrite(path, self._frames, duration=int(1000 / self._fps), loop=0)
            else:
                iio.imwrite(
                    path, self._frames, fps=self._fps, codec="libx264", plugin="pyav"
                )
        except ImportError:
            try:
                from PIL import Image

                frame_dir = os.path.join(self._video_dir, self._episode_name)
                os.makedirs(frame_dir, exist_ok=True)
                for idx, frame in enumerate(self._frames):
                    Image.fromarray(frame).save(
                        os.path.join(frame_dir, f"frame_{idx:05d}.png")
                    )
                path = frame_dir
            except ImportError:
                np_path = path.replace(f".{fmt}", ".npy")
                np.save(np_path, np.stack(self._frames))
                path = np_path
        self._frames = []
        return path


def record_episode(
    env,
    actions_fn=None,
    max_steps=500,
    video_dir="recordings",
    episode_name="episode",
    fps=10,
    fmt="mp4",
    seed=42,
) -> str:
    """Convenience: reset -> run -> record -> save.  Returns path to video."""
    recorder = VideoRecorder(video_dir, fps)
    recorder.start(episode_name)
    obs, infos = env.reset(seed=seed)
    frame = env.render()
    recorder.capture(frame)
    rng = np.random.default_rng(seed + 1)
    for _ in range(max_steps):
        if actions_fn is not None:
            actions = actions_fn(obs, infos, env)
        else:
            actions = {a: int(rng.integers(0, 6)) for a in env.agents}
        if not actions:
            break
        obs, _, terminated, truncated, infos = env.step(actions)
        frame = env.render()
        recorder.capture(frame)
        if any(terminated.values()) or any(truncated.values()):
            break
    return recorder.finish(fmt=fmt)
