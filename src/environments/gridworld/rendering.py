from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Tuple

import numpy as np
import pygame
import pygame.surfarray

from environments.gridworld.blindspot.constants import _BLINDSPOT_STYLE
from environments.gridworld.coingame.constants import _COINGAME_STYLE

if TYPE_CHECKING:
    from pygame import Surface, Rect

try:
    import pygame.font

    _HAS_FONT = True
except (ImportError, NotImplementedError):
    _HAS_FONT = False


from .rendering_utils import (
    GridworldStyle,
    DEFAULT_CELL_SIZE,
    HUD_HEIGHT,
    DEFAULT_FPS,
    Color,
)


_STYLES: dict[str, GridworldStyle] = {
    "coingame": _COINGAME_STYLE,
    "blindspot": _BLINDSPOT_STYLE,
}


class GridworldRenderer:
    """Unified pygame renderer for gridworld environments."""

    def __init__(
        self,
        grid_size: int,
        render_mode: str,
        cell_size: int = DEFAULT_CELL_SIZE,
        fps: int = DEFAULT_FPS,
        style: str | GridworldStyle = "coingame",
    ) -> None:
        self._grid_size = grid_size
        self._render_mode = render_mode
        self._cell_size = cell_size
        self._fps = fps

        if isinstance(style, GridworldStyle):
            self._style = style
        else:
            self._style = _STYLES.get(style, _COINGAME_STYLE)

        self._hud_height = HUD_HEIGHT if self._style.show_hud else 0
        self._win_w = grid_size * cell_size
        self._win_h = grid_size * cell_size + self._hud_height

        self._screen = None
        self._surface = None
        self._clock = None
        self._font = None

        self._init_pygame()

    def _init_pygame(self) -> None:
        pygame.init()
        pygame.display.init()

        if _HAS_FONT:
            pygame.font.init()

        if self._render_mode == "human":
            self._screen = pygame.display.set_mode((self._win_w, self._win_h))
            pygame.display.set_caption(self._style.title)
            self._clock = pygame.time.Clock()
        else:
            self._surface = pygame.Surface((self._win_w, self._win_h))

        if _HAS_FONT:
            try:
                self._font = pygame.font.SysFont("monospace", 16)
            except Exception:
                self._font = pygame.font.Font(None, 18)
        else:
            self._font = None

    def render(
        self,
        *,
        agent_positions: dict[str, Tuple[int, int]],
        step: int,
        max_steps: int,
        coins: dict[str, Tuple[int, int] | None] | None = None,
        traps: Iterable[Tuple[int, int]] | None = None,
        goal: Tuple[int, int] | None = None,
        vision_center: Tuple[int, int] | None = None,
        vision_range: int | None = None,
        reached_goal: dict[str, bool] | None = None,
        messages: dict[str, int] | None = None,
        hud_text: str | None = None,
    ) -> np.ndarray | None:
        canvas = self._screen if self._render_mode == "human" else self._surface

        self._draw_background(canvas)
        if vision_center is not None and vision_range is not None:
            self._draw_vision(canvas, vision_center, vision_range)
        self._draw_grid(canvas)

        if goal is not None:
            self._draw_goal(canvas, goal)
        if traps is not None:
            self._draw_traps(canvas, traps)
        if coins is not None:
            self._draw_coins(canvas, coins)

        self._draw_agents(canvas, agent_positions, reached_goal)

        if self._style.show_hud:
            if hud_text is None:
                hud_text = f"Step: {step} / {max_steps}"
                if messages:
                    msg_text = ", ".join(f"{k}:{v}" for k, v in messages.items())
                    hud_text = f"{hud_text} | Msg {msg_text}"
            self._draw_hud(canvas, hud_text)

        if self._render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass
            pygame.display.flip()
            if self._clock is not None:
                self._clock.tick(self._fps)
            return None
        else:
            frame = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            ).copy()
            return frame

    def close(self) -> None:
        if _HAS_FONT:
            pygame.font.quit()
        pygame.display.quit()
        pygame.quit()
        self._screen = None
        self._surface = None
        self._clock = None
        self._font = None

    def _cell_rect(self, x: int, y: int) -> Rect:
        cs = self._cell_size
        return pygame.Rect(x * cs, y * cs, cs, cs)

    def _cell_center(self, x: int, y: int) -> Tuple[int, int]:
        cs = self._cell_size
        return (x * cs + cs // 2, y * cs + cs // 2)

    def _draw_background(self, canvas: Surface) -> None:
        if self._style.background == "checker" and self._style.bg_dark is not None:
            gs = self._grid_size
            for row in range(gs):
                for col in range(gs):
                    colour = (
                        self._style.bg_light
                        if (row + col) % 2 == 0
                        else self._style.bg_dark
                    )
                    rect = self._cell_rect(col, row)
                    pygame.draw.rect(canvas, colour, rect)
        else:
            canvas.fill(self._style.bg_light)

        if self._style.show_hud and self._hud_height > 0:
            hud_rect = pygame.Rect(
                0, self._grid_size * self._cell_size, self._win_w, self._hud_height
            )
            pygame.draw.rect(canvas, self._style.hud_bg, hud_rect)

    def _draw_grid(self, canvas: Surface) -> None:
        cs = self._cell_size
        gs = self._grid_size
        for i in range(gs + 1):
            pygame.draw.line(
                canvas, self._style.grid_line, (i * cs, 0), (i * cs, gs * cs), 1
            )
            pygame.draw.line(
                canvas, self._style.grid_line, (0, i * cs), (gs * cs, i * cs), 1
            )

    def _draw_vision(
        self, canvas: Surface, center: Tuple[int, int], vision_range: int
    ) -> None:
        vx, vy = center
        for dx in range(-vision_range, vision_range + 1):
            for dy in range(-vision_range, vision_range + 1):
                px, py = vx + dx, vy + dy
                if 0 <= px < self._grid_size and 0 <= py < self._grid_size:
                    rect = self._cell_rect(px, py)
                    pygame.draw.rect(canvas, self._style.vision_color, rect)

    def _draw_goal(self, canvas: Surface, goal: Tuple[int, int]) -> None:
        gx, gy = goal
        pygame.draw.circle(
            canvas,
            self._style.goal_color,
            self._cell_center(gx, gy),
            max(self._cell_size // 3, 4),
        )

    def _draw_traps(self, canvas: Surface, traps: Iterable[Tuple[int, int]]) -> None:
        cs = self._cell_size
        for tx, ty in traps:
            rect = pygame.Rect(tx * cs + cs // 4, ty * cs + cs // 4, cs // 2, cs // 2)
            pygame.draw.rect(canvas, self._style.trap_color, rect)

    def _draw_diamond(
        self,
        canvas: Surface,
        cx: int,
        cy: int,
        half: int,
        fill: Color,
        outline: Color,
    ) -> None:
        points = [(cx, cy - half), (cx + half, cy), (cx, cy + half), (cx - half, cy)]
        pygame.draw.polygon(canvas, fill, points)
        pygame.draw.polygon(canvas, outline, points, 2)

    def _draw_coins(
        self, canvas: Surface, coins: dict[str, Tuple[int, int] | None]
    ) -> None:
        cs = self._cell_size
        half = max(cs // 5, 4)
        for name, pos in coins.items():
            if pos is None:
                continue
            colour = self._style.coin_colors.get(name, (200, 200, 200))
            cx, cy = self._cell_center(*pos)
            self._draw_diamond(canvas, cx, cy, half, colour, self._style.coin_outline)

    def _draw_agents(
        self,
        canvas: Surface,
        agent_positions: dict[str, Tuple[int, int]],
        reached_goal: dict[str, bool] | None,
    ) -> None:
        cs = self._cell_size
        radius = max(cs // 3, 6)

        for agent, pos in agent_positions.items():
            if reached_goal and reached_goal.get(agent, False):
                continue
            colour = self._style.agent_colors.get(agent, (0, 0, 0))
            cx, cy = self._cell_center(*pos)
            pygame.draw.circle(canvas, colour, (cx, cy), radius)
            if self._style.agent_outline is not None:
                pygame.draw.circle(
                    canvas, self._style.agent_outline, (cx, cy), radius, 2
                )

            if self._font is not None and self._style.agent_labels is not None:
                label = self._style.agent_labels.get(agent)
                if label:
                    text_surf = self._font.render(label, True, (255, 255, 255))
                    text_rect = text_surf.get_rect(center=(cx, cy))
                    canvas.blit(text_surf, text_rect)

    def _draw_hud(self, canvas: Surface, text: str) -> None:
        if self._font is None or self._hud_height <= 0:
            return
        hud_y = self._grid_size * self._cell_size + self._hud_height // 2
        text_surf = self._font.render(text, True, self._style.hud_text)
        text_rect = text_surf.get_rect(center=(self._win_w // 2, hud_y))
        canvas.blit(text_surf, text_rect)
