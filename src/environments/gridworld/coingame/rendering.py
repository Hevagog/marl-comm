from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from pygame import Surface, Rect

import numpy as np
import pygame
import pygame.surfarray

try:
    import pygame.font

    _HAS_FONT = True
except (ImportError, NotImplementedError):
    _HAS_FONT = False

_BG_LIGHT = (240, 240, 240)
_BG_DARK = (210, 210, 210)
_GRID_LINE = (50, 50, 50)
_RED_AGENT = (220, 50, 50)
_BLUE_AGENT = (50, 50, 220)
_RED_COIN = (255, 130, 130)
_BLUE_COIN = (130, 130, 255)
_COIN_OUTLINE = (20, 20, 20)
_HUD_BG = (30, 30, 30)
_HUD_TEXT = (220, 220, 220)

_DEFAULT_CELL_SIZE = 80  # pixels per grid cell
_HUD_HEIGHT = 30  # pixels for the HUD strip at the bottom
_DEFAULT_FPS = 10


class CoinGameRenderer:
    """Pygame-backed renderer for :class:`CoinGameEnv`.

    Parameters
    ----------
    grid_size : int
        Number of cells along each axis of the square grid.
    render_mode : {"human", "rgb_array"}
        ``"human"`` opens a pygame window; ``"rgb_array"`` renders off-screen.
    cell_size : int
        Pixel width/height of each grid cell.
    fps : int
        Target frames per second (only used in ``"human"`` mode to throttle the
        display clock).
    """

    def __init__(
        self,
        grid_size: int,
        render_mode: str,
        cell_size: int = _DEFAULT_CELL_SIZE,
        fps: int = _DEFAULT_FPS,
    ) -> None:
        self._grid_size = grid_size
        self._render_mode = render_mode
        self._cell_size = cell_size
        self._fps = fps

        self._win_w = grid_size * cell_size
        self._win_h = grid_size * cell_size + _HUD_HEIGHT

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
            pygame.display.set_caption("Coin Game")
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
        agent_positions: dict[str, Tuple[int, int]],
        red_coin_pos: Tuple[int, int] | None,
        blue_coin_pos: Tuple[int, int] | None,
        step: int,
        max_steps: int,
    ) -> np.ndarray | None:
        """Draw one frame."""
        canvas = self._screen if self._render_mode == "human" else self._surface

        self._draw_background(canvas)
        self._draw_coins(canvas, red_coin_pos, blue_coin_pos)
        self._draw_agents(canvas, agent_positions)
        self._draw_hud(canvas, step, max_steps)

        if self._render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass
            pygame.display.flip()
            self._clock.tick(self._fps)
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            ).copy()

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
        cs = self._cell_size
        gs = self._grid_size

        for row in range(gs):
            for col in range(gs):
                colour = _BG_LIGHT if (row + col) % 2 == 0 else _BG_DARK
                rect = self._cell_rect(col, row)
                pygame.draw.rect(canvas, colour, rect)

        for i in range(gs + 1):
            pygame.draw.line(canvas, _GRID_LINE, (i * cs, 0), (i * cs, gs * cs), 1)
            pygame.draw.line(canvas, _GRID_LINE, (0, i * cs), (gs * cs, i * cs), 1)

        hud_rect = pygame.Rect(0, gs * cs, self._win_w, _HUD_HEIGHT)
        pygame.draw.rect(canvas, _HUD_BG, hud_rect)

    def _draw_diamond(
        self,
        canvas: Surface,
        cx: int,
        cy: int,
        half: int,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int],
    ) -> None:
        points = [(cx, cy - half), (cx + half, cy), (cx, cy + half), (cx - half, cy)]
        pygame.draw.polygon(canvas, fill, points)
        pygame.draw.polygon(canvas, outline, points, 2)

    def _draw_coins(
        self,
        canvas: Surface,
        red_coin_pos: Tuple[int, int] | None,
        blue_coin_pos: Tuple[int, int] | None,
    ) -> None:
        cs = self._cell_size
        half = max(cs // 5, 4)

        if red_coin_pos is not None:
            cx, cy = self._cell_center(*red_coin_pos)
            self._draw_diamond(canvas, cx, cy, half, _RED_COIN, _COIN_OUTLINE)

        if blue_coin_pos is not None:
            cx, cy = self._cell_center(*blue_coin_pos)
            self._draw_diamond(canvas, cx, cy, half, _BLUE_COIN, _COIN_OUTLINE)

    def _draw_agents(
        self,
        canvas: Surface,
        agent_positions: dict[str, Tuple[int, int]],
    ) -> None:
        cs = self._cell_size
        radius = max(cs // 3, 6)

        colours = {
            "agent_0": _RED_AGENT,
            "agent_1": _BLUE_AGENT,
        }

        for agent, colour in colours.items():
            if agent not in agent_positions:
                continue
            cx, cy = self._cell_center(*agent_positions[agent])
            pygame.draw.circle(canvas, colour, (cx, cy), radius)
            pygame.draw.circle(canvas, _GRID_LINE, (cx, cy), radius, 2)

            # Draw a small letter to distinguish agents if font is loaded
            if self._font is not None:
                label = "R" if agent == "agent_0" else "B"
                text_surf = self._font.render(label, True, (255, 255, 255))
                text_rect = text_surf.get_rect(center=(cx, cy))
                canvas.blit(text_surf, text_rect)

    def _draw_hud(self, canvas: Surface, step: int, max_steps: int) -> None:
        # Skip drawing HUD text if fonts are missing
        if self._font is None:
            return

        gs = self._grid_size
        cs = self._cell_size
        hud_y = gs * cs + _HUD_HEIGHT // 2

        text = f"Step: {step} / {max_steps}"
        text_surf = self._font.render(text, True, _HUD_TEXT)
        text_rect = text_surf.get_rect(center=(self._win_w // 2, hud_y))
        canvas.blit(text_surf, text_rect)
