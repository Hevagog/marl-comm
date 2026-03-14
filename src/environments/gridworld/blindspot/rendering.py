from __future__ import annotations

from typing import Tuple, TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from pygame import Surface

import numpy as np
import pygame

_BG_LIGHT = (250, 250, 250)
_GRID_LINE = (200, 200, 200)
_AGENT_A_COLOR = (50, 150, 250)  # Blue-ish
_AGENT_B_COLOR = (250, 150, 50)  # Orange-ish
_TRAP_COLOR = (250, 50, 50)  # Red
_GOAL_COLOR = (50, 200, 50)  # Green
_VISION_BG = (250, 230, 200)  # highlight for Agent B vision

_DEFAULT_CELL_SIZE = 80


class BlindSpotRenderer:
    def __init__(
        self,
        grid_size: int,
        render_mode: str,
        cell_size: int = _DEFAULT_CELL_SIZE,
        fps: int = 10,
    ):
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.cell_size = cell_size
        self.fps = fps
        self.width = grid_size * cell_size
        self.height = grid_size * cell_size
        self.screen: Optional[Surface] = None
        self.clock = None

    def render(
        self,
        positions: Dict[str, Tuple[int, int]],
        traps: List[Tuple[int, int]],
        goal: Tuple[int, int],
        reached_goal: Dict[str, bool],
        step: int,
        max_steps: int,
        vision_range: int,
        messages: Optional[Dict[str, int]] = None,
    ) -> np.ndarray | None:
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Blind-Spot Navigation")
                self.clock = pygame.time.Clock()
            else:
                self.screen = pygame.Surface((self.width, self.height))

        self.screen.fill(_BG_LIGHT)

        # Draw vision range of agent 1
        a1_pos = positions.get("agent_1")
        if a1_pos:
            vx, vy = a1_pos
            for dx in range(-vision_range, vision_range + 1):
                for dy in range(-vision_range, vision_range + 1):
                    px, py = vx + dx, vy + dy
                    if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                        rect = pygame.Rect(
                            px * self.cell_size,
                            py * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        )
                        pygame.draw.rect(self.screen, _VISION_BG, rect)

        # Draw grid
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                self.screen,
                _GRID_LINE,
                (x * self.cell_size, 0),
                (x * self.cell_size, self.height),
            )
            pygame.draw.line(
                self.screen,
                _GRID_LINE,
                (0, x * self.cell_size),
                (self.width, x * self.cell_size),
            )

        # Draw goal
        pygame.draw.circle(
            self.screen,
            _GOAL_COLOR,
            (
                int((goal[0] + 0.5) * self.cell_size),
                int((goal[1] + 0.5) * self.cell_size),
            ),
            self.cell_size // 3,
        )

        # Draw traps
        for tx, ty in traps:
            rect = pygame.Rect(
                tx * self.cell_size + self.cell_size // 4,
                ty * self.cell_size + self.cell_size // 4,
                self.cell_size // 2,
                self.cell_size // 2,
            )
            pygame.draw.rect(self.screen, _TRAP_COLOR, rect)

        # Draw agents
        colors = {"agent_0": _AGENT_A_COLOR, "agent_1": _AGENT_B_COLOR}
        for aid, pos in positions.items():
            if not reached_goal.get(aid, False):
                color = colors.get(aid, (0, 0, 0))
                pygame.draw.circle(
                    self.screen,
                    color,
                    (
                        int((pos[0] + 0.5) * self.cell_size),
                        int((pos[1] + 0.5) * self.cell_size),
                    ),
                    self.cell_size // 3,
                )

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            if self.clock is not None:
                self.clock.tick(self.fps)
            return None
        else:
            return np.transpose(pygame.surfarray.pixels3d(self.screen), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
