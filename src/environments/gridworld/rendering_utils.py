from dataclasses import dataclass

Color = tuple[int, int, int]

DEFAULT_CELL_SIZE = 80
DEFAULT_FPS = 10
HUD_HEIGHT = 30


@dataclass(frozen=True)
class GridworldStyle:
    title: str
    background: str
    bg_light: Color
    bg_dark: Color | None
    grid_line: Color
    hud_bg: Color
    hud_text: Color
    agent_colors: dict[str, Color]
    agent_outline: Color | None
    agent_labels: dict[str, str] | None
    coin_colors: dict[str, Color]
    coin_outline: Color
    trap_color: Color
    goal_color: Color
    vision_color: Color
    show_hud: bool
