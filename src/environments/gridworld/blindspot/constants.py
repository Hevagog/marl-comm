from environments.gridworld.rendering_utils import GridworldStyle

_BLINDSPOT_STYLE = GridworldStyle(
    title="Blind-Spot Navigation",
    background="solid",
    bg_light=(250, 250, 250),
    bg_dark=None,
    grid_line=(200, 200, 200),
    hud_bg=(30, 30, 30),
    hud_text=(220, 220, 220),
    agent_colors={
        "agent_0": (50, 150, 250),
        "agent_1": (250, 150, 50),
    },
    agent_outline=None,
    agent_labels=None,
    coin_colors={},
    coin_outline=(20, 20, 20),
    trap_color=(250, 50, 50),
    goal_color=(50, 200, 50),
    vision_color=(250, 230, 200),
    show_hud=False,
)
