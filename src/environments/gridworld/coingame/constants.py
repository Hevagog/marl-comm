from environments.gridworld.rendering_utils import GridworldStyle

_COINGAME_STYLE = GridworldStyle(
    title="Coin Game",
    background="checker",
    bg_light=(240, 240, 240),
    bg_dark=(210, 210, 210),
    grid_line=(50, 50, 50),
    hud_bg=(30, 30, 30),
    hud_text=(220, 220, 220),
    agent_colors={
        "agent_0": (220, 50, 50),
        "agent_1": (50, 50, 220),
    },
    agent_outline=(50, 50, 50),
    agent_labels={
        "agent_0": "R",
        "agent_1": "B",
    },
    coin_colors={
        "red": (255, 130, 130),
        "blue": (130, 130, 255),
    },
    coin_outline=(20, 20, 20),
    trap_color=(250, 50, 50),
    goal_color=(50, 200, 50),
    vision_color=(250, 230, 200),
    show_hud=True,
)
