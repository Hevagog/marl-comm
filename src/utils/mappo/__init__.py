"""MAPPO analysis and visualisation utilities."""

from .coingame_data import (
    RED_AGENT,
    BLUE_AGENT,
    CoinEvent,
    StepRecord,
    EpisodeData as CoinGameEpisodeData,
    EvalData as CoinGameEvalData,
)
from .coingame_analysis import EvalCollector
from .coingame_visualizer import save_all_figures

from .warehouse_data import EvalData as WarehouseEvalData
from .warehouse_analysis import WarehouseEvalCollector
from .warehouse_visualizer import save_all_warehouse_figures

from .continuous_coord_data import EvalData as ContinuousCoordEvalData
from .continuous_coord_analysis import ContinuousCoordEvalCollector
from .continuous_coord_visualizer import save_all_cc_figures

from .altruism_analysis import CoinGameAltruismCollector, load_prototypes
from .altruism_visualizer import save_all_altruism_figures

__all__ = [
    # Coin Game
    "RED_AGENT",
    "BLUE_AGENT",
    "CoinEvent",
    "StepRecord",
    "CoinGameEpisodeData",
    "CoinGameEvalData",
    "EvalCollector",
    "save_all_figures",
    # Warehouse
    "WarehouseEvalData",
    "WarehouseEvalCollector",
    "save_all_warehouse_figures",
    # Continuous Coordination
    "ContinuousCoordEvalData",
    "ContinuousCoordEvalCollector",
    "save_all_cc_figures",
    # Altruism
    "CoinGameAltruismCollector",
    "load_prototypes",
    "save_all_altruism_figures",
]
