"""MAGIC communication analysis utilities."""

from .coingame_analysis import (
    MAGICCommCollector,
    MAGICAnalysisData,
    compute_mean_adj,
    compute_topology_stability,
    compute_spectral_properties,
    compute_role_distribution,
    compute_edge_bimodality,
    compute_round_agreement,
    compute_message_information_gain,
    compute_graph_density_over_time,
)
from .coingame_visualizer import save_all_magic_figures_extended
from .warehouse_analysis import MAGICWarehouseCommCollector
from .warehouse_visualizer import save_all_magic_warehouse_figures
from .runner import run_magic_analysis

__all__ = [
    # Coin Game / generic analysis
    "MAGICCommCollector",
    "MAGICAnalysisData",
    "compute_mean_adj",
    "compute_topology_stability",
    "compute_spectral_properties",
    "compute_role_distribution",
    "compute_edge_bimodality",
    "compute_round_agreement",
    "compute_message_information_gain",
    "compute_graph_density_over_time",
    "save_all_magic_figures_extended",
    # Warehouse
    "MAGICWarehouseCommCollector",
    "save_all_magic_warehouse_figures",
    # Runner
    "run_magic_analysis",
]
