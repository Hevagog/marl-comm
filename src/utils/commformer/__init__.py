"""CommFormer communication analysis utilities."""

from .analysis import (
    CommFormerCommCollector,
    CommFormerAnalysisData,
    extract_alpha_from_agent,
    compute_hard_adj_from_alpha,
    compute_graph_metrics,
    compute_encoder_pca,
    compute_encoder_tsne,
    compute_alpha_stats,
    compute_information_flow,
    compute_laplacian_spectrum,
    compute_representation_alignment,
    compute_encoder_variance_dynamics,
    compute_alpha_concentration,
    print_summary,
)
from .visualizer import save_all_commformer_figures
from .runner import run_commformer_analysis

__all__ = [
    "CommFormerCommCollector",
    "CommFormerAnalysisData",
    "extract_alpha_from_agent",
    "compute_hard_adj_from_alpha",
    "compute_graph_metrics",
    "compute_encoder_pca",
    "compute_encoder_tsne",
    "compute_alpha_stats",
    "compute_information_flow",
    "compute_laplacian_spectrum",
    "compute_representation_alignment",
    "compute_encoder_variance_dynamics",
    "compute_alpha_concentration",
    "print_summary",
    "save_all_commformer_figures",
    "run_commformer_analysis",
]
