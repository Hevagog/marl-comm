"""MAM (Multi-Agent Mamba) analysis utilities.

Submodules
----------
analysis           : lightweight base MAM performance collector
hopfield_analysis  : MAMHM Hopfield memory collector (MAMHMCollector)
hopfield_visualizer: MAMHM figure generators
runner             : run_mam_analysis dispatcher
hopfield_runner    : run_mamhm_analysis dispatcher (Hopfield-specific)
"""

from .analysis import collect_episode_returns
from .hopfield_analysis import (
    MAMHMCollector,
    MAMHMAnalysisData,
    extract_hopfield_params,
    compute_memory_utilization,
    compute_agent_memory_profiles,
    compute_memory_attention_dynamics,
    compute_xi_similarity_matrix,
    compute_xi_pca,
    compute_memory_reward_correlation,
    compute_attention_concentration,
    print_summary as hopfield_print_summary,
)
from .hopfield_visualizer import save_all_mamhm_figures
from .runner import run_mam_analysis
from .hopfield_runner import run_mamhm_analysis

__all__ = [
    # Base MAM
    "collect_episode_returns",
    "run_mam_analysis",
    # MAMHM
    "MAMHMCollector",
    "MAMHMAnalysisData",
    "extract_hopfield_params",
    "compute_memory_utilization",
    "compute_agent_memory_profiles",
    "compute_memory_attention_dynamics",
    "compute_xi_similarity_matrix",
    "compute_xi_pca",
    "compute_memory_reward_correlation",
    "compute_attention_concentration",
    "hopfield_print_summary",
    "save_all_mamhm_figures",
    "run_mamhm_analysis",
]
