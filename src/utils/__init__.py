from utils.utils import load_config

from utils.mappo.coingame_analysis import EvalCollector
from utils.mappo.coingame_visualizer import save_all_figures
from utils.mappo.warehouse_analysis import WarehouseEvalCollector
from utils.mappo.warehouse_visualizer import save_all_warehouse_figures
from utils.mappo.continuous_coord_analysis import ContinuousCoordEvalCollector
from utils.mappo.continuous_coord_visualizer import save_all_cc_figures
from utils.mappo.altruism_analysis import CoinGameAltruismCollector, load_prototypes
from utils.mappo.altruism_visualizer import save_all_altruism_figures

from utils.magic.warehouse_analysis import MAGICWarehouseCommCollector
from utils.magic.warehouse_visualizer import save_all_magic_warehouse_figures
from utils.magic.coingame_visualizer import save_all_magic_figures_extended
from utils.magic.coingame_analysis import (
    compute_topology_stability,
    compute_spectral_properties as magic_compute_spectral_properties,
    compute_role_distribution,
    compute_edge_bimodality,
    compute_round_agreement,
    compute_message_information_gain,
    compute_graph_density_over_time,
)

from utils.commformer.analysis import (
    CommFormerCommCollector,
    CommFormerAnalysisData,
    extract_alpha_from_agent,
    compute_hard_adj_from_alpha,
    compute_graph_metrics,
    compute_encoder_pca,
    compute_encoder_tsne,
    compute_alpha_stats,
    print_summary as commformer_print_summary,
    compute_information_flow,
    compute_laplacian_spectrum,
    compute_representation_alignment,
    compute_encoder_variance_dynamics,
    compute_alpha_concentration,
)
from utils.commformer.visualizer import save_all_commformer_figures

from utils.mam.hopfield_analysis import (
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
    print_summary as mamhm_print_summary,
)
from utils.mam.hopfield_visualizer import save_all_mamhm_figures

from utils.shared.stats import (
    iqm,
    bootstrap_ci,
    optimality_gap,
    mann_whitney_u,
    aggregate_metrics,
    pairwise_tests,
    print_aggregate_table,
)

from utils.comm_graph_renderer import render_comm_graph_frame, make_split_frame

__all__ = [
    "load_config",
    # MAPPO / coingame
    "EvalCollector",
    "save_all_figures",
    # MAPPO / warehouse
    "WarehouseEvalCollector",
    "save_all_warehouse_figures",
    # MAPPO / continuous coordination
    "ContinuousCoordEvalCollector",
    "save_all_cc_figures",
    # MAPPO / altruism
    "CoinGameAltruismCollector",
    "load_prototypes",
    "save_all_altruism_figures",
    # MAGIC / warehouse
    "MAGICWarehouseCommCollector",
    "save_all_magic_warehouse_figures",
    # MAGIC / comm analysis
    "save_all_magic_figures_extended",
    "compute_topology_stability",
    "magic_compute_spectral_properties",
    "compute_role_distribution",
    "compute_edge_bimodality",
    "compute_round_agreement",
    "compute_message_information_gain",
    "compute_graph_density_over_time",
    # CommFormer
    "CommFormerCommCollector",
    "CommFormerAnalysisData",
    "extract_alpha_from_agent",
    "compute_hard_adj_from_alpha",
    "compute_graph_metrics",
    "compute_encoder_pca",
    "compute_encoder_tsne",
    "compute_alpha_stats",
    "commformer_print_summary",
    "save_all_commformer_figures",
    "compute_information_flow",
    "compute_laplacian_spectrum",
    "compute_representation_alignment",
    "compute_encoder_variance_dynamics",
    "compute_alpha_concentration",
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
    "mamhm_print_summary",
    "save_all_mamhm_figures",
    # rliable stats
    "iqm",
    "bootstrap_ci",
    "optimality_gap",
    "mann_whitney_u",
    "aggregate_metrics",
    "pairwise_tests",
    "print_aggregate_table",
    # Comm graph renderer
    "render_comm_graph_frame",
    "make_split_frame",
]
