"""
commformer_comm_visualizer.py
==============================
Rich visualizations for CommFormer's learned communication graph.

Figures produced
----------------
 1.  fig_alpha_heatmap       – Learned α parameter as a heatmap
 2.  fig_adj_heatmap         – Hard binary k-hot adjacency heatmap
 3.  fig_graph_structure     – Directed graph with edge weights
 4.  fig_degree_distribution – In/out degree per agent
 5.  fig_alpha_stats         – α row/column statistics (sender/receiver strength)
 6.  fig_encoder_pca         – PCA of encoder outputs coloured by agent
 7.  fig_encoder_tsne        – t-SNE of encoder outputs coloured by agent
 8.  fig_encoder_norm        – L2-norm of encoder outputs over episode time
 9.  fig_reward_distribution – Distribution of episode rewards
10.  fig_episode_lengths     – Episode length distribution
11.  fig_reward_trajectory   – Per-episode cumulative reward curves
12.  fig_summary_table       – Text summary of key metrics

All sub-figures saved as both PDF and PNG to *output_dir*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from .commformer_comm_analysis import (
    CommFormerAnalysisData,
    compute_graph_metrics,
    compute_encoder_pca,
    compute_encoder_tsne,
    compute_encoder_norm_per_agent,
    compute_alpha_stats,
    compute_reward_vs_graph_density,
    print_summary,
)

# ──────────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────────

# Agent colours — up to 8 agents
_AGENT_COLORS = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B2",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # grey
]

CMAP_ALPHA = "RdYlGn"   # diverging: strong negative → neutral → strong positive
CMAP_ADJ = "Blues"       # sequential: 0 → 1

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(str(out_dir / f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def _agent_labels(n: int) -> list[str]:
    return [f"A{i}" for i in range(n)]


def _heatmap_ax(
    ax: plt.Axes,
    mat: np.ndarray,
    title: str,
    agent_labels: list[str],
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    annot: bool = True,
) -> None:
    """Draw a labelled heatmap on the given axes."""
    im = ax.imshow(mat, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    n = mat.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(agent_labels, fontsize=9)
    ax.set_yticklabels(agent_labels, fontsize=9)
    ax.set_xlabel("Receiver")
    ax.set_ylabel("Sender")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if annot:
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, f"{mat[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(mat[i, j]) > 0.6 * (vmax or mat.max() + 1e-9) else "black",
                )


# ──────────────────────────────────────────────────────────────────────────────
# Individual figure generators
# ──────────────────────────────────────────────────────────────────────────────


def fig_alpha_heatmap(data: CommFormerAnalysisData, prefix: str) -> plt.Figure | None:
    """Heatmap of the learned α parameter (N×N)."""
    if data.alpha is None:
        return None

    n = data.num_agents
    labels = _agent_labels(n)
    alpha = data.alpha

    fig, ax = plt.subplots(figsize=(max(4, n + 1), max(4, n + 1)))
    vabs = max(abs(alpha.min()), abs(alpha.max()), 1e-9)
    _heatmap_ax(
        ax, alpha,
        f"Learned α — CommFormer Communication Graph\n({prefix})",
        labels, cmap=CMAP_ALPHA, vmin=-vabs, vmax=vabs,
    )
    fig.tight_layout()
    return fig


def fig_adj_heatmap(data: CommFormerAnalysisData, prefix: str) -> plt.Figure | None:
    """Heatmap of the hard k-hot binary adjacency derived from α."""
    if data.hard_adj is None:
        return None

    n = data.num_agents
    labels = _agent_labels(n)
    adj = data.hard_adj

    fig, axes = plt.subplots(1, 2, figsize=(max(8, 2 * (n + 1)), max(4, n + 1)))

    # Left: hard adjacency
    _heatmap_ax(
        axes[0], adj,
        f"Hard k-hot Adjacency  (S={data.sparsity})\n({prefix})",
        labels, cmap=CMAP_ADJ, vmin=0.0, vmax=1.0,
    )

    # Right: α soft heatmap for comparison (if available)
    if data.alpha is not None:
        alpha = data.alpha
        vabs = max(abs(alpha.min()), abs(alpha.max()), 1e-9)
        _heatmap_ax(
            axes[1], alpha,
            "Soft α parameter",
            labels, cmap=CMAP_ALPHA, vmin=-vabs, vmax=vabs,
        )
    else:
        axes[1].axis("off")

    fig.tight_layout()
    return fig


def fig_graph_structure(data: CommFormerAnalysisData, prefix: str) -> plt.Figure | None:
    """Directed graph visualization of the hard adjacency.

    Nodes represent agents; arrows show active communication edges.
    Edge width is proportional to α value (if available).
    """
    if data.hard_adj is None:
        return None

    try:
        import networkx as nx
    except ImportError:
        return None

    n = data.num_agents
    adj = data.hard_adj
    alpha = data.alpha

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0.5:
                weight = float(alpha[i, j]) if alpha is not None else 1.0
                G.add_edge(i, j, weight=weight)

    fig, ax = plt.subplots(figsize=(5, 5))

    # Circular layout
    pos = nx.circular_layout(G)
    labels = {i: f"A{i}" for i in range(n)}

    node_colors = [_AGENT_COLORS[i % len(_AGENT_COLORS)] for i in range(n)]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=800)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_color="white", font_size=11)

    if G.edges:
        weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
        # Normalise edge widths
        w_arr = np.asarray(weights)
        w_norm = 1.0 + 3.0 * (w_arr - w_arr.min()) / (w_arr.max() - w_arr.min() + 1e-9)
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=w_norm.tolist(),
            arrowsize=20,
            edge_color="steelblue",
            arrows=True,
            connectionstyle="arc3,rad=0.15",
        )

    ax.set_title(f"Learned Communication Graph\n({prefix})", pad=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_degree_distribution(data: CommFormerAnalysisData, prefix: str) -> plt.Figure | None:
    """Bar chart of in-degree and out-degree per agent."""
    if data.hard_adj is None:
        return None

    metrics = compute_graph_metrics(data)
    n = data.num_agents
    labels = _agent_labels(n)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(5, n + 2), 4))
    bars_in = ax.bar(x - width / 2, metrics["in_degree"], width,
                     label="In-degree (receives from)", color="#4C72B0", alpha=0.85)
    bars_out = ax.bar(x + width / 2, metrics["out_degree"], width,
                      label="Out-degree (sends to)", color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Degree")
    ax.set_title(f"Agent Communication Degree\n({prefix})")
    ax.legend()
    ax.set_ylim(0, n + 0.5)

    # Annotate
    for bar in list(bars_in) + list(bars_out):
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.05,
                str(int(h)), ha="center", va="bottom", fontsize=9,
            )

    # Add density and reciprocity as text
    ax.text(
        0.98, 0.97,
        f"density={metrics['density']:.2f}  reciprocity={metrics['reciprocity']:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="gray",
    )
    fig.tight_layout()
    return fig


def fig_alpha_stats(data: CommFormerAnalysisData, prefix: str) -> plt.Figure | None:
    """Bar chart of per-agent row (sender) and column (receiver) mean α values."""
    if data.alpha is None:
        return None

    stats = compute_alpha_stats(data)
    n = data.num_agents
    labels = _agent_labels(n)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(5, n + 2), 4))
    ax.bar(x - width / 2, stats["row_means"], width,
           label="Row mean (sender strength)", color="#55A868", alpha=0.85)
    ax.bar(x + width / 2, stats["col_means"], width,
           label="Col mean (receiver popularity)", color="#C44E52", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean α value")
    ax.set_title(f"α Row/Column Statistics\n({prefix})")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    return fig


def fig_encoder_pca(data: CommFormerAnalysisData, prefix: str) -> plt.Figure | None:
    """2-D PCA of encoder output embeddings, coloured by agent."""
    projected, labels, evr = compute_encoder_pca(data)
    if projected is None:
        return None

    n = data.num_agents
    agent_labels = _agent_labels(n)

    fig, ax = plt.subplots(figsize=(6, 5))
    for agent_idx in range(n):
        mask = labels == agent_idx
        if mask.sum() == 0:
            continue
        color = _AGENT_COLORS[agent_idx % len(_AGENT_COLORS)]
        ax.scatter(
            projected[mask, 0], projected[mask, 1],
            c=color, label=agent_labels[agent_idx], s=15, alpha=0.6,
        )
    ax.set_xlabel(f"PC1 ({evr[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%} var)")
    ax.set_title(f"Encoder Output PCA\n({prefix})")
    ax.legend(markerscale=2, fontsize=9)
    fig.tight_layout()
    return fig


def fig_encoder_tsne(data: CommFormerAnalysisData, prefix: str) -> plt.Figure | None:
    """2-D t-SNE of encoder output embeddings, coloured by agent."""
    projected, labels = compute_encoder_tsne(data)
    if projected is None:
        return None

    n = data.num_agents
    agent_labels = _agent_labels(n)

    fig, ax = plt.subplots(figsize=(6, 5))
    for agent_idx in range(n):
        mask = labels == agent_idx
        if mask.sum() == 0:
            continue
        color = _AGENT_COLORS[agent_idx % len(_AGENT_COLORS)]
        ax.scatter(
            projected[mask, 0], projected[mask, 1],
            c=color, label=agent_labels[agent_idx], s=15, alpha=0.6,
        )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title(f"Encoder Output t-SNE\n({prefix})")
    ax.legend(markerscale=2, fontsize=9)
    fig.tight_layout()
    return fig


def fig_encoder_norm(data: CommFormerAnalysisData, prefix: str) -> plt.Figure | None:
    """L2-norm of encoder outputs per agent over episode time."""
    norms = compute_encoder_norm_per_agent(data)
    if not any(len(v) > 0 for v in norms.values()):
        return None

    n = data.num_agents
    agent_labels = _agent_labels(n)

    fig, ax = plt.subplots(figsize=(8, 4))
    for agent_idx, norm_arr in norms.items():
        if len(norm_arr) == 0:
            continue
        color = _AGENT_COLORS[agent_idx % len(_AGENT_COLORS)]
        ax.plot(norm_arr, color=color, alpha=0.7, linewidth=1.2,
                label=agent_labels[agent_idx])

    ax.set_xlabel("Step (across all episodes)")
    ax.set_ylabel("L2-norm")
    ax.set_title(f"Encoder Output L2-Norm Over Time\n({prefix})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_reward_distribution(data: CommFormerAnalysisData, prefix: str) -> plt.Figure | None:
    """Histogram of episode-level mean rewards."""
    rewards = data.episode_rewards()
    if len(rewards) == 0:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rewards, bins=min(20, max(5, len(rewards) // 3)),
            color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(rewards.mean(), color="#C44E52", linestyle="--", linewidth=1.5,
               label=f"Mean: {rewards.mean():+.2f}")
    ax.set_xlabel("Mean Episode Reward")
    ax.set_ylabel("Count")
    ax.set_title(f"Episode Reward Distribution\n({prefix})")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_episode_lengths(data: CommFormerAnalysisData, prefix: str) -> plt.Figure | None:
    """Histogram of episode lengths."""
    lengths = data.episode_lengths()
    if len(lengths) == 0:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lengths, bins=min(20, max(5, len(lengths) // 3)),
            color="#55A868", alpha=0.8, edgecolor="white")
    ax.axvline(lengths.mean(), color="#C44E52", linestyle="--", linewidth=1.5,
               label=f"Mean: {lengths.mean():.1f}")
    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Count")
    ax.set_title(f"Episode Length Distribution\n({prefix})")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_reward_trajectory(data: CommFormerAnalysisData, prefix: str) -> plt.Figure | None:
    """Cumulative reward curves for individual episodes."""
    if not data.episodes:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    max_show = min(20, len(data.episodes))

    for ep in data.episodes[:max_show]:
        cum = np.cumsum([s.total_reward for s in ep.steps])
        color = "#55A868" if ep.success else "#C44E52"
        ax.plot(cum, color=color, alpha=0.4, linewidth=0.8)

    # Mean curve
    max_len = max(ep.length for ep in data.episodes[:max_show])
    mean_curve = np.zeros(max_len)
    cnt = np.zeros(max_len)
    for ep in data.episodes[:max_show]:
        cum = np.cumsum([s.total_reward for s in ep.steps])
        mean_curve[:len(cum)] += cum
        cnt[:len(cum)] += 1
    safe = np.where(cnt > 0, cnt, 1)
    mean_curve /= safe
    valid = cnt > 0
    ax.plot(np.where(valid)[0], mean_curve[valid], color="black",
            linewidth=2, label="Mean", zorder=5)

    success_patch = mpatches.Patch(color="#55A868", alpha=0.6, label="Success")
    failure_patch = mpatches.Patch(color="#C44E52", alpha=0.6, label="Failure")
    ax.legend(handles=[success_patch, failure_patch, plt.Line2D([], [], color="black", linewidth=2, label="Mean")],
              fontsize=9)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(f"Episode Reward Trajectories (first {max_show})\n({prefix})")
    fig.tight_layout()
    return fig


def fig_summary_table(data: CommFormerAnalysisData, prefix: str) -> plt.Figure:
    """Text figure summarising key metrics."""
    metrics = compute_graph_metrics(data)
    alpha_stats = compute_alpha_stats(data)

    rows = [
        ("Episodes", str(data.n_episodes)),
        ("Success rate", f"{data.success_rate:.1%}"),
        ("Mean length", f"{data.mean_episode_length:.1f}"),
        ("Mean reward", f"{data.mean_episode_reward:+.3f}"),
        ("Num agents", str(data.num_agents)),
        ("Sparsity S", str(data.sparsity)),
        ("Graph density", f"{metrics.get('density', 0):.3f}"),
        ("Reciprocity", f"{metrics.get('reciprocity', 0):.3f}"),
        ("Symmetric", str(metrics.get('is_symmetric', '?'))),
        ("α mean", f"{alpha_stats.get('mean', 0):.3f}" if alpha_stats else "N/A"),
        ("α std", f"{alpha_stats.get('std', 0):.3f}" if alpha_stats else "N/A"),
        ("Comm data", "Yes" if data.has_comm_data else "No"),
    ]

    fig, ax = plt.subplots(figsize=(5, 0.35 * len(rows) + 1.0))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    ax.set_title(f"CommFormer Analysis Summary\n({prefix})", pad=10)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────


def save_all_commformer_figures(
    data: CommFormerAnalysisData,
    output_dir: str = "eval_plots/commformer",
    prefix: str = "commformer",
) -> None:
    """Generate and save all CommFormer analysis figures.

    Parameters
    ----------
    data       : result from CommFormerCommCollector.collect().
    output_dir : directory to write figures (created if absent).
    prefix     : filename prefix (e.g. experiment name).
    """
    print_summary(data)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _try_save(gen_fn, name: str) -> None:
        try:
            fig = gen_fn()
            if fig is not None:
                _save(fig, out, f"{prefix}_{name}")
        except Exception as exc:
            print(f"  [CommFormerViz] Skipped {name}: {exc}")

    _try_save(lambda: fig_alpha_heatmap(data, prefix), "alpha_heatmap")
    _try_save(lambda: fig_adj_heatmap(data, prefix), "adj_heatmap")
    _try_save(lambda: fig_graph_structure(data, prefix), "graph_structure")
    _try_save(lambda: fig_degree_distribution(data, prefix), "degree_distribution")
    _try_save(lambda: fig_alpha_stats(data, prefix), "alpha_stats")
    _try_save(lambda: fig_encoder_pca(data, prefix), "encoder_pca")
    _try_save(lambda: fig_encoder_tsne(data, prefix), "encoder_tsne")
    _try_save(lambda: fig_encoder_norm(data, prefix), "encoder_norm")
    _try_save(lambda: fig_reward_distribution(data, prefix), "reward_distribution")
    _try_save(lambda: fig_episode_lengths(data, prefix), "episode_lengths")
    _try_save(lambda: fig_reward_trajectory(data, prefix), "reward_trajectory")
    _try_save(lambda: fig_summary_table(data, prefix), "summary_table")

    print(f"\n  Figures saved to: {out}/")
