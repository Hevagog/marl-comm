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

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .analysis import (
    CommFormerAnalysisData,
    compute_graph_metrics,
    compute_encoder_pca,
    compute_encoder_tsne,
    compute_encoder_norm_per_agent,
    compute_alpha_stats,
    print_summary,
    compute_information_flow,
    compute_laplacian_spectrum,
    compute_representation_alignment,
    compute_encoder_variance_dynamics,
    compute_alpha_concentration,
)

# ──────────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────────

from utils.shared.style import _AGENT_COLORS, save_figure

CMAP_ALPHA = "RdYlGn"
CMAP_ADJ = "Blues"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    save_figure(fig, out_dir, name)


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
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white"
                    if abs(mat[i, j]) > 0.6 * (vmax or mat.max() + 1e-9)
                    else "black",
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
        ax,
        alpha,
        f"Learned α — CommFormer Communication Graph\n({prefix})",
        labels,
        cmap=CMAP_ALPHA,
        vmin=-vabs,
        vmax=vabs,
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
        axes[0],
        adj,
        f"Hard k-hot Adjacency  (S={data.sparsity})\n({prefix})",
        labels,
        cmap=CMAP_ADJ,
        vmin=0.0,
        vmax=1.0,
    )

    # Right: α soft heatmap for comparison (if available)
    if data.alpha is not None:
        alpha = data.alpha
        vabs = max(abs(alpha.min()), abs(alpha.max()), 1e-9)
        _heatmap_ax(
            axes[1],
            alpha,
            "Soft α parameter",
            labels,
            cmap=CMAP_ALPHA,
            vmin=-vabs,
            vmax=vabs,
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
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax, font_color="white", font_size=11
    )

    if G.edges:
        weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
        # Normalise edge widths
        w_arr = np.asarray(weights)
        w_norm = 1.0 + 3.0 * (w_arr - w_arr.min()) / (w_arr.max() - w_arr.min() + 1e-9)
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            width=w_norm.tolist(),
            arrowsize=20,
            edge_color="steelblue",
            arrows=True,
            connectionstyle="arc3,rad=0.15",
        )

    # nx.draw_networkx_edges silently ignores self-loops with arrows=True.
    # Draw them explicitly as small dashed arcs offset radially from each node.
    for i in range(n):
        if adj[i, i] <= 0.5:
            continue
        xi, yi = pos[i]
        loop = mpatches.Arc(
            (xi * 1.35, yi * 1.35),
            width=0.24,
            height=0.24,
            color="#888888",
            alpha=0.7,
            lw=1.5,
            linestyle="--",
            zorder=3,
        )
        ax.add_patch(loop)
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.65, 1.65)

    ax.set_title(f"Learned Communication Graph\n({prefix})", pad=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_degree_distribution(
    data: CommFormerAnalysisData, prefix: str
) -> plt.Figure | None:
    """Bar chart of in-degree and out-degree per agent."""
    if data.hard_adj is None:
        return None

    metrics = compute_graph_metrics(data)
    n = data.num_agents
    labels = _agent_labels(n)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(5, n + 2), 4))
    bars_in = ax.bar(
        x - width / 2,
        metrics["in_degree"],
        width,
        label="In-degree (receives from)",
        color="#4C72B0",
        alpha=0.85,
    )
    bars_out = ax.bar(
        x + width / 2,
        metrics["out_degree"],
        width,
        label="Out-degree (sends to)",
        color="#DD8452",
        alpha=0.85,
    )
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
                bar.get_x() + bar.get_width() / 2,
                h + 0.05,
                str(int(h)),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Add density and reciprocity as text
    ax.text(
        0.98,
        0.97,
        f"density={metrics['density']:.2f}  reciprocity={metrics['reciprocity']:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="gray",
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
    ax.bar(
        x - width / 2,
        stats["row_means"],
        width,
        label="Row mean (sender strength)",
        color="#55A868",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        stats["col_means"],
        width,
        label="Col mean (receiver popularity)",
        color="#C44E52",
        alpha=0.85,
    )
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
            projected[mask, 0],
            projected[mask, 1],
            c=color,
            label=agent_labels[agent_idx],
            s=15,
            alpha=0.6,
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
            projected[mask, 0],
            projected[mask, 1],
            c=color,
            label=agent_labels[agent_idx],
            s=15,
            alpha=0.6,
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
        ax.plot(
            norm_arr,
            color=color,
            alpha=0.7,
            linewidth=1.2,
            label=agent_labels[agent_idx],
        )

    ax.set_xlabel("Step (across all episodes)")
    ax.set_ylabel("L2-norm")
    ax.set_title(f"Encoder Output L2-Norm Over Time\n({prefix})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_reward_distribution(
    data: CommFormerAnalysisData, prefix: str
) -> plt.Figure | None:
    """Histogram of episode-level mean rewards."""
    rewards = data.episode_rewards()
    if len(rewards) == 0:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        rewards,
        bins=min(20, max(5, len(rewards) // 3)),
        color="#4C72B0",
        alpha=0.8,
        edgecolor="white",
    )
    ax.axvline(
        rewards.mean(),
        color="#C44E52",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {rewards.mean():+.2f}",
    )
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
    ax.hist(
        lengths,
        bins=min(20, max(5, len(lengths) // 3)),
        color="#55A868",
        alpha=0.8,
        edgecolor="white",
    )
    ax.axvline(
        lengths.mean(),
        color="#C44E52",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {lengths.mean():.1f}",
    )
    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Count")
    ax.set_title(f"Episode Length Distribution\n({prefix})")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_reward_trajectory(
    data: CommFormerAnalysisData, prefix: str
) -> plt.Figure | None:
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
        mean_curve[: len(cum)] += cum
        cnt[: len(cum)] += 1
    safe = np.where(cnt > 0, cnt, 1)
    mean_curve /= safe
    valid = cnt > 0
    ax.plot(
        np.where(valid)[0],
        mean_curve[valid],
        color="black",
        linewidth=2,
        label="Mean",
        zorder=5,
    )

    success_patch = mpatches.Patch(color="#55A868", alpha=0.6, label="Success")
    failure_patch = mpatches.Patch(color="#C44E52", alpha=0.6, label="Failure")
    ax.legend(
        handles=[
            success_patch,
            failure_patch,
            plt.Line2D([], [], color="black", linewidth=2, label="Mean"),
        ],
        fontsize=9,
    )
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
        ("Symmetric", str(metrics.get("is_symmetric", "?"))),
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

    # Extended analysis figures
    extended_fns = [
        (lambda: fig_information_flow(data, prefix), "information_flow"),
        (lambda: fig_laplacian_spectrum(data, prefix), "laplacian_spectrum"),
        (lambda: fig_representation_alignment(data, prefix), "rep_alignment"),
        (lambda: fig_encoder_variance(data, prefix), "encoder_variance"),
        (lambda: fig_alpha_concentration(data, prefix), "alpha_concentration"),
    ]
    for gen_fn, name in extended_fns:
        _try_save(gen_fn, name)

    print(f"  Extended figures saved to: {out}/")


# ──────────────────────────────────────────────────────────────────────────────
# Extended figures (graph-theoretic analysis)
# ──────────────────────────────────────────────────────────────────────────────


def fig_information_flow(data: CommFormerAnalysisData, prefix: str):
    """Bar chart of random-walk stationary distribution (PageRank) per agent.

    The stationary distribution reveals which agents are the information hubs
    in the learned static communication graph.  High-stationary agents receive
    and pass on more messages in expectation.
    """
    if data.hard_adj is None:
        return None

    flow = compute_information_flow(data)
    n = data.num_agents
    labels = _agent_labels(n)
    pi = flow["stationary"]
    hub = flow["hub_agent"]
    mixing = flow["mixing_steps"]

    fig, axes = plt.subplots(1, 2, figsize=(max(8, n + 3), 4.5))
    fig.suptitle(
        f"Information Flow Analysis — Random Walk on Learned Graph\n"
        f"Mixing time: {mixing} steps  |  Hub agent: {labels[hub]}",
        fontsize=11,
    )

    # Stationary distribution bar chart
    colors = [_AGENT_COLORS[i % len(_AGENT_COLORS)] for i in range(n)]
    axes[0].bar(range(n), pi, color=colors, alpha=0.85, width=0.6)
    axes[0].axhline(
        1.0 / n, color="gray", linewidth=1.2, linestyle="--", label=f"Uniform (1/{n})"
    )
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Stationary probability")
    axes[0].set_title("Stationary distribution π\n(higher = more central to info flow)")
    axes[0].legend(fontsize=9)

    # Transition matrix heatmap
    P = flow["row_stochastic"]
    im = axes[1].imshow(P, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_yticks(range(n))
    axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].set_xlabel("To")
    axes[1].set_ylabel("From")
    axes[1].set_title(
        "Row-stochastic transition matrix P\n(P[i,j] = prob. info flows i→j)"
    )
    for i in range(n):
        for j in range(n):
            axes[1].text(
                j,
                i,
                f"{P[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if P[i, j] > 0.5 else "black",
            )
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


def fig_laplacian_spectrum(data: CommFormerAnalysisData, prefix: str):
    """Eigenvalue spectrum of the graph Laplacians.

    The normalised Laplacian eigenvalues characterise graph connectivity:
    - λ₀ = 0 always (constant eigenvector)
    - λ₁ (Fiedler) = 0 iff graph is disconnected; larger = better connected
    - Spectral gap λ₂ − λ₁ governs how quickly random walks mix
    """
    if data.hard_adj is None:
        return None

    spec = compute_laplacian_spectrum(data)
    n = data.num_agents
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.suptitle(
        f"Graph Laplacian Spectrum\n"
        f"Fiedler value = {spec['fiedler_value']:.4f}  |  "
        f"Spectral gap = {spec['spectral_gap']:.4f}",
        fontsize=11,
    )

    x = range(n)
    axes[0].stem(
        x, spec["unnorm_eigenvalues"], markerfmt="C0o", linefmt="C0-", basefmt="k-"
    )
    axes[0].set_xticks(x)
    axes[0].set_xlabel("Eigenvalue index")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title("Unnormalised Laplacian L = D − A")

    axes[1].stem(
        x, spec["norm_eigenvalues"], markerfmt="C1o", linefmt="C1-", basefmt="k-"
    )
    axes[1].set_xticks(x)
    axes[1].set_xlabel("Eigenvalue index")
    axes[1].set_ylabel("Eigenvalue")
    axes[1].set_title(
        "Normalised Laplacian L_norm\n"
        f"(λ₁={spec['fiedler_value']:.3f} — higher = better connected)"
    )
    if n > 1:
        axes[1].axhline(
            spec["fiedler_value"],
            color=_AGENT_COLORS[2],
            linewidth=1.2,
            linestyle="--",
            label=f"λ₁ (Fiedler) = {spec['fiedler_value']:.3f}",
        )
        axes[1].legend(fontsize=9)

    fig.tight_layout()
    return fig


def fig_representation_alignment(data: CommFormerAnalysisData, prefix: str):
    """Cosine similarity for edge vs non-edge agent pairs.

    If communication is effective, connected agents should develop more similar
    encoder representations over the episode (positive alignment advantage).
    """
    if data.hard_adj is None or not data.has_comm_data:
        return None

    align = compute_representation_alignment(data)
    es = align["edge_similarities"]
    ns_arr = align["non_edge_similarities"]

    if len(es) == 0 and len(ns_arr) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        f"Representation Alignment: Edge vs Non-Edge Agent Pairs\n"
        f"edge mean={align['edge_mean']:.3f}  "
        f"non-edge mean={align['non_edge_mean']:.3f}  "
        f"advantage={align['alignment_advantage']:+.3f}  "
        f"({'comm helps' if align['alignment_advantage'] > 0 else 'no alignment effect'})",
        fontsize=11,
    )

    # Histogram
    lo = min(
        float(np.percentile(es, 1)) if len(es) > 0 else -1.0,
        float(np.percentile(ns_arr, 1)) if len(ns_arr) > 0 else -1.0,
    )
    hi = (
        max(
            float(np.percentile(es, 99)) if len(es) > 0 else 1.0,
            float(np.percentile(ns_arr, 99)) if len(ns_arr) > 0 else 1.0,
        )
        + 1e-6
    )
    bins = np.linspace(lo, hi, 35)

    if len(es) > 0:
        axes[0].hist(
            es,
            bins=bins,
            color=_AGENT_COLORS[0],
            alpha=0.65,
            label=f"Edge pairs (N={len(es)})",
            density=True,
        )
    if len(ns_arr) > 0:
        axes[0].hist(
            ns_arr,
            bins=bins,
            color=_AGENT_COLORS[1],
            alpha=0.65,
            label=f"Non-edge pairs (N={len(ns_arr)})",
            density=True,
        )
    axes[0].axvline(
        align["edge_mean"], color=_AGENT_COLORS[0], linewidth=2, linestyle="--"
    )
    axes[0].axvline(
        align["non_edge_mean"], color=_AGENT_COLORS[1], linewidth=2, linestyle="--"
    )
    axes[0].set_xlabel("Cosine similarity")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution of pairwise cosine similarities")
    axes[0].legend(fontsize=9)

    # Violin plot
    plot_data = []
    plot_labels = []
    if len(es) > 0:
        plot_data.append(es)
        plot_labels.append("Edge")
    if len(ns_arr) > 0:
        plot_data.append(ns_arr)
        plot_labels.append("Non-edge")

    if plot_data:
        vlns = axes[1].violinplot(
            plot_data, positions=range(len(plot_data)), showmedians=True
        )
        for i, body in enumerate(vlns["bodies"]):
            body.set_facecolor(_AGENT_COLORS[i])
            body.set_alpha(0.65)
        axes[1].set_xticks(range(len(plot_labels)))
        axes[1].set_xticklabels(plot_labels)
        axes[1].set_ylabel("Cosine similarity")
        axes[1].set_title("Edge vs non-edge similarity distributions")

    fig.tight_layout()
    return fig


def fig_encoder_variance(data: CommFormerAnalysisData, prefix: str):
    """Per-agent encoder representation variance over episode time.

    Increasing variance across the episode indicates agents are developing
    more diverse representations (state-responsive).  Decreasing variance
    may indicate representation collapse or very stable policies.
    """
    if not data.has_comm_data:
        return None

    var_dynamics = compute_encoder_variance_dynamics(data)
    if not var_dynamics:
        return None

    n = data.num_agents
    labels = _agent_labels(n)
    fig, ax = plt.subplots(figsize=(10, 4.2))

    for agent_idx, variances in var_dynamics.items():
        if len(variances) == 0:
            continue
        color = _AGENT_COLORS[agent_idx % len(_AGENT_COLORS)]
        label = labels[agent_idx] if agent_idx < len(labels) else f"A{agent_idx}"
        smoothed = np.convolve(variances, np.ones(10) / 10, mode="same")
        ax.plot(variances, color=color, alpha=0.2, linewidth=0.6)
        ax.plot(smoothed, color=color, alpha=0.9, linewidth=1.8, label=label)

    ax.set_xlabel("Step (across all episodes)")
    ax.set_ylabel("Feature variance (within-step)")
    ax.set_title(
        "Encoder Representation Variance Over Episode Time\n"
        "(increasing = more dynamic/role-differentiated; decreasing = stable/collapsed)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_alpha_concentration(data: CommFormerAnalysisData, prefix: str):
    """α selectivity: how concentrated are the learned communication weights?

    Low row entropy = agent is selective about who it listens to.
    High row entropy = agent attends nearly uniformly to all others.
    High Gini coefficient = few partners dominate the communication weight.
    """
    if data.alpha is None:
        return None

    conc = compute_alpha_concentration(data)
    n = data.num_agents
    labels = _agent_labels(n)
    x = np.arange(n)
    width = 0.28

    fig, axes = plt.subplots(1, 3, figsize=(max(12, 3 * n + 2), 4.5))
    fig.suptitle(
        "α Parameter Concentration — How Selective is the Communication?\n"
        "(lower entropy / higher Gini = more selective, fewer partners)",
        fontsize=11,
    )

    # Row entropies (sender selectivity)
    axes[0].bar(
        x,
        conc["row_entropies"],
        color=_AGENT_COLORS[0],
        alpha=0.8,
        width=0.55,
        label="Row entropy",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Entropy (nats)")
    axes[0].set_title("Row entropy — Sender selectivity\n(low = sends to few)")
    max_ent = float(np.log(n)) if n > 1 else 1.0
    axes[0].axhline(
        max_ent,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label=f"Max entropy (uniform) = {max_ent:.2f}",
    )
    axes[0].legend(fontsize=8)

    # Col entropies (receiver selectivity)
    axes[1].bar(x, conc["col_entropies"], color=_AGENT_COLORS[1], alpha=0.8, width=0.55)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Entropy (nats)")
    axes[1].set_title("Col entropy — Receiver popularity\n(low = listened to by few)")
    axes[1].axhline(max_ent, color="gray", linestyle="--", linewidth=0.8)

    # Gini coefficient
    axes[2].bar(x, conc["gini_row"], color=_AGENT_COLORS[2], alpha=0.8, width=0.55)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("Gini coefficient")
    axes[2].set_ylim(0, 1.0)
    axes[2].set_title(
        "Gini of α rows — Concentration\n(higher = weights concentrated on few partners)"
    )
    axes[2].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    return fig
