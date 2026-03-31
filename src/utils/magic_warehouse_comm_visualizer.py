"""
magic_warehouse_comm_visualizer.py
====================================
Publication-quality figures for MAGIC's communication mechanism in the
multi-robot warehouse (4 agents, 2 comm rounds, 128-dim messages).

Figures produced
----------------
 1.  fig_adj_heatmaps        – Mean soft 4×4 adjacency matrices (both rounds)
 2.  fig_network_diagram     – Directed graph with arrow-weight visualisation
 3.  fig_pairwise_barplot    – Per-pair directed weights comparison (R1 vs R2)
 4.  fig_phase_conditional   – Adj heatmaps conditioned on resource phase (4 panels)
 5.  fig_battery_conditional – Adj: battery-safe vs battery-emergency
 6.  fig_comm_trajectory     – Comm density over episode time
 7.  fig_context_correlations– Comm density vs battery / pending_tasks / phase
 8.  fig_hub_scores          – In-degree (who gets most messages) over time
 9.  fig_spatial_comm        – Comm density projected onto the 12×16 grid
10.  fig_round_divergence    – Round-1 vs Round-2 adj divergence over time
11.  fig_message_pca_phase   – PCA of messages coloured by resource phase
12.  fig_message_pca_role    – PCA of messages coloured by agent role (speed)
13.  fig_message_norms       – Message L2 norms per agent role over episode time
14.  fig_rescue_aligned      – Comm graph aligned around rescue events
15.  fig_interference_effect – Comm density vs distance to treatment stations
16.  fig_summary_dashboard   – One-page overview combining key findings
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe
import numpy as np

matplotlib.use("Agg")

from .magic_warehouse_comm_analysis import (
    MAGICWarehouseData,
    compute_mean_adj,
    compute_conditional_adj,
    compute_comm_vs_context,
    compute_pairwise_weights,
    compute_hub_scores,
    compute_message_pca,
    compute_message_pca_with_context,
    compute_round_divergence,
    compute_message_norm_over_time,
    align_comm_around_rescue,
    print_summary,
)

# ─── Style constants ──────────────────────────────────────────────────────────

_AGENT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]  # A0 A1 A2 A3


def _agent_labels(N: int) -> list:
    """Generate agent labels A0, A1, … for any N."""
    return [f"A{i}" for i in range(N)]


def _agent_colors(N: int) -> list:
    """Return N distinct agent colors, cycling through tab10 for N > 4."""
    if N <= len(_AGENT_COLORS):
        return _AGENT_COLORS[:N]
    cmap = matplotlib.colormaps.get_cmap("tab10")
    return [mcolors.to_hex(cmap(i % 10)) for i in range(N)]
_PHASE_COLORS = ["#95B9D4", "#F4A261", "#E76F51", "#2A9D8F"]
_PHASE_NAMES = {0: "Searching", 1: "To Treatment", 2: "Treating", 3: "To Goal"}
CMAP_ADJ = "Blues"
CMAP_HEAT = "YlOrRd"
_BG = "#F8F9FA"

plt.rcParams.update(
    {
        "figure.facecolor": _BG,
        "axes.facecolor": "#FFFFFF",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)


def _save(fig: plt.Figure, out_dir: Path, name: str, dpi: int = 150) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(str(out_dir / f"{name}.{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_dir / name}.png/pdf")


def _adj_heatmap(
    ax: plt.Axes, mat: np.ndarray, title: str, N: int = 0, vmax: float = 1.0
) -> None:
    # Derive N from the matrix; the N parameter is kept for call-site compatibility.
    N = mat.shape[0]
    labels = _agent_labels(N)
    # Robust vmax: use the actual matrix maximum so the colour scale is informative.
    actual_max = float(mat.max()) if mat.size > 0 else 0.0
    vmax = max(actual_max, 1e-6)
    im = ax.imshow(mat, cmap=CMAP_ADJ, vmin=0, vmax=vmax, aspect="equal")
    ax.set_xticks(range(N))
    ax.set_xticklabels(labels, rotation=45 if N > 6 else 0, ha="right" if N > 6 else "center")
    ax.set_yticks(range(N))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Receiver")
    ax.set_ylabel("Sender")
    ax.set_title(title)
    fontsize = max(5, 8 - max(0, N - 4))
    for i in range(N):
        for j in range(N):
            val = mat[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center", fontsize=fontsize,
                color="white" if val > 0.55 * vmax else "black",
            )
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes("right", size="7%", pad=0.04)
    plt.colorbar(im, cax=cax)


# ─── Figure 1: Mean adjacency heatmaps ───────────────────────────────────────


def fig_adj_heatmaps(data: MAGICWarehouseData, out_dir: Path, prefix: str) -> None:
    R = data.num_comm_rounds
    fig, axes = plt.subplots(1, R, figsize=(5.5 * R, 5.5), squeeze=False)
    fig.suptitle(
        "MAGIC Scheduler — Mean Soft Adjacency Matrices\n"
        "(rows=sender, cols=receiver; darker=more frequent comm)",
        fontsize=12, y=1.03,
    )
    for r in range(R):
        adj = compute_mean_adj(data, r)
        _adj_heatmap(axes[0, r], adj, f"Round {r + 1}")
    fig.tight_layout()
    _save(fig, out_dir, f"{prefix}_01_adj_heatmaps")


# ─── Figure 2: Directed network diagram ──────────────────────────────────────


def fig_network_diagram(data: MAGICWarehouseData, out_dir: Path, prefix: str) -> None:
    """
    Visualise the mean communication graph as a directed network.
    Arrow thickness ∝ edge weight. Node colour = agent speed role.
    Both rounds shown side by side.
    """
    R = data.num_comm_rounds
    fig, axes = plt.subplots(1, R, figsize=(7 * R, 6.5), squeeze=False)
    fig.suptitle(
        "MAGIC Communication Network — Mean Directed Graph\n"
        "(arrow thickness = communication strength; colour = speed role)",
        fontsize=12, y=1.03,
    )

    roles = data.agent_roles()

    for r in range(R):
        ax = axes[0, r]
        ax.set_aspect("equal")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.axis("off")
        ax.set_title(f"Round {r + 1}")

        adj = compute_mean_adj(data, r)
        N = adj.shape[0]
        agent_colors = _agent_colors(N)
        labels = _agent_labels(N)

        # Node positions: circular layout
        angles = [np.pi / 2 + i * 2 * np.pi / N for i in range(N)]
        node_x = np.cos(angles)
        node_y = np.sin(angles)

        max_weight = adj[~np.eye(N, dtype=bool)].max() if N > 1 else 1.0
        max_weight = max(max_weight, 1e-6)

        # Draw edges
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                w = adj[i, j]
                if w < 0.05:
                    continue
                norm_w = w / max_weight
                ax.annotate(
                    "",
                    xy=(node_x[j] * 0.82, node_y[j] * 0.82),
                    xytext=(node_x[i] * 0.82, node_y[i] * 0.82),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=agent_colors[i],
                        alpha=0.5 + 0.5 * norm_w,
                        lw=0.5 + 4.0 * norm_w,
                        mutation_scale=12 + 8 * norm_w,
                        connectionstyle="arc3,rad=0.12",
                    ),
                )
                # Weight label near midpoint
                mx = 0.5 * (node_x[i] + node_x[j]) * 0.65
                my = 0.5 * (node_y[i] + node_y[j]) * 0.65
                ax.text(mx, my, f"{w:.2f}", ha="center", va="center", fontsize=7, color="gray")

        # Draw nodes
        for i in range(N):
            role = roles.get(f"agent_{i}", {})
            node_color = "#E07B39" if role.get("is_fast", False) else agent_colors[i]
            circle = plt.Circle(
                (node_x[i], node_y[i]), 0.18, color=node_color, ec="black", lw=1.5, zorder=5
            )
            ax.add_patch(circle)
            sp = role.get("speed", 1.0)
            cap = role.get("capacity", 1)
            ax.text(
                node_x[i], node_y[i],
                f"{labels[i]}\nsp={int(sp)}\ncap={cap}",
                ha="center", va="center", fontsize=7, fontweight="bold",
                color="white", zorder=6,
            )

        # Legend
        fast_patch = mpatches.Patch(color="#E07B39", label="Fast (speed=2)")
        slow_patch = mpatches.Patch(color=_AGENT_COLORS[0], label="Slow (speed=1)")
        ax.legend(handles=[fast_patch, slow_patch], loc="lower right", fontsize=8)

    fig.tight_layout()
    _save(fig, out_dir, f"{prefix}_02_network_diagram")


# ─── Figure 3: Per-pair directional weights barplot ───────────────────────────


def fig_pairwise_barplot(data: MAGICWarehouseData, out_dir: Path, prefix: str) -> None:
    R = data.num_comm_rounds
    N = data.actual_num_agents
    agent_labels = _agent_labels(N)
    agent_colors = _agent_colors(N)
    pairs = [f"{agent_labels[i]}→{agent_labels[j]}" for i in range(N) for j in range(N) if i != j]

    fig, axes = plt.subplots(1, R, figsize=(max(10, 1.5 * len(pairs)), 5), sharey=True)
    if R == 1:
        axes = [axes]
    fig.suptitle(
        "Per-Pair Directed Communication Weights by Round\n"
        "(each bar = mean soft adjacency weight for that directed edge)",
        fontsize=12,
    )

    for r, ax in enumerate(axes):
        pw = compute_pairwise_weights(data, r)
        vals = [pw.get(p, 0.0) for p in pairs]
        bar_colors = [agent_colors[int(p.split("→")[0][1:])] for p in pairs]
        ax.bar(range(len(pairs)), vals, color=bar_colors, alpha=0.8)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(pairs, rotation=45, ha="right")
        ax.set_title(f"Round {r + 1}")
        ax.set_ylabel("Mean soft adjacency weight")
        ax.set_ylim(0, 1.0)
        ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.6, label="0.5 threshold")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, out_dir, f"{prefix}_03_pairwise_barplot")


# ─── Figure 4: Phase-conditioned adjacency ────────────────────────────────────


def fig_phase_conditional(data: MAGICWarehouseData, out_dir: Path, prefix: str) -> None:
    N = data.actual_num_agents
    phase_conditions = [
        ("Phase 0: Searching\n(all agents)", lambda s: True, "#95B9D4"),
        ("Phase 0 dominant\n(≥3 agents searching)", lambda s: s.searching_count >= 3, "#4C72B0"),
        ("Phase 2: Any treating\n(at treatment station)", lambda s: s.treating_count >= 1, "#E76F51"),
        ("Battery emergency\n(any agent <30)", lambda s: s.battery_emergency, "#C44E52"),
        ("Rescue active\n(any agent dragging)", lambda s: s.any_rescuing, "#8172B2"),
        ("High task pressure\n(pending_tasks > 8)", lambda s: s.pending_tasks > 8, "#DD8452"),
    ]

    ncols = 3
    nrows = (len(phase_conditions) + ncols - 1) // ncols
    R = data.num_comm_rounds

    for r in range(R):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows))
        axes = np.array(axes).ravel()
        fig.suptitle(
            f"Communication Graph under Warehouse Scenarios — Round {r + 1}\n"
            "(Mean soft adjacency conditioned on environment state)",
            fontsize=12, y=1.01,
        )
        for ax, (title, cond_fn, color) in zip(axes, phase_conditions):
            adj = compute_conditional_adj(data, cond_fn, round_idx=r)
            n_steps = sum(1 for s in data.comm_steps() if cond_fn(s))
            _adj_heatmap(ax, adj, f"{title}\n(n={n_steps} steps)", N)

        for ax in axes[len(phase_conditions):]:
            ax.axis("off")

        fig.tight_layout()
        _save(fig, out_dir, f"{prefix}_04_phase_conditional_round{r + 1}")


# ─── Figure 5: Battery-conditioned adjacency ─────────────────────────────────


def fig_battery_conditional(data: MAGICWarehouseData, out_dir: Path, prefix: str) -> None:
    N = data.actual_num_agents
    R = data.num_comm_rounds

    conditions = [
        ("All batteries safe\n(all > 50)", lambda s: all(v > 50 for v in s.battery.values())),
        ("Low battery present\n(any < 50)", lambda s: any(v < 50 for v in s.battery.values())),
        ("Critical battery\n(any < 30)", lambda s: s.battery_emergency),
        ("Agent charging\n(any charging=True)", lambda s: True),  # placeholder
    ]

    fig, axes = plt.subplots(R, 4, figsize=(5.5 * 4, 5.5 * R))
    if R == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        "Communication Graph Conditioned on Battery State\n"
        "(rows=comm round, cols=battery condition)",
        fontsize=12, y=1.01,
    )

    for r in range(R):
        for col, (title, cond_fn) in enumerate(conditions):
            adj = compute_conditional_adj(data, cond_fn, round_idx=r)
            n_steps = sum(1 for s in data.comm_steps() if cond_fn(s))
            _adj_heatmap(axes[r, col], adj, f"{title}\n(n={n_steps})", N)

    fig.tight_layout()
    _save(fig, out_dir, f"{prefix}_05_battery_conditional")


# ─── Figure 6: Communication density trajectory ───────────────────────────────


def fig_comm_trajectory(
    data: MAGICWarehouseData, out_dir: Path, prefix: str, max_episodes: int = 20
) -> None:
    episodes = [ep for ep in data.episodes if ep.has_comm_data][:max_episodes]
    if not episodes:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"hspace": 0.45})
    fig.suptitle("MAGIC Communication Density — Warehouse Episodes", fontsize=13, fontweight="bold")

    # Top: individual episode curves coloured by delivery count
    ax = axes[0]
    max_d = max(ep.total_deliveries for ep in episodes) or 1
    cmap = matplotlib.colormaps.get_cmap("RdYlGn")
    for ep in episodes:
        steps = [s.step for s in ep.steps if s.adj_per_round is not None]
        dens = [s.comm_density for s in ep.steps if s.adj_per_round is not None]
        if steps:
            color = cmap(ep.total_deliveries / max_d)
            ax.plot(steps, dens, alpha=0.4, color=color, linewidth=0.9)
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=max_d))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Deliveries/episode", fraction=0.03, pad=0.02)
    ax.set_xlabel("Step")
    ax.set_ylabel("Comm density (off-diagonal)")
    ax.set_title("Per-Episode Comm Density (colour = delivery count)")
    ax.set_ylim(0, 1.05)

    # Bottom: mean ± std with rescue event markers
    ax = axes[1]
    max_len = data.episodes[0].length if data.episodes else 500
    mat = np.full((len(episodes), max_len), np.nan)
    for i, ep in enumerate(episodes):
        for s in ep.steps:
            if s.adj_per_round is not None and s.step < max_len:
                mat[i, s.step] = s.comm_density
    t = np.arange(max_len)
    mean_d = np.nanmean(mat, axis=0)
    std_d = np.nanstd(mat, axis=0)
    valid = ~np.isnan(mean_d)
    ax.plot(t[valid], mean_d[valid], color="#4C72B0", linewidth=2, label="Mean")
    ax.fill_between(t[valid], (mean_d - std_d)[valid], (mean_d + std_d)[valid],
                    alpha=0.25, color="#4C72B0", label="±1 SD")
    ax.set_xlabel("Step")
    ax.set_ylabel("Comm density")
    ax.set_title("Mean Communication Density (± 1 SD)")
    ax.set_ylim(0, 1.05)
    ax.legend()

    _save(fig, out_dir, f"{prefix}_06_comm_trajectory")


# ─── Figure 7: Context correlations ─────────────────────────────────────────


def fig_context_correlations(data: MAGICWarehouseData, out_dir: Path, prefix: str) -> None:
    context_vars = [
        ("min_battery", "Min battery level", "Battery level"),
        ("pending_tasks", "Pending task queue", "Pending tasks"),
        ("treating_count", "Agents at treatment station", "Treating agents"),
        ("searching_count", "Agents searching (phase=0)", "Searching agents"),
        ("step_fraction", "Episode time fraction", "Step fraction (t/max)"),
    ]

    ncols = 3
    nrows = (len(context_vars) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), constrained_layout=True)
    fig.suptitle(
        "Communication Density vs Warehouse Context Variables\n"
        "(Does MAGIC adapt its topology based on task state?)",
        fontsize=12,
    )
    axes_flat = np.array(axes).ravel()

    for ax, (key, title, xlabel) in zip(axes_flat, context_vars):
        centres, means, stds = compute_comm_vs_context(data, key)
        valid = means > 0
        if not valid.any():
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            continue
        width = (centres[1] - centres[0]) if len(centres) > 1 else 0.5
        ax.bar(centres[valid], means[valid], width=width * 0.8, color="#4C72B0", alpha=0.75)
        ax.errorbar(centres[valid], means[valid], yerr=stds[valid],
                    fmt="none", ecolor="black", capsize=3, linewidth=0.9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean comm density")
        ax.set_title(title)
        ax.set_ylim(0, 1.0)

    for ax in axes_flat[len(context_vars):]:
        ax.axis("off")

    _save(fig, out_dir, f"{prefix}_07_context_correlations")


# ─── Figure 8: Hub scores over time ──────────────────────────────────────────


def fig_hub_scores(data: MAGICWarehouseData, out_dir: Path, prefix: str) -> None:
    scores = compute_hub_scores(data, round_idx=data.num_comm_rounds - 1)
    if scores.size == 0:
        return
    N = scores.shape[1]
    agent_colors = _agent_colors(N)
    labels = _agent_labels(N)

    # Smooth with rolling mean
    def _smooth(x: np.ndarray, w: int = 20) -> np.ndarray:
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)
    fig.suptitle(
        "MAGIC Hub Scores — Who Receives the Most Messages?\n"
        "(in-degree = number of other agents sending to this agent)",
        fontsize=12, fontweight="bold",
    )

    # Top: per-agent in-degree over time (smoothed)
    ax = axes[0]
    roles = data.agent_roles()
    for i in range(N):
        agent = f"agent_{i}"
        y = scores[:, i]
        ys = _smooth(y)
        xs = np.arange(len(ys))
        role = roles.get(agent, {})
        label = f"{labels[i]} (sp={int(role.get('speed',1))}, cap={role.get('capacity',1)})"
        ax.plot(xs, ys, color=agent_colors[i], linewidth=2, label=label, alpha=0.85)
    ax.set_xlabel("Step (across all episodes)")
    ax.set_ylabel("In-degree (messages received)")
    ax.set_title("Per-Agent In-Degree (Smoothed, Window=20)")
    ax.set_ylim(-0.1, N)
    ax.legend()

    # Bottom: stacked bar (mean in-degree per agent)
    ax = axes[1]
    mean_scores = scores.mean(axis=0)
    bar_labels = []
    for i in range(N):
        role = roles.get(f"agent_{i}", {})
        bar_labels.append(f"{labels[i]}\n(sp={int(role.get('speed',1))}, cap={role.get('capacity',1)})")
    ax.bar(bar_labels, mean_scores, color=agent_colors, alpha=0.85)
    ax.set_ylabel("Mean in-degree")
    ax.set_title("Mean In-Degree per Agent\n(higher = more often the target of communication)")
    for i, v in enumerate(mean_scores):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    _save(fig, out_dir, f"{prefix}_08_hub_scores")


# ─── Figure 9: Spatial communication density heatmap ────────────────────────


def fig_spatial_comm(data: MAGICWarehouseData, out_dir: Path, prefix: str) -> None:
    H, W, N = data.grid_height, data.grid_width, data.actual_num_agents
    labels = _agent_labels(N)

    # Build per-agent sender spatial heatmap: position of sending agent vs comm density
    spatial = np.zeros((N, H, W), dtype=np.float32)
    counts = np.zeros((N, H, W), dtype=np.float32)

    for s in data.comm_steps():
        for i, agent in enumerate([f"agent_{k}" for k in range(N)]):
            pos = s.positions.get(agent)
            if pos is None:
                continue
            r, c = pos
            if 0 <= r < H and 0 <= c < W:
                spatial[i, r, c] += s.comm_density
                counts[i, r, c] += 1

    with np.errstate(invalid="ignore"):
        spatial_mean = np.where(counts > 0, spatial / counts, np.nan)

    # Global mean across agents
    global_mean = np.nanmean(spatial_mean, axis=0)

    roles = data.agent_roles()
    fig, axes = plt.subplots(2, N // 2 + 1, figsize=(6 * (N // 2 + 1), 9), constrained_layout=True)
    fig.suptitle(
        "Spatial Communication Density — Where Does the Agent Communicate Most?\n"
        "(mean comm density when agent is at each grid cell)",
        fontsize=12, fontweight="bold",
    )
    axes_flat = np.array(axes).ravel()

    # Per-agent heatmaps
    for i in range(N):
        ax = axes_flat[i]
        im = ax.imshow(spatial_mean[i], origin="upper", cmap=CMAP_HEAT,
                       aspect="auto", vmin=0, vmax=1.0)
        role = roles.get(f"agent_{i}", {})
        ax.set_title(
            f"{labels[i]} (sp={int(role.get('speed',1))}, cap={role.get('capacity',1)})"
        )
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Comm density")

    # Global heatmap
    ax = axes_flat[N]
    im = ax.imshow(global_mean, origin="upper", cmap=CMAP_HEAT, aspect="auto", vmin=0, vmax=1.0)
    ax.set_title("Global Mean Comm Density")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Comm density")

    for ax in axes_flat[N + 1:]:
        ax.axis("off")

    _save(fig, out_dir, f"{prefix}_09_spatial_comm")


# ─── Figure 10: Round divergence ─────────────────────────────────────────────


def fig_round_divergence(data: MAGICWarehouseData, out_dir: Path, prefix: str) -> None:
    if data.num_comm_rounds < 2:
        return

    diffs = compute_round_divergence(data)
    if diffs is None or len(diffs) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(
        "Round-1 vs Round-2 Topology Divergence\n"
        "High divergence = Round 2 meaningfully refines Round 1's graph",
        fontsize=12, fontweight="bold",
    )

    # Left: divergence over time
    ax = axes[0]
    ax.plot(diffs, color="#8172B2", linewidth=1.2, alpha=0.6)
    # Smooth
    if len(diffs) > 30:
        w = 30
        smooth = np.convolve(diffs, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(w // 2, w // 2 + len(smooth)), smooth, color="#8172B2",
                linewidth=2.5, label="Smoothed (w=30)")
    ax.set_xlabel("Step (across all episodes)")
    ax.set_ylabel("‖G² − G¹‖_F  (Frobenius norm)")
    ax.set_title("Frobenius Norm of (Round-2 − Round-1) Adjacency")
    ax.legend()

    # Right: per-round heatmaps side by side for comparison
    ax = axes[1]
    adj_r1 = compute_mean_adj(data, 0)
    adj_r2 = compute_mean_adj(data, 1)
    N = adj_r1.shape[0]
    labels = _agent_labels(N)
    delta = adj_r2 - adj_r1  # positive = stronger in R2, negative = weaker

    im = ax.imshow(delta, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="equal")
    ax.set_xticks(range(N))
    ax.set_xticklabels(labels, rotation=45 if N > 6 else 0, ha="right" if N > 6 else "center")
    ax.set_yticks(range(N))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Receiver")
    ax.set_ylabel("Sender")
    ax.set_title("Round-2 − Round-1 Mean Adjacency\n(red=stronger in R2, blue=weaker)")
    fontsize = max(5, 8 - max(0, N - 4))
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{delta[i,j]:+.2f}", ha="center", va="center",
                    fontsize=fontsize, color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _save(fig, out_dir, f"{prefix}_10_round_divergence")


# ─── Figure 11 & 12: Message PCA ─────────────────────────────────────────────


def fig_message_pca(
    data: MAGICWarehouseData, out_dir: Path, prefix: str, use_agg: bool = True
) -> None:
    tag = "agg" if use_agg else "raw"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    fig.suptitle(
        f"MAGIC Message Embeddings — PCA ({'aggregated' if use_agg else 'raw'})\n"
        "Each point = one agent's message at one step",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: coloured by agent identity
    proj, var_ratio, pca_labels = compute_message_pca(data, use_agg=use_agg)
    ax = axes[0]
    if proj is not None and pca_labels is not None:
        agent_ids = sorted(set(int(x) for x in pca_labels))
        act_colors = _agent_colors(max(agent_ids) + 1)
        act_labels = _agent_labels(max(agent_ids) + 1)
        for i in agent_ids:
            mask = pca_labels == i
            if mask.any():
                ax.scatter(proj[mask, 0], proj[mask, 1], c=act_colors[i],
                           s=4, alpha=0.4, label=act_labels[i])
        ax.set_title(
            f"Coloured by Agent\n(PC1={var_ratio[0]:.1%}, PC2={var_ratio[1]:.1%})"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(markerscale=3)
    else:
        ax.text(0.5, 0.5, "sklearn not available", ha="center", va="center")
        ax.axis("off")

    # Panel 2: coloured by resource phase
    proj2, var2, ctx2 = compute_message_pca_with_context(data, "phase", use_agg=use_agg)
    ax = axes[1]
    if proj2 is not None:
        scatter = ax.scatter(proj2[:, 0], proj2[:, 1], c=ctx2, cmap="RdYlGn",
                             s=4, alpha=0.4, vmin=0, vmax=3)
        plt.colorbar(scatter, ax=ax, label="Resource phase (0–3)", ticks=[0, 1, 2, 3])
        ax.set_title(f"Coloured by Resource Phase\n(0=Search, 1=Transit, 2=Treat, 3=Deliver)")
        ax.set_xlabel("PC1")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")

    # Panel 3: coloured by battery level
    proj3, var3, ctx3 = compute_message_pca_with_context(data, "battery", use_agg=use_agg)
    ax = axes[2]
    if proj3 is not None:
        scatter = ax.scatter(proj3[:, 0], proj3[:, 1], c=ctx3, cmap="RdYlBu",
                             s=4, alpha=0.4)
        plt.colorbar(scatter, ax=ax, label="Battery level")
        ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax.set_title("Coloured by Battery Level\n(blue=high, red=critical)")
        ax.set_xlabel("PC1")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")

    _save(fig, out_dir, f"{prefix}_11_message_pca_{tag}")


# ─── Figure 13: Message norm dynamics ────────────────────────────────────────


def fig_message_norms(data: MAGICWarehouseData, out_dir: Path, prefix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    fig.suptitle(
        "Message Embedding L2 Norms — Raw vs Aggregated\n"
        "(high norm delta = agent synthesised significant information from neighbours)",
        fontsize=12, fontweight="bold",
    )

    roles = data.agent_roles()
    raw_norms = compute_message_norm_over_time(data, use_agg=False)
    agg_norms = compute_message_norm_over_time(data, use_agg=True)

    def _smooth(x: np.ndarray, w: int = 25) -> np.ndarray:
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    # Left: raw norms
    ax = axes[0]
    N_raw = data.actual_num_agents
    raw_labels = _agent_labels(N_raw)
    raw_colors = _agent_colors(N_raw)
    for i, (agent, norms) in enumerate(raw_norms.items()):
        if len(norms) == 0:
            continue
        role = roles.get(agent, {})
        lbl = raw_labels[i] if i < len(raw_labels) else f"A{i}"
        color = raw_colors[i] if i < len(raw_colors) else "#888888"
        label = f"{lbl} sp={int(role.get('speed',1))}"
        ys = _smooth(norms)
        ax.plot(np.arange(len(ys)), ys, color=color, linewidth=1.8, label=label)
    ax.set_xlabel("Step (across episodes)")
    ax.set_ylabel("L2 norm")
    ax.set_title("Raw Message Norms (pre-GAT aggregation)")
    ax.legend()

    # Right: aggregated norms + delta (agg - raw)
    ax = axes[1]
    N_norms = data.actual_num_agents
    delta_labels = _agent_labels(N_norms)
    delta_colors = _agent_colors(N_norms)
    for i, agent in enumerate([f"agent_{k}" for k in range(N_norms)]):
        raw = raw_norms.get(agent, np.array([]))
        agg = agg_norms.get(agent, np.array([]))
        min_len = min(len(raw), len(agg))
        if min_len == 0:
            continue
        delta = agg[:min_len] - raw[:min_len]
        ys = _smooth(delta)
        role = roles.get(agent, {})
        label = f"Δ{delta_labels[i]} sp={int(role.get('speed',1))}"
        ax.plot(np.arange(len(ys)), ys, color=delta_colors[i], linewidth=1.8, label=label)
    ax.axhline(0, color="gray", linestyle="--", lw=1.0, alpha=0.6)
    ax.set_xlabel("Step (across episodes)")
    ax.set_ylabel("Agg norm − Raw norm")
    ax.set_title("Message Norm Delta (aggregated − raw)\n(positive = GAT added information)")
    ax.legend()

    _save(fig, out_dir, f"{prefix}_13_message_norms")


# ─── Figure 14: Rescue-aligned communication ─────────────────────────────────


def fig_rescue_aligned(
    data: MAGICWarehouseData, out_dir: Path, prefix: str
) -> None:
    R = data.num_comm_rounds
    window_before, window_after = 10, 15

    fig, axes = plt.subplots(R, 3, figsize=(18, 5.5 * R), constrained_layout=True)
    if R == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        f"Communication Graph Aligned Around Rescue Events\n"
        f"(window: −{window_before} to +{window_after} steps from rescue start)",
        fontsize=12, fontweight="bold",
    )

    for r in range(R):
        aligned = align_comm_around_rescue(data, window_before, window_after, r)
        if aligned is None:
            for c in range(3):
                axes[r, c].text(0.5, 0.5, "No rescue events found", ha="center", va="center")
                axes[r, c].axis("off")
            continue

        W = window_before + window_after

        # Left: pre-rescue mean adj
        pre_adj = aligned[:window_before].mean(axis=0)
        _adj_heatmap(axes[r, 0], pre_adj, f"Pre-rescue\n(−{window_before} to −1 steps)")

        # Centre: post-rescue mean adj
        post_adj = aligned[window_before:].mean(axis=0)
        _adj_heatmap(axes[r, 1], post_adj, f"Post-rescue\n(0 to +{window_after} steps)")

        # Right: comm density evolution around event
        ax = axes[r, 2]
        timeline = aligned.mean(axis=(1, 2))   # (W,) mean density
        t = np.arange(-window_before, window_after)
        ax.plot(t, timeline, color="#8172B2", linewidth=2)
        ax.axvline(0, color="red", linestyle="--", lw=1.5, label="Rescue start")
        ax.set_xlabel("Steps relative to rescue")
        ax.set_ylabel("Mean adj weight")
        ax.set_title(f"Comm Density Timeline (Round {r + 1})")
        ax.legend()

    _save(fig, out_dir, f"{prefix}_14_rescue_aligned")


# ─── Figure 15: Interference zone effect ─────────────────────────────────────


def fig_interference_effect(
    data: MAGICWarehouseData,
    out_dir: Path,
    prefix: str,
    treatment_cols: Optional[List[int]] = None,
) -> None:
    """
    Compare comm density for agents near treatment stations (interference_radius=2)
    vs agents far from treatment stations.

    treatment_cols: list of column indices of treatment stations. If None, uses
    columns at roughly 1/3 and 2/3 of the grid width.
    """
    if treatment_cols is None:
        W = data.grid_width
        treatment_cols = [W // 3, 2 * W // 3]

    def _dist_to_treatment(row: int, col: int) -> int:
        return min(abs(col - tc) for tc in treatment_cols)

    csteps = data.comm_steps()
    if not csteps:
        return

    # Per-step: minimum distance any active agent is from a treatment station
    near_dists, near_dens = [], []
    far_dists, far_dens = [], []
    all_dists, all_dens = [], []

    for s in csteps:
        agents_list = list(s.positions.keys())
        if not agents_list:
            continue
        dists = [_dist_to_treatment(s.positions[a][0], s.positions[a][1]) for a in agents_list]
        min_dist = min(dists)
        all_dists.append(min_dist)
        all_dens.append(s.comm_density)
        if min_dist <= 2:
            near_dists.append(min_dist)
            near_dens.append(s.comm_density)
        else:
            far_dists.append(min_dist)
            far_dens.append(s.comm_density)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.suptitle(
        "Communication Density vs Proximity to Treatment Stations\n"
        "(interference_treatment_boost=0.35 — does MAGIC compensate for noisy obs?)",
        fontsize=12, fontweight="bold",
    )

    # Left: scatter + binned mean
    ax = axes[0]
    ax.scatter(all_dists, all_dens, alpha=0.15, s=5, color="#4C72B0", label="All steps")
    # Bin by distance
    max_d = max(all_dists) if all_dists else 1
    bins = np.arange(0, max_d + 2)
    bin_means, bin_stds = [], []
    for d in range(max_d + 1):
        mask = [i for i, x in enumerate(all_dists) if x == d]
        if mask:
            vals = [all_dens[i] for i in mask]
            bin_means.append(float(np.mean(vals)))
            bin_stds.append(float(np.std(vals)))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
    t = np.arange(max_d + 1)
    valid = ~np.isnan(bin_means)
    ax.plot(t[valid], np.array(bin_means)[valid], color="#E07B39", linewidth=2, label="Mean")
    ax.axvline(2, color="red", linestyle="--", lw=1.5, label="Interference radius (2)")
    ax.set_xlabel("Min agent distance to nearest treatment station")
    ax.set_ylabel("Comm density")
    ax.set_title("Comm Density vs Treatment Station Distance")
    ax.set_ylim(0, 1.05)
    ax.legend()

    # Right: box comparison (near ≤ 2 vs far > 2)
    ax = axes[1]
    plot_data = []
    plot_labels = []
    if near_dens:
        plot_data.append(near_dens)
        plot_labels.append(f"Near\n(≤2, n={len(near_dens)})")
    if far_dens:
        plot_data.append(far_dens)
        plot_labels.append(f"Far\n(>2, n={len(far_dens)})")
    if plot_data:
        bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True,
                        medianprops=dict(color="black", lw=2))
        colors = ["#C44E52", "#4C72B0"]
        for patch, c in zip(bp["boxes"], colors[:len(bp["boxes"])]):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_ylabel("Comm density")
        ax.set_title("Near vs Far from Treatment Station\n(interference zone effect)")
        ax.set_ylim(0, 1.05)

    _save(fig, out_dir, f"{prefix}_15_interference_effect")


# ─── Figure 16: Summary dashboard ────────────────────────────────────────────


def fig_summary_dashboard(data: MAGICWarehouseData, out_dir: Path, prefix: str) -> None:
    """One-page overview combining the most informative findings."""
    R = data.num_comm_rounds
    N = data.actual_num_agents
    agent_colors = _agent_colors(N)
    labels = _agent_labels(N)
    fig = plt.figure(figsize=(22, 14), constrained_layout=True)
    fig.suptitle(
        f"MAGIC Warehouse Communication Summary Dashboard\n"
        f"({data.n_episodes} episodes, {N} agents, {R} comm rounds, msg_dim={data.message_dim})",
        fontsize=14, fontweight="bold",
    )
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    # Row 0: Adjacency heatmaps (one per round)
    for r in range(min(R, 2)):
        ax = fig.add_subplot(gs[0, r])
        adj = compute_mean_adj(data, r)
        _adj_heatmap(ax, adj, f"Mean Adj — Round {r + 1}")

    # Row 0, col 2: Hub scores bar
    ax = fig.add_subplot(gs[0, 2])
    scores = compute_hub_scores(data)
    if scores.size > 0:
        mean_scores = scores.mean(axis=0)
        hub_N = scores.shape[1]
        hub_colors = _agent_colors(hub_N)
        roles = data.agent_roles()
        hub_labels = [
            f"{labels[i] if i < len(labels) else f'A{i}'}\nsp={int(roles.get(f'agent_{i}',{}).get('speed',1))}"
            for i in range(hub_N)
        ]
        ax.bar(hub_labels, mean_scores, color=hub_colors, alpha=0.85)
        ax.set_ylabel("Mean in-degree")
        ax.set_title("Hub Scores\n(who receives most messages)")
        for i, v in enumerate(mean_scores):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    # Row 0, col 3: Phase-conditioned adj (Phase 2: Treating)
    ax = fig.add_subplot(gs[0, 3])
    adj_treating = compute_conditional_adj(
        data, lambda s: s.treating_count >= 1, round_idx=R - 1
    )
    n_treating = sum(1 for s in data.comm_steps() if s.treating_count >= 1)
    _adj_heatmap(ax, adj_treating, f"Treating condition\n(n={n_treating})")

    # Row 1: Context correlations
    ctx_vars = [
        ("min_battery", "Min battery", "#4C72B0"),
        ("pending_tasks", "Pending tasks", "#DD8452"),
        ("treating_count", "Treating agents", "#E76F51"),
        ("step_fraction", "Episode progress", "#8172B2"),
    ]
    for col, (key, label, color) in enumerate(ctx_vars):
        ax = fig.add_subplot(gs[1, col])
        centres, means, stds = compute_comm_vs_context(data, key)
        valid = means > 0
        if valid.any():
            width = (centres[1] - centres[0]) if len(centres) > 1 else 0.5
            ax.bar(centres[valid], means[valid], width=width * 0.8, color=color, alpha=0.75)
            ax.errorbar(centres[valid], means[valid], yerr=stds[valid],
                        fmt="none", ecolor="black", capsize=2, lw=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Comm density")
        ax.set_title(f"Comm vs {label}")
        ax.set_ylim(0, 1.0)

    # Row 2: Message PCA (phase coloured)
    ax = fig.add_subplot(gs[2, :2])
    proj, var_ratio, pca_labels = compute_message_pca(data, use_agg=True)
    if proj is not None and pca_labels is not None:
        pca_agent_ids = sorted(set(int(x) for x in pca_labels))
        pca_colors = _agent_colors(max(pca_agent_ids) + 1)
        pca_agent_labels = _agent_labels(max(pca_agent_ids) + 1)
        for i in pca_agent_ids:
            mask = pca_labels == i
            if mask.any():
                ax.scatter(proj[mask, 0], proj[mask, 1], c=pca_colors[i],
                           s=3, alpha=0.35, label=pca_agent_labels[i])
        ax.set_title(
            f"Message PCA (aggregated)\n(PC1={var_ratio[0]:.1%}, PC2={var_ratio[1]:.1%})"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(markerscale=3, fontsize=8)

    # Row 2: Round divergence
    ax = fig.add_subplot(gs[2, 2])
    diffs = compute_round_divergence(data)
    if diffs is not None and len(diffs) > 0:
        ax.plot(diffs, color="#8172B2", alpha=0.5, lw=0.8)
        if len(diffs) > 30:
            smooth = np.convolve(diffs, np.ones(30) / 30, mode="valid")
            ax.plot(np.arange(15, 15 + len(smooth)), smooth, color="#8172B2", lw=2.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("‖G² − G¹‖_F")
        ax.set_title("Round-1 vs Round-2 Divergence")
    else:
        ax.text(0.5, 0.5, "Single round", ha="center", va="center")
        ax.axis("off")

    # Row 2, col 3: Summary stats text box
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    summary_lines = [
        "Summary Statistics",
        "──────────────────",
        f"Episodes: {data.n_episodes}",
        f"Mean deliveries/ep: {data.mean_deliveries:.1f}",
        f"Mean ep length: {data.mean_episode_length:.0f}",
        "",
        "Comm rounds analysis:",
    ]
    for r in range(R):
        adj = compute_mean_adj(data, r)
        adj_N = adj.shape[0]
        off_diag = adj[~np.eye(adj_N, dtype=bool)]
        summary_lines.append(f"  R{r+1} mean edge weight: {off_diag.mean():.3f}")
        summary_lines.append(f"  R{r+1} edge density: {(off_diag > 0.5).mean():.1%}")
    pw = compute_pairwise_weights(data, round_idx=R - 1)
    if pw:
        top_pair = max(pw.items(), key=lambda x: x[1])
        summary_lines += ["", f"Top comm pair: {top_pair[0]}", f"  weight: {top_pair[1]:.3f}"]
    ax.text(0.05, 0.95, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=9, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    _save(fig, out_dir, f"{prefix}_16_summary_dashboard")


# ─── Master save function ─────────────────────────────────────────────────────


def save_all_magic_warehouse_figures(
    data: MAGICWarehouseData,
    output_dir: str | Path = "eval_plots/warehouse/magic",
    prefix: str = "magic_warehouse",
    dpi: int = 150,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print_summary(data)
    print(f"\n{'=' * 65}")
    print(f"  Generating MAGIC warehouse comm plots → {out}")
    print(f"{'=' * 65}\n")

    if not data.has_comm_data:
        print("  WARNING: No communication data found. Only basic figures will be generated.")

    fig_adj_heatmaps(data, out, prefix)
    fig_network_diagram(data, out, prefix)
    fig_pairwise_barplot(data, out, prefix)
    fig_phase_conditional(data, out, prefix)
    fig_battery_conditional(data, out, prefix)
    fig_comm_trajectory(data, out, prefix)
    fig_context_correlations(data, out, prefix)
    fig_hub_scores(data, out, prefix)
    fig_spatial_comm(data, out, prefix)
    fig_round_divergence(data, out, prefix)
    fig_message_pca(data, out, prefix, use_agg=True)
    fig_message_pca(data, out, prefix, use_agg=False)
    fig_message_norms(data, out, prefix)
    fig_rescue_aligned(data, out, prefix)
    fig_interference_effect(data, out, prefix)
    fig_summary_dashboard(data, out, prefix)

    print(f"\nAll MAGIC warehouse plots saved to {out}/")
