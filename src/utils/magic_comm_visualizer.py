"""
magic_comm_visualizer.py
========================
Rich visualizations for MAGIC's communication mechanism on BlindSpot Navigation.

Figures produced
----------------
 1.  fig_adj_heatmap          – Mean soft adjacency matrices per comm round
 2.  fig_comm_trajectory      – Per-episode communication density timeline
 3.  fig_directional_bar      – A→B vs B→A asymmetry per comm round
 4.  fig_comm_vs_trap_dist    – Comm density binned by nearest-trap distance
 5.  fig_comm_vs_goal_dist    – Comm density binned by nearest-goal distance
 6.  fig_conditional_adj      – Adj heatmaps conditioned on scenario phase
 7.  fig_event_aligned        – Comm aligned around trap encounters / goal reach
 8.  fig_message_pca          – PCA of raw message embeddings with context overlay
 9.  fig_message_tsne         – t-SNE of raw message embeddings
10.  fig_round_comparison     – Adj heatmap evolution across comm rounds
11.  fig_policy_entropy       – Policy entropy vs comm density scatter
12.  fig_success_vs_comm      – Successful vs failed episode comm patterns
13.  fig_comm_density_hist    – Distribution of per-step comm density
14.  fig_message_norm         – Message L2-norm over episode time
15.  fig_spatial_comm         – Mean comm density on the grid (spatial heatmap)
16.  fig_episode_gallery      – Gallery of 6 sample episode comm timelines

All sub-figures saved as both PDF and PNG to *output_dir*.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .magic_comm_analysis import (
    MAGICAnalysisData,
    CommEpisodeData,
    CommStepRecord,
    AGENT_A,
    AGENT_B,
    AGENTS,
    compute_mean_adj,
    compute_conditional_adj,
    compute_comm_vs_context,
    compute_message_pca,
    compute_message_tsne,
    compute_directional_asymmetry,
    align_around_event,
    compute_action_entropy,
    print_summary,
)

# ──────────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "a": "#4C72B0",
    "b": "#DD8452",
    "success": "#55A868",
    "failure": "#C44E52",
    "neutral": "#8172B2",
}
CMAP_COMM = "Blues"
CMAP_HEAT = "RdYlGn_r"
AGENT_LABELS = {AGENT_A: "Agent A", AGENT_B: "Agent B"}
_TICK = ["A", "B"]

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


def _adj_heatmap(ax: plt.Axes, mat: np.ndarray, title: str, vmax: float = 1.0) -> None:
    N = mat.shape[0]
    im = ax.imshow(mat, cmap=CMAP_COMM, vmin=0, vmax=vmax, aspect="equal")
    ax.set_xticks(range(N))
    ax.set_xticklabels(_TICK[:N])
    ax.set_yticks(range(N))
    ax.set_yticklabels(_TICK[:N])
    ax.set_xlabel("Message Receiver")
    ax.set_ylabel("Message Sender")
    ax.set_title(title)
    for i in range(N):
        for j in range(N):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=11,
                color="white" if mat[i, j] > 0.5 * vmax else "black",
            )
    cax = make_axes_locatable(ax).append_axes("right", size="8%", pad=0.05)
    plt.colorbar(im, cax=cax)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 – Mean adjacency heatmaps
# ──────────────────────────────────────────────────────────────────────────────


def fig_adj_heatmap(data: MAGICAnalysisData, out_dir: Path, prefix: str) -> None:
    R = data.num_comm_rounds
    fig, axes = plt.subplots(1, R, figsize=(4 * R, 4.2), squeeze=False)
    fig.suptitle(
        "MAGIC Scheduler — Mean Soft Adjacency Matrix\n"
        "(darker = more frequent communication)",
        fontsize=12,
        y=1.02,
    )
    for r in range(R):
        _adj_heatmap(axes[0, r], compute_mean_adj(data, r), f"Round {r + 1}")
    fig.tight_layout()
    _save(fig, out_dir, f"{prefix}_adj_heatmap")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 – Communication density timelines
# ──────────────────────────────────────────────────────────────────────────────


def fig_comm_trajectory(
    data: MAGICAnalysisData,
    out_dir: Path,
    prefix: str,
    max_episodes: int = 20,
) -> None:
    episodes = data.episodes[:max_episodes]
    if not any(e.has_comm_data for e in episodes):
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"hspace": 0.42})
    ax_top, ax_bot = axes

    for ep in episodes:
        if not ep.has_comm_data:
            continue
        steps = [s.step for s in ep.steps if s.adj_per_round is not None]
        dens = [s.comm_density for s in ep.steps if s.adj_per_round is not None]
        if steps:
            color = PALETTE["success"] if ep.success else PALETTE["failure"]
            ax_top.plot(steps, dens, alpha=0.35, color=color, linewidth=0.8)

    ax_top.set_xlabel("Environment step")
    ax_top.set_ylabel("Comm density")
    ax_top.set_title("Communication density per step (green=success, red=failure)")
    ax_top.set_ylim(0, 1.05)
    legend_handles = [
        mpatches.Patch(color=PALETTE["success"], alpha=0.6, label="Success"),
        mpatches.Patch(color=PALETTE["failure"], alpha=0.6, label="Failure"),
    ]
    ax_top.legend(handles=legend_handles, fontsize=9, loc="upper right")

    max_len = max((e.length for e in episodes), default=0)
    if max_len > 0:
        acc = np.full((len(episodes), max_len), np.nan)
        for i, ep in enumerate(episodes):
            for s in ep.steps:
                if s.adj_per_round is not None and s.step < max_len:
                    acc[i, s.step] = s.comm_density
        mean_d = np.nanmean(acc, axis=0)
        std_d = np.nanstd(acc, axis=0)
        t = np.arange(max_len)
        ax_bot.plot(t, mean_d, color=PALETTE["neutral"], linewidth=1.5, label="Mean")
        ax_bot.fill_between(
            t,
            mean_d - std_d,
            mean_d + std_d,
            alpha=0.25,
            color=PALETTE["neutral"],
            label="±1 SD",
        )
        ax_bot.set_xlabel("Environment step")
        ax_bot.set_ylabel("Comm density")
        ax_bot.set_title("Mean communication density across episodes")
        ax_bot.set_ylim(0, 1.05)
        ax_bot.legend(fontsize=9)

    _save(fig, out_dir, f"{prefix}_comm_trajectory")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 – Directional asymmetry bar chart
# ──────────────────────────────────────────────────────────────────────────────


def fig_directional_bar(data: MAGICAnalysisData, out_dir: Path, prefix: str) -> None:
    R = data.num_comm_rounds
    x = np.arange(R)
    w = 0.22
    a2b = [compute_directional_asymmetry(data, r)["a_to_b_mean"] for r in range(R)]
    b2a = [compute_directional_asymmetry(data, r)["b_to_a_mean"] for r in range(R)]
    sla = [compute_directional_asymmetry(data, r)["self_loop_a"] for r in range(R)]
    slb = [compute_directional_asymmetry(data, r)["self_loop_b"] for r in range(R)]

    fig, ax = plt.subplots(figsize=(max(5, 3 * R), 4.5))
    ax.bar(x - 1.5 * w, a2b, width=w, label="A → B", color=PALETTE["a"])
    ax.bar(x - 0.5 * w, b2a, width=w, label="B → A", color=PALETTE["b"])
    ax.bar(
        x + 0.5 * w,
        sla,
        width=w,
        label="A self",
        color=PALETTE["a"],
        alpha=0.45,
        hatch="//",
    )
    ax.bar(
        x + 1.5 * w,
        slb,
        width=w,
        label="B self",
        color=PALETTE["b"],
        alpha=0.45,
        hatch="//",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Round {r + 1}" for r in range(R)])
    ax.set_ylabel("Mean soft adjacency weight")
    ax.set_ylim(0, 1.0)
    ax.set_title("Directed Communication Weights per Round\n(who sends to whom)")
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    _save(fig, out_dir, f"{prefix}_directional_bar")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 & 5 – Comm density vs context distance
# ──────────────────────────────────────────────────────────────────────────────


def _context_plot(
    data: MAGICAnalysisData,
    out_dir: Path,
    prefix: str,
    key_a: str,
    key_b: str,
    xlabel: str,
    title: str,
    fig_name: str,
) -> None:
    if not data.comm_steps():
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    fig.suptitle(title)
    for ax, key, agent_label, color in [
        (axes[0], key_a, "Agent A", PALETTE["a"]),
        (axes[1], key_b, "Agent B", PALETTE["b"]),
    ]:
        try:
            centres, means, stds = compute_comm_vs_context(data, key)
        except ValueError:
            continue
        valid = means > 0
        ax.bar(
            centres[valid],
            means[valid],
            width=(centres[1] - centres[0]) if len(centres) > 1 else 0.5,
            color=color,
            alpha=0.65,
            label=agent_label,
        )
        ax.errorbar(
            centres[valid],
            means[valid],
            yerr=stds[valid],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=0.9,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean comm density")
        ax.set_title(agent_label)
        ax.set_ylim(0, 1.0)
    fig.tight_layout()
    _save(fig, out_dir, f"{prefix}_{fig_name}")


def fig_comm_vs_trap_dist(data: MAGICAnalysisData, out_dir: Path, prefix: str) -> None:
    _context_plot(
        data,
        out_dir,
        prefix,
        key_a="trap_dist_a",
        key_b="trap_dist_b",
        xlabel="Manhattan distance to nearest trap",
        title="Communication Density vs Trap Proximity\n"
        "(Does danger trigger communication?)",
        fig_name="comm_vs_trap_dist",
    )


def fig_comm_vs_goal_dist(data: MAGICAnalysisData, out_dir: Path, prefix: str) -> None:
    _context_plot(
        data,
        out_dir,
        prefix,
        key_a="dist_goal_a",
        key_b="dist_goal_b",
        xlabel="Manhattan distance to goal",
        title="Communication Density vs Goal Proximity",
        fig_name="comm_vs_goal_dist",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6 – Conditional adjacency heatmaps by scenario phase
# ──────────────────────────────────────────────────────────────────────────────


def fig_conditional_adj(
    data: MAGICAnalysisData,
    out_dir: Path,
    prefix: str,
    round_idx: int = -1,
) -> None:
    r = round_idx if round_idx >= 0 else data.num_comm_rounds - 1
    conditions = [
        ("A near trap\n(dist ≤ 2)", lambda s: s.min_trap_dist.get(AGENT_A, 99) <= 2),
        (
            "A far from traps\n(dist > 4)",
            lambda s: s.min_trap_dist.get(AGENT_A, 99) > 4,
        ),
        ("Early episode\n(step < 30%)", lambda s: s.step < 0.3 * data.max_cycles),
        ("Late episode\n(step > 70%)", lambda s: s.step > 0.7 * data.max_cycles),
        (
            "None reached goal",
            lambda s: not s.reached.get(AGENT_A, False)
            and not s.reached.get(AGENT_B, False),
        ),
        (
            "≥1 agent at goal",
            lambda s: s.reached.get(AGENT_A, False) or s.reached.get(AGENT_B, False),
        ),
    ]

    n_cond = len(conditions)
    ncols = 3
    nrows = (n_cond + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(
        f"Communication Graph under Different Scenarios (Round {r+1})\n"
        "Darker = stronger communication link",
        fontsize=12,
        y=1.01,
    )
    for idx, (label, fn) in enumerate(conditions):
        _adj_heatmap(axes[idx], compute_conditional_adj(data, fn, round_idx=r), label)
    for idx in range(n_cond, len(axes)):
        axes[idx].set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir, f"{prefix}_conditional_adj")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 7 – Event-aligned communication
# ──────────────────────────────────────────────────────────────────────────────


def fig_event_aligned(
    data: MAGICAnalysisData,
    out_dir: Path,
    prefix: str,
    window_before: int = 5,
    window_after: int = 10,
) -> None:
    t = np.arange(-window_before, window_after)
    events = {
        "Trap encounter (Agent A)": lambda ep: ep.trap_hit_steps,
        "Goal reached (either agent)": lambda ep: [
            s.step
            for s in ep.steps
            if s.reached.get(AGENT_A, False) or s.reached.get(AGENT_B, False)
        ],
    }

    R = data.num_comm_rounds
    fig, axes = plt.subplots(
        len(events),
        R,
        figsize=(4 * R, 3.8 * len(events)),
        squeeze=False,
    )
    fig.suptitle("Communication Graph Aligned Around Key Events", fontsize=12, y=1.01)

    for row, (ev_label, ev_fn) in enumerate(events.items()):
        for r in range(R):
            result = align_around_event(
                data,
                ev_fn,
                window_before=window_before,
                window_after=window_after,
                round_idx=r,
            )
            ax = axes[row, r]
            if result is None:
                ax.text(
                    0.5,
                    0.5,
                    "No events found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="gray",
                )
                ax.set_title(f"{ev_label}\nRound {r+1}")
                ax.axis("off")
                continue

            ax.plot(
                t, result[:, 0, 1], color=PALETTE["a"], linewidth=2.0, label="A → B"
            )
            ax.plot(
                t, result[:, 1, 0], color=PALETTE["b"], linewidth=2.0, label="B → A"
            )
            ax.axvline(
                0,
                color="gray",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label="Event t=0",
            )
            ax.set_xlim(t[0], t[-1])
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Steps relative to event")
            ax.set_ylabel("Soft adjacency weight")
            ax.set_title(f"{ev_label}\nRound {r+1}")
            if r == R - 1:
                ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, out_dir, f"{prefix}_event_aligned")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 8 – PCA of message embeddings
# ──────────────────────────────────────────────────────────────────────────────


def fig_message_pca(data: MAGICAnalysisData, out_dir: Path, prefix: str) -> None:
    if not data.has_comm_data:
        return

    proj, ev = compute_message_pca(data, n_components=2)
    if proj is None:
        return

    csteps = data.comm_steps()
    N, T = 2, len(csteps)
    if len(proj) != T * N:
        return

    ctx = data.context_at_comm_steps()
    idx_a = np.arange(0, T * N, N)
    idx_b = np.arange(1, T * N, N)

    plots = [
        ("trap_dist_a", "Trap distance (A)", idx_a),
        ("trap_dist_b", "Trap distance (B)", idx_b),
        ("dist_goal_a", "Goal distance (A)", idx_a),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), squeeze=False)
    fig.suptitle(
        f"PCA of Message Embeddings  (PC1={ev[0]:.1%}, PC2={ev[1]:.1%} var)\n"
        "Each point = one agent message at one timestep",
        fontsize=11,
        y=1.02,
    )
    for col, (ctx_key, clabel, idx) in enumerate(plots):
        ax = axes[0, col]
        c_vals = ctx[ctx_key][: len(idx)]
        sc = ax.scatter(
            proj[idx, 0],
            proj[idx, 1],
            c=c_vals,
            cmap=CMAP_HEAT,
            s=12,
            alpha=0.55,
            linewidths=0,
        )
        plt.colorbar(sc, ax=ax, label=clabel, shrink=0.85)
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
        ax.set_title(clabel)
    fig.tight_layout()
    _save(fig, out_dir, f"{prefix}_message_pca")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 9 – t-SNE of message embeddings
# ──────────────────────────────────────────────────────────────────────────────


def fig_message_tsne(data: MAGICAnalysisData, out_dir: Path, prefix: str) -> None:
    if not data.has_comm_data:
        return

    emb = compute_message_tsne(data, perplexity=30.0, max_samples=1500)
    if emb is None:
        return

    csteps = data.comm_steps()
    N = 2
    T_used = min(len(csteps), len(emb) // N)
    ctx = data.context_at_comm_steps()

    idx_a = np.arange(0, T_used * N, N)
    idx_b = np.arange(1, T_used * N, N)
    idx_a = idx_a[idx_a < len(emb)]
    idx_b = idx_b[idx_b < len(emb)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), squeeze=False)
    fig.suptitle(
        "t-SNE of Message Embeddings\n"
        "Clusters reveal distinct communicative 'modes' the agents learned",
        fontsize=11,
        y=1.02,
    )
    plots = [
        ("trap_dist_a", "Trap distance (A)", idx_a),
        ("trap_dist_b", "Trap distance (B)", idx_b),
        ("step_fraction", "Episode progress", np.arange(len(emb))),
    ]
    for col, (ctx_key, clabel, idx) in enumerate(plots):
        ax = axes[0, col]
        if ctx_key == "step_fraction":
            c_vals = np.tile(ctx[ctx_key], N)[: len(emb)]
        else:
            c_vals = ctx[ctx_key][: len(idx)]
        sc = ax.scatter(
            emb[idx, 0],
            emb[idx, 1],
            c=c_vals,
            cmap=CMAP_HEAT,
            s=10,
            alpha=0.5,
            linewidths=0,
        )
        plt.colorbar(sc, ax=ax, label=clabel, shrink=0.85)
        ax.set_title(f"Colour: {clabel}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    _save(fig, out_dir, f"{prefix}_message_tsne")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 10 – Round-by-round comparison
# ──────────────────────────────────────────────────────────────────────────────


def fig_round_comparison(data: MAGICAnalysisData, out_dir: Path, prefix: str) -> None:
    R = data.num_comm_rounds
    if R < 2:
        return

    fig, axes = plt.subplots(1, R + 1, figsize=(4.5 * (R + 1), 4.2), squeeze=False)
    fig.suptitle(
        "How Communication Graphs Evolve Across Rounds\n"
        "(Right-most: change from Round 1 → last)",
        fontsize=12,
        y=1.02,
    )
    adjs = [compute_mean_adj(data, r) for r in range(R)]
    for r, adj in enumerate(adjs):
        _adj_heatmap(axes[0, r], adj, f"Round {r+1}")

    delta = adjs[-1] - adjs[0]
    ax = axes[0, R]
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="equal")
    N = delta.shape[0]
    ax.set_xticks(range(N))
    ax.set_xticklabels(_TICK[:N])
    ax.set_yticks(range(N))
    ax.set_yticklabels(_TICK[:N])
    ax.set_xlabel("Receiver")
    ax.set_ylabel("Sender")
    ax.set_title("Δ (Last − First round)")
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{delta[i,j]:+.2f}", ha="center", va="center", fontsize=11)
    plt.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", "8%", pad=0.05))
    fig.tight_layout()
    _save(fig, out_dir, f"{prefix}_round_comparison")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 11 – Policy entropy vs comm density
# ──────────────────────────────────────────────────────────────────────────────


def fig_policy_entropy(data: MAGICAnalysisData, out_dir: Path, prefix: str) -> None:
    csteps = data.comm_steps()
    ent = compute_action_entropy(data)
    dens = np.array([s.comm_density for s in csteps])

    if len(ent) == 0 or len(dens) == 0:
        return
    n = min(len(ent), len(dens))
    ent, dens = ent[:n], dens[:n]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.scatter(dens, ent, s=8, alpha=0.4, color=PALETTE["neutral"])
    if len(dens) > 2 and dens.std() > 1e-6:
        z = np.polyfit(dens, ent, 1)
        xl = np.linspace(dens.min(), dens.max(), 100)
        ax.plot(xl, np.polyval(z, xl), color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Communication density")
    ax.set_ylabel("Mean policy entropy (nats)")
    ax.set_title(
        "Policy Entropy vs Communication Density\n"
        "(Does communication reduce action uncertainty?)"
    )

    ax2 = axes[1]
    bins = np.linspace(0, 1, 11)
    centres = 0.5 * (bins[:-1] + bins[1:])
    means, stds = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (dens >= lo) & (dens < hi)
        means.append(ent[mask].mean() if mask.sum() > 0 else np.nan)
        stds.append(ent[mask].std() if mask.sum() > 0 else 0.0)
    means, stds = np.array(means), np.array(stds)
    valid = ~np.isnan(means)
    ax2.bar(
        centres[valid], means[valid], width=0.08, color=PALETTE["neutral"], alpha=0.65
    )
    ax2.errorbar(
        centres[valid],
        means[valid],
        yerr=stds[valid],
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    ax2.set_xlabel("Communication density (binned)")
    ax2.set_ylabel("Mean policy entropy")
    ax2.set_title("Entropy by Comm Density (binned)")

    fig.tight_layout()
    _save(fig, out_dir, f"{prefix}_policy_entropy")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 12 – Success vs failure communication comparison
# ──────────────────────────────────────────────────────────────────────────────


def fig_success_vs_comm(data: MAGICAnalysisData, out_dir: Path, prefix: str) -> None:
    if not data.has_comm_data:
        return

    succ = [e for e in data.episodes if e.success]
    fail = [e for e in data.episodes if not e.success]

    def _mean_adj_eps(eps, r=0):
        mats = [
            s.adj_per_round[r]
            for e in eps
            for s in e.steps
            if s.adj_per_round and len(s.adj_per_round) > r
        ]
        return np.stack(mats).mean(axis=0) if mats else np.zeros((2, 2))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Successful vs Failed Episodes — Communication Patterns", fontsize=12, y=1.02
    )

    # Δ adjacency heatmap
    r = data.num_comm_rounds - 1
    diff = _mean_adj_eps(succ, r) - _mean_adj_eps(fail, r)
    ax = axes[0]
    im = ax.imshow(diff, cmap="RdBu", vmin=-0.4, vmax=0.4, aspect="equal")
    N = diff.shape[0]
    ax.set_xticks(range(N))
    ax.set_xticklabels(_TICK[:N])
    ax.set_yticks(range(N))
    ax.set_yticklabels(_TICK[:N])
    ax.set_xlabel("Receiver")
    ax.set_ylabel("Sender")
    ax.set_title("Δ adjacency\n(Success − Failure)")
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{diff[i,j]:+.2f}", ha="center", va="center", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Episode length violin
    ax2 = axes[1]
    ls = [e.length for e in succ] or [0]
    lf = [e.length for e in fail] or [0]
    parts = ax2.violinplot([ls, lf], positions=[1, 2], showmedians=True)
    for i, c in enumerate([PALETTE["success"], PALETTE["failure"]]):
        parts["bodies"][i].set_facecolor(c)
        parts["bodies"][i].set_alpha(0.6)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Success", "Failure"])
    ax2.set_ylabel("Episode length (steps)")
    ax2.set_title("Episode Length Distribution")

    # Mean comm density violin
    ax3 = axes[2]
    ds = [e.mean_comm_density for e in succ if not np.isnan(e.mean_comm_density)] or [
        0.0
    ]
    df = [e.mean_comm_density for e in fail if not np.isnan(e.mean_comm_density)] or [
        0.0
    ]
    parts2 = ax3.violinplot([ds, df], positions=[1, 2], showmedians=True)
    for i, c in enumerate([PALETTE["success"], PALETTE["failure"]]):
        parts2["bodies"][i].set_facecolor(c)
        parts2["bodies"][i].set_alpha(0.6)
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(["Success", "Failure"])
    ax3.set_ylabel("Mean comm density")
    ax3.set_title("Communication Density\nby Outcome")

    fig.tight_layout()
    _save(fig, out_dir, f"{prefix}_success_vs_comm")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 13 – Comm density histogram
# ──────────────────────────────────────────────────────────────────────────────


def fig_comm_density_hist(data: MAGICAnalysisData, out_dir: Path, prefix: str) -> None:
    csteps = data.comm_steps()
    if not csteps:
        return

    dens = np.array([s.comm_density for s in csteps])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.suptitle("Distribution of Communication Density (per step)", fontsize=12)

    ax = axes[0]
    lo = float(np.percentile(dens, 1))
    hi = float(np.percentile(dens, 99)) + 1e-6
    ax.hist(
        dens,
        bins="auto",
        range=(lo, hi),
        color=PALETTE["neutral"],
        alpha=0.75,
        edgecolor="white",
    )
    ax.axvline(
        dens.mean(),
        color="black",
        linewidth=1.5,
        linestyle="--",
        label=f"Mean {dens.mean():.2f}",
    )
    ax.set_xlabel("Comm density")
    ax.set_ylabel("Count")
    ax.set_title("Histogram")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    for r in range(data.num_comm_rounds):
        stack = data.adj_stack(r)
        if stack.size == 0:
            continue
        a2b = stack[:, 0, 1]
        b2a = stack[:, 1, 0]
        lo_a = float(np.percentile(a2b, 1))
        hi_a = float(np.percentile(a2b, 99)) + 1e-6
        lo_b = float(np.percentile(b2a, 1))
        hi_b = float(np.percentile(b2a, 99)) + 1e-6
        ax2.hist(
            a2b,
            bins="auto",
            range=(lo_a, hi_a),
            alpha=0.5,
            label=f"A→B R{r+1}",
            color=PALETTE["a"],
            density=True,
        )
        ax2.hist(
            b2a,
            bins="auto",
            range=(lo_b, hi_b),
            alpha=0.5,
            label=f"B→A R{r+1}",
            color=PALETTE["b"],
            density=True,
        )
    ax2.set_xlabel("Edge weight")
    ax2.set_ylabel("Density")
    ax2.set_title("Per-edge Weight Distribution")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, out_dir, f"{prefix}_comm_density_hist")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 14 – Message L2 norm over episode time
# ──────────────────────────────────────────────────────────────────────────────


def fig_message_norm(data: MAGICAnalysisData, out_dir: Path, prefix: str) -> None:
    msgs_a: Dict[int, list] = {}
    msgs_b: Dict[int, list] = {}

    for s in data.comm_steps():
        if s.messages is None:
            continue
        msgs_a.setdefault(s.step, []).append(float(np.linalg.norm(s.messages[0])))
        msgs_b.setdefault(s.step, []).append(float(np.linalg.norm(s.messages[1])))

    if not msgs_a:
        return

    ts = sorted(msgs_a)
    mn_a = [np.mean(msgs_a[t]) for t in ts]
    mn_b = [np.mean(msgs_b.get(t, [0.0])) for t in ts]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(ts, mn_a, color=PALETTE["a"], linewidth=1.5, label="Agent A")
    ax.plot(ts, mn_b, color=PALETTE["b"], linewidth=1.5, label="Agent B")
    ax.fill_between(ts, mn_a, alpha=0.15, color=PALETTE["a"])
    ax.fill_between(ts, mn_b, alpha=0.15, color=PALETTE["b"])
    ax.set_xlabel("Step within episode")
    ax.set_ylabel("Mean message L2 norm")
    ax.set_title(
        "Message Embedding Magnitude Over Episode\n"
        "(Proxy for 'information intensity' of communication)"
    )
    ax.legend(fontsize=9)
    _save(fig, out_dir, f"{prefix}_message_norm")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 15 – Spatial communication heatmap
# ──────────────────────────────────────────────────────────────────────────────


def fig_spatial_comm(data: MAGICAnalysisData, out_dir: Path, prefix: str) -> None:
    G = data.grid_size
    acc_a, cnt_a = np.zeros((G, G)), np.zeros((G, G))
    acc_b, cnt_b = np.zeros((G, G)), np.zeros((G, G))
    trap_acc = np.zeros((G, G))

    for ep in data.episodes:
        for tx, ty in ep.traps:
            trap_acc[ty, tx] += 1
        for s in ep.steps:
            cd = s.comm_density
            for pos, acc, cnt in [
                (s.positions.get(AGENT_A), acc_a, cnt_a),
                (s.positions.get(AGENT_B), acc_b, cnt_b),
            ]:
                if pos:
                    acc[pos[1], pos[0]] += cd
                    cnt[pos[1], pos[0]] += 1

    heat_a = np.where(cnt_a > 0, acc_a / cnt_a, np.nan)
    heat_b = np.where(cnt_b > 0, acc_b / cnt_b, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        "Mean Communication Density by Agent Position\n"
        "(Reveals spatial 'trigger zones' for communication)",
        fontsize=12,
        y=1.02,
    )

    for ax, heat, label in [
        (axes[0], heat_a, "Agent A"),
        (axes[1], heat_b, "Agent B"),
    ]:
        im = ax.imshow(
            heat, cmap="YlOrRd", vmin=0, vmax=1, origin="lower", aspect="equal"
        )
        for ty in range(G):
            for tx in range(G):
                if trap_acc[ty, tx] > 0:
                    ax.add_patch(
                        plt.Circle((tx, ty), 0.3, color="navy", alpha=0.6, zorder=5)
                    )
        ax.set_title(
            f"{label} — comm density at position\n(blue dots = trap locations)"
        )
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        plt.colorbar(im, ax=ax, label="Mean comm density", shrink=0.8)

    fig.tight_layout()
    _save(fig, out_dir, f"{prefix}_spatial_comm")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 16 – Episode gallery
# ──────────────────────────────────────────────────────────────────────────────


def fig_episode_gallery(
    data: MAGICAnalysisData,
    out_dir: Path,
    prefix: str,
    n_show: int = 6,
) -> None:
    eps = [e for e in data.episodes if e.has_comm_data][:n_show]
    if not eps:
        return

    R = data.num_comm_rounds
    last_r = R - 1
    ncols = 2
    nrows = (len(eps) + ncols - 1) // ncols

    fig = plt.figure(figsize=(13, 5 * nrows))
    fig.suptitle(
        "Episode Gallery — Per-step Communication Breakdown", fontsize=12, y=1.01
    )

    for ep_i, ep in enumerate(eps):
        ax = fig.add_subplot(nrows, ncols, ep_i + 1)
        steps = [s.step for s in ep.steps]
        dens = [s.comm_density for s in ep.steps]
        a2b = [
            (
                s.adj_per_round[last_r][0, 1]
                if s.adj_per_round and len(s.adj_per_round) > last_r
                else 0.0
            )
            for s in ep.steps
        ]
        b2a = [
            (
                s.adj_per_round[last_r][1, 0]
                if s.adj_per_round and len(s.adj_per_round) > last_r
                else 0.0
            )
            for s in ep.steps
        ]
        trap_d = [
            min(s.min_trap_dist.get(AGENT_A, 99), s.min_trap_dist.get(AGENT_B, 99))
            for s in ep.steps
        ]

        ax.fill_between(steps, dens, alpha=0.12, color=PALETTE["neutral"])
        ax.plot(steps, dens, color=PALETTE["neutral"], linewidth=1.2, label="Density")
        ax.plot(steps, a2b, color=PALETTE["a"], linewidth=1.5, label="A→B")
        ax.plot(steps, b2a, color=PALETTE["b"], linewidth=1.5, label="B→A")

        td_norm = 1.0 - np.clip(
            np.array(trap_d, float) / max(data.grid_size - 1, 1), 0, 1
        )
        ax.plot(
            steps,
            td_norm,
            color="red",
            linewidth=0.8,
            linestyle=":",
            alpha=0.7,
            label="Trap prox.",
        )

        trap_set = set(ep.traps)
        for s in ep.steps:
            if any(s.positions.get(a) in trap_set for a in AGENTS):
                ax.axvline(s.step, color="red", linewidth=0.5, alpha=0.4)

        outcome = "✓" if ep.success else "✗"
        ax.set_title(f"Ep {ep.episode_idx+1} {outcome}  len={ep.length}", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Step")
        ax.set_ylabel("Weight")
        if ep_i == 0:
            ax.legend(fontsize=6, loc="upper right", ncol=2)

    fig.tight_layout()
    _save(fig, out_dir, f"{prefix}_episode_gallery")


# ──────────────────────────────────────────────────────────────────────────────
# Master entry point
# ──────────────────────────────────────────────────────────────────────────────


def save_all_magic_figures(
    data: MAGICAnalysisData,
    output_dir: str = "eval_plots/magic",
    prefix: str = "magic",
) -> None:
    """
    Generate and save all MAGIC communication analysis figures.

    Parameters
    ----------
    data       : MAGICAnalysisData returned by MAGICCommCollector.collect()
    output_dir : directory for output files
    prefix     : filename prefix for all figures
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_summary(data)

    figure_fns = [
        ("Adjacency heatmaps", fig_adj_heatmap),
        ("Communication trajectory", fig_comm_trajectory),
        ("Directional asymmetry", fig_directional_bar),
        ("Comm vs trap distance", fig_comm_vs_trap_dist),
        ("Comm vs goal distance", fig_comm_vs_goal_dist),
        ("Conditional adjacency", fig_conditional_adj),
        ("Event-aligned comm", fig_event_aligned),
        ("Message PCA", fig_message_pca),
        ("Message t-SNE", fig_message_tsne),
        ("Round comparison", fig_round_comparison),
        ("Policy entropy vs comm", fig_policy_entropy),
        ("Success vs failure comm", fig_success_vs_comm),
        ("Comm density histogram", fig_comm_density_hist),
        ("Message L2 norm", fig_message_norm),
        ("Spatial communication heatmap", fig_spatial_comm),
        ("Episode gallery", fig_episode_gallery),
    ]

    for label, fn in figure_fns:
        print(f"  Generating: {label} …", end=" ", flush=True)
        try:
            fn(data, out_dir, prefix)
            print("done")
        except Exception as exc:
            print(f"SKIPPED ({type(exc).__name__}: {exc})")

    print(f"\n  All figures saved to: {out_dir.resolve()}")
