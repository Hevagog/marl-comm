"""
mamhm_visualizer.py
====================
Rich visualizations for MAMHM's Hopfield Memory Bank explainability.

Figures produced
----------------
 1.  fig_memory_utilization     — Bar chart of mean attention per memory slot
 2.  fig_agent_memory_profiles  — Per-agent attention profiles over memory slots
 3.  fig_memory_heatmap         — (N x K) heatmap of agent-memory attention
 4.  fig_attention_dynamics     — Attention evolution over episode timesteps
 5.  fig_attention_entropy      — Attention entropy over time (concentration)
 6.  fig_xi_similarity          — Cosine similarity between memory prototypes
 7.  fig_xi_pca                 — 2D PCA of memory prototypes
 8.  fig_reward_memory_corr     — Memory slots associated with high/low reward
 9.  fig_gate_info              — Gate value and effective memory contribution
10.  fig_reward_distribution    — Episode reward histogram
11.  fig_episode_lengths        — Episode length histogram
12.  fig_reward_trajectory      — Per-episode cumulative reward curves
13.  fig_summary_table          — Text summary of key metrics

All sub-figures saved as both PDF and PNG to *output_dir*.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .hopfield_analysis import (
    MAMHMAnalysisData,
    compute_memory_utilization,
    compute_agent_memory_profiles,
    compute_memory_attention_dynamics,
    compute_xi_similarity_matrix,
    compute_xi_pca,
    compute_memory_reward_correlation,
    compute_attention_concentration,
    print_summary,
)

from utils.shared.style import _AGENT_COLORS, CMAP_ATTN, CMAP_SIM, save_figure

CMAP_DIFF = "coolwarm"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    save_figure(fig, out_dir, name)


def _agent_labels(n: int) -> list[str]:
    return [f"A{i}" for i in range(n)]


# ──────────────────────────────────────────────────────────────────────────────
# Individual figure generators
# ──────────────────────────────────────────────────────────────────────────────


def fig_memory_utilization(data: MAMHMAnalysisData, prefix: str) -> plt.Figure | None:
    """Bar chart showing mean attention per memory slot (sorted descending)."""
    util = compute_memory_utilization(data)
    mean_attn = util["mean_attention"]
    if mean_attn.sum() == 0:
        return None

    K = len(mean_attn)
    sorted_idx = np.argsort(mean_attn)[::-1]
    sorted_attn = mean_attn[sorted_idx]

    fig, ax = plt.subplots(figsize=(max(8, K * 0.15), 4))
    colors = np.where(sorted_attn > 1.0 / K, "#4C72B0", "#C0C0C0")
    ax.bar(range(K), sorted_attn, color=colors, edgecolor="none", width=0.8)
    ax.axhline(
        1.0 / K,
        color="#C44E52",
        linestyle="--",
        linewidth=1,
        label=f"Uniform = 1/K = {1.0 / K:.4f}",
    )
    ax.set_xlabel("Memory Slot (sorted by attention)")
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title(
        f"Hopfield Memory Utilization — {util['active_memories']}/{K} active\n({prefix})"
    )
    ax.legend(fontsize=9)

    if K <= 32:
        ax.set_xticks(range(K))
        ax.set_xticklabels(
            [str(sorted_idx[i]) for i in range(K)], fontsize=7, rotation=90
        )

    fig.tight_layout()
    return fig


def fig_agent_memory_profiles(
    data: MAMHMAnalysisData, prefix: str
) -> plt.Figure | None:
    """Per-agent attention profiles overlaid on the same axes."""
    profiles = compute_agent_memory_profiles(data)
    if not profiles or all(p.sum() == 0 for p in profiles.values()):
        return None

    N = data.num_agents
    K = data.num_memories
    agent_labels = _agent_labels(N)

    fig, ax = plt.subplots(figsize=(max(8, K * 0.15), 4))
    x = np.arange(K)
    width = 0.8 / N

    for i in range(N):
        color = _AGENT_COLORS[i % len(_AGENT_COLORS)]
        ax.bar(
            x + i * width - 0.4 + width / 2,
            profiles[i],
            width,
            label=agent_labels[i],
            color=color,
            alpha=0.75,
        )

    ax.set_xlabel("Memory Slot")
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title(f"Per-Agent Memory Attention Profiles\n({prefix})")
    ax.legend(fontsize=9)

    if K <= 32:
        ax.set_xticks(range(K))

    fig.tight_layout()
    return fig


def fig_memory_heatmap(data: MAMHMAnalysisData, prefix: str) -> plt.Figure | None:
    """Heatmap of (N x K) mean attention weights."""
    profiles = compute_agent_memory_profiles(data)
    if not profiles or all(p.sum() == 0 for p in profiles.values()):
        return None

    N = data.num_agents
    K = data.num_memories
    agent_labels = _agent_labels(N)

    mat = np.stack([profiles[i] for i in range(N)], axis=0)  # (N, K)

    # Only show top-M most active memory slots for readability
    mean_per_slot = mat.mean(axis=0)
    M = min(K, 32)
    top_slots = np.argsort(mean_per_slot)[::-1][:M]
    mat_top = mat[:, top_slots]

    fig, ax = plt.subplots(figsize=(max(6, M * 0.3), max(3, N * 0.6)))
    im = ax.imshow(mat_top, cmap=CMAP_ATTN, aspect="auto")
    ax.set_yticks(range(N))
    ax.set_yticklabels(agent_labels)
    ax.set_xlabel(f"Memory Slot (top {M} by mean attention)")
    ax.set_ylabel("Agent")
    ax.set_title(f"Agent-Memory Attention Heatmap\n({prefix})")

    if M <= 32:
        ax.set_xticks(range(M))
        ax.set_xticklabels([str(s) for s in top_slots], fontsize=7, rotation=90)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention")

    # Annotate cells if small enough
    if M <= 16 and N <= 8:
        for i in range(N):
            for j in range(M):
                val = mat_top[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if val > mat_top.max() * 0.6 else "black",
                )

    fig.tight_layout()
    return fig


def fig_attention_dynamics(data: MAMHMAnalysisData, prefix: str) -> plt.Figure | None:
    """Attention evolution over timesteps for first episode — top-K memory slots."""
    attn_dyn = compute_memory_attention_dynamics(data, episode_idx=0)
    if attn_dyn is None:
        return None

    T, N, K = attn_dyn.shape
    agent_labels = _agent_labels(N)

    # Find top-8 most active memory slots across the episode
    mean_per_slot = attn_dyn.mean(axis=(0, 1))
    n_show = min(8, K)
    top_slots = np.argsort(mean_per_slot)[::-1][:n_show]

    fig, axes = plt.subplots(N, 1, figsize=(10, 2.5 * N), sharex=True)
    if N == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        for rank, slot_idx in enumerate(top_slots):
            alpha_val = 0.9 - rank * 0.08
            ax.plot(
                range(T),
                attn_dyn[:, i, slot_idx],
                label=f"Slot {slot_idx}",
                alpha=max(alpha_val, 0.3),
                linewidth=1.2,
            )
        ax.set_ylabel(f"{agent_labels[i]}\nAttention")
        ax.set_ylim(0, None)
        if i == 0:
            ax.legend(fontsize=7, ncol=n_show, loc="upper right")

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(
        f"Hopfield Memory Attention Dynamics (Episode 0)\n({prefix})",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def fig_attention_entropy(data: MAMHMAnalysisData, prefix: str) -> plt.Figure | None:
    """Attention entropy over time — measures concentration vs. diffusion."""
    util = compute_memory_utilization(data)
    entropy = util["entropy_per_step"]
    if len(entropy) == 0:
        return None

    K = data.num_memories
    max_entropy = np.log(K)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(entropy, color="#4C72B0", alpha=0.6, linewidth=0.8)

    # Smoothed line
    if len(entropy) > 20:
        window = max(5, len(entropy) // 50)
        smoothed = np.convolve(entropy, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window // 2, window // 2 + len(smoothed)),
            smoothed,
            color="#C44E52",
            linewidth=2,
            label="Smoothed",
        )

    ax.axhline(
        max_entropy,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Uniform H = ln({K}) = {max_entropy:.2f}",
    )
    ax.set_xlabel("Step (across all episodes)")
    ax.set_ylabel("Attention Entropy (nats)")
    ax.set_title(f"Hopfield Attention Entropy Over Time\n({prefix})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_xi_similarity(data: MAMHMAnalysisData, prefix: str) -> plt.Figure | None:
    """Cosine similarity matrix between learned memory prototypes."""
    sim = compute_xi_similarity_matrix(data)
    if sim is None:
        return None

    K = sim.shape[0]
    # For large K, downsample for readability
    if K > 32:
        # Show only the top-32 most distinctive (lowest mean off-diagonal similarity)
        off_diag_mean = (sim.sum(axis=1) - 1) / (K - 1)
        show_idx = np.argsort(off_diag_mean)[:32]
        sim = sim[np.ix_(show_idx, show_idx)]
        K_show = 32
        title_extra = f" (top {K_show} most distinctive)"
    else:
        show_idx = np.arange(K)
        K_show = K
        title_extra = ""

    fig, ax = plt.subplots(figsize=(max(5, K_show * 0.25), max(5, K_show * 0.25)))
    im = ax.imshow(sim, cmap=CMAP_SIM, vmin=-1, vmax=1, aspect="equal")
    ax.set_xlabel("Memory Prototype")
    ax.set_ylabel("Memory Prototype")
    ax.set_title(f"Memory Prototype Cosine Similarity{title_extra}\n({prefix})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if K_show <= 20:
        ax.set_xticks(range(K_show))
        ax.set_yticks(range(K_show))
        ax.set_xticklabels([str(s) for s in show_idx], fontsize=7, rotation=90)
        ax.set_yticklabels([str(s) for s in show_idx], fontsize=7)

    fig.tight_layout()
    return fig


def fig_xi_pca(data: MAMHMAnalysisData, prefix: str) -> plt.Figure | None:
    """2D PCA of learned memory prototypes, colored by utilization."""
    projected, evr = compute_xi_pca(data)
    if projected is None:
        return None

    util = compute_memory_utilization(data)
    mean_attn = util["mean_attention"]

    K = projected.shape[0]
    fig, ax = plt.subplots(figsize=(6, 5))

    # Color by mean attention (utilization)
    scatter = ax.scatter(
        projected[:, 0],
        projected[:, 1],
        c=mean_attn,
        cmap="YlOrRd",
        s=40,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="Mean Attention")

    # Label top-5 most attended
    top5 = np.argsort(mean_attn)[::-1][:5]
    for idx in top5:
        ax.annotate(
            str(idx),
            (projected[idx, 0], projected[idx, 1]),
            fontsize=8,
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_xlabel(f"PC1 ({evr[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%} var)")
    ax.set_title(f"Memory Prototypes PCA (K={K})\n({prefix})")
    fig.tight_layout()
    return fig


def fig_reward_memory_corr(data: MAMHMAnalysisData, prefix: str) -> plt.Figure | None:
    """Memory slots associated with high vs. low reward episodes."""
    corr = compute_memory_reward_correlation(data)
    diff = corr["diff_profile"]
    high = corr["high_reward_profile"]
    low = corr["low_reward_profile"]

    if diff.sum() == 0 and high.sum() == 0:
        return None

    K = len(diff)
    # Sort by absolute difference
    sorted_idx = np.argsort(np.abs(diff))[::-1]
    n_show = min(20, K)
    top_idx = sorted_idx[:n_show]

    fig, axes = plt.subplots(2, 1, figsize=(max(8, n_show * 0.4), 7))

    # Top: High vs Low reward attention profiles
    ax = axes[0]
    x = np.arange(n_show)
    width = 0.35
    ax.bar(
        x - width / 2,
        high[top_idx],
        width,
        label="High reward (Q4)",
        color="#55A868",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        low[top_idx],
        width,
        label="Low reward (Q1)",
        color="#C44E52",
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in top_idx], fontsize=7, rotation=90)
    ax.set_ylabel("Mean Attention")
    ax.set_title(f"Memory Attention: High vs Low Reward Episodes\n({prefix})")
    ax.legend(fontsize=9)

    # Bottom: Difference profile
    ax = axes[1]
    colors = np.where(diff[top_idx] > 0, "#55A868", "#C44E52")
    ax.bar(x, diff[top_idx], color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in top_idx], fontsize=7, rotation=90)
    ax.set_xlabel("Memory Slot")
    ax.set_ylabel("Attention Difference (High - Low)")
    ax.set_title("Memory Slots Predictive of Reward")
    ax.axhline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    return fig


def fig_gate_info(data: MAMHMAnalysisData, prefix: str) -> plt.Figure | None:
    """Visualization of the learned gate value and memory contribution scaling."""
    if data.gate_logit is None:
        return None

    gate_logit = data.gate_logit
    gate_val = data.gate_value
    gamma = 0.1  # default gamma from config

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Gate sigmoid curve with current value marked
    ax = axes[0]
    logits = np.linspace(-6, 6, 200)
    sigmoid = 1.0 / (1.0 + np.exp(-logits))
    ax.plot(logits, sigmoid, color="#4C72B0", linewidth=2)
    ax.axvline(
        gate_logit,
        color="#C44E52",
        linestyle="--",
        linewidth=1.5,
        label=f"Current logit = {gate_logit:.3f}",
    )
    ax.axhline(gate_val, color="#C44E52", linestyle=":", linewidth=1, alpha=0.5)
    ax.plot(gate_logit, gate_val, "o", color="#C44E52", markersize=10, zorder=5)
    ax.axvline(
        -3.0,
        color="gray",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
        label="Init logit = -3.0",
    )
    ax.set_xlabel("Gate Logit")
    ax.set_ylabel("Gate Value (sigmoid)")
    ax.set_title(f"Learned Gate: {gate_val:.4f}")
    ax.legend(fontsize=9)

    # Right: Effective memory contribution breakdown
    ax = axes[1]
    init_gate = 1.0 / (1.0 + np.exp(3.0))  # sigmoid(-3)
    init_contrib = init_gate * gamma
    current_contrib = gate_val * gamma

    bars = ax.bar(
        ["Init\n(sigmoid(-3))", "Learned\n(current)"],
        [init_contrib * 100, current_contrib * 100],
        color=["#C0C0C0", "#4C72B0"],
        width=0.5,
    )
    for bar, val in zip(bars, [init_contrib, current_contrib]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Effective Memory Scale (gate * gamma) [%]")
    ax.set_title(f"Memory Contribution: gate * gamma = {current_contrib:.4f}")
    ax.set_ylim(0, max(current_contrib * 100, init_contrib * 100) * 1.5 + 1)

    fig.suptitle(f"Hopfield Memory Gate Analysis\n({prefix})", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def fig_reward_distribution(data: MAMHMAnalysisData, prefix: str) -> plt.Figure | None:
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


def fig_episode_lengths(data: MAMHMAnalysisData, prefix: str) -> plt.Figure | None:
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


def fig_reward_trajectory(data: MAMHMAnalysisData, prefix: str) -> plt.Figure | None:
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


def fig_summary_table(data: MAMHMAnalysisData, prefix: str) -> plt.Figure:
    """Text figure summarising key metrics."""
    util = compute_memory_utilization(data)
    conc = compute_attention_concentration(data)

    rows = [
        ("Episodes", str(data.n_episodes)),
        ("Success rate", f"{data.success_rate:.1%}"),
        ("Mean length", f"{data.mean_episode_length:.1f}"),
        ("Mean reward", f"{data.mean_episode_reward:+.3f}"),
        ("Num agents", str(data.num_agents)),
        ("d_model", str(data.d_model)),
        ("Num memories K", str(data.num_memories)),
        (
            "Active memories",
            f"{util['active_memories']}/{data.num_memories} ({util['utilization_ratio']:.1%})",
        ),
        (
            "Effective K",
            f"{conc['effective_k_overall']:.1f} ({conc['concentration_ratio']:.1%} of uniform)",
        ),
        (
            "Gate logit",
            f"{data.gate_logit:.3f}" if data.gate_logit is not None else "N/A",
        ),
        (
            "Gate value",
            f"{data.gate_value:.4f}" if data.gate_value is not None else "N/A",
        ),
        ("Memory data", "Yes" if data.has_memory_data else "No"),
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
    ax.set_title(f"MAMHM Analysis Summary\n({prefix})", pad=10)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────


def save_all_mamhm_figures(
    data: MAMHMAnalysisData,
    output_dir: str = "eval_plots/mamhm",
    prefix: str = "mamhm",
) -> None:
    """Generate and save all MAMHM Hopfield memory analysis figures.

    Parameters
    ----------
    data       : result from MAMHMCollector.collect().
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
            print(f"  [MAMHMViz] Skipped {name}: {exc}")

    _try_save(lambda: fig_memory_utilization(data, prefix), "memory_utilization")
    _try_save(lambda: fig_agent_memory_profiles(data, prefix), "agent_memory_profiles")
    _try_save(lambda: fig_memory_heatmap(data, prefix), "memory_heatmap")
    _try_save(lambda: fig_attention_dynamics(data, prefix), "attention_dynamics")
    _try_save(lambda: fig_attention_entropy(data, prefix), "attention_entropy")
    _try_save(lambda: fig_xi_similarity(data, prefix), "xi_similarity")
    _try_save(lambda: fig_xi_pca(data, prefix), "xi_pca")
    _try_save(lambda: fig_reward_memory_corr(data, prefix), "reward_memory_correlation")
    _try_save(lambda: fig_gate_info(data, prefix), "gate_info")
    _try_save(lambda: fig_reward_distribution(data, prefix), "reward_distribution")
    _try_save(lambda: fig_episode_lengths(data, prefix), "episode_lengths")
    _try_save(lambda: fig_reward_trajectory(data, prefix), "reward_trajectory")
    _try_save(lambda: fig_summary_table(data, prefix), "summary_table")

    print(f"\n  Figures saved to: {out}/")
