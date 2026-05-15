"""Figures for the CoinGame altruism-persistence experiment."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from utils.mappo.altruism_analysis import save_summary_csv


_PALETTE = {
    "HSC Phase 1 (altruistic)": "#455A64",
    "HSC frozen_proto": "#1976D2",
    "HSC full_finetune": "#E64A19",
    "GRU full_finetune": "#388E3C",
    "MAPPO": "#7B1FA2",
}
_DEFAULT_COLOR = "#607D8B"
_DPI = 150


def _rolling_mean(arr: np.ndarray, window: int = 5) -> np.ndarray:
    if len(arr) < window:
        return arr
    valid = np.nan_to_num(arr, nan=0.0)
    out = np.convolve(valid, np.ones(window) / window, mode="valid")
    pad = np.full(len(arr) - len(out), np.nan)
    return np.concatenate([pad, out])


def _mean_agent_series(series: dict[str, np.ndarray]) -> np.ndarray:
    if not series:
        return np.array([], dtype=float)
    stacked = np.stack(list(series.values()), axis=0).astype(float)
    valid = ~np.isnan(stacked)
    counts = valid.sum(axis=0)
    sums = np.where(valid, stacked, 0.0).sum(axis=0)
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def _save_both(fig, base_path: str) -> None:
    os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
    fig.savefig(base_path + ".pdf", dpi=_DPI, bbox_inches="tight")
    fig.savefig(base_path + ".png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def _color(label: str) -> str:
    return _PALETTE.get(label, _DEFAULT_COLOR)


def plot_cooperation_rate(
    data_dict: dict[str, Any],
    output_path: str,
    window: int = 5,
    title: str = "Cooperation Persistence after Reward Shift",
) -> None:
    if not _HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, data in data_dict.items():
        y = _mean_agent_series(data.cooperation_rates)
        if y.size == 0:
            continue
        x = np.arange(len(y))
        ax.plot(x, y, alpha=0.18, color=_color(label), linewidth=0.8)
        ax.plot(
            x, _rolling_mean(y, window), label=label, color=_color(label), linewidth=2.0
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Evaluation Episode")
    ax.set_ylabel("Own-Coin Fraction")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    _save_both(fig, output_path)


def plot_steal_and_victimization(
    data_dict: dict[str, Any],
    output_path: str,
    window: int = 5,
) -> None:
    if not _HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=False)
    for label, data in data_dict.items():
        steal_rate = _mean_agent_series(data.steal_rates)
        victim_counts = _mean_agent_series(data.victimization_counts)
        if steal_rate.size:
            x = np.arange(len(steal_rate))
            axes[0].plot(
                x,
                _rolling_mean(steal_rate, window),
                label=label,
                color=_color(label),
                linewidth=2.0,
            )
        if victim_counts.size:
            x = np.arange(len(victim_counts))
            axes[1].plot(
                x,
                _rolling_mean(victim_counts, window),
                label=label,
                color=_color(label),
                linewidth=2.0,
            )
    axes[0].set_title("(A) Steal Rate", fontweight="bold")
    axes[0].set_ylabel("Steal Fraction")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_title("(B) Victimizations", fontweight="bold")
    axes[1].set_ylabel("Steals Suffered per Agent")
    for ax in axes:
        ax.set_xlabel("Evaluation Episode")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=9)
    _save_both(fig, output_path)


def plot_reward_decomposition(data_dict: dict[str, Any], output_path: str) -> None:
    if not _HAS_MPL:
        return
    labels = list(data_dict.keys())
    own = []
    steal_gain = []
    penalty = []
    social = []
    for data in data_dict.values():
        comps = data.reward_component_series()
        own.append(float(np.nanmean(comps["own_coin"])))
        steal_gain.append(float(np.nanmean(comps["steal_gain"])))
        penalty.append(float(np.nanmean(comps["steal_penalty_received"])))
        social.append(float(np.nanmean(comps["social_welfare_transfer"])))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.55
    ax.bar(x, own, width, label="Own coin", color="#2E7D32", alpha=0.88)
    ax.bar(
        x,
        steal_gain,
        width,
        bottom=own,
        label="Steal gain",
        color="#EF6C00",
        alpha=0.88,
    )
    ax.bar(x, penalty, width, label="Penalty suffered", color="#C62828", alpha=0.88)
    ax.scatter(
        x, social, label="Reward mixing transfer", color="#1565C0", marker="D", zorder=3
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Mean Reward per Agent per Episode")
    ax.set_title("Exact Reward Decomposition", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    _save_both(fig, output_path)


def plot_welfare_inequality(
    data_dict: dict[str, Any], output_path: str, window: int = 5
) -> None:
    if not _HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, data in data_dict.items():
        if data.welfare.size:
            x = np.arange(len(data.welfare))
            axes[0].plot(
                x,
                _rolling_mean(data.welfare, window),
                label=label,
                color=_color(label),
                linewidth=2.0,
            )
        if data.reward_inequality.size:
            x = np.arange(len(data.reward_inequality))
            axes[1].plot(
                x,
                _rolling_mean(data.reward_inequality, window),
                label=label,
                color=_color(label),
                linewidth=2.0,
            )
    axes[0].set_title("(A) Team Welfare", fontweight="bold")
    axes[0].set_ylabel("Sum of Agent Rewards")
    axes[1].set_title("(B) Reward Inequality", fontweight="bold")
    axes[1].set_ylabel("|Return Red - Return Blue|")
    for ax in axes:
        ax.set_xlabel("Evaluation Episode")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=9)
    _save_both(fig, output_path)


def plot_prototype_diagnostics(
    data_dict: dict[str, Any], output_path: str, window: int = 3
) -> None:
    if not _HAS_MPL:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    any_plotted = False
    for label, data in data_dict.items():
        drift = np.array(
            [x for x in data.prototype_cosine_drift if x is not None], dtype=float
        )
        entropy = np.array(
            [x for x in data.prototype_entropy if x is not None], dtype=float
        )
        max_prob = np.array(
            [x for x in data.prototype_max_prob if x is not None], dtype=float
        )
        if drift.size:
            axes[0].plot(
                np.arange(len(drift)),
                _rolling_mean(drift, window),
                label=label,
                color=_color(label),
                linewidth=2.0,
            )
            any_plotted = True
        if entropy.size:
            axes[1].plot(
                np.arange(len(entropy)),
                _rolling_mean(entropy, window),
                label=label,
                color=_color(label),
                linewidth=2.0,
            )
            any_plotted = True
        if max_prob.size:
            axes[2].plot(
                np.arange(len(max_prob)),
                _rolling_mean(max_prob, window),
                label=label,
                color=_color(label),
                linewidth=2.0,
            )
            any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return
    axes[0].set_title("(A) Prototype Drift", fontweight="bold")
    axes[0].set_ylabel("1 - Cosine Similarity")
    axes[1].set_title("(B) Attention Entropy", fontweight="bold")
    axes[1].set_ylabel("Entropy")
    axes[2].set_title("(C) Attention Peakiness", fontweight="bold")
    axes[2].set_ylabel("Mean Max Probability")
    for ax in axes:
        ax.set_xlabel("Evaluation Episode")
        ax.grid(True, alpha=0.25)
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
            break
    _save_both(fig, output_path)


def plot_prototype_usage_heatmap(data_dict: dict[str, Any], output_path: str) -> None:
    if not _HAS_MPL:
        return
    rows = []
    labels = []
    for label, data in data_dict.items():
        usage = [u for u in data.prototype_usage if u is not None]
        if not usage:
            continue
        rows.append(np.mean(np.stack(usage, axis=0), axis=0))
        labels.append(label)
    if not rows:
        return
    mat = np.stack(rows, axis=0)
    fig, ax = plt.subplots(
        figsize=(max(8, mat.shape[1] * 0.45), 1.2 + len(labels) * 0.55)
    )
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels([str(i) for i in range(mat.shape[1])])
    ax.set_xlabel("Prototype Index")
    ax.set_title("Mean HSC Prototype Usage", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Mean attention probability")
    _save_both(fig, output_path)


def plot_time_to_defection(data_dict: dict[str, Any], output_path: str) -> None:
    if not _HAS_MPL:
        return
    labels = list(data_dict.keys())
    n_episodes = max((len(d.episodes) for d in data_dict.values()), default=0)
    vals = [
        d.time_to_defection if d.time_to_defection is not None else n_episodes
        for d in data_dict.values()
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(labels))
    for i, (label, val, data) in enumerate(zip(labels, vals, data_dict.values())):
        hatch = "///" if data.time_to_defection is None else ""
        ax.bar(i, val, color=_color(label), edgecolor="white", hatch=hatch, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Episode of First Defection")
    ax.set_title("Time-to-Defection", fontweight="bold")
    ax.set_ylim(0, max(1, n_episodes) * 1.12)
    ax.grid(True, axis="y", alpha=0.25)
    _save_both(fig, output_path)


def plot_panel_summary(
    data_dict: dict[str, Any], output_path: str, window: int = 5
) -> None:
    if not _HAS_MPL:
        return
    fig = plt.figure(figsize=(15, 7))
    gs = GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.38)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    for label, data in data_dict.items():
        coop = _mean_agent_series(data.cooperation_rates)
        steal = _mean_agent_series(data.steal_rates)
        if coop.size:
            x = np.arange(len(coop))
            axes[0].plot(
                x,
                _rolling_mean(coop, window),
                label=label,
                color=_color(label),
                linewidth=2.0,
            )
        if steal.size:
            x = np.arange(len(steal))
            axes[1].plot(
                x,
                _rolling_mean(steal, window),
                label=label,
                color=_color(label),
                linewidth=2.0,
            )
        if data.welfare.size:
            x = np.arange(len(data.welfare))
            axes[2].plot(
                x,
                _rolling_mean(data.welfare, window),
                label=label,
                color=_color(label),
                linewidth=2.0,
            )
        entropy = np.array(
            [x for x in data.prototype_entropy if x is not None], dtype=float
        )
        if entropy.size:
            axes[3].plot(
                np.arange(len(entropy)),
                _rolling_mean(entropy, window),
                label=label,
                color=_color(label),
                linewidth=2.0,
            )

    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    titles = ["(A) Cooperation", "(B) Stealing", "(C) Team Welfare", "(D) HSC Entropy"]
    ylabels = [
        "Own-Coin Fraction",
        "Steal Fraction",
        "Sum of Rewards",
        "Prototype Entropy",
    ]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Evaluation Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=8)
    _save_both(fig, output_path)


def save_all_altruism_figures(
    data_dict: dict[str, Any],
    output_dir: str = "eval_plots/altruism",
    prefix: str = "altruism",
) -> None:
    if not _HAS_MPL:
        print("[altruism_visualizer] matplotlib not available; skipping figures.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"[altruism_visualizer] Saving figures to {output_dir}/")

    plot_cooperation_rate(data_dict, f"{output_dir}/{prefix}_cooperation_rate")
    plot_steal_and_victimization(
        data_dict, f"{output_dir}/{prefix}_steal_victimization"
    )
    plot_reward_decomposition(data_dict, f"{output_dir}/{prefix}_reward_decomp")
    plot_welfare_inequality(data_dict, f"{output_dir}/{prefix}_welfare_inequality")
    plot_prototype_diagnostics(
        data_dict, f"{output_dir}/{prefix}_prototype_diagnostics"
    )
    plot_prototype_usage_heatmap(data_dict, f"{output_dir}/{prefix}_prototype_usage")
    plot_time_to_defection(data_dict, f"{output_dir}/{prefix}_time_to_defection")
    plot_panel_summary(data_dict, f"{output_dir}/{prefix}_panel_summary")
    save_summary_csv(data_dict, f"{output_dir}/{prefix}_summary.csv")

    print("[altruism_visualizer] Done; figures and summary CSV written.")
