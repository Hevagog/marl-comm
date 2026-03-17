from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .magic_sa_comm_analysis import (
    MAGICSAAnalysisData,
    compute_mean_adj,
    compute_directional_asymmetry,
    compute_role_comm_over_time,
    compute_comm_by_goal_pressure,
    compute_message_norms,
)

matplotlib.use("Agg")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)

_TICK = ["Adv", "Ag0", "Ag1"]
CMAP_COMM = "Blues"
PALETTE = {
    "adv": "#C0392B",
    "ag0": "#2471A3",
    "ag1": "#27AE60",
    "neutral": "#8172B2",
}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
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
                fontsize=10,
                color="white" if mat[i, j] > 0.5 * vmax else "black",
            )
    cax = make_axes_locatable(ax).append_axes("right", size="8%", pad=0.05)
    plt.colorbar(im, cax=cax)


def fig_adj_heatmap(data: MAGICSAAnalysisData, out_dir: Path, prefix: str) -> None:
    R = data.num_comm_rounds
    fig, axes = plt.subplots(1, R, figsize=(5 * R, 4.2), squeeze=False)
    fig.suptitle(
        "MAGIC Scheduler — Mean Soft Adjacency Matrix (Simple Adversary)",
        fontsize=13,
        y=1.02,
    )
    for r in range(R):
        _adj_heatmap(axes[0, r], compute_mean_adj(data, r), f"Round {r + 1}")
    fig.tight_layout()
    _save(fig, out_dir, f"{prefix}_adj_heatmap")


def fig_comm_trajectory(data: MAGICSAAnalysisData, out_dir: Path, prefix: str) -> None:
    episodes = [e for e in data.episodes if e.has_comm_data]
    if not episodes:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"hspace": 0.32})
    ax_top, ax_bot = axes
    for ep in episodes[:24]:
        xs = [s.step for s in ep.steps if s.adj_per_round is not None]
        ys = [s.comm_density for s in ep.steps if s.adj_per_round is not None]
        if xs:
            ax_top.plot(xs, ys, alpha=0.25, color=PALETTE["neutral"], linewidth=0.9)

    ax_top.set_title("Per-Episode Communication Density")
    ax_top.set_xlabel("Environment Step")
    ax_top.set_ylabel("Comm Density")
    ax_top.set_ylim(-0.05, 1.05)

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
        ax_bot.plot(
            t, mean_d, color=PALETTE["neutral"], linewidth=2, label="Mean Density"
        )
        ax_bot.fill_between(
            t,
            mean_d - std_d,
            mean_d + std_d,
            alpha=0.25,
            color=PALETTE["neutral"],
            label="±1 SD",
        )
        ax_bot.set_xlabel("Environment Step")
        ax_bot.set_ylabel("Comm Density")
        ax_bot.set_title("Mean Communication Density Across Episodes")
        ax_bot.set_ylim(-0.05, 1.05)
        ax_bot.legend()

    _save(fig, out_dir, f"{prefix}_comm_trajectory")


def fig_directional_bar(data: MAGICSAAnalysisData, out_dir: Path, prefix: str) -> None:
    rounds = data.num_comm_rounds
    x = np.arange(rounds)
    width = 0.18

    adv_to_good = []
    good_to_adv = []
    good_internal = []
    self_loops = []

    for r in range(rounds):
        asym = compute_directional_asymmetry(data, r)
        adv_to_good.append(float(np.mean([asym["adv_to_ag0"], asym["adv_to_ag1"]])))
        good_to_adv.append(float(np.mean([asym["ag0_to_adv"], asym["ag1_to_adv"]])))
        good_internal.append(float(np.mean([asym["ag0_to_ag1"], asym["ag1_to_ag0"]])))
        self_loops.append(
            float(np.mean([asym["self_adv"], asym["self_ag0"], asym["self_ag1"]]))
        )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(
        x - 1.5 * width,
        adv_to_good,
        width=width,
        color=PALETTE["adv"],
        label="Adversary → Good",
    )
    ax.bar(
        x - 0.5 * width,
        good_to_adv,
        width=width,
        color=PALETTE["ag0"],
        label="Good → Adversary",
    )
    ax.bar(
        x + 0.5 * width,
        good_internal,
        width=width,
        color=PALETTE["ag1"],
        label="Good ↔ Good",
    )
    ax.bar(
        x + 1.5 * width,
        self_loops,
        width=width,
        color=PALETTE["neutral"],
        alpha=0.75,
        label="Self-loops",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Round {r + 1}" for r in range(rounds)])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean soft adjacency weight")
    ax.set_title("Directional Communication by Role")
    ax.legend(fontsize=9)
    _save(fig, out_dir, f"{prefix}_directional_roles")


def fig_role_comm_over_time(
    data: MAGICSAAnalysisData, out_dir: Path, prefix: str
) -> None:
    trace = compute_role_comm_over_time(data)
    if trace["steps"].size == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        trace["steps"],
        trace["adv_out"],
        color=PALETTE["adv"],
        linewidth=2,
        label="Adversary outgoing",
    )
    ax.plot(
        trace["steps"],
        trace["good_out"],
        color=PALETTE["ag0"],
        linewidth=2,
        label="Good outgoing",
    )
    ax.plot(
        trace["steps"],
        trace["cross_team"],
        color=PALETTE["neutral"],
        linewidth=2,
        label="Cross-team links",
    )
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Mean adjacency weight")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Role-Level Communication Over Episode Time")
    ax.legend(fontsize=9)
    _save(fig, out_dir, f"{prefix}_role_comm_timeline")


def fig_comm_vs_goal_pressure(
    data: MAGICSAAnalysisData, out_dir: Path, prefix: str
) -> None:
    centres, means, stds = compute_comm_by_goal_pressure(data)
    if centres.size == 0:
        return

    valid = np.isfinite(means)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    width = centres[1] - centres[0] if centres.size > 1 else 0.5
    ax.bar(
        centres[valid], means[valid], width=width, alpha=0.75, color=PALETTE["neutral"]
    )
    ax.errorbar(
        centres[valid],
        means[valid],
        yerr=stds[valid],
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Goal pressure = dist(adversary, goal) - min dist(good, goal)")
    ax.set_ylabel("Mean communication density")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Communication vs Tactical Goal Pressure")
    _save(fig, out_dir, f"{prefix}_comm_vs_goal_pressure")


def fig_message_norms(data: MAGICSAAnalysisData, out_dir: Path, prefix: str) -> None:
    norms = compute_message_norms(data)
    if norms["adv"].size == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    ax.hist(norms["adv"], bins=24, alpha=0.65, color=PALETTE["adv"], label="Adversary")
    ax.hist(norms["ag0"], bins=24, alpha=0.55, color=PALETTE["ag0"], label="Agent 0")
    ax.hist(norms["ag1"], bins=24, alpha=0.55, color=PALETTE["ag1"], label="Agent 1")
    ax.set_xlabel("L2 norm of raw message vector")
    ax.set_ylabel("Count")
    ax.set_title("Message Magnitude Distribution")
    ax.legend(fontsize=9)

    ax = axes[1]
    means = [
        float(np.mean(norms["adv"])),
        float(np.mean(norms["ag0"])),
        float(np.mean(norms["ag1"])),
    ]
    stds = [
        float(np.std(norms["adv"])),
        float(np.std(norms["ag0"])),
        float(np.std(norms["ag1"])),
    ]
    x = np.arange(3)
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=[PALETTE["adv"], PALETTE["ag0"], PALETTE["ag1"]],
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Adversary", "Agent 0", "Agent 1"])
    ax.set_ylabel("Mean L2 norm")
    ax.set_title("Message Magnitude by Role")

    _save(fig, out_dir, f"{prefix}_message_norms")


def fig_episode_gallery(
    data: MAGICSAAnalysisData, out_dir: Path, prefix: str, n_episodes: int = 6
) -> None:
    episodes = [ep for ep in data.episodes if ep.steps][:n_episodes]
    if not episodes:
        return

    cols = 3
    rows = int(np.ceil(len(episodes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
    fig.suptitle("Sample Episode Communication Timelines", fontsize=13, y=1.01)

    for idx, ep in enumerate(episodes):
        ax = axes[idx // cols, idx % cols]
        xs = [s.step for s in ep.steps if s.adj_per_round is not None]
        ys = [s.comm_density for s in ep.steps if s.adj_per_round is not None]
        rew = [sum(s.rewards.values()) for s in ep.steps]
        if xs:
            ax.plot(
                xs, ys, color=PALETTE["neutral"], linewidth=1.8, label="Comm density"
            )
        if rew:
            rew_arr = np.asarray(rew, dtype=np.float32)
            rew_norm = (rew_arr - rew_arr.min()) / (rew_arr.ptp() + 1e-8)
            ax.plot(
                np.arange(len(rew_norm)),
                rew_norm,
                color="black",
                linewidth=1.1,
                alpha=0.7,
                label="Norm. team reward",
            )
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Ep {ep.episode_idx + 1} | len={ep.length}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)

    total_axes = rows * cols
    for idx in range(len(episodes), total_axes):
        axes[idx // cols, idx % cols].axis("off")

    _save(fig, out_dir, f"{prefix}_episode_gallery")


def save_all_magic_sa_figures(
    data: MAGICSAAnalysisData,
    output_dir: str | Path,
    prefix: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 55}")
    print(f" Generating MAGIC simple_adversary plots → {out}")
    print(f"{'=' * 55}")

    fig_adj_heatmap(data, out, prefix)
    fig_comm_trajectory(data, out, prefix)
    fig_directional_bar(data, out, prefix)
    fig_role_comm_over_time(data, out, prefix)
    fig_comm_vs_goal_pressure(data, out, prefix)
    fig_message_norms(data, out, prefix)
    fig_episode_gallery(data, out, prefix)

    print(f"All simple_adversary communication plots saved to {out}/")
