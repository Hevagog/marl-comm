from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .simple_adversary_eval_data import (
    ADVERSARY,
    AGENT_0,
    AGENT_1,
    ALL_AGENTS,
    GOOD_AGENTS,
    EvalData,
)

_ADV_COLOR = "#C0392B"  # Red
_AG0_COLOR = "#2471A3"  # Blue
_AG1_COLOR = "#27AE60"  # Green
_GOAL_COLOR = "#F1C40F"  # Yellow
_SPOOF_COLOR = "#95A5A6"  # Grey

_AGENT_COLORS = {ADVERSARY: _ADV_COLOR, AGENT_0: _AG0_COLOR, AGENT_1: _AG1_COLOR}
_AGENT_LABELS = {ADVERSARY: "Adversary", AGENT_0: "Agent 0", AGENT_1: "Agent 1"}

matplotlib.rcParams.update(
    {
        "figure.facecolor": "#F8F9FA",
        "axes.facecolor": "#FFFFFF",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_overview(data: EvalData, save_path: Optional[Path] = None) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    fig.suptitle(
        "Simple Adversary – Evaluation Overview", fontsize=16, fontweight="bold"
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)

    # (0,0) Reward distributions
    ax = fig.add_subplot(gs[0, 0])
    rewards_per_ep = [[ep.total_rewards[a] for ep in data.episodes] for a in ALL_AGENTS]
    x = np.arange(len(ALL_AGENTS))
    parts = ax.violinplot(rewards_per_ep, positions=x, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(_AGENT_COLORS[ALL_AGENTS[i]])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    ax.set_xticks(x)
    ax.set_xticklabels([_AGENT_LABELS[a] for a in ALL_AGENTS])
    ax.set_ylabel("Episode Total Reward")
    ax.set_title("Reward Distribution Summary")

    # (0,1) Mean Distance to Goal Over Time
    ax = fig.add_subplot(gs[0, 1])
    max_len = max((len(ep.steps) for ep in data.episodes), default=0)
    for a in ALL_AGENTS:
        mat = []
        for ep in data.episodes:
            row = [s.dist_to_goal.get(a, np.nan) for s in ep.steps]
            row += [np.nan] * (max_len - len(row))
            mat.append(row)
        mat = np.array(mat)
        mean_dist = np.nanmean(mat, axis=0)
        std_dist = np.nanstd(mat, axis=0)
        steps_x = np.arange(max_len)
        ax.plot(
            steps_x, mean_dist, color=_AGENT_COLORS[a], label=_AGENT_LABELS[a], lw=2
        )
        ax.fill_between(
            steps_x,
            mean_dist - std_dist,
            mean_dist + std_dist,
            color=_AGENT_COLORS[a],
            alpha=0.2,
        )
    ax.set_title("Average Distance to True Goal")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance")
    ax.legend()

    # (1,0) Mean Distance to Spoof Over Time (For Good Agents)
    ax = fig.add_subplot(gs[1, 0])
    for a in GOOD_AGENTS:
        mat = []
        for ep in data.episodes:
            row = [s.dist_to_spoof.get(a, np.nan) for s in ep.steps]
            row += [np.nan] * (max_len - len(row))
            mat.append(row)
        mat = np.array(mat)
        mean_dist = np.nanmean(mat, axis=0)
        std_dist = np.nanstd(mat, axis=0)
        steps_x = np.arange(max_len)
        ax.plot(
            steps_x, mean_dist, color=_AGENT_COLORS[a], label=_AGENT_LABELS[a], lw=2
        )
        ax.fill_between(
            steps_x,
            mean_dist - std_dist,
            mean_dist + std_dist,
            color=_AGENT_COLORS[a],
            alpha=0.2,
        )
    ax.set_title("Average Distance to Spoof Landmark")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance")
    ax.legend()

    # (1,1) Action distribution
    ax = fig.add_subplot(gs[1, 1])
    action_names = ["NONE", "LEFT", "RIGHT", "DOWN", "UP"]  # MPE Discrete actions
    n_actions = len(action_names)
    action_x = np.arange(n_actions)
    bar_w = 0.25
    for i, a in enumerate(ALL_AGENTS):
        all_actions = [s.actions.get(a, 0) for ep in data.episodes for s in ep.steps]
        counts = np.array(
            [all_actions.count(act) for act in range(n_actions)], dtype=float
        )
        counts /= counts.sum() + 1e-9
        ax.bar(
            action_x + (i - 1) * bar_w,
            counts,
            bar_w,
            color=_AGENT_COLORS[a],
            label=_AGENT_LABELS[a],
        )
    ax.set_xticks(action_x)
    ax.set_xticklabels(action_names)
    ax.set_ylabel("Action frequency")
    ax.set_title("Action Distribution")
    ax.legend()

    if save_path:
        _savefig(fig, save_path)
    return fig


def plot_trajectories(
    data: EvalData, n_episodes: int = 6, save_path: Optional[Path] = None
) -> plt.Figure:
    episodes = data.episodes[:n_episodes]
    n = len(episodes)
    if n == 0:
        return plt.figure()

    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 5, rows * 5), constrained_layout=True
    )
    fig.suptitle("Sample Episode Trajectories", fontsize=15, fontweight="bold")

    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])

    for idx, ep in enumerate(episodes):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        # Draw landmarks first
        if ep.steps:
            lms = ep.steps[0].landmarks
            goal_idx = ep.goal_landmark_idx
            for i, lm in enumerate(lms):
                color = _GOAL_COLOR if i == goal_idx else _SPOOF_COLOR
                marker = "*" if i == goal_idx else "o"
                size = 200 if i == goal_idx else 100
                ax.scatter(
                    *lm,
                    color=color,
                    s=size,
                    edgecolors="black",
                    marker=marker,
                    zorder=4,
                )

        for a in ALL_AGENTS:
            path = ep.get_agent_trajectories(a)
            if not path:
                continue
            color = _AGENT_COLORS[a]
            n_steps = len(path)
            for i in range(1, len(path)):
                alpha = 0.15 + 0.8 * (i / n_steps)
                ax.plot(
                    [path[i - 1][0], path[i][0]],
                    [path[i - 1][1], path[i][1]],
                    color=color,
                    alpha=alpha,
                    linewidth=2,
                    zorder=3,
                )
            ax.scatter(*path[0], color=color, s=80, marker="x", zorder=5)  # Start
            ax.scatter(
                *path[-1], color=color, s=80, marker="D", edgecolors="white", zorder=5
            )  # End

        adv_rew = ep.total_rewards.get(ADVERSARY, 0.0)
        ag_rew = (
            ep.total_rewards.get(AGENT_0, 0.0) + ep.total_rewards.get(AGENT_1, 0.0)
        ) / 2
        ax.set_title(
            f"Episode {ep.episode_idx + 1}\nAdv R={adv_rew:+.1f} | Ag R={ag_rew:+.1f}",
            fontsize=10,
        )
        ax.set_aspect("equal")

    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    legend_elements = [
        mpatches.Patch(color=_ADV_COLOR, label="Adversary path"),
        mpatches.Patch(color=_AG0_COLOR, label="Agent 0 path"),
        mpatches.Patch(color=_AG1_COLOR, label="Agent 1 path"),
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=_GOAL_COLOR,
            markersize=12,
            label="Goal Landmark",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_SPOOF_COLOR,
            markersize=8,
            label="Spoof Landmark",
        ),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=10)

    if save_path:
        _savefig(fig, save_path)
    return fig


def plot_role_diagnostics(
    data: EvalData, save_path: Optional[Path] = None
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Simple Adversary – Role Diagnostics", fontsize=14, fontweight="bold")

    ax = axes[0]
    max_len = max((len(ep.steps) for ep in data.episodes), default=0)
    gap_mat = []
    for ep in data.episodes:
        row = []
        for s in ep.steps:
            d_adv = s.dist_to_goal.get(ADVERSARY, np.nan)
            d_best_good = min(
                s.dist_to_goal.get(AGENT_0, np.nan),
                s.dist_to_goal.get(AGENT_1, np.nan),
            )
            row.append(d_adv - d_best_good)
        row += [np.nan] * (max_len - len(row))
        gap_mat.append(row)

    if gap_mat and max_len > 0:
        gap_arr = np.asarray(gap_mat, dtype=np.float32)
        mean_gap = np.nanmean(gap_arr, axis=0)
        std_gap = np.nanstd(gap_arr, axis=0)
        xs = np.arange(max_len)
        ax.plot(xs, mean_gap, color="#5D6D7E", linewidth=2)
        ax.fill_between(
            xs, mean_gap - std_gap, mean_gap + std_gap, color="#5D6D7E", alpha=0.2
        )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Goal Pressure Gap Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("dist(adversary, goal) - min dist(good, goal)")

    ax = axes[1]
    final_gap = []
    for ep in data.episodes:
        if not ep.steps:
            continue
        last = ep.steps[-1]
        d_adv = last.dist_to_goal.get(ADVERSARY, np.nan)
        d_best_good = min(
            last.dist_to_goal.get(AGENT_0, np.nan),
            last.dist_to_goal.get(AGENT_1, np.nan),
        )
        final_gap.append(d_adv - d_best_good)
    if final_gap:
        ax.hist(
            final_gap,
            bins=min(20, max(8, len(final_gap) // 2)),
            color="#7F8C8D",
            alpha=0.85,
        )
        ax.axvline(
            float(np.mean(final_gap)),
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Mean",
        )
        ax.legend()
    ax.axvline(0.0, color="#C0392B", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.set_title("Final-Step Goal Pressure Distribution")
    ax.set_xlabel("Final dist gap")
    ax.set_ylabel("Episodes")

    if save_path:
        _savefig(fig, save_path)
    return fig


def save_all_sa_figures(
    data: EvalData,
    output_dir: str | Path = "eval_plots",
    prefix: str = "",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pfx = f"{prefix}_" if prefix else ""

    print(f"\n{'=' * 55}")
    print(f" Generating simple_adversary analysis plots → {out}")
    print(f"{'=' * 55}")

    plot_overview(data, save_path=out / f"{pfx}1_sa_overview.png")
    plot_trajectories(
        data,
        n_episodes=min(6, len(data.episodes)),
        save_path=out / f"{pfx}2_sa_trajectories.png",
    )
    plot_role_diagnostics(data, save_path=out / f"{pfx}3_sa_role_diagnostics.png")

    print(f"\nAll plots saved to {out}/")
