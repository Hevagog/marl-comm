"""Behavioral visualizer for Coin Game evaluation data.

Generates five publication-quality figure panels:

  Figure 1 - Behavioral Overview Dashboard
  Figure 2 - Spatial Position Heatmaps
  Figure 3 - Episode Event Timelines (sample episodes)
  Figure 4 - Sample Episode Trajectories
  Figure 5 - Coin-Chase Distance Analysis
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from matplotlib.gridspec import GridSpec

from .eval_data import BLUE_AGENT, RED_AGENT, EvalData

_RED_DARK = "#C0392B"
_RED_LIGHT = "#F1948A"
_BLUE_DARK = "#2471A3"
_BLUE_LIGHT = "#85C1E9"
_GOLD = "#F39C12"
_GREY = "#7F8C8D"
_BG = "#F8F9FA"
_PANEL_BG = "#FFFFFF"

matplotlib.rcParams.update(
    {
        "figure.facecolor": _BG,
        "axes.facecolor": _PANEL_BG,
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
    }
)

_AGENT_COLORS = {RED_AGENT: _RED_DARK, BLUE_AGENT: _BLUE_DARK}
_AGENT_LABELS = {RED_AGENT: "Red (agent_0)", BLUE_AGENT: "Blue (agent_1)"}


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_overview(data: EvalData, save_path: Optional[Path] = None) -> plt.Figure:
    """Six-panel behavioral overview dashboard."""
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    fig.suptitle(
        "Coin Game – Behavioral Overview Dashboard", fontsize=14, fontweight="bold"
    )
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    agents = [RED_AGENT, BLUE_AGENT]
    labels = ["Red", "Blue"]
    colors = [_RED_DARK, _BLUE_DARK]
    x = np.arange(2)
    bar_w = 0.35

    # (0,0) Cooperation rate
    ax = fig.add_subplot(gs[0, 0])
    coop = [data.overall_cooperation_rate(a) * 100 for a in agents]
    steal = [data.overall_steal_rate(a) * 100 for a in agents]
    bars_c = ax.bar(
        x - bar_w / 2, coop, bar_w, label="Own-coin %", color=colors, alpha=0.85
    )
    bars_s = ax.bar(
        x + bar_w / 2,
        steal,
        bar_w,
        label="Steal %",
        color=[_RED_LIGHT, _BLUE_LIGHT],
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of pickups")
    ax.set_ylim(0, 110)
    ax.set_title("Cooperation vs Steal Rate")
    ax.legend(loc="upper right")
    for bar in list(bars_c) + list(bars_s):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # (0,1) Pickup counts breakdown
    ax = fig.add_subplot(gs[0, 1])
    own_counts = [sum(ep.own_pickups_by(a) for ep in data.episodes) for a in agents]
    steal_counts = [sum(ep.steals_by(a) for ep in data.episodes) for a in agents]
    ax.bar(x, own_counts, bar_w * 2, label="Own coin", color=colors, alpha=0.85)
    ax.bar(
        x,
        steal_counts,
        bar_w * 2,
        bottom=own_counts,
        label="Stolen",
        color=[_RED_LIGHT, _BLUE_LIGHT],
        alpha=0.85,
        hatch="//",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total coin pickups")
    ax.set_title(f"Pickup Breakdown ({len(data)} episodes)")
    ax.legend()

    # (0,2) Victim rate (steals suffered)
    ax = fig.add_subplot(gs[0, 2])
    victim_counts = [
        sum(ep.steals_suffered_by(a) for ep in data.episodes) for a in agents
    ]
    ax.bar(x, victim_counts, bar_w * 2, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Times victimized")
    ax.set_title("Steal Victimization Count")

    # (1,0) Per-episode total reward distribution
    ax = fig.add_subplot(gs[1, 0])
    rewards_per_ep = [data.per_episode_rewards(a) for a in agents]
    parts = ax.violinplot(
        rewards_per_ep, positions=x, widths=0.6, showmedians=True, showextrema=True
    )
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Episode total reward")
    ax.set_title("Reward Distribution per Episode")

    # (1,1) Steal rate trend over episodes
    ax = fig.add_subplot(gs[1, 1])
    window = max(1, len(data) // 10)
    for agent, color, label in zip(agents, colors, labels):
        rates = data.per_episode_steal_rates(agent)
        smoothed = np.convolve(rates, np.ones(window) / window, mode="valid")
        ep_x = np.arange(len(smoothed)) + window // 2
        ax.plot(ep_x, smoothed, color=color, label=label, linewidth=2)
        ax.scatter(range(len(rates)), rates, color=color, alpha=0.2, s=10)
    ax.axhline(0.5, color=_GREY, linestyle="--", linewidth=1, label="50% threshold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steal rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Steal Rate Over Episodes (smoothed)")
    ax.legend()

    # (1,2) Reward component breakdown (pick vs penalty received)
    ax = fig.add_subplot(gs[1, 2])
    for i, (agent, color, label) in enumerate(zip(agents, colors, labels)):
        # Pick reward portion: own_pickups * pick_reward + steals_made * pick_reward
        total_pickups = sum(ep.total_pickups_by(agent) for ep in data.episodes)
        total_steals_made = sum(ep.steals_by(agent) for ep in data.episodes)
        total_victimized = sum(ep.steals_suffered_by(agent) for ep in data.episodes)
        n_ep = len(data)
        pick_component = total_pickups * data.pick_reward / n_ep
        penalty_component = total_victimized * data.steal_penalty / n_ep  # negative
        ax.bar(
            i - bar_w / 2,
            pick_component,
            bar_w,
            color=color,
            alpha=0.85,
            label="Avg pick reward",
        )
        ax.bar(
            i + bar_w / 2,
            penalty_component,
            bar_w,
            color=color,
            alpha=0.4,
            hatch="xx",
            label="Avg penalty received",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Avg reward component / episode")
    ax.set_title("Reward Component Breakdown")

    # (2,0-2) Action distribution
    ax = fig.add_subplot(gs[2, :])
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    n_actions = 4
    action_x = np.arange(n_actions)
    offset = bar_w / 2
    for i, (agent, color, label) in enumerate(zip(agents, colors, labels)):
        all_actions = [
            s.actions.get(agent, -1) for ep in data.episodes for s in ep.steps
        ]
        counts = np.array([all_actions.count(a) for a in range(n_actions)], dtype=float)
        counts /= counts.sum() + 1e-9  # normalize
        ax.bar(
            action_x + (i - 0.5) * bar_w,
            counts,
            bar_w,
            color=color,
            alpha=0.85,
            label=label,
        )
    ax.set_xticks(action_x)
    ax.set_xticklabels(action_names)
    ax.set_ylabel("Action frequency")
    ax.set_title("Action Distribution Across All Episodes")
    ax.legend()

    if save_path:
        _savefig(fig, save_path)
    return fig


def plot_heatmaps(data: EvalData, save_path: Optional[Path] = None) -> plt.Figure:
    """2D position heatmaps for each agent + coin pickup locations."""
    gs_size = data.grid_size
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle("Spatial Position Heatmaps", fontsize=14, fontweight="bold")

    agents = [RED_AGENT, BLUE_AGENT]
    cmaps = ["Reds", "Blues"]
    titles = ["Red Agent Positions", "Blue Agent Positions", "Coin Pickup Locations"]

    for ax, agent, cmap in zip(axes[:2], agents, cmaps):
        heatmap = np.zeros((gs_size, gs_size), dtype=float)
        for ep in data.episodes:
            for step in ep.steps:
                pos = step.red_pos if agent == RED_AGENT else step.blue_pos
                if 0 <= pos[0] < gs_size and 0 <= pos[1] < gs_size:
                    heatmap[pos[1], pos[0]] += 1  # y=row, x=col
        heatmap /= heatmap.max() + 1e-9

        im = ax.imshow(
            heatmap,
            cmap=cmap,
            vmin=0,
            vmax=1,
            origin="upper",
            extent=[-0.5, gs_size - 0.5, gs_size - 0.5, -0.5],
        )
        ax.set_title(titles[agents.index(agent)])
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        ax.set_xticks(range(gs_size))
        ax.set_yticks(range(gs_size))
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, label="Relative frequency")

    # Third panel: coin pickup density
    ax = axes[2]
    red_pickup_map = np.zeros((gs_size, gs_size), dtype=float)
    blue_pickup_map = np.zeros((gs_size, gs_size), dtype=float)

    for ep in data.episodes:
        for event in ep.events:
            # Find the position of the picker at event.step
            step_rec = ep.steps[event.step] if event.step < len(ep.steps) else None
            if step_rec is None:
                continue
            if event.picker == RED_AGENT:
                pos = step_rec.red_pos
                red_pickup_map[pos[1], pos[0]] += 1
            else:
                pos = step_rec.blue_pos
                blue_pickup_map[pos[1], pos[0]] += 1

    # Show as overlaid scatter
    ax.set_xlim(-0.5, gs_size - 0.5)
    ax.set_ylim(gs_size - 0.5, -0.5)
    ax.set_facecolor("#EEEEEE")
    ax.grid(False)

    for y in range(gs_size):
        for x in range(gs_size):
            if red_pickup_map[y, x] > 0:
                ax.scatter(
                    x,
                    y,
                    s=red_pickup_map[y, x] * 40,
                    color=_RED_DARK,
                    alpha=0.7,
                    zorder=3,
                )
            if blue_pickup_map[y, x] > 0:
                ax.scatter(
                    x,
                    y,
                    s=blue_pickup_map[y, x] * 40,
                    color=_BLUE_DARK,
                    alpha=0.7,
                    zorder=3,
                    marker="^",
                )

    # Draw grid lines
    for i in range(gs_size + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.5)
        ax.axvline(i - 0.5, color="white", linewidth=0.5)

    ax.set_title(titles[2])
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.set_xticks(range(gs_size))
    ax.set_yticks(range(gs_size))
    red_patch = mpatches.Patch(color=_RED_DARK, label="Red pickups (circle)")
    blue_patch = mpatches.Patch(color=_BLUE_DARK, label="Blue pickups (triangle)")
    ax.legend(handles=[red_patch, blue_patch], loc="lower right", fontsize=8)

    if save_path:
        _savefig(fig, save_path)
    return fig


_EVENT_SYMBOLS = {
    ("red", RED_AGENT, False): ("●", _RED_DARK, "Red picks own coin"),
    ("blue", BLUE_AGENT, False): ("●", _BLUE_DARK, "Blue picks own coin"),
    ("red", BLUE_AGENT, True): ("★", _RED_DARK, "Blue STEALS red coin"),
    ("blue", RED_AGENT, True): ("★", _BLUE_DARK, "Red STEALS blue coin"),
}


def plot_event_timelines(
    data: EvalData,
    n_episodes: int = 8,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Horizontal event timelines for a sample of episodes."""
    episodes = data.episodes[:n_episodes]
    n = len(episodes)
    if n == 0:
        return plt.figure()

    fig, axes = plt.subplots(n, 1, figsize=(16, n * 1.6 + 1.5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "Episode Event Timelines (coin pickup events)", fontsize=14, fontweight="bold"
    )

    max_steps = max((ep.length for ep in episodes), default=1)

    for ax, ep in zip(axes, episodes):
        ep_len = ep.length
        ax.set_xlim(-1, max_steps + 1)
        ax.set_ylim(-0.8, 0.8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.grid(False)
        ax.axhline(0, color=_GREY, linewidth=1.2, alpha=0.5)

        # Background: shade agent reward regions
        # Draw reward trace as thin bar
        cum_red = 0.0
        cum_blue = 0.0

        for event in ep.events:
            key = (event.coin_color, event.picker, event.is_steal)
            symbol, color, _ = _EVENT_SYMBOLS.get(key, ("?", _GREY, "unknown"))
            y_pos = 0.25 if event.picker == RED_AGENT else -0.25
            ax.text(
                event.step,
                y_pos,
                symbol,
                ha="center",
                va="center",
                fontsize=14 if not event.is_steal else 16,
                color=color,
                fontweight="bold" if event.is_steal else "normal",
                zorder=5,
            )

        # Cumulative reward trace as thin line
        red_rewards = [s.rewards.get(RED_AGENT, 0) for s in ep.steps]
        blue_rewards = [s.rewards.get(BLUE_AGENT, 0) for s in ep.steps]
        cum_red = np.cumsum(red_rewards)
        cum_blue = np.cumsum(blue_rewards)

        # Normalize traces to [-0.6, -0.1] and [0.1, 0.6]
        def _scale(arr, lo, hi):
            mn, mx = arr.min(), arr.max()
            if mx == mn:
                return np.full_like(arr, (lo + hi) / 2)
            return lo + (arr - mn) / (mx - mn) * (hi - lo)

        steps_x = np.arange(len(ep.steps))
        ax2 = ax.twinx()
        ax2.set_ylim(-1, 1)
        ax2.set_yticks([])
        ax2.plot(
            steps_x,
            cum_red / (abs(cum_red).max() + 1e-6) * 0.55,
            color=_RED_DARK,
            alpha=0.3,
            linewidth=1.5,
        )
        ax2.plot(
            steps_x,
            cum_blue / (abs(cum_blue).max() + 1e-6) * 0.55,
            color=_BLUE_DARK,
            alpha=0.3,
            linewidth=1.5,
        )

        n_steals = sum(1 for e in ep.events if e.is_steal)
        r_red = ep.total_rewards.get(RED_AGENT, 0.0)
        r_blue = ep.total_rewards.get(BLUE_AGENT, 0.0)
        ax.set_ylabel(
            f"Ep {ep.episode_idx + 1}\nR={r_red:+.0f} B={r_blue:+.0f}\n✩{n_steals}",
            fontsize=8,
            rotation=0,
            labelpad=55,
            va="center",
        )
        ax.set_xlabel("Step" if ep == episodes[-1] else "")

    # Legend
    legend_elements = [
        mpatches.Patch(color=_RED_DARK, label="● Red picks own coin"),
        mpatches.Patch(color=_BLUE_DARK, label="● Blue picks own coin"),
        mpatches.Patch(color=_RED_DARK, label="★ Blue steals red coin", hatch="//"),
        mpatches.Patch(color=_BLUE_DARK, label="★ Red steals blue coin", hatch="//"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", ncol=2, fontsize=9)

    if save_path:
        _savefig(fig, save_path)
    return fig


def plot_trajectories(
    data: EvalData,
    n_episodes: int = 6,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Grid trajectory plots for a sample of episodes."""
    episodes = data.episodes[:n_episodes]
    n = len(episodes)
    if n == 0:
        return plt.figure()

    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 5, rows * 5), constrained_layout=True
    )
    fig.suptitle("Sample Episode Trajectories", fontsize=14, fontweight="bold")

    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])

    gs_size = data.grid_size

    def _draw_grid(ax):
        for i in range(gs_size + 1):
            ax.axhline(i - 0.5, color="#CCCCCC", linewidth=0.5)
            ax.axvline(i - 0.5, color="#CCCCCC", linewidth=0.5)
        ax.set_xlim(-0.5, gs_size - 0.5)
        ax.set_ylim(gs_size - 0.5, -0.5)
        ax.set_xticks(range(gs_size))
        ax.set_yticks(range(gs_size))
        ax.set_facecolor("#F5F5F5")
        ax.grid(False)

    for idx, ep in enumerate(episodes):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        _draw_grid(ax)

        red_path = [(s.red_pos[0], s.red_pos[1]) for s in ep.steps]
        blue_path = [(s.blue_pos[0], s.blue_pos[1]) for s in ep.steps]

        n_steps = len(red_path)
        # Draw fading path
        for path, color in [(red_path, _RED_DARK), (blue_path, _BLUE_DARK)]:
            for i in range(1, len(path)):
                alpha = 0.15 + 0.7 * (i / n_steps)
                ax.plot(
                    [path[i - 1][0], path[i][0]],
                    [path[i - 1][1], path[i][1]],
                    color=color,
                    alpha=alpha,
                    linewidth=1.5,
                )

        # Start/end markers
        if red_path:
            ax.scatter(*red_path[0], color=_RED_DARK, s=120, zorder=6, marker="s")
            ax.scatter(*red_path[-1], color=_RED_DARK, s=120, zorder=6, marker="D")
        if blue_path:
            ax.scatter(*blue_path[0], color=_BLUE_DARK, s=120, zorder=6, marker="s")
            ax.scatter(*blue_path[-1], color=_BLUE_DARK, s=120, zorder=6, marker="D")

        # Coin pickup events
        for event in ep.events:
            step_rec = ep.steps[event.step] if event.step < len(ep.steps) else None
            if step_rec is None:
                continue
            pos = step_rec.red_pos if event.picker == RED_AGENT else step_rec.blue_pos
            coin_color = _RED_LIGHT if event.coin_color == "red" else _BLUE_LIGHT
            marker = "★" if event.is_steal else "●"
            ax.text(
                pos[0],
                pos[1],
                marker,
                color=coin_color,
                fontsize=14 if event.is_steal else 10,
                ha="center",
                va="center",
                zorder=7,
                fontweight="bold",
            )

        r_red = ep.total_rewards.get(RED_AGENT, 0.0)
        r_blue = ep.total_rewards.get(BLUE_AGENT, 0.0)
        n_steals = sum(1 for e in ep.events if e.is_steal)
        ax.set_title(
            f"Episode {ep.episode_idx + 1}  "
            f"R={r_red:+.1f}  B={r_blue:+.1f}  ✩steals={n_steals}",
            fontsize=9,
        )

    # Hide unused axes
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    # Legend
    red_sq = mpatches.Patch(color=_RED_DARK, label="Red path")
    blue_sq = mpatches.Patch(color=_BLUE_DARK, label="Blue path")
    star_p = plt.Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        markerfacecolor=_GREY,
        markersize=10,
        label="Steal event",
    )
    dot_p = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=_GREY,
        markersize=8,
        label="Own pickup",
    )
    sq_p = plt.Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor=_GREY,
        markersize=8,
        label="Start pos",
    )
    fig.legend(
        handles=[red_sq, blue_sq, star_p, dot_p, sq_p],
        loc="lower center",
        ncol=5,
        fontsize=9,
        frameon=True,
    )

    if save_path:
        _savefig(fig, save_path)
    return fig


def plot_distance_analysis(
    data: EvalData, save_path: Optional[Path] = None
) -> plt.Figure:
    """Shows average Manhattan distance to own vs opponent coin over time.

    A cooperative agent should consistently move *toward its own coin*.
    A defecting agent will often be closer to the *opponent's coin*.
    """
    agents = [RED_AGENT, BLUE_AGENT]
    agent_labels = ["Red Agent", "Blue Agent"]
    own_coin_colors = [_RED_DARK, _BLUE_DARK]
    opp_coin_colors = [_BLUE_LIGHT, _RED_LIGHT]

    # Gather per-timestep distances across episodes
    max_steps = max((ep.length for ep in data.episodes), default=0)
    if max_steps == 0:
        return plt.figure()

    # Matrices: shape [n_episodes, max_steps]
    def _dist_matrix(agent, coin_type):
        """coin_type: 'own' | 'opp'"""
        mat = []
        for ep in data.episodes:
            row = []
            for step in ep.steps:
                if agent == RED_AGENT:
                    agent_pos = step.red_pos
                    own_pos = step.red_coin_pos
                    opp_pos = step.blue_coin_pos
                else:
                    agent_pos = step.blue_pos
                    own_pos = step.blue_coin_pos
                    opp_pos = step.red_coin_pos

                coin_pos = own_pos if coin_type == "own" else opp_pos
                if coin_pos is None:
                    row.append(np.nan)
                else:
                    row.append(
                        float(
                            abs(agent_pos[0] - coin_pos[0])
                            + abs(agent_pos[1] - coin_pos[1])
                        )
                    )
            mat.append(row)
        return mat

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        "Coin-Chase Distance Analysis\n"
        "(Manhattan distance from agent to coin over episode steps)",
        fontsize=13,
        fontweight="bold",
    )

    for row_idx, (agent, label, own_c, opp_c) in enumerate(
        zip(agents, agent_labels, own_coin_colors, opp_coin_colors)
    ):
        own_mat = _dist_matrix(agent, "own")
        opp_mat = _dist_matrix(agent, "opp")

        # Left: mean ± std over episodes
        ax_left = axes[row_idx][0]
        max_ep_len = max((len(r) for r in own_mat), default=0)

        def _padded_stats(mat):
            padded = np.full((len(mat), max_ep_len), np.nan)
            for i, row in enumerate(mat):
                padded[i, : len(row)] = row
            mean = np.nanmean(padded, axis=0)
            std = np.nanstd(padded, axis=0)
            return mean, std

        own_mean, own_std = _padded_stats(own_mat)
        opp_mean, opp_std = _padded_stats(opp_mat)
        steps_x = np.arange(max_ep_len)

        ax_left.plot(steps_x, own_mean, color=own_c, linewidth=2, label="To own coin")
        ax_left.fill_between(
            steps_x, own_mean - own_std, own_mean + own_std, color=own_c, alpha=0.2
        )
        ax_left.plot(
            steps_x,
            opp_mean,
            color=opp_c,
            linewidth=2,
            linestyle="--",
            label="To opponent's coin",
        )
        ax_left.fill_between(
            steps_x, opp_mean - opp_std, opp_mean + opp_std, color=opp_c, alpha=0.2
        )
        ax_left.set_xlabel("Step")
        ax_left.set_ylabel("Manhattan distance")
        ax_left.set_title(f"{label}: Distance to Coins Over Time")
        ax_left.legend()
        ax_left.yaxis.set_minor_locator(mticker.AutoMinorLocator())

        # Right: histogram of "preference ratio" = (d_opp - d_own) per step
        ax_right = axes[row_idx][1]
        diffs = []
        for own_row, opp_row in zip(own_mat, opp_mat):
            for d_own, d_opp in zip(own_row, opp_row):
                if not (math.isnan(d_own) or math.isnan(d_opp)):
                    diffs.append(d_opp - d_own)

        if diffs:
            diffs_arr = np.array(diffs)
            n_coop = int(np.sum(diffs_arr > 0))
            n_def = int(np.sum(diffs_arr <= 0))
            frac_coop = n_coop / len(diffs_arr)

            ax_right.hist(diffs_arr, bins=30, color=own_c, edgecolor="white", alpha=0.8)
            ax_right.axvline(
                0, color="black", linewidth=1.2, linestyle="--", label="d_opp = d_own"
            )
            ax_right.axvline(
                np.mean(diffs_arr),
                color=_GOLD,
                linewidth=1.5,
                linestyle="-",
                label=f"Mean = {np.mean(diffs_arr):.2f}",
            )
            ax_right.set_xlabel("d(opponent coin) − d(own coin)")
            ax_right.set_ylabel("Count")
            ax_right.set_title(
                f"{label}: Coin Preference\n"
                f"Cooperative steps (d_opp>0): {frac_coop * 100:.1f}%"
            )
            ax_right.legend(fontsize=8)

    if save_path:
        _savefig(fig, save_path)
    return fig


def save_all_figures(
    data: EvalData,
    output_dir: str | Path = "eval_plots",
    prefix: str = "",
    dpi: int = 150,
) -> None:
    """Generate and save all five behavioral analysis figures.

    Parameters
    ----------
    data : EvalData
        Collected evaluation data.
    output_dir : str or Path
        Directory to write PNG files into (created if absent).
    prefix : str
        Optional filename prefix (e.g. experiment name).
    dpi : int
        Output resolution.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pfx = f"{prefix}_" if prefix else ""

    print(f"\n{'=' * 55}")
    print(f" Generating behavioral analysis plots → {out}")
    print(f"{'=' * 55}")
    print(f" Episodes collected : {len(data)}")
    for agent, label in [(RED_AGENT, "Red"), (BLUE_AGENT, "Blue")]:
        print(
            f"  {label:4s} | cooperation: {data.overall_cooperation_rate(agent) * 100:.1f}%"
            f"  | steal rate: {data.overall_steal_rate(agent) * 100:.1f}%"
            f"  | mean reward: {data.mean_total_reward(agent):+.2f}"
        )
    print(f"{'=' * 55}\n")

    plot_overview(data, save_path=out / f"{pfx}1_overview.png")
    plot_heatmaps(data, save_path=out / f"{pfx}2_heatmaps.png")
    plot_event_timelines(
        data, n_episodes=min(8, len(data)), save_path=out / f"{pfx}3_timelines.png"
    )
    plot_trajectories(
        data, n_episodes=min(6, len(data)), save_path=out / f"{pfx}4_trajectories.png"
    )
    plot_distance_analysis(data, save_path=out / f"{pfx}5_distances.png")

    print(f"\nAll plots saved to {out}/")
