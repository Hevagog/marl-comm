"""
continuous_coord_eval_visualizer.py
=====================================

Six figures produced:
  1. Performance Dashboard       – 6-panel overview of key metrics
  2. Spatial Heatmaps            – Agent position density on [0,1]² space
  3. Capture & Synchrony Analysis– Capture timing, sync bonus, spread histograms
  4. Coordination Dynamics       – Inter-agent distance, collision rate, entropy over time
  5. Sample Episode Trajectories – 2D path plots for a handful of episodes
  6. Target Lifecycle & Pressure – Active targets over time, capture vs expiry rates
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .continuous_coord_data import ACTION_NAMES, EvalData
from utils.shared.style import _AGENT_COLORS, save_figure

_GOLD = "#F4A261"
_TEAL = "#2A9D8F"


def _savefig(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    save_figure(fig, path.parent, path.stem, dpi=dpi)


def _agent_color(agent: str, agents: list[str]) -> str:
    try:
        idx = agents.index(agent)
    except ValueError:
        idx = 0
    return _AGENT_COLORS[idx % len(_AGENT_COLORS)]


# ─── Figure 1: Performance Dashboard ─────────────────────────────────────────


def plot_overview(data: EvalData, save_path: Path | None = None) -> plt.Figure:
    """Six-panel performance dashboard."""
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    fig.suptitle(
        "Continuous Coordination – Performance Overview Dashboard",
        fontsize=14,
        fontweight="bold",
    )
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    n_ep = len(data)
    agents = data.agents or [f"agent_{i}" for i in range(data.num_agents)]

    # (0,0) Episode return distribution (violin)
    ax = fig.add_subplot(gs[0, 0])
    returns = data.episode_returns()
    parts = ax.violinplot(
        [returns], positions=[0], widths=0.6, showmedians=True, showextrema=True
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(_TEAL)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    ax.scatter(
        np.zeros(n_ep) + np.random.uniform(-0.05, 0.05, n_ep),
        returns,
        alpha=0.4,
        s=15,
        color=_TEAL,
        zorder=4,
    )
    ax.axhline(
        np.mean(returns),
        color=_GOLD,
        linewidth=1.5,
        linestyle="--",
        label=f"Mean={np.mean(returns):.1f}",
    )
    ax.set_xticks([0])
    ax.set_xticklabels(["Episodes"])
    ax.set_ylabel("Mean per-agent return")
    ax.set_title(f"Episode Return Distribution\n({n_ep} episodes)")
    ax.legend(fontsize=8)

    # (0,1) Captures per episode distribution
    ax = fig.add_subplot(gs[0, 1])
    caps = [ep.total_captures for ep in data.episodes]
    ax.hist(
        caps,
        bins=max(1, max(caps) + 1),
        color=_TEAL,
        edgecolor="white",
        alpha=0.8,
        rwidth=0.85,
    )
    ax.axvline(
        np.mean(caps),
        color=_GOLD,
        linewidth=1.8,
        linestyle="--",
        label=f"Mean={np.mean(caps):.1f}",
    )
    ax.set_xlabel("Captures per episode")
    ax.set_ylabel("Count")
    ax.set_title("Capture Count Distribution")
    ax.legend(fontsize=8)

    # (0,2) Collision rate per episode
    ax = fig.add_subplot(gs[0, 2])
    coll_rates = [ep.collision_rate * 100 for ep in data.episodes]
    ax.hist(coll_rates, bins=20, color="#E63946", edgecolor="white", alpha=0.8)
    ax.axvline(
        np.mean(coll_rates),
        color=_GOLD,
        linewidth=1.8,
        linestyle="--",
        label=f"Mean={np.mean(coll_rates):.1f}%",
    )
    ax.set_xlabel("Collision rate (% of steps)")
    ax.set_ylabel("Count")
    ax.set_title("Collision Rate Distribution")
    ax.legend(fontsize=8)

    # (1,0) Per-agent return fairness
    ax = fig.add_subplot(gs[1, 0])
    per_agent = data.per_agent_returns()
    positions = np.arange(len(agents))
    for i, a in enumerate(agents):
        vals = per_agent.get(a, [0.0])
        color = _agent_color(a, agents)
        parts = ax.violinplot(
            [vals], positions=[i], widths=0.6, showmedians=True, showextrema=False
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        ax.scatter([i] * len(vals), vals, alpha=0.3, s=12, color=color, zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels([a.split("_")[-1] for a in agents])
    ax.set_xlabel("Agent")
    ax.set_ylabel("Episode total reward")
    ax.set_title("Per-Agent Reward Distribution\n(fairness)")

    # (1,1) Sync bonus distribution
    ax = fig.add_subplot(gs[1, 1])
    bonuses = data.sync_bonuses()
    if bonuses:
        ax.hist(bonuses, bins=20, color=_GOLD, edgecolor="white", alpha=0.85)
        ax.axvline(
            np.mean(bonuses),
            color="black",
            linewidth=1.5,
            linestyle="--",
            label=f"Mean={np.mean(bonuses):.2f}",
        )
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "No capture events",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="grey",
        )
    ax.set_xlabel("Estimated sync reward")
    ax.set_ylabel("Count")
    ax.set_title("Sync Bonus Distribution\n(higher = better coordination)")

    # (1,2) Cumulative return over episode steps
    ax = fig.add_subplot(gs[1, 2])
    t, mean_ret, std_ret = data.mean_cumulative_return_over_time()
    ax.plot(t, mean_ret, color=_TEAL, linewidth=2, label="Mean cumulative return")
    ax.fill_between(
        t,
        mean_ret - std_ret,
        mean_ret + std_ret,
        color=_TEAL,
        alpha=0.2,
        label="±1 std",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative return")
    ax.set_title("Mean Cumulative Return over Episode")
    ax.legend(fontsize=8)

    # (2,0-2) Action distribution
    ax = fig.add_subplot(gs[2, :])
    action_dist = data.action_distribution()
    action_labels = [ACTION_NAMES.get(i, str(i)) for i in range(9)]
    colors_bar = [_TEAL] * 8 + ["#ADB5BD"]
    bars = ax.bar(
        range(9), action_dist * 100, color=colors_bar, edgecolor="white", alpha=0.85
    )
    for bar, val in zip(bars, action_dist * 100):
        if val > 1:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(range(9))
    ax.set_xticklabels(action_labels)
    ax.set_ylabel("Action frequency (%)")
    ax.set_title("Action Distribution Across All Episodes (all agents)")
    ax.axhline(
        100 / 9, color="grey", linestyle="--", linewidth=1, label="Uniform (11.1%)"
    )
    ax.legend(fontsize=8)

    if save_path:
        _savefig(fig, save_path)
    return fig


# ─── Figure 2: Spatial Heatmaps ──────────────────────────────────────────────


def plot_spatial_heatmaps(
    data: EvalData,
    grid_res: int = 25,
    save_path: Path | None = None,
) -> plt.Figure:
    """Per-agent position density heatmaps on [0,1]²."""
    agents = data.agents or [f"agent_{i}" for i in range(data.num_agents)]
    n = len(agents)
    cols = min(n, 4)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4.5, rows * 4), constrained_layout=True
    )
    fig.suptitle("Agent Position Density Heatmaps", fontsize=14, fontweight="bold")

    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])

    for idx, agent in enumerate(agents):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        hmap = data.position_heatmap(agent, grid_res=grid_res)
        max_val = hmap.max()
        if max_val > 0:
            hmap = hmap / max_val

        color_idx = idx % len(_AGENT_COLORS)
        # Build custom colormap from white to agent color
        from matplotlib.colors import LinearSegmentedColormap

        agent_hex = _AGENT_COLORS[color_idx]
        cmap = LinearSegmentedColormap.from_list(
            f"agent_{idx}", ["#FFFFFF", agent_hex], N=256
        )

        im = ax.imshow(
            hmap,
            cmap=cmap,
            vmin=0,
            vmax=1,
            origin="upper",
            extent=[0, 1, 0, 1],
            aspect="auto",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, label="Relative freq.")

        ax.set_title(f"Agent {idx} ({agent})")
        ax.set_xlabel("x [0,1]")
        ax.set_ylabel("y [0,1]")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(False)

        # Overlay capture event positions
        cap_positions = []
        for ep in data.episodes:
            for ev in ep.capture_events:
                if ev.step < len(ep.steps):
                    state = ep.steps[ev.step].agent_states.get(agent)
                    if state:
                        cap_positions.append((state.px, state.py))
        if cap_positions:
            cpx, cpy = zip(*cap_positions)
            ax.scatter(
                cpx,
                cpy,
                c="gold",
                s=30,
                zorder=5,
                marker="*",
                edgecolors="black",
                linewidths=0.4,
                alpha=0.7,
                label="Capture events",
            )
            ax.legend(fontsize=7, loc="lower right")

    # Hide unused axes
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    if save_path:
        _savefig(fig, save_path)
    return fig


# ─── Figure 3: Capture & Synchrony Analysis ──────────────────────────────────


def plot_capture_analysis(
    data: EvalData,
    save_path: Path | None = None,
) -> plt.Figure:
    """Capture timing, sync bonus, and k_req distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Capture & Synchrony Analysis", fontsize=14, fontweight="bold")

    all_cap_steps = data.capture_steps()
    all_sync = data.sync_bonuses()
    all_k = [e.k_required for ep in data.episodes for e in ep.capture_events]

    # (0,0) Capture timing: when during the episode do captures happen?
    ax = axes[0][0]
    if all_cap_steps:
        t_norm = [s / data.max_cycles for s in all_cap_steps]
        ax.hist(t_norm, bins=20, color=_TEAL, edgecolor="white", alpha=0.85)
        ax.set_xlabel("Episode progress (0=start, 1=end)")
        ax.set_ylabel("Capture count")
        ax.set_title(
            "Capture Timing Distribution\n(when in episode do captures happen?)"
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No captures recorded",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="grey",
        )
        ax.set_title("Capture Timing Distribution")

    # (0,1) Sync bonus vs capture step (scatter)
    ax = axes[0][1]
    if all_cap_steps and all_sync:
        t_norm = [s / data.max_cycles for s in all_cap_steps]
        sc = ax.scatter(
            t_norm, all_sync, c=all_k, cmap="plasma", alpha=0.6, s=40, edgecolors="none"
        )
        fig.colorbar(sc, ax=ax, label="k_required")
        ax.set_xlabel("Episode progress")
        ax.set_ylabel("Estimated sync reward")
        ax.set_title("Sync Bonus vs Capture Timing\n(color = k required)")
    else:
        ax.text(
            0.5,
            0.5,
            "No captures recorded",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="grey",
        )
        ax.set_title("Sync Bonus vs Capture Timing")

    # (1,0) k_req distribution (captured targets)
    ax = axes[1][0]
    if all_k:
        unique_k = sorted(set(all_k))
        counts = [all_k.count(k) for k in unique_k]
        colors = [_AGENT_COLORS[i % len(_AGENT_COLORS)] for i in range(len(unique_k))]
        ax.bar(
            [str(k) for k in unique_k],
            counts,
            color=colors,
            edgecolor="white",
            alpha=0.85,
        )
        ax.set_xlabel("k_required (agents needed)")
        ax.set_ylabel("Count of captured targets")
        ax.set_title("Captured Target k-Requirement Distribution")
    else:
        ax.text(
            0.5,
            0.5,
            "No captures",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="grey",
        )
        ax.set_title("k-Requirement Distribution")

    # (1,1) Captures per episode over evaluation episodes (trend)
    ax = axes[1][1]
    caps_per_ep = [ep.total_captures for ep in data.episodes]
    ep_x = np.arange(len(caps_per_ep))
    ax.bar(ep_x, caps_per_ep, color=_TEAL, alpha=0.7, edgecolor="none")
    window = max(1, len(caps_per_ep) // 8)
    smoothed = np.convolve(caps_per_ep, np.ones(window) / window, mode="valid")
    ax.plot(
        np.arange(len(smoothed)) + window // 2,
        smoothed,
        color=_GOLD,
        linewidth=2,
        label=f"Smoothed (w={window})",
    )
    ax.axhline(
        np.mean(caps_per_ep),
        color="#E63946",
        linewidth=1.5,
        linestyle="--",
        label=f"Mean={np.mean(caps_per_ep):.1f}",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Captures")
    ax.set_title("Captures per Evaluation Episode")
    ax.legend(fontsize=8)

    if save_path:
        _savefig(fig, save_path)
    return fig


# ─── Figure 4: Coordination Dynamics ─────────────────────────────────────────


def plot_coordination_dynamics(
    data: EvalData,
    save_path: Path | None = None,
) -> plt.Figure:
    """Inter-agent distance, collision rate, coordination entropy over episode steps."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(
        "Coordination Dynamics Over Episode Steps", fontsize=14, fontweight="bold"
    )

    # (0,0) Mean inter-agent distance over time
    ax = axes[0][0]
    t, mean_d, std_d = data.mean_inter_agent_dist_over_time()
    valid = ~np.isnan(mean_d)
    ax.plot(
        t[valid], mean_d[valid], color=_TEAL, linewidth=2, label="Mean inter-agent dist"
    )
    ax.fill_between(
        t[valid],
        (mean_d - std_d)[valid],
        (mean_d + std_d)[valid],
        color=_TEAL,
        alpha=0.2,
        label="±1 std",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Euclidean distance")
    ax.set_title("Mean Pairwise Agent Distance Over Time\n(lower = more coordinated)")
    ax.legend(fontsize=8)

    # (0,1) Per-step collision frequency (smoothed)
    ax = axes[0][1]
    collision_counts = np.zeros(data.max_cycles, dtype=float)
    for ep in data.episodes:
        for ev in ep.collision_events:
            if ev.step < data.max_cycles:
                collision_counts[ev.step] += 1
    collision_rate = collision_counts / max(len(data.episodes), 1)
    window = max(1, data.max_cycles // 20)
    smoothed = np.convolve(collision_rate, np.ones(window) / window, mode="same")
    ax.plot(np.arange(data.max_cycles), smoothed, color="#E63946", linewidth=2)
    ax.fill_between(
        np.arange(data.max_cycles), 0, smoothed, color="#E63946", alpha=0.15
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Avg collisions per step")
    ax.set_title("Collision Frequency Over Episode\n(smoothed)")

    # (1,0) Coordination entropy over time
    ax = axes[1][0]
    t_ent, mean_ent = data.coord_entropy_over_time()
    valid = ~np.isnan(mean_ent)
    ax.plot(t_ent[valid], mean_ent[valid], color=_GOLD, linewidth=2)
    ax.fill_between(t_ent[valid], 0, mean_ent[valid], color=_GOLD, alpha=0.2)
    ax.axhline(
        np.log2(9),
        color="grey",
        linestyle="--",
        linewidth=1,
        label="Max entropy (uniform)",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Action entropy (bits)")
    ax.set_title(
        "Joint-Action Coordination Entropy Over Time\n(low = coordinated, high = diverse)"
    )
    ax.legend(fontsize=8)

    # (1,1) Mean nearest-target distance over time (proximity shaping progress)
    ax = axes[1][1]
    # Compute from target obs: for each step, collect all agents' distance to nearest active target
    near_dist_mat = np.full((len(data.episodes), data.max_cycles), np.nan)
    for ep_i, ep in enumerate(data.episodes):
        for step in ep.steps:
            t_idx = step.step
            if t_idx >= data.max_cycles:
                continue
            active_tgts = [tgt for tgt in step.targets if tgt.active]
            if not active_tgts:
                continue
            per_agent_min = []
            for a_state in step.agent_states.values():
                min_d = min(
                    math.sqrt(
                        (a_state.px - tgt.abs_x) ** 2 + (a_state.py - tgt.abs_y) ** 2
                    )
                    for tgt in active_tgts
                )
                per_agent_min.append(min_d)
            near_dist_mat[ep_i, t_idx] = float(np.mean(per_agent_min))

    mean_near = np.nanmean(near_dist_mat, axis=0)
    std_near = np.nanstd(near_dist_mat, axis=0)
    valid = ~np.isnan(mean_near)
    if valid.any():
        ax.plot(
            t[valid],
            mean_near[valid],
            color="#6D6875",
            linewidth=2,
            label="Mean nearest-target dist",
        )
        ax.fill_between(
            t[valid],
            (mean_near - std_near)[valid],
            (mean_near + std_near)[valid],
            color="#6D6875",
            alpha=0.2,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to nearest active target")
    ax.set_title(
        "Mean Proximity to Nearest Active Target\n(lower = agents approaching targets)"
    )
    ax.legend(fontsize=8)

    if save_path:
        _savefig(fig, save_path)
    return fig


# ─── Figure 5: Sample Episode Trajectories ────────────────────────────────────


def plot_trajectories(
    data: EvalData,
    n_episodes: int = 6,
    save_path: Path | None = None,
) -> plt.Figure:
    """2D agent trajectories for a sample of evaluation episodes."""
    episodes = data.episodes[:n_episodes]
    n = len(episodes)
    if n == 0:
        return plt.figure()

    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 5, rows * 5), constrained_layout=True
    )
    fig.suptitle(
        "Sample Episode Trajectories (continuous [0,1]² space)",
        fontsize=14,
        fontweight="bold",
    )

    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])

    agents = data.agents or [f"agent_{i}" for i in range(data.num_agents)]

    for ep_idx, ep in enumerate(episodes):
        row, col = divmod(ep_idx, cols)
        ax = axes[row][col]
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_facecolor("#F0F4F8")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Draw capture events as circles
        for ev in ep.capture_events:
            if ev.step < len(ep.steps):
                # Get centroid of agents near the target
                tgt = (
                    ep.steps[ev.step].targets[ev.target_idx]
                    if ev.target_idx < len(ep.steps[ev.step].targets)
                    else None
                )
                if tgt and tgt.active is False:  # just captured
                    pass
                # Just mark where it happened using active target from prev step
                if ev.step > 0:
                    prev_targets = ep.steps[ev.step - 1].targets
                    if (
                        ev.target_idx < len(prev_targets)
                        and prev_targets[ev.target_idx].active
                    ):
                        tgt = prev_targets[ev.target_idx]
                        circle = plt.Circle(
                            (tgt.abs_x, tgt.abs_y),
                            0.04,
                            fill=True,
                            color=_GOLD,
                            alpha=0.3,
                            zorder=2,
                        )
                        ax.add_patch(circle)
                        ax.scatter(
                            tgt.abs_x,
                            tgt.abs_y,
                            c=_GOLD,
                            s=80,
                            zorder=5,
                            marker="*",
                            edgecolors="black",
                            linewidths=0.4,
                        )

        # Draw agent trajectories with fading alpha
        n_steps = len(ep.steps)
        for a_idx, agent in enumerate(agents):
            path = ep.agent_positions(agent)
            if len(path) < 2:
                continue
            color = _agent_color(agent, agents)
            for i in range(1, len(path)):
                alpha = 0.15 + 0.75 * (i / n_steps)
                ax.plot(
                    [path[i - 1][0], path[i][0]],
                    [path[i - 1][1], path[i][1]],
                    color=color,
                    alpha=alpha,
                    linewidth=1.2,
                )
            # Start/end markers
            ax.scatter(
                *path[0],
                color=color,
                s=80,
                zorder=6,
                marker="s",
                edgecolors="white",
                linewidths=0.5,
            )
            ax.scatter(
                *path[-1],
                color=color,
                s=80,
                zorder=6,
                marker="D",
                edgecolors="white",
                linewidths=0.5,
            )

        ep_ret = ep.episode_return
        caps = ep.total_captures
        ax.set_title(
            f"Ep {ep.episode_idx + 1} | Return={ep_ret:+.1f} | Caps={caps}",
            fontsize=9,
        )

    # Hide unused axes
    for ep_idx in range(n, rows * cols):
        row, col = divmod(ep_idx, cols)
        axes[row][col].set_visible(False)

    # Legend
    patches = [
        mpatches.Patch(color=_agent_color(a, agents), label=f"Agent {i}")
        for i, a in enumerate(agents)
    ]
    patches += [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="grey",
            markersize=8,
            label="Start",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="grey",
            markersize=8,
            label="End",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=_GOLD,
            markersize=10,
            label="Capture zone",
        ),
    ]
    fig.legend(
        handles=patches, loc="lower center", ncol=len(patches), fontsize=9, frameon=True
    )

    if save_path:
        _savefig(fig, save_path)
    return fig


# ─── Figure 6: Target Lifecycle & Pressure ────────────────────────────────────


def plot_target_lifecycle(
    data: EvalData,
    save_path: Path | None = None,
) -> plt.Figure:
    """Active targets over time, capture vs expiry per episode."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Target Lifecycle & Task Pressure", fontsize=14, fontweight="bold")

    # (0,0) Mean active targets over episode steps
    ax = axes[0][0]
    t, mean_act = data.mean_active_targets_over_time()
    valid = ~np.isnan(mean_act)
    ax.plot(t[valid], mean_act[valid], color=_TEAL, linewidth=2)
    ax.fill_between(t[valid], 0, mean_act[valid], color=_TEAL, alpha=0.15)
    ax.set_xlabel("Step")
    ax.set_ylabel("Active targets (avg)")
    ax.set_title(
        "Mean Active Target Count Over Episode\n(higher pressure = more targets)"
    )
    ax.set_ylim(0, data.max_targets + 0.5)

    # (0,1) Capture count vs episode return (scatter – do captures improve return?)
    ax = axes[0][1]
    caps = [ep.total_captures for ep in data.episodes]
    rets = data.episode_returns()
    ax.scatter(caps, rets, alpha=0.6, s=40, color=_TEAL, edgecolors="none")
    if len(caps) > 2:
        z = np.polyfit(caps, rets, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(caps), max(caps), 100)
        ax.plot(
            x_line,
            p(x_line),
            color=_GOLD,
            linewidth=2,
            linestyle="--",
            label=f"Trend (slope={z[0]:.2f})",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("Captures per episode")
    ax.set_ylabel("Mean per-agent return")
    ax.set_title("Captures vs Episode Return\n(trend shows coordination value)")

    # (1,0) Captures vs collisions (coordination efficiency)
    ax = axes[1][0]
    colls = [ep.total_collisions for ep in data.episodes]
    ax.scatter(colls, caps, alpha=0.6, s=40, color="#E63946", edgecolors="none")
    if len(colls) > 2 and len(set(colls)) > 1:
        z = np.polyfit(colls, caps, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(colls), max(colls), 100)
        ax.plot(x_line, p(x_line), color=_GOLD, linewidth=2, linestyle="--")
    ax.set_xlabel("Collisions per episode")
    ax.set_ylabel("Captures per episode")
    ax.set_title("Collisions vs Captures\n(ideally: low collision, high capture)")

    # (1,1) Summary statistics bar chart
    ax = axes[1][1]
    stats = {
        "Mean return": data.mean_return(),
        "Mean captures": data.mean_captures_per_episode(),
        "Coll. rate (%)": data.mean_collision_rate() * 100,
        "Mean sync": float(np.mean(data.sync_bonuses()))
        if data.sync_bonuses()
        else 0.0,
    }
    keys = list(stats.keys())
    vals = list(stats.values())
    colors_bar = [_TEAL, _TEAL, "#E63946", _GOLD]
    bars = ax.bar(
        range(len(keys)), vals, color=colors_bar, edgecolor="white", alpha=0.85
    )
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + abs(bar.get_height()) * 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=15, ha="right")
    ax.set_title("Summary Metrics")
    ax.axhline(0, color="black", linewidth=0.8)

    if save_path:
        _savefig(fig, save_path)
    return fig


# ─── Master save function ─────────────────────────────────────────────────────


def _print_summary(data: EvalData) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  Continuous Coordination Evaluation Summary")
    print(sep)
    print(f"  Episodes          : {len(data)}")
    print(f"  Mean return       : {data.mean_return():.2f} ± {data.std_return():.2f}")
    print(f"  Mean captures/ep  : {data.mean_captures_per_episode():.2f}")
    print(f"  Mean collisions/ep: {data.mean_collisions_per_episode():.2f}")
    print(f"  Mean coll. rate   : {data.mean_collision_rate() * 100:.1f}%")
    bonuses = data.sync_bonuses()
    if bonuses:
        print(f"  Mean sync bonus   : {float(np.mean(bonuses)):.3f}")
    print(sep + "\n")


def save_all_cc_figures(
    data: EvalData,
    output_dir: str | Path = "eval_plots/continuous_coord",
    prefix: str = "",
    dpi: int = 150,
) -> None:
    """Generate and save all 6 evaluation figures.

    Parameters
    ----------
    data       : EvalData from ContinuousCoordEvalCollector.collect()
    output_dir : Directory to write PNG files into (created if absent).
    prefix     : Optional filename prefix (e.g. experiment name).
    dpi        : Output resolution.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pfx = f"{prefix}_" if prefix else ""

    _print_summary(data)

    print(f"  Generating plots → {out}")
    plot_overview(data, save_path=out / f"{pfx}1_overview.png")
    plot_spatial_heatmaps(data, save_path=out / f"{pfx}2_spatial_heatmaps.png")
    plot_capture_analysis(data, save_path=out / f"{pfx}3_capture_analysis.png")
    plot_coordination_dynamics(
        data, save_path=out / f"{pfx}4_coordination_dynamics.png"
    )
    plot_trajectories(
        data, n_episodes=min(6, len(data)), save_path=out / f"{pfx}5_trajectories.png"
    )
    plot_target_lifecycle(data, save_path=out / f"{pfx}6_target_lifecycle.png")
    print(f"\nAll plots saved to {out}/\n")
