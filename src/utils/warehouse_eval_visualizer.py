"""
warehouse_eval_visualizer.py
============================
Publication-quality figures for multi-robot warehouse evaluation.

Figures produced
----------------
 1. fig_overview           – 6-panel performance dashboard
 2. fig_delivery_trajectory– Cumulative deliveries over episode time
 3. fig_phase_heatmap      – Phase distribution over episode time (time × phase)
 4. fig_spatial_heatmaps   – Agent visitation heatmaps on the 12×16 grid
 5. fig_battery_dynamics   – Battery level trajectories per agent
 6. fig_role_comparison    – Delivery & battery stats by heterogeneous role
 7. fig_task_pressure      – Pending & expired tasks over episode time
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .warehouse_eval_data import (
    ACTION_NAMES,
    PHASE_NAMES,
    EvalData,
    EpisodeData,
)

matplotlib.use("Agg")

# ─── Style ────────────────────────────────────────────────────────────────────

_AGENT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
_PHASE_COLORS = ["#95B9D4", "#F4A261", "#E76F51", "#2A9D8F"]
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
    }
)


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(str(path.with_suffix(f".{ext}")), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}.png/pdf")


# ─── Figure 1: Overview dashboard ─────────────────────────────────────────────


def fig_overview(data: EvalData, out_dir: Path, prefix: str) -> None:
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    fig.suptitle("Warehouse Evaluation — Performance Overview", fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 3)

    # (0,0) Deliveries & expirations per episode
    ax = fig.add_subplot(gs[0, 0])
    deliveries = [ep.total_deliveries for ep in data.episodes]
    expired = [ep.expired_tasks_total for ep in data.episodes]
    x = np.arange(len(data.episodes))
    ax.bar(x, deliveries, color="#2A9D8F", alpha=0.85, label="Deliveries")
    ax.bar(x, expired, bottom=deliveries, color="#E76F51", alpha=0.75, label="Expired tasks")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Deliveries vs Expired Tasks\n"
        f"Mean deliveries: {data.mean_deliveries_per_episode():.1f}  |  "
        f"Mean expired: {data.mean_expired_per_episode():.1f}"
    )
    ax.legend()

    # (0,1) Episode length distribution
    ax = fig.add_subplot(gs[0, 1])
    lengths = [ep.length for ep in data.episodes]
    ax.hist(lengths, bins=min(20, max(4, len(set(lengths)))), color="#7F8C8D", alpha=0.85)
    ax.axvline(float(np.mean(lengths)), color="black", linestyle="--", linewidth=1.5,
               label=f"Mean: {np.mean(lengths):.0f}")
    ax.set_xlabel("Episode length (steps)")
    ax.set_ylabel("Count")
    ax.set_title("Episode Length Distribution")
    ax.legend()

    # (0,2) Mean reward per agent
    ax = fig.add_subplot(gs[0, 2])
    mean_rewards = [data.mean_total_reward(a) for a in data.agents]
    ax.bar(
        [f"A{i}" for i in range(len(data.agents))],
        mean_rewards,
        color=_AGENT_COLORS[: len(data.agents)],
        alpha=0.85,
    )
    ax.set_ylabel("Mean total reward")
    ax.set_title("Mean Episode Return per Agent")
    for i, v in enumerate(mean_rewards):
        ax.text(i, v + abs(v) * 0.02, f"{v:.0f}", ha="center", fontsize=8)

    # (1,0) Rescue & stranded events
    ax = fig.add_subplot(gs[1, 0])
    rescues = [len(ep.rescue_event_steps()) for ep in data.episodes]
    stranded = [len(ep.stranded_event_steps()) for ep in data.episodes]
    x = np.arange(len(data.episodes))
    ax.bar(x, rescues, color="#8172B2", alpha=0.85, label="Rescue steps")
    ax.bar(x, stranded, color="#C44E52", alpha=0.6, label="Stranded steps")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title(
        f"Rescue Activity\nMean rescue steps/ep: {np.mean(rescues):.1f}"
    )
    ax.legend()

    # (1,1) Deliveries by agent (box/violin)
    ax = fig.add_subplot(gs[1, 1])
    per_agent_deliveries = []
    for a in data.agents:
        vals = [ep.deliveries_by_agent.get(a, 0) for ep in data.episodes]
        per_agent_deliveries.append(vals)
    parts = ax.violinplot(per_agent_deliveries, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(_AGENT_COLORS[i % len(_AGENT_COLORS)])
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(data.agents) + 1))
    ax.set_xticklabels([f"A{i}" for i in range(len(data.agents))])
    ax.set_ylabel("Deliveries")
    ax.set_title("Deliveries per Agent (Distribution)")

    # (1,2) Role comparison (fast vs slow agent deliveries)
    ax = fig.add_subplot(gs[1, 2])
    role_d = data.role_deliveries()
    if role_d:
        ax.bar(list(role_d.keys()), list(role_d.values()),
               color=["#E07B39", "#4C72B0"], alpha=0.85)
        ax.set_ylabel("Mean deliveries/episode")
        ax.set_title("Deliveries by Heterogeneous Role")
        for i, (k, v) in enumerate(role_d.items()):
            ax.text(i, v + 0.05, f"{v:.1f}", ha="center", fontsize=9)

    _save(fig, out_dir / f"{prefix}_1_overview")


# ─── Figure 2: Cumulative delivery trajectory ─────────────────────────────────


def fig_delivery_trajectory(data: EvalData, out_dir: Path, prefix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Warehouse — Cumulative Delivery Trajectory", fontsize=13, fontweight="bold")

    t, mean_d, std_d = data.mean_cumulative_deliveries_over_time()
    # Trim trailing NaN
    valid = ~np.isnan(mean_d)
    t, mean_d, std_d = t[valid], mean_d[valid], std_d[valid]

    # Left: mean ± std band
    ax = axes[0]
    ax.plot(t, mean_d, color="#2A9D8F", linewidth=2, label="Mean")
    ax.fill_between(t, mean_d - std_d, mean_d + std_d, alpha=0.25, color="#2A9D8F", label="±1 SD")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total deliveries")
    ax.set_title("Cumulative Deliveries Over Episode Time")
    ax.legend()

    # Right: individual episode curves
    ax = axes[1]
    for ep in data.episodes[:20]:
        series = ep.cumulative_deliveries_series()
        steps = np.arange(len(series))
        ax.plot(steps, series, alpha=0.35, linewidth=0.9, color="#2A9D8F")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total deliveries")
    ax.set_title("Individual Episode Delivery Curves")

    _save(fig, out_dir / f"{prefix}_2_delivery_trajectory")


# ─── Figure 3: Phase distribution heatmap ─────────────────────────────────────


def fig_phase_heatmap(data: EvalData, out_dir: Path, prefix: str) -> None:
    phase_frac = data.phase_fraction_over_time()  # (max_cycles, 4)
    # Trim trailing zeros
    last_nonzero = np.max(np.nonzero(phase_frac.sum(axis=1))[0]) + 1 if phase_frac.any() else 1
    phase_frac = phase_frac[:last_nonzero]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    fig.suptitle(
        "Resource Phase Distribution Over Episode Time",
        fontsize=13,
        fontweight="bold",
    )

    # Left: stacked area chart
    ax = axes[0]
    t = np.arange(len(phase_frac))
    bottom = np.zeros(len(t))
    for p_idx, (name, color) in enumerate(zip(PHASE_NAMES.values(), _PHASE_COLORS)):
        frac = phase_frac[:, p_idx]
        ax.fill_between(t, bottom, bottom + frac, alpha=0.75, color=color, label=name)
        bottom += frac
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction of active agents")
    ax.set_title("Agent Phase Distribution (Stacked Area)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")

    # Right: heatmap (phase × time)
    ax = axes[1]
    im = ax.imshow(
        phase_frac.T,
        aspect="auto",
        origin="lower",
        cmap="YlOrRd",
        extent=[0, len(phase_frac), -0.5, 3.5],
    )
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(list(PHASE_NAMES.values()))
    ax.set_xlabel("Step")
    ax.set_title("Phase Fraction Heatmap")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of agents")

    _save(fig, out_dir / f"{prefix}_3_phase_heatmap")


# ─── Figure 4: Spatial visitation heatmaps ────────────────────────────────────


def fig_spatial_heatmaps(data: EvalData, out_dir: Path, prefix: str) -> None:
    n_agents = len(data.agents)
    ncols = min(n_agents, 4)
    nrows = (n_agents + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), constrained_layout=True)
    fig.suptitle(
        "Agent Visitation Heatmaps — 12×16 Warehouse Grid",
        fontsize=13,
        fontweight="bold",
    )
    axes_flat = np.array(axes).ravel() if n_agents > 1 else [axes]

    for ax, agent, color in zip(axes_flat, data.agents, _AGENT_COLORS):
        hmap = data.position_heatmap(agent)
        im = ax.imshow(hmap, origin="upper", cmap="Blues", aspect="auto")
        ax.set_title(f"{agent}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Visits")

        # Get speed/role from first non-empty episode
        speed_label = ""
        for ep in data.episodes:
            if ep.steps:
                state = ep.steps[-1].agent_states.get(agent)
                if state:
                    speed_label = f" (speed={int(state.speed)}, cap={state.capacity})"
                    break
        ax.set_title(f"{agent}{speed_label}")

    for ax in axes_flat[n_agents:]:
        ax.axis("off")

    _save(fig, out_dir / f"{prefix}_4_spatial_heatmaps")


# ─── Figure 5: Battery dynamics ───────────────────────────────────────────────


def fig_battery_dynamics(data: EvalData, out_dir: Path, prefix: str) -> None:
    n_agents = len(data.agents)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    fig.suptitle("Battery Dynamics Over Episode Time", fontsize=13, fontweight="bold")

    # Top: mean battery per agent
    ax = axes[0]
    for agent, color in zip(data.agents, _AGENT_COLORS):
        t, mean_batt = data.mean_battery_over_time(agent)
        valid = ~np.isnan(mean_batt)
        ax.plot(t[valid], mean_batt[valid], color=color, linewidth=2, label=agent)
    ax.axhline(25, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Critical (25)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Battery level")
    ax.set_title("Mean Battery Level per Agent")
    ax.legend()

    # Bottom: fraction of agents with critical battery over time
    ax = axes[1]
    max_len = data.max_cycles
    critical_frac = np.full(max_len, np.nan)
    counts = np.zeros(max_len)
    for ep in data.episodes:
        for step in ep.steps:
            t = step.step
            if t >= max_len:
                continue
            active = [s for s in step.agent_states.values() if s.active]
            if active:
                crit = sum(1 for s in active if s.is_battery_critical)
                critical_frac[t] = (critical_frac[t] if not np.isnan(critical_frac[t]) else 0) + crit / len(active)
                counts[t] += 1
    valid_t = counts > 0
    if valid_t.any():
        t_arr = np.arange(max_len)
        frac = np.where(valid_t, critical_frac / np.where(counts > 0, counts, 1), np.nan)
        ax.plot(t_arr[valid_t], frac[valid_t], color="#C44E52", linewidth=2)
        ax.fill_between(t_arr[valid_t], 0, frac[valid_t], alpha=0.3, color="#C44E52")
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction of agents")
    ax.set_ylim(0, 1)
    ax.set_title("Fraction of Active Agents with Critical Battery (< 25)")

    _save(fig, out_dir / f"{prefix}_5_battery_dynamics")


# ─── Figure 6: Heterogeneous role comparison ──────────────────────────────────


def fig_role_comparison(data: EvalData, out_dir: Path, prefix: str) -> None:
    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    fig.suptitle("Heterogeneous Agent Role Comparison", fontsize=13, fontweight="bold")
    gs = fig.add_gridspec(1, 3)

    # Collect per-agent role stats across episodes
    roles: dict = {}  # agent_name → {speed, capacity, deliveries_list, mean_battery_list}
    for ep in data.episodes:
        if not ep.steps:
            continue
        last = ep.steps[-1]
        for a in ep.agents:
            state = last.agent_states.get(a)
            if state is None:
                continue
            if a not in roles:
                roles[a] = {
                    "speed": state.speed,
                    "capacity": state.capacity,
                    "deliveries": [],
                    "battery": [],
                }
            roles[a]["deliveries"].append(float(state.total_deliveries))
            # mean battery over episode
            batt_series = ep.battery_series(a)
            if len(batt_series) > 0:
                roles[a]["battery"].append(float(batt_series.mean()))

    agent_names = sorted(roles.keys())
    labels = [f"{a}\n(sp={int(roles[a]['speed'])}, cap={roles[a]['capacity']})" for a in agent_names]
    colors = [_AGENT_COLORS[i % len(_AGENT_COLORS)] for i in range(len(agent_names))]

    # (0) Deliveries
    ax = fig.add_subplot(gs[0])
    means = [float(np.mean(roles[a]["deliveries"])) if roles[a]["deliveries"] else 0.0 for a in agent_names]
    stds = [float(np.std(roles[a]["deliveries"])) if roles[a]["deliveries"] else 0.0 for a in agent_names]
    bars = ax.bar(labels, means, color=colors, alpha=0.85, yerr=stds, capsize=4)
    ax.set_ylabel("Mean deliveries/episode")
    ax.set_title("Deliveries by Agent Role")

    # (1) Mean battery
    ax = fig.add_subplot(gs[1])
    batt_means = [float(np.mean(roles[a]["battery"])) if roles[a]["battery"] else 0.0 for a in agent_names]
    ax.bar(labels, batt_means, color=colors, alpha=0.85)
    ax.axhline(25, color="red", linestyle="--", linewidth=1.2, label="Critical threshold")
    ax.set_ylabel("Mean battery level")
    ax.set_title("Mean Battery by Agent Role")
    ax.legend()

    # (2) Deliveries per role group (fast vs slow, high-cap vs low-cap)
    ax = fig.add_subplot(gs[2])
    role_labels_all, role_vals = [], []
    fast_d = [d for a in agent_names for d in roles[a]["deliveries"] if roles[a]["speed"] >= 2]
    slow_d = [d for a in agent_names for d in roles[a]["deliveries"] if roles[a]["speed"] < 2]
    highcap_d = [d for a in agent_names for d in roles[a]["deliveries"] if roles[a]["capacity"] >= 2]
    lowcap_d = [d for a in agent_names for d in roles[a]["deliveries"] if roles[a]["capacity"] < 2]

    groups = {
        "Fast\n(speed=2)": fast_d,
        "Slow\n(speed=1)": slow_d,
        "High-cap\n(cap=2)": highcap_d,
        "Low-cap\n(cap=1)": lowcap_d,
    }
    for label, vals in groups.items():
        if vals:
            role_labels_all.append(label)
            role_vals.append(vals)

    if role_vals:
        parts = ax.violinplot(role_vals, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(role_labels_all) + 1))
        ax.set_xticklabels(role_labels_all)
        ax.set_ylabel("Deliveries/episode")
        ax.set_title("Deliveries by Role Group")

    _save(fig, out_dir / f"{prefix}_6_role_comparison")


# ─── Figure 7: Task deadline pressure ─────────────────────────────────────────


def fig_task_pressure(data: EvalData, out_dir: Path, prefix: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)
    fig.suptitle("Task Deadline Pressure Over Episode Time", fontsize=13, fontweight="bold")

    max_len = data.max_cycles
    pending_mat = np.full((len(data.episodes), max_len), np.nan)
    expired_mat = np.full((len(data.episodes), max_len), np.nan)

    for i, ep in enumerate(data.episodes):
        prev_expired = 0
        for step in ep.steps:
            t = step.step
            if t >= max_len:
                continue
            pending_mat[i, t] = float(step.pending_tasks)
            # Convert cumulative expired to per-step delta (show increases)
            expired_mat[i, t] = float(step.expired_tasks_cumulative)
            prev_expired = step.expired_tasks_cumulative

    # Top: pending tasks
    ax = axes[0]
    mean_pending = np.nanmean(pending_mat, axis=0)
    std_pending = np.nanstd(pending_mat, axis=0)
    t = np.arange(max_len)
    valid = ~np.isnan(mean_pending)
    ax.plot(t[valid], mean_pending[valid], color="#E07B39", linewidth=2, label="Mean pending")
    ax.fill_between(
        t[valid],
        (mean_pending - std_pending)[valid],
        (mean_pending + std_pending)[valid],
        alpha=0.25,
        color="#E07B39",
    )
    ax.axhline(15, color="red", linestyle="--", linewidth=1.0, alpha=0.7, label="Queue cap (15)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Pending tasks")
    ax.set_title("Mean Pending Task Queue Size")
    ax.legend()

    # Bottom: cumulative expired tasks
    ax = axes[1]
    mean_exp = np.nanmean(expired_mat, axis=0)
    std_exp = np.nanstd(expired_mat, axis=0)
    valid = ~np.isnan(mean_exp)
    ax.plot(t[valid], mean_exp[valid], color="#C44E52", linewidth=2, label="Mean expired (cumulative)")
    ax.fill_between(
        t[valid],
        (mean_exp - std_exp)[valid],
        (mean_exp + std_exp)[valid],
        alpha=0.25,
        color="#C44E52",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Expired tasks (cumulative)")
    ax.set_title("Cumulative Expired Tasks Over Episode")
    ax.legend()

    _save(fig, out_dir / f"{prefix}_7_task_pressure")


# ─── Master save function ─────────────────────────────────────────────────────


def save_all_warehouse_figures(
    data: EvalData,
    output_dir: str | Path = "eval_plots/warehouse",
    prefix: str = "warehouse",
    dpi: int = 150,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Warehouse Evaluation Analysis → {out}")
    print(f"{'=' * 60}")
    print(f"  Episodes         : {len(data)}")
    print(f"  Mean deliveries  : {data.mean_deliveries_per_episode():.1f}")
    print(f"  Mean expired     : {data.mean_expired_per_episode():.1f}")
    print(f"  Mean ep. length  : {data.mean_episode_length():.0f} / {data.max_cycles}")
    print(f"  Mean rescues/ep  : {data.mean_rescue_events_per_episode():.1f} steps")
    for a in data.agents:
        print(f"    {a}: mean reward {data.mean_total_reward(a):+.1f}")
    print(f"{'=' * 60}\n")

    fig_overview(data, out, prefix)
    fig_delivery_trajectory(data, out, prefix)
    fig_phase_heatmap(data, out, prefix)
    fig_spatial_heatmaps(data, out, prefix)
    fig_battery_dynamics(data, out, prefix)
    fig_role_comparison(data, out, prefix)
    fig_task_pressure(data, out, prefix)

    print(f"\nAll warehouse plots saved to {out}/")
