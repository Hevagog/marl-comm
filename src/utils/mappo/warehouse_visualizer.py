"""
warehouse_eval_visualizer.py
============================

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .warehouse_data import (
    ACTION_NAMES,
    PHASE_NAMES,
    EvalData,
)

from utils.shared.style import _AGENT_COLORS, _PHASE_COLORS, save_figure


def _save(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    save_figure(fig, path.parent, path.name, dpi=dpi)


# ─── Figure 1: Overview dashboard ─────────────────────────────────────────────


def fig_overview(data: EvalData, out_dir: Path, prefix: str) -> None:
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    fig.suptitle(
        "Warehouse Evaluation — Performance Overview", fontsize=14, fontweight="bold"
    )
    gs = fig.add_gridspec(2, 3)

    # (0,0) Deliveries & expirations per episode
    ax = fig.add_subplot(gs[0, 0])
    deliveries = [ep.total_deliveries for ep in data.episodes]
    expired = [ep.expired_tasks_total for ep in data.episodes]
    x = np.arange(len(data.episodes))
    ax.bar(x, deliveries, color="#2A9D8F", alpha=0.85, label="Deliveries")
    ax.bar(
        x,
        expired,
        bottom=deliveries,
        color="#E76F51",
        alpha=0.75,
        label="Expired tasks",
    )
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
    ax.hist(
        lengths, bins=min(20, max(4, len(set(lengths)))), color="#7F8C8D", alpha=0.85
    )
    ax.axvline(
        float(np.mean(lengths)),
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {np.mean(lengths):.0f}",
    )
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
    ax.set_title(f"Rescue Activity\nMean rescue steps/ep: {np.mean(rescues):.1f}")
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
    ax.set_title("Deliveries per Agent (true counts; see Fig. 10 for details)")

    # (1,2) Role comparison (fast vs slow agent deliveries)
    ax = fig.add_subplot(gs[1, 2])
    role_d = data.role_deliveries()
    if role_d:
        ax.bar(
            list(role_d.keys()),
            list(role_d.values()),
            color=["#E07B39", "#4C72B0"],
            alpha=0.85,
        )
        ax.set_ylabel("Mean deliveries/episode")
        ax.set_title("Deliveries by Heterogeneous Role")
        for i, (k, v) in enumerate(role_d.items()):
            ax.text(i, v + 0.05, f"{v:.1f}", ha="center", fontsize=9)

    _save(fig, out_dir / f"{prefix}_1_overview")


# ─── Figure 2: Cumulative delivery trajectory ─────────────────────────────────


def fig_delivery_trajectory(data: EvalData, out_dir: Path, prefix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(
        "Warehouse — Cumulative Delivery Trajectory", fontsize=13, fontweight="bold"
    )

    t, mean_d, std_d = data.mean_cumulative_deliveries_over_time()
    # Trim trailing NaN
    valid = ~np.isnan(mean_d)
    t, mean_d, std_d = t[valid], mean_d[valid], std_d[valid]

    # Left: mean ± std band
    ax = axes[0]
    ax.plot(t, mean_d, color="#2A9D8F", linewidth=2, label="Mean")
    ax.fill_between(
        t, mean_d - std_d, mean_d + std_d, alpha=0.25, color="#2A9D8F", label="±1 SD"
    )
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
    last_nonzero = (
        np.max(np.nonzero(phase_frac.sum(axis=1))[0]) + 1 if phase_frac.any() else 1
    )
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
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), constrained_layout=True
    )
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
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    fig.suptitle("Battery Dynamics Over Episode Time", fontsize=13, fontweight="bold")

    # Top: mean battery per agent
    ax = axes[0]
    for agent, color in zip(data.agents, _AGENT_COLORS):
        t, mean_batt = data.mean_battery_over_time(agent)
        valid = ~np.isnan(mean_batt)
        ax.plot(t[valid], mean_batt[valid], color=color, linewidth=2, label=agent)
    ax.axhline(
        25, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Critical (25)"
    )
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
                critical_frac[t] = (
                    critical_frac[t] if not np.isnan(critical_frac[t]) else 0
                ) + crit / len(active)
                counts[t] += 1
    valid_t = counts > 0
    if valid_t.any():
        t_arr = np.arange(max_len)
        frac = np.where(
            valid_t, critical_frac / np.where(counts > 0, counts, 1), np.nan
        )
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
            roles[a]["deliveries"].append(float(state.deliveries))
            # mean battery over episode
            batt_series = ep.battery_series(a)
            if len(batt_series) > 0:
                roles[a]["battery"].append(float(batt_series.mean()))

    agent_names = sorted(roles.keys())
    labels = [
        f"{a}\n(sp={int(roles[a]['speed'])}, cap={roles[a]['capacity']})"
        for a in agent_names
    ]
    colors = [_AGENT_COLORS[i % len(_AGENT_COLORS)] for i in range(len(agent_names))]

    # (0) Deliveries
    ax = fig.add_subplot(gs[0])
    means = [
        float(np.mean(roles[a]["deliveries"])) if roles[a]["deliveries"] else 0.0
        for a in agent_names
    ]
    stds = [
        float(np.std(roles[a]["deliveries"])) if roles[a]["deliveries"] else 0.0
        for a in agent_names
    ]
    ax.bar(labels, means, color=colors, alpha=0.85, yerr=stds, capsize=4)
    ax.set_ylabel("Mean deliveries/episode")
    ax.set_title("Deliveries by Agent Role")

    # (1) Mean battery
    ax = fig.add_subplot(gs[1])
    batt_means = [
        float(np.mean(roles[a]["battery"])) if roles[a]["battery"] else 0.0
        for a in agent_names
    ]
    ax.bar(labels, batt_means, color=colors, alpha=0.85)
    ax.axhline(
        25, color="red", linestyle="--", linewidth=1.2, label="Critical threshold"
    )
    ax.set_ylabel("Mean battery level")
    ax.set_title("Mean Battery by Agent Role")
    ax.legend()

    # (2) Deliveries per role group (fast vs slow, high-cap vs low-cap)
    ax = fig.add_subplot(gs[2])
    role_labels_all, role_vals = [], []
    fast_d = [
        d for a in agent_names for d in roles[a]["deliveries"] if roles[a]["speed"] >= 2
    ]
    slow_d = [
        d for a in agent_names for d in roles[a]["deliveries"] if roles[a]["speed"] < 2
    ]
    highcap_d = [
        d
        for a in agent_names
        for d in roles[a]["deliveries"]
        if roles[a]["capacity"] >= 2
    ]
    lowcap_d = [
        d
        for a in agent_names
        for d in roles[a]["deliveries"]
        if roles[a]["capacity"] < 2
    ]

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
        ax.violinplot(role_vals, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(role_labels_all) + 1))
        ax.set_xticklabels(role_labels_all)
        ax.set_ylabel("Deliveries/episode")
        ax.set_title("Deliveries by Role Group")

    _save(fig, out_dir / f"{prefix}_6_role_comparison")


# ─── Figure 7: Task deadline pressure ─────────────────────────────────────────


def fig_task_pressure(data: EvalData, out_dir: Path, prefix: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)
    fig.suptitle(
        "Task Deadline Pressure Over Episode Time", fontsize=13, fontweight="bold"
    )

    max_len = data.max_cycles
    pending_mat = np.full((len(data.episodes), max_len), np.nan)
    expired_mat = np.full((len(data.episodes), max_len), np.nan)

    for i, ep in enumerate(data.episodes):
        for step in ep.steps:
            t = step.step
            if t >= max_len:
                continue
            pending_mat[i, t] = float(step.pending_tasks)
            # Convert cumulative expired to per-step delta (show increases)
            expired_mat[i, t] = float(step.expired_tasks_cumulative)

    # Top: pending tasks
    ax = axes[0]
    mean_pending = np.nanmean(pending_mat, axis=0)
    std_pending = np.nanstd(pending_mat, axis=0)
    t = np.arange(max_len)
    valid = ~np.isnan(mean_pending)
    ax.plot(
        t[valid],
        mean_pending[valid],
        color="#E07B39",
        linewidth=2,
        label="Mean pending",
    )
    ax.fill_between(
        t[valid],
        (mean_pending - std_pending)[valid],
        (mean_pending + std_pending)[valid],
        alpha=0.25,
        color="#E07B39",
    )
    ax.axhline(
        15,
        color="red",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label="Queue cap (15)",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Pending tasks")
    ax.set_title("Mean Pending Task Queue Size")
    ax.legend()

    # Bottom: cumulative expired tasks
    ax = axes[1]
    mean_exp = np.nanmean(expired_mat, axis=0)
    std_exp = np.nanstd(expired_mat, axis=0)
    valid = ~np.isnan(mean_exp)
    ax.plot(
        t[valid],
        mean_exp[valid],
        color="#C44E52",
        linewidth=2,
        label="Mean expired (cumulative)",
    )
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


# ─── Figure 8: Throughput & Delivery Efficiency ───────────────────────────────


def fig_throughput(data: EvalData, out_dir: Path, prefix: str) -> None:
    """Delivery rate over time (picks per N steps) and delivery success rate."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle(
        "Warehouse Throughput & Delivery Efficiency", fontsize=13, fontweight="bold"
    )

    max_len = data.max_cycles
    window = max(1, max_len // 20)  # rolling window ~5% of episode

    # (0) Rolling delivery rate (deliveries per `window` steps)
    ax = axes[0]
    t_vals = np.arange(max_len)
    rate_mat = np.full((len(data.episodes), max_len), np.nan)
    for ep_i, ep in enumerate(data.episodes):
        cumulative = np.zeros(max_len)
        for step in ep.steps:
            t = step.step
            if t < max_len and step.agent_states:
                # env-global counter replicated to every agent → max, not sum
                # (analysis.md §4 Bug 1).
                cumulative[t] = max(
                    s.total_deliveries for s in step.agent_states.values()
                )
        # Fill forward any zeros (episode may end early)
        last_val = 0.0
        for t in range(max_len):
            if cumulative[t] == 0:
                cumulative[t] = last_val
            else:
                last_val = cumulative[t]
        delta = np.diff(np.concatenate([[0], cumulative]))
        rate = np.convolve(delta, np.ones(window), mode="same")
        rate_mat[ep_i] = rate

    mean_rate = np.nanmean(rate_mat, axis=0)
    std_rate = np.nanstd(rate_mat, axis=0)
    valid = ~np.isnan(mean_rate)
    ax.plot(
        t_vals[valid], mean_rate[valid], color="#2A9D8F", linewidth=2, label="Mean rate"
    )
    ax.fill_between(
        t_vals[valid],
        (mean_rate - std_rate)[valid],
        (mean_rate + std_rate)[valid],
        alpha=0.2,
        color="#2A9D8F",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel(f"Deliveries per {window} steps")
    ax.set_title(f"Rolling Delivery Rate\n(window={window} steps)")
    ax.legend()

    # (1) Delivery success rate = deliveries / (deliveries + expired) per episode
    ax = axes[1]
    success_rates = []
    for ep in data.episodes:
        d = ep.total_deliveries
        e = ep.expired_tasks_total
        rate = d / (d + e) if (d + e) > 0 else 0.0
        success_rates.append(rate * 100)
    ax.hist(success_rates, bins=20, color="#55A868", edgecolor="white", alpha=0.85)
    ax.axvline(
        float(np.mean(success_rates)),
        color="#E07B39",
        linestyle="--",
        linewidth=2,
        label=f"Mean={np.mean(success_rates):.1f}%",
    )
    ax.set_xlabel("Success rate (%)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Delivery Success Rate\n= deliveries / (deliveries + expired)\nMean: {np.mean(success_rates):.1f}%"
    )

    # (2) Deliveries vs episode length scatter (efficiency: more deliveries in fewer steps?)
    ax = axes[2]
    deliveries = [ep.total_deliveries for ep in data.episodes]
    lengths = [ep.length for ep in data.episodes]
    efficiency = [d / max(ln, 1) * 1000 for d, ln in zip(deliveries, lengths)]
    ax.scatter(
        lengths,
        deliveries,
        c=efficiency,
        cmap="viridis",
        s=40,
        alpha=0.7,
        edgecolors="none",
    )
    if len(lengths) > 2:
        z = np.polyfit(lengths, deliveries, 1)
        p = np.poly1d(z)
        xl = np.linspace(min(lengths), max(lengths), 100)
        ax.plot(xl, p(xl), color="#C44E52", linewidth=2, linestyle="--", label="Trend")
        ax.legend()
    ax.set_xlabel("Episode length (steps)")
    ax.set_ylabel("Total deliveries")
    ax.set_title("Deliveries vs Episode Length\n(color = deliveries/1000 steps)")

    _save(fig, out_dir / f"{prefix}_8_throughput")


# ─── Figure 9: Action Distribution & Agent Congestion ─────────────────────────


def fig_action_congestion(data: EvalData, out_dir: Path, prefix: str) -> None:
    """Action usage distribution and pairwise co-location density (congestion)."""
    fig = plt.figure(figsize=(18, 9), constrained_layout=True)
    fig.suptitle(
        "Action Distribution & Spatial Congestion", fontsize=13, fontweight="bold"
    )
    gs = fig.add_gridspec(2, 3)

    # (0,0-1) Action distribution per agent
    ax = fig.add_subplot(gs[0, :2])
    n_agents = len(data.agents)
    x = np.arange(len(ACTION_NAMES))
    bar_w = 0.8 / max(n_agents, 1)
    for a_idx, agent in enumerate(data.agents):
        counts = np.zeros(len(ACTION_NAMES), dtype=float)
        for ep in data.episodes:
            for step in ep.steps:
                a_act = step.actions.get(agent, -1)
                if 0 <= a_act < len(ACTION_NAMES):
                    counts[a_act] += 1
        total = counts.sum()
        freq = counts / total if total > 0 else counts
        offset = (a_idx - n_agents / 2 + 0.5) * bar_w
        ax.bar(
            x + offset,
            freq * 100,
            bar_w * 0.9,
            color=_AGENT_COLORS[a_idx % len(_AGENT_COLORS)],
            alpha=0.82,
            label=agent,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(list(ACTION_NAMES.values()))
    ax.set_ylabel("Action frequency (%)")
    ax.set_title(
        "Action Distribution per Agent\n(INTERACT includes pick/treat/deliver/rescue)"
    )
    ax.axhline(
        100 / len(ACTION_NAMES), color="grey", linestyle="--", linewidth=1, alpha=0.5
    )
    ax.legend(loc="upper right", ncol=min(4, n_agents), fontsize=8)

    # (0,2) Rescue vs normal activity ratio per agent
    ax = fig.add_subplot(gs[0, 2])
    rescue_fracs = []
    for agent in data.agents:
        rescue_steps = interact_steps = 0
        for ep in data.episodes:
            for step in ep.steps:
                a_act = step.actions.get(agent, -1)
                if a_act == 5:  # INTERACT
                    interact_steps += 1
                state = step.agent_states.get(agent)
                if state and state.dragging:
                    rescue_steps += 1
        total_steps = sum(ep.length for ep in data.episodes)
        rescue_fracs.append(rescue_steps / max(total_steps, 1) * 100)

    bars = ax.bar(
        [f"A{i}" for i in range(n_agents)],
        rescue_fracs,
        color=_AGENT_COLORS[:n_agents],
        alpha=0.85,
    )
    for bar, v in zip(bars, rescue_fracs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{v:.1f}%",
            ha="center",
            fontsize=8,
        )
    ax.set_ylabel("% of steps in rescue")
    ax.set_title("Rescue Activity per Agent")

    # (1,0-2) Pairwise co-location density (agents on same/adjacent cell)
    ax = fig.add_subplot(gs[1, :])
    if len(data.agents) >= 2:
        pair_labels = []
        congestion_counts = []
        for i in range(len(data.agents)):
            for j in range(i + 1, len(data.agents)):
                a, b = data.agents[i], data.agents[j]
                pair_labels.append(f"{a[-1]}-{b[-1]}")
                count = 0
                total = 0
                for ep in data.episodes:
                    for step in ep.steps:
                        sa = step.agent_states.get(a)
                        sb = step.agent_states.get(b)
                        if sa and sb and sa.active and sb.active:
                            total += 1
                            dr = abs(sa.position[0] - sb.position[0])
                            dc = abs(sa.position[1] - sb.position[1])
                            if dr <= 1 and dc <= 1:  # within 1-cell neighborhood
                                count += 1
                congestion_counts.append(count / max(total, 1) * 100)
        colors = [
            _AGENT_COLORS[i % len(_AGENT_COLORS)] for i in range(len(pair_labels))
        ]
        bars = ax.bar(
            pair_labels, congestion_counts, color=colors, alpha=0.85, edgecolor="white"
        )
        for bar, v in zip(bars, congestion_counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{v:.1f}%",
                ha="center",
                fontsize=8,
            )
        ax.set_ylabel("% of steps within 1-cell")
        ax.set_title(
            "Pairwise Agent Co-location Rate\n(% steps where agents within 1 cell = congestion metric)"
        )
        ax.axhline(
            float(np.mean(congestion_counts)),
            color="#C44E52",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean={np.mean(congestion_counts):.1f}%",
        )
        ax.legend(fontsize=8)

    _save(fig, out_dir / f"{prefix}_9_action_congestion")


# ─── Figure 10: Scenario-specific cooperation diagnostics ────────────────────


def fig_scenario_metrics(data: EvalData, out_dir: Path, prefix: str) -> None:
    """Per-agent deliveries (true counts), team-synchronized delivery events,
    and rendezvous occupancy — the three diagnostics that distinguish the
    Scenario 1 / Scenario 2 configs introduced for the comm-method ablation.

    Skipped silently when the env did not enable any of the new features.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle(
        "Cooperation Diagnostics (per-agent deliveries · team-sync · rendezvous)",
        fontsize=13,
        fontweight="bold",
    )

    # ---- (0) Per-agent deliveries — uses true per-agent counter ----------
    ax = axes[0]
    per_agent: list[list[int]] = []
    for a in data.agents:
        vals = [int(ep.deliveries_by_agent.get(a, 0)) for ep in data.episodes]
        per_agent.append(vals)
    means = [float(np.mean(v)) if v else 0.0 for v in per_agent]
    stds = [float(np.std(v)) if v else 0.0 for v in per_agent]
    xpos = np.arange(len(data.agents))
    ax.bar(
        xpos,
        means,
        yerr=stds,
        capsize=4,
        color=[_AGENT_COLORS[i % len(_AGENT_COLORS)] for i in range(len(data.agents))],
        alpha=0.85,
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"A{i}" for i in range(len(data.agents))])
    ax.set_ylabel("Deliveries / episode")
    ax.set_title(f"Per-Agent Deliveries (true count)\nTotal mean/ep: {sum(means):.1f}")
    for i, m in enumerate(means):
        ax.text(i, m + max(stds + [0.1]) * 0.1, f"{m:.1f}", ha="center", fontsize=9)

    # ---- (1) Team-synchronized delivery events ---------------------------
    ax = axes[1]
    sync_counts = [ep.synchronized_delivery_events(window=20) for ep in data.episodes]
    total_deliveries = [ep.total_deliveries for ep in data.episodes]
    sync_frac = [
        (s / d) if d > 0 else 0.0 for s, d in zip(sync_counts, total_deliveries)
    ]
    if any(total_deliveries):
        ax.hist(
            sync_frac,
            bins=10,
            range=(0.0, 1.0),
            color="#9467BD",
            alpha=0.85,
            edgecolor="white",
        )
        ax.axvline(
            float(np.mean(sync_frac)),
            color="#E07B39",
            linestyle="--",
            linewidth=2,
            label=f"Mean fraction = {np.mean(sync_frac):.2f}",
        )
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no deliveries", transform=ax.transAxes, ha="center")
    ax.set_xlabel("Fraction of deliveries within ±20 steps of a peer's delivery")
    ax.set_ylabel("Episodes")
    ax.set_title("Team-Synchronized Delivery Rate\n(Scenario 1 metric)")

    # ---- (2) Rendezvous occupancy over time ------------------------------
    ax = axes[2]
    max_len = data.max_cycles
    occupancy = np.zeros(max_len, dtype=np.float32)
    counts = np.zeros(max_len, dtype=np.float32)
    for ep in data.episodes:
        for s in ep.steps:
            t = s.step
            if t >= max_len:
                continue
            n_on = sum(1 for st in s.agent_states.values() if st.on_rendezvous)
            occupancy[t] += n_on
            counts[t] += 1
    safe = np.where(counts > 0, counts, 1)
    mean_occ = occupancy / safe
    if mean_occ.sum() > 0:
        ax.plot(np.arange(max_len), mean_occ, color="#2A9D8F", linewidth=2)
        ax.set_title(
            f"Rendezvous Occupancy (Scenario 2)\nMean agents on cell: {mean_occ.mean():.2f}"
        )
    else:
        ax.text(
            0.5,
            0.5,
            "rendezvous disabled or never visited",
            transform=ax.transAxes,
            ha="center",
        )
        ax.set_title("Rendezvous Occupancy (Scenario 2)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean agents on rendezvous cell")

    _save(fig, out_dir / f"{prefix}_10_scenario_metrics")


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
    fig_throughput(data, out, prefix)
    fig_action_congestion(data, out, prefix)
    fig_scenario_metrics(data, out, prefix)

    print(f"\nAll warehouse plots saved to {out}/")
