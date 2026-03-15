from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .blindspot_eval_data import AGENT_A, AGENT_B, MOVEMENT_NAMES, EvalData

_COLOR_A = "#C0392B"
_COLOR_B = "#2471A3"
_COLOR_GOAL = "#F39C12"
_COLOR_TRAP = "#2C3E50"
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


def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def _episode_lengths(data: EvalData) -> List[int]:
    return [ep.length for ep in data.episodes]


def _time_to_goal(data: EvalData, agent: str) -> List[int]:
    times = []
    for ep in data.episodes:
        t = ep.first_reach_step(agent)
        if t is not None:
            times.append(t + 1)
    return times


def _mean_distance_over_time(data: EvalData, agent: str) -> Tuple[np.ndarray, np.ndarray]:
    if not data.episodes:
        return np.array([]), np.array([])
    max_len = max(ep.length for ep in data.episodes)
    sums = np.zeros(max_len, dtype=np.float32)
    counts = np.zeros(max_len, dtype=np.float32)
    for ep in data.episodes:
        for step in ep.steps:
            idx = step.step
            sums[idx] += step.dist_to_goal[agent]
            counts[idx] += 1
    counts[counts == 0] = np.nan
    means = sums / counts
    return np.arange(max_len), means


def _message_entropy(counts: List[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = np.array(counts, dtype=np.float64) / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _message_action_counts(
    data: EvalData, sender: str, receiver: str, num_actions: int = 5
) -> np.ndarray:
    counts = np.zeros((data.num_message_tokens, num_actions), dtype=np.float64)
    if not data.use_communication:
        return counts
    for ep in data.episodes:
        for idx in range(len(ep.steps) - 1):
            step = ep.steps[idx]
            next_step = ep.steps[idx + 1]
            msg = step.messages.get(sender)
            if msg is None or not (0 <= msg < data.num_message_tokens):
                continue
            recv_action = next_step.actions.get(receiver)
            if recv_action is None or not (0 <= recv_action < num_actions):
                continue
            counts[msg, recv_action] += 1
    return counts


def _mutual_info_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    pxy = counts / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pxy / (px * py)
        log_term = np.where(pxy > 0, np.log2(ratio), 0.0)
    mi = float(np.sum(pxy * log_term))
    return mi


def plot_overview(data: EvalData, save_path: Optional[Path] = None) -> plt.Figure:
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    fig.suptitle("Blind-Spot Navigation - Evaluation Overview", fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # (0,0) Success vs timeout
    ax = fig.add_subplot(gs[0, 0])
    success = sum(1 for ep in data.episodes if ep.success)
    timeout = len(data.episodes) - success
    bars = ax.bar(
        ["Success", "Timeout"],
        [success, timeout],
        color=["#2ECC71", "#E67E22"],
        alpha=0.85,
    )
    ax.set_ylabel("Episodes")
    ax.set_title("Termination Outcomes")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{int(h)}", ha="center")

    # (0,1) Episode length distribution
    ax = fig.add_subplot(gs[0, 1])
    lengths = _episode_lengths(data)
    if lengths:
        ax.hist(lengths, bins=min(20, max(5, len(set(lengths)))), color="#7F8C8D", alpha=0.8)
        ax.axvline(np.mean(lengths), color="black", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Episode length (steps)")
        ax.set_ylabel("Count")
    ax.set_title("Episode Length Distribution")

    # (0,2) Mean reward per agent
    ax = fig.add_subplot(gs[0, 2])
    means = [data.mean_total_reward(AGENT_A), data.mean_total_reward(AGENT_B)]
    ax.bar(["agent_0", "agent_1"], means, color=[_COLOR_A, _COLOR_B], alpha=0.85)
    ax.set_ylabel("Mean total reward")
    ax.set_title("Average Episode Return")

    # (1,0) Trap hits
    ax = fig.add_subplot(gs[1, 0])
    hits = [data.total_trap_hits(AGENT_A), data.total_trap_hits(AGENT_B)]
    ax.bar(["agent_0", "agent_1"], hits, color=[_COLOR_A, _COLOR_B], alpha=0.85)
    ax.set_ylabel("Trap hits")
    ax.set_title("Trap Hits Across Episodes")

    # (1,1) Time to goal distribution
    ax = fig.add_subplot(gs[1, 1])
    times_a = _time_to_goal(data, AGENT_A)
    times_b = _time_to_goal(data, AGENT_B)
    if times_a or times_b:
        ax.violinplot([times_a or [0], times_b or [0]], showmeans=True, showmedians=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["agent_0", "agent_1"])
        ax.set_ylabel("Steps to first goal")
    ax.set_title("Time to First Goal (Reached Episodes)")

    # (1,2) Mean distance to goal over time
    ax = fig.add_subplot(gs[1, 2])
    xs_a, means_a = _mean_distance_over_time(data, AGENT_A)
    xs_b, means_b = _mean_distance_over_time(data, AGENT_B)
    if len(xs_a) > 0:
        ax.plot(xs_a, means_a, color=_COLOR_A, label="agent_0", linewidth=2)
    if len(xs_b) > 0:
        ax.plot(xs_b, means_b, color=_COLOR_B, label="agent_1", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Manhattan distance")
    ax.set_title("Distance to Goal Over Time")
    ax.legend()

    if save_path:
        _savefig(fig, save_path)
    return fig


def plot_heatmaps(data: EvalData, save_path: Optional[Path] = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    fig.suptitle("Blind-Spot Navigation - Visitation Heatmaps", fontsize=13, fontweight="bold")

    for ax, agent, color in zip(axes, [AGENT_A, AGENT_B], [_COLOR_A, _COLOR_B]):
        counts = np.zeros((data.grid_size, data.grid_size), dtype=np.float32)
        for (x, y) in data.position_counts(agent):
            counts[y, x] += 1
        im = ax.imshow(counts, origin="upper", cmap="viridis")
        ax.set_title(f"{agent} visitation")
        ax.set_xticks(range(data.grid_size))
        ax.set_yticks(range(data.grid_size))
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if data.episodes:
            traps = data.episodes[0].traps
            goal = data.episodes[0].goal
            for (tx, ty) in traps:
                ax.scatter(tx, ty, marker="x", color=_COLOR_TRAP, s=40)
            ax.scatter(goal[0], goal[1], marker="*", color=_COLOR_GOAL, s=80, edgecolor="black")

    if save_path:
        _savefig(fig, save_path)
    return fig


def plot_trajectories(
    data: EvalData, n_episodes: int = 6, save_path: Optional[Path] = None
) -> plt.Figure:
    episodes = data.episodes[:n_episodes]
    cols = 3
    rows = math.ceil(len(episodes) / cols) if episodes else 1
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    fig.suptitle("Blind-Spot Navigation - Sample Trajectories", fontsize=13, fontweight="bold")

    for ax, ep in zip(axes, episodes):
        ax.set_title(f"Episode {ep.episode_idx} (len={ep.length})")
        ax.set_xlim(-0.5, data.grid_size - 0.5)
        ax.set_ylim(data.grid_size - 0.5, -0.5)
        ax.set_xticks(range(data.grid_size))
        ax.set_yticks(range(data.grid_size))
        ax.grid(True, linestyle="--", alpha=0.3)

        for (tx, ty) in ep.traps:
            ax.scatter(tx, ty, marker="x", color=_COLOR_TRAP, s=40)
        ax.scatter(ep.goal[0], ep.goal[1], marker="*", color=_COLOR_GOAL, s=80, edgecolor="black")

        path_a = np.array([s.positions[AGENT_A] for s in ep.steps])
        path_b = np.array([s.positions[AGENT_B] for s in ep.steps])
        if len(path_a) > 0:
            ax.plot(path_a[:, 0], path_a[:, 1], color=_COLOR_A, linewidth=2, label="agent_0")
            ax.scatter(path_a[0, 0], path_a[0, 1], color=_COLOR_A, s=30)
            ax.scatter(path_a[-1, 0], path_a[-1, 1], color=_COLOR_A, s=30, marker="s")
        if len(path_b) > 0:
            ax.plot(path_b[:, 0], path_b[:, 1], color=_COLOR_B, linewidth=2, label="agent_1")
            ax.scatter(path_b[0, 0], path_b[0, 1], color=_COLOR_B, s=30)
            ax.scatter(path_b[-1, 0], path_b[-1, 1], color=_COLOR_B, s=30, marker="s")
        ax.legend(loc="upper right")

    for ax in axes[len(episodes) :]:
        ax.axis("off")

    if save_path:
        _savefig(fig, save_path)
    return fig


def plot_messages(data: EvalData, save_path: Optional[Path] = None) -> Optional[plt.Figure]:
    if not data.use_communication:
        return None

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    fig.suptitle("Blind-Spot Navigation - Message Analysis", fontsize=13, fontweight="bold")
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # (0,0) Message token usage
    ax = fig.add_subplot(gs[0, 0])
    counts_a = data.message_counts(AGENT_A)
    counts_b = data.message_counts(AGENT_B)
    x = np.arange(data.num_message_tokens)
    width = 0.35
    ax.bar(x - width / 2, counts_a, width, color=_COLOR_A, alpha=0.85, label="agent_0")
    ax.bar(x + width / 2, counts_b, width, color=_COLOR_B, alpha=0.85, label="agent_1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"m{idx}" for idx in x])
    ax.set_ylabel("Count")
    ax.set_title("Message Token Usage")
    ax.legend()

    # (0,1) Message entropy
    ax = fig.add_subplot(gs[0, 1])
    ent_a = _message_entropy(counts_a)
    ent_b = _message_entropy(counts_b)
    ax.bar(["agent_0", "agent_1"], [ent_a, ent_b], color=[_COLOR_A, _COLOR_B], alpha=0.85)
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Message Entropy")

    # (1,0) Sender A -> Listener B
    ax = fig.add_subplot(gs[1, 0])
    counts_ab = _message_action_counts(data, AGENT_A, AGENT_B)
    mi_ab = _mutual_info_from_counts(counts_ab)
    if counts_ab.sum() > 0:
        row_sums = counts_ab.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        probs_ab = counts_ab / row_sums
        im = ax.imshow(probs_ab, cmap="viridis")
        ax.set_title(f"agent_0 msg -> agent_1 action (MI={mi_ab:.3f})")
        ax.set_xlabel("Movement action")
        ax.set_ylabel("Message token")
        ax.set_xticks(range(5))
        ax.set_xticklabels([MOVEMENT_NAMES[i] for i in range(5)], rotation=45, ha="right")
        ax.set_yticks(range(data.num_message_tokens))
        ax.set_yticklabels([f"m{i}" for i in range(data.num_message_tokens)])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "No messages", ha="center", va="center")
        ax.axis("off")

    # (1,1) Sender B -> Listener A
    ax = fig.add_subplot(gs[1, 1])
    counts_ba = _message_action_counts(data, AGENT_B, AGENT_A)
    mi_ba = _mutual_info_from_counts(counts_ba)
    if counts_ba.sum() > 0:
        row_sums = counts_ba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        probs_ba = counts_ba / row_sums
        im = ax.imshow(probs_ba, cmap="viridis")
        ax.set_title(f"agent_1 msg -> agent_0 action (MI={mi_ba:.3f})")
        ax.set_xlabel("Movement action")
        ax.set_ylabel("Message token")
        ax.set_xticks(range(5))
        ax.set_xticklabels([MOVEMENT_NAMES[i] for i in range(5)], rotation=45, ha="right")
        ax.set_yticks(range(data.num_message_tokens))
        ax.set_yticklabels([f"m{i}" for i in range(data.num_message_tokens)])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "No messages", ha="center", va="center")
        ax.axis("off")

    if save_path:
        _savefig(fig, save_path)
    return fig


def save_all_blindspot_figures(
    data: EvalData,
    output_dir: str | Path = "eval_plots",
    prefix: str = "",
    dpi: int = 150,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pfx = f"{prefix}_" if prefix else ""

    print(f"\n{'=' * 55}")
    print(f" Generating blindspot analysis plots -> {out}")
    print(f"{'=' * 55}")
    print(f" Episodes collected : {len(data)}")
    print(f" Success rate      : {data.success_rate() * 100:.1f}%")
    print(f" Mean ep length    : {data.mean_episode_length():.1f}")
    for agent, label in [(AGENT_A, "agent_0"), (AGENT_B, "agent_1")]:
        print(
            f"  {label:7s} | mean reward: {data.mean_total_reward(agent):+.2f}"
            f" | trap hits: {data.total_trap_hits(agent)}"
        )
    print(f"{'=' * 55}\n")

    plot_overview(data, save_path=out / f"{pfx}1_overview.png")
    plot_heatmaps(data, save_path=out / f"{pfx}2_heatmaps.png")
    plot_trajectories(data, n_episodes=min(6, len(data)), save_path=out / f"{pfx}3_trajectories.png")
    if data.use_communication:
        plot_messages(data, save_path=out / f"{pfx}4_messages.png")

    print(f"\nAll plots saved to {out}/")
