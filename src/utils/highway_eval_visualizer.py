"""
Highway Intersection Evaluation Visualizer
===========================================

Visualization functions for highway intersection evaluation metrics.
Creates comprehensive plots for analyzing multi-agent traffic behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .highway_eval_data import ACTION_NAMES, IntersectionEvalData

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


def _savefig(fig: Figure, path: Path, dpi: int = 150) -> None:
    """Save figure to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_overview(
    data: IntersectionEvalData, save_path: Optional[Path] = None
) -> Figure:
    """
    Plot overview of intersection performance metrics.

    Shows:
    - Episode rewards over time
    - Collision and arrival rates
    - Episode lengths
    - Per-agent average rewards
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        "Highway Intersection – Evaluation Overview", fontsize=15, fontweight="bold"
    )

    episodes = np.arange(len(data.episodes))
    rewards = np.array(data.per_episode_rewards(), dtype=np.float32)
    collisions = np.array(data.per_episode_collisions(), dtype=np.float32)
    arrivals = np.array(data.per_episode_arrivals(), dtype=np.float32)
    lengths = np.array([ep.episode_length for ep in data.episodes], dtype=np.float32)

    # Episode rewards
    ax = axes[0, 0]
    ax.plot(episodes, rewards, color="#2471A3", linewidth=2)
    ax.axhline(
        y=np.mean(rewards),
        color="#E74C3C",
        linestyle="--",
        alpha=0.7,
        label=f"Mean: {np.mean(rewards):.2f}",
    )
    ax.set_title("Total Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()

    # Collisions and arrivals
    ax = axes[0, 1]
    ax.plot(
        episodes,
        collisions,
        color="#E74C3C",
        linewidth=2,
        alpha=0.8,
        label="Collisions",
    )
    ax.plot(
        episodes, arrivals, color="#27AE60", linewidth=2, alpha=0.8, label="Arrivals"
    )
    ax.set_title("Collisions and Arrivals per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Count")
    ax.legend()

    # Episode lengths
    ax = axes[1, 0]
    ax.plot(episodes, lengths, color="#7F8C8D", linewidth=2)
    ax.axhline(
        y=data.duration,
        color="#E67E22",
        linestyle="--",
        alpha=0.7,
        label=f"Max: {data.duration}",
    )
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend()

    # Per-agent average rewards
    ax = axes[1, 1]
    agent_names = data.agent_names
    means = [data.mean_reward_per_agent(name) for name in agent_names]
    x = np.arange(len(agent_names))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(agent_names)))
    ax.bar(x, means, color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=45, ha="right")
    ax.set_title("Average Reward per Agent")
    ax.set_ylabel("Reward")

    if save_path:
        _savefig(fig, save_path)
    return fig


def plot_safety_metrics(
    data: IntersectionEvalData, save_path: Optional[Path] = None
) -> Figure:
    """
    Plot safety metrics: collision and arrival rates per agent.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(
        "Highway Intersection – Safety Metrics", fontsize=15, fontweight="bold"
    )

    agent_names = data.agent_names
    x = np.arange(len(agent_names))

    # Collision rates
    ax = axes[0]
    collision_rates = [data.collision_rate_per_agent(name) for name in agent_names]
    ax.bar(x, collision_rates, color="#E74C3C", alpha=0.85)
    ax.axhline(
        y=data.collision_rate(),
        color="#C0392B",
        linestyle="--",
        alpha=0.7,
        label=f"Overall: {data.collision_rate():.2%}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=45, ha="right")
    ax.set_title("Collision Rate per Agent")
    ax.set_ylabel("Collision Rate")
    ax.set_ylim(0, 1.0)
    ax.legend()

    # Arrival rates
    ax = axes[1]
    arrival_rates = [data.arrival_rate_per_agent(name) for name in agent_names]
    ax.bar(x, arrival_rates, color="#27AE60", alpha=0.85)
    ax.axhline(
        y=data.arrival_rate(),
        color="#229954",
        linestyle="--",
        alpha=0.7,
        label=f"Overall: {data.arrival_rate():.2%}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=45, ha="right")
    ax.set_title("Arrival Rate per Agent")
    ax.set_ylabel("Arrival Rate")
    ax.set_ylim(0, 1.0)
    ax.legend()

    if save_path:
        _savefig(fig, save_path)
    return fig


def plot_action_distributions(
    data: IntersectionEvalData, save_path: Optional[Path] = None
) -> Figure:
    """
    Plot action distribution for each agent.

    Shows the frequency of each discrete action:
    LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER
    """
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    fig.suptitle(
        "Highway Intersection – Action Distributions", fontsize=14, fontweight="bold"
    )

    agent_names = data.agent_names
    actions = list(ACTION_NAMES.keys())
    action_labels = [ACTION_NAMES[a] for a in actions]

    x = np.arange(len(actions))
    width = 0.8 / len(agent_names)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(agent_names)))

    for idx, (name, color) in enumerate(zip(agent_names, colors)):
        dist = data.overall_action_distribution(name)
        freqs = [dist.get(action, 0.0) for action in actions]
        offset = (idx - len(agent_names) / 2 + 0.5) * width
        ax.bar(x + offset, freqs, width=width, alpha=0.85, label=name, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(action_labels, rotation=45, ha="right")
    ax.set_xlabel("Action")
    ax.set_ylabel("Frequency")
    ax.set_title("Policy Action Frequencies")
    ax.legend()

    if save_path:
        _savefig(fig, save_path)
    return fig


def plot_speed_analysis(
    data: IntersectionEvalData, save_path: Optional[Path] = None
) -> Figure:
    """
    Plot speed analysis for each agent.

    Shows average speeds maintained by each vehicle.
    """
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fig.suptitle(
        "Highway Intersection – Speed Analysis", fontsize=14, fontweight="bold"
    )

    agent_names = data.agent_names
    x = np.arange(len(agent_names))
    speeds = [data.mean_speed_per_agent(name) for name in agent_names]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(agent_names)))

    ax.bar(x, speeds, color=colors, alpha=0.9)
    ax.axhline(
        y=np.mean(speeds),
        color="#E74C3C",
        linestyle="--",
        alpha=0.7,
        label=f"Mean: {np.mean(speeds):.2f} m/s",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=45, ha="right")
    ax.set_title("Average Speed per Agent")
    ax.set_ylabel("Speed (m/s)")
    ax.legend()

    if save_path:
        _savefig(fig, save_path)
    return fig


def plot_trajectory_heatmap(
    data: IntersectionEvalData, save_path: Optional[Path] = None
) -> Figure:
    """
    Plot heatmap of vehicle positions across all episodes.

    Shows where vehicles spend most of their time in the intersection.
    """
    fig, axes = plt.subplots(
        1,
        len(data.agent_names),
        figsize=(4 * len(data.agent_names), 4),
        constrained_layout=True,
    )
    if len(data.agent_names) == 1:
        axes = [axes]

    fig.suptitle(
        "Highway Intersection – Vehicle Position Heatmaps",
        fontsize=14,
        fontweight="bold",
    )

    for ax, agent_name in zip(axes, data.agent_names):
        # Collect all positions
        positions = []
        for ep in data.episodes:
            if agent_name in ep.vehicle_trajectories:
                for step in ep.vehicle_trajectories[agent_name]:
                    positions.append(step.position)

        if not positions:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(agent_name)
            continue

        positions = np.array(positions)
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]

        # Create 2D histogram
        ax.hist2d(
            x_coords,
            y_coords,
            bins=30,
            cmap="YlOrRd",
            cmin=1,
        )
        ax.set_title(agent_name)
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
        ax.set_aspect("equal")

    if save_path:
        _savefig(fig, save_path)
    return fig


def save_all_highway_figures(
    data: IntersectionEvalData,
    output_dir: str | Path = "eval_plots",
    prefix: str = "",
) -> None:
    """
    Generate and save all highway intersection analysis plots.

    Args:
        data: Collected evaluation data
        output_dir: Directory to save plots
        prefix: Prefix for filenames
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pfx = f"{prefix}_" if prefix else ""

    print(f"\n{'=' * 60}")
    print(f" Generating highway intersection analysis plots → {out}")
    print(f"{'=' * 60}")

    plot_overview(data, save_path=out / f"{pfx}1_highway_overview.png")
    plot_safety_metrics(data, save_path=out / f"{pfx}2_highway_safety.png")
    plot_action_distributions(data, save_path=out / f"{pfx}3_highway_actions.png")
    plot_speed_analysis(data, save_path=out / f"{pfx}4_highway_speeds.png")
    plot_trajectory_heatmap(data, save_path=out / f"{pfx}5_highway_trajectories.png")

    print(f"\nAll plots saved to {out}/")
