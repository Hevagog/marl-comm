from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .flatland_eval_data import FlatlandEvalData

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
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_overview(data: FlatlandEvalData, save_path: Optional[Path] = None) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    fig.suptitle("Flatland – Evaluation Overview", fontsize=15, fontweight="bold")

    episodes = np.arange(len(data.episodes))
    rewards = np.asarray(data.per_episode_rewards(), dtype=np.float32)
    completions = np.asarray(data.per_episode_completion_ratios(), dtype=np.float32)
    lengths = np.asarray(data.per_episode_lengths(), dtype=np.float32)

    ax = axes[0]
    ax.plot(episodes, rewards, color="#2471A3", linewidth=2)
    ax.axhline(rewards.mean() if rewards.size else 0.0, color="#E74C3C", linestyle="--", alpha=0.7)
    ax.set_title("Team Return per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")

    ax = axes[1]
    ax.plot(episodes, completions, color="#27AE60", linewidth=2)
    ax.set_title("Completion Ratio")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Completed Agents / Team")
    ax.set_ylim(0.0, 1.05)

    ax = axes[2]
    ax.plot(episodes, lengths, color="#7F8C8D", linewidth=2)
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")

    if save_path:
        _savefig(fig, save_path)
    return fig


def plot_completion_histogram(
    data: FlatlandEvalData, save_path: Optional[Path] = None
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    fig.suptitle("Flatland – Per-Agent Completion Histogram", fontsize=14, fontweight="bold")

    histogram = data.completion_histogram()
    agent_names = list(histogram.keys())
    counts = [histogram[name] for name in agent_names]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, max(1, len(agent_names))))

    ax.bar(np.arange(len(agent_names)), counts, color=colors, alpha=0.9)
    ax.set_xticks(np.arange(len(agent_names)))
    ax.set_xticklabels(agent_names, rotation=45, ha="right")
    ax.set_ylabel("Completed Episodes")
    ax.set_title("How often each agent reached its target")

    if save_path:
        _savefig(fig, save_path)
    return fig


def save_all_flatland_figures(
    data: FlatlandEvalData,
    output_dir: str | Path = "eval_plots/flatland",
    prefix: str = "flatland",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pfx = f"{prefix}_" if prefix else ""
    plot_overview(data, save_path=out / f"{pfx}1_flatland_overview.png")
    plot_completion_histogram(
        data, save_path=out / f"{pfx}2_flatland_completion_histogram.png"
    )

