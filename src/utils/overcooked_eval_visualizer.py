from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .overcooked_eval_analysis import OvercookedEvalData

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


def plot_overview(data: OvercookedEvalData, save_path: Optional[Path] = None) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("Overcooked – Evaluation Overview", fontsize=15, fontweight="bold")

    episodes = np.arange(len(data.episodes))
    joint_returns = np.array(
        [ep.joint_return for ep in data.episodes], dtype=np.float32
    )
    sparse_returns = np.array(
        [ep.sparse_return for ep in data.episodes], dtype=np.float32
    )
    deliveries = np.array([ep.deliveries for ep in data.episodes], dtype=np.float32)
    lengths = np.array([ep.steps for ep in data.episodes], dtype=np.float32)

    # Joint and sparse returns by episode
    ax = axes[0, 0]
    ax.plot(episodes, joint_returns, label="Joint return", color="#2471A3", linewidth=2)
    ax.plot(
        episodes,
        sparse_returns,
        label="Sparse return",
        color="#C0392B",
        linewidth=2,
        alpha=0.9,
    )
    ax.set_title("Returns per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.legend()

    # Deliveries by episode
    ax = axes[0, 1]
    ax.bar(episodes, deliveries, color="#27AE60", alpha=0.85)
    ax.set_title("Deliveries per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Count")

    # Episode lengths
    ax = axes[1, 0]
    ax.plot(episodes, lengths, color="#7F8C8D", linewidth=2)
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")

    # Per-agent average return
    ax = axes[1, 1]
    means = []
    labels = []
    for name in data.agent_names:
        labels.append(name)
        means.append(np.mean([ep.returns.get(name, 0.0) for ep in data.episodes]))
    x = np.arange(len(labels))
    ax.bar(x, means, color=["#5DADE2", "#F5B7B1"][: len(labels)], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Average Return per Agent")
    ax.set_ylabel("Return")

    if save_path:
        _savefig(fig, save_path)
    return fig


def plot_action_distribution(
    data: OvercookedEvalData, save_path: Optional[Path] = None
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    fig.suptitle("Overcooked – Action Distribution", fontsize=14, fontweight="bold")

    action_names = ["N", "S", "E", "W", "Stay", "Interact"]
    action_x = np.arange(data.num_actions)
    width = 0.35

    for idx, name in enumerate(data.agent_names):
        counts = np.zeros(data.num_actions, dtype=np.float32)
        for ep in data.episodes:
            for action, value in ep.action_counts[name].items():
                counts[action] += value
        total = np.sum(counts)
        freqs = counts / (total + 1e-9)
        ax.bar(
            action_x + (idx - 0.5) * width,
            freqs,
            width=width,
            alpha=0.85,
            label=name,
        )

    ax.set_xticks(action_x)
    ax.set_xticklabels(action_names[: data.num_actions])
    ax.set_xlabel("Action")
    ax.set_ylabel("Frequency")
    ax.set_title("Policy Action Frequencies")
    ax.legend()

    if save_path:
        _savefig(fig, save_path)
    return fig


def save_all_overcooked_figures(
    data: OvercookedEvalData,
    output_dir: str | Path = "eval_plots",
    prefix: str = "",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pfx = f"{prefix}_" if prefix else ""

    print(f"\n{'=' * 55}")
    print(f" Generating overcooked analysis plots → {out}")
    print(f"{'=' * 55}")

    plot_overview(data, save_path=out / f"{pfx}1_overcooked_overview.png")
    plot_action_distribution(
        data,
        save_path=out / f"{pfx}2_overcooked_actions.png",
    )

    print(f"\nAll plots saved to {out}/")
