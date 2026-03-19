"""
magic_highway_comm_visualizer.py
=================================
Visualization functions for MAGIC communication analysis on Highway Intersection.

Creates comprehensive plots for analyzing:
- Communication graph structure (who talks to whom)
- Message semantics (what messages mean in different contexts)
- Communication patterns across traffic scenarios
- Correlation between communication and safety/performance metrics
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from .magic_highway_comm_analysis import MAGICHighwayAnalysisData

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


def plot_communication_graph(
    data: MAGICHighwayAnalysisData, save_path: Optional[Path] = None
) -> Optional[Figure]:
    """
    Plot average communication graph showing who communicates with whom.

    Visualizes the learned communication topology:
    - Node size: proportional to total messages sent
    - Edge width: proportional to communication frequency
    - Edge color: intensity of attention weights
    """
    if not data.has_comm_data:
        print("  Skipping communication graph (no comm data)")
        return None

    fig, axes = plt.subplots(
        1, data.num_comm_rounds, figsize=(6 * data.num_comm_rounds, 6)
    )
    if data.num_comm_rounds == 1:
        axes = [axes]

    fig.suptitle(
        "MAGIC Communication Graph – Highway Intersection",
        fontsize=15,
        fontweight="bold",
    )

    agent_names = data.agent_names
    n = len(agent_names)

    for round_idx, ax in enumerate(axes):
        adj_stack = data.adj_stack(round_idx=round_idx)
        if adj_stack.size == 0:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"Round {round_idx + 1}")
            continue

        # Average adjacency matrix across all timesteps
        avg_adj = np.mean(adj_stack, axis=0)  # (N, N)

        # Plot as heatmap
        sns.heatmap(
            avg_adj,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            xticklabels=agent_names,
            yticklabels=agent_names,
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": "Avg Attention"},
        )
        ax.set_title(f"Round {round_idx + 1}")
        ax.set_xlabel("Receiver")
        ax.set_ylabel("Sender")

    if save_path:
        _savefig(fig, save_path)
    return fig


def plot_communication_density(
    data: MAGICHighwayAnalysisData, save_path: Optional[Path] = None
) -> Optional[Figure]:
    """
    Plot communication density over time and across episodes.

    Shows:
    - Communication density per episode
    - Communication density vs. episode step
    - Histogram of communication density
    """
    if not data.has_comm_data:
        print("  Skipping communication density (no comm data)")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(
        "MAGIC Communication Density – Highway Intersection",
        fontsize=15,
        fontweight="bold",
    )

    # Communication density per episode
    ax = axes[0, 0]
    episode_densities = [
        ep.mean_comm_density for ep in data.episodes if ep.has_comm_data
    ]
    if episode_densities:
        episodes = np.arange(len(episode_densities))
        ax.plot(episodes, episode_densities, color="#2471A3", linewidth=2)
        ax.axhline(
            y=np.mean(episode_densities),
            color="#E74C3C",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {np.mean(episode_densities):.2f}",
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Communication Density")
        ax.set_title("Communication Density per Episode")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    # Communication density vs. step
    ax = axes[0, 1]
    comm_steps = data.comm_steps()
    if comm_steps:
        steps = np.array([s.step for s in comm_steps])
        densities = np.array([s.comm_density for s in comm_steps])
        ax.scatter(steps, densities, alpha=0.3, s=10, color="#27AE60")
        # Running average
        window = 10
        if len(densities) >= window:
            running_avg = np.convolve(densities, np.ones(window) / window, mode="valid")
            ax.plot(
                steps[window - 1 :],
                running_avg,
                color="#E74C3C",
                linewidth=2,
                label="Running avg",
            )
            ax.legend()
        ax.set_xlabel("Episode Step")
        ax.set_ylabel("Communication Density")
        ax.set_title("Communication Density vs. Step")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    # Histogram of communication density
    ax = axes[1, 0]
    if comm_steps:
        densities = np.array([s.comm_density for s in comm_steps])
        ax.hist(densities, bins=30, color="#9B59B6", alpha=0.7, edgecolor="black")
        ax.axvline(
            x=np.mean(densities),
            color="#E74C3C",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(densities):.2f}",
        )
        ax.set_xlabel("Communication Density")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Communication Density")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    # Communication density vs. reward
    ax = axes[1, 1]
    if comm_steps:
        densities = np.array([s.comm_density for s in comm_steps])
        rewards = np.array([s.total_reward for s in comm_steps])
        ax.scatter(densities, rewards, alpha=0.3, s=10, color="#E67E22")
        # Linear fit
        if len(densities) > 1:
            z = np.polyfit(densities, rewards, 1)
            p = np.poly1d(z)
            x_fit = np.linspace(densities.min(), densities.max(), 100)
            ax.plot(
                x_fit,
                p(x_fit),
                "r--",
                linewidth=2,
                label=f"Fit: {z[0]:.2f}x + {z[1]:.2f}",
            )
            ax.legend()
        ax.set_xlabel("Communication Density")
        ax.set_ylabel("Total Reward")
        ax.set_title("Communication Density vs. Reward")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    if save_path:
        _savefig(fig, save_path)
    return fig


def plot_message_semantics_pca(
    data: MAGICHighwayAnalysisData, save_path: Optional[Path] = None
) -> Optional[Figure]:
    """
    Plot message embeddings in 2D PCA space, colored by context features.

    This helps understand what messages encode:
    - Speed of vehicle
    - Reward signal
    - Collision events
    - Arrival events
    """
    if not data.has_comm_data:
        print("  Skipping message semantics PCA (no comm data)")
        return None

    messages = data.message_stack()  # (T, N, message_dim)
    if messages.size == 0:
        print("  Skipping message semantics PCA (no messages)")
        return None

    # Flatten to (T*N, message_dim)
    T, N, D = messages.shape
    messages_flat = messages.reshape(-1, D)

    # PCA to 2D
    pca = PCA(n_components=2)
    messages_2d = pca.fit_transform(messages_flat)

    # Get context features
    context = data.context_at_comm_steps()
    if not context:
        print("  Skipping message semantics PCA (no context)")
        return None

    # Replicate context features per agent
    context_flat = {}
    for key, vals in context.items():
        if (
            key.startswith("speed_")
            or key.startswith("reward_")
            or key.startswith("crashed_")
            or key.startswith("arrived_")
        ):
            # Per-agent feature: already (T,) for each agent
            # We need to replicate for all agents at each timestep
            # Actually, we need to match the message sender
            # For simplicity, let's use the first agent's features as a proxy
            pass
        else:
            # Global feature: replicate N times per timestep
            context_flat[key] = np.repeat(vals, N)

    # For per-agent features, we need to flatten them correctly
    # Let's extract the first agent's features for now
    agent_names = data.agent_names
    if agent_names:
        first_agent = agent_names[0]
        for prefix in ["speed", "reward", "crashed", "arrived"]:
            key = f"{prefix}_{first_agent}"
            if key in context:
                context_flat[key] = np.repeat(context[key], N)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(
        f"MAGIC Message Semantics (PCA) – Highway Intersection\n"
        f"Variance explained: {pca.explained_variance_ratio_.sum():.1%}",
        fontsize=15,
        fontweight="bold",
    )

    # Plot 1: colored by communication density
    ax = axes[0, 0]
    if "comm_density" in context_flat:
        scatter = ax.scatter(
            messages_2d[:, 0],
            messages_2d[:, 1],
            c=context_flat["comm_density"],
            cmap="viridis",
            alpha=0.5,
            s=10,
        )
        plt.colorbar(scatter, ax=ax, label="Comm Density")
        ax.set_title("Messages colored by Comm Density")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    # Plot 2: colored by speed (first agent)
    ax = axes[0, 1]
    agent_keys = [k for k in context_flat.keys() if k.startswith("speed_")]
    if agent_keys:
        scatter = ax.scatter(
            messages_2d[:, 0],
            messages_2d[:, 1],
            c=context_flat[agent_keys[0]],
            cmap="coolwarm",
            alpha=0.5,
            s=10,
        )
        plt.colorbar(scatter, ax=ax, label="Speed (m/s)")
        ax.set_title(f"Messages colored by {agent_keys[0]}")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    # Plot 3: colored by reward (first agent)
    ax = axes[1, 0]
    agent_keys = [k for k in context_flat.keys() if k.startswith("reward_")]
    if agent_keys:
        scatter = ax.scatter(
            messages_2d[:, 0],
            messages_2d[:, 1],
            c=context_flat[agent_keys[0]],
            cmap="RdYlGn",
            alpha=0.5,
            s=10,
        )
        plt.colorbar(scatter, ax=ax, label="Reward")
        ax.set_title(f"Messages colored by {agent_keys[0]}")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    # Plot 4: colored by crashed status (first agent)
    ax = axes[1, 1]
    agent_keys = [k for k in context_flat.keys() if k.startswith("crashed_")]
    if agent_keys:
        scatter = ax.scatter(
            messages_2d[:, 0],
            messages_2d[:, 1],
            c=context_flat[agent_keys[0]],
            cmap="Reds",
            alpha=0.5,
            s=10,
        )
        plt.colorbar(scatter, ax=ax, label="Crashed")
        ax.set_title(f"Messages colored by {agent_keys[0]}")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    if save_path:
        _savefig(fig, save_path)
    return fig


def plot_message_clusters(
    data: MAGICHighwayAnalysisData,
    n_clusters: int = 5,
    save_path: Optional[Path] = None,
) -> Optional[Figure]:
    """
    Cluster messages and analyze cluster characteristics.

    Uses K-means clustering to identify distinct message types and
    analyzes what contexts they appear in.
    """
    if not data.has_comm_data:
        print("  Skipping message clusters (no comm data)")
        return None

    messages = data.message_stack()  # (T, N, message_dim)
    if messages.size == 0:
        print("  Skipping message clusters (no messages)")
        return None

    # Flatten to (T*N, message_dim)
    T, N, D = messages.shape
    messages_flat = messages.reshape(-1, D)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(messages_flat)

    # PCA for visualization
    pca = PCA(n_components=2)
    messages_2d = pca.fit_transform(messages_flat)

    # Get context features
    context = data.context_at_comm_steps()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(
        f"MAGIC Message Clusters (K={n_clusters}) – Highway Intersection",
        fontsize=15,
        fontweight="bold",
    )

    # Plot 1: Clusters in PCA space
    ax = axes[0, 0]
    scatter = ax.scatter(
        messages_2d[:, 0],
        messages_2d[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.5,
        s=10,
    )
    ax.set_title("Message Clusters in PCA Space")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.colorbar(scatter, ax=ax, label="Cluster")

    # Plot 2: Cluster sizes
    ax = axes[0, 1]
    cluster_sizes = [np.sum(labels == k) for k in range(n_clusters)]
    ax.bar(
        range(n_clusters),
        cluster_sizes,
        color=plt.cm.tab10(np.linspace(0, 1, n_clusters)),
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Sizes")
    ax.set_xticks(range(n_clusters))

    # Plot 3: Average speed per cluster (if available)
    ax = axes[1, 0]
    agent_names = data.agent_names
    if agent_names and f"speed_{agent_names[0]}" in context:
        speed_key = f"speed_{agent_names[0]}"
        speeds_flat = np.repeat(context[speed_key], N)
        avg_speeds = [speeds_flat[labels == k].mean() for k in range(n_clusters)]
        ax.bar(
            range(n_clusters),
            avg_speeds,
            color=plt.cm.tab10(np.linspace(0, 1, n_clusters)),
        )
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Avg Speed (m/s)")
        ax.set_title(f"Average Speed per Cluster ({agent_names[0]})")
        ax.set_xticks(range(n_clusters))
    else:
        ax.text(
            0.5, 0.5, "No speed data", ha="center", va="center", transform=ax.transAxes
        )

    # Plot 4: Average reward per cluster (if available)
    ax = axes[1, 1]
    if agent_names and f"reward_{agent_names[0]}" in context:
        reward_key = f"reward_{agent_names[0]}"
        rewards_flat = np.repeat(context[reward_key], N)
        avg_rewards = [rewards_flat[labels == k].mean() for k in range(n_clusters)]
        ax.bar(
            range(n_clusters),
            avg_rewards,
            color=plt.cm.tab10(np.linspace(0, 1, n_clusters)),
        )
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Avg Reward")
        ax.set_title(f"Average Reward per Cluster ({agent_names[0]})")
        ax.set_xticks(range(n_clusters))
    else:
        ax.text(
            0.5, 0.5, "No reward data", ha="center", va="center", transform=ax.transAxes
        )

    if save_path:
        _savefig(fig, save_path)
    return fig


def save_all_magic_highway_figures(
    data: MAGICHighwayAnalysisData,
    output_dir: str | Path = "eval_plots/magic_highway",
    prefix: str = "",
) -> None:
    """
    Generate and save all MAGIC highway communication analysis plots.

    Args:
        data: Collected MAGIC communication data
        output_dir: Directory to save plots
        prefix: Prefix for filenames
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pfx = f"{prefix}_" if prefix else ""

    print(f"\n{'=' * 70}")
    print(f" Generating MAGIC highway communication analysis plots → {out}")
    print(f"{'=' * 70}")

    plot_communication_graph(data, save_path=out / f"{pfx}1_comm_graph.png")
    plot_communication_density(data, save_path=out / f"{pfx}2_comm_density.png")
    plot_message_semantics_pca(
        data, save_path=out / f"{pfx}3_message_semantics_pca.png"
    )
    plot_message_clusters(
        data, n_clusters=5, save_path=out / f"{pfx}4_message_clusters.png"
    )

    print(f"\nAll plots saved to {out}/")
