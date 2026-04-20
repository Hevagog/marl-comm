"""
commformer_comm_analysis.py
============================
Data collection and statistical analysis for CommFormer's learned
communication graph in any supported environment.

Architecture recap (CommFormer, Hu et al. ICLR 2024)
-----------------------------------------------------
 1. CommGraph  – learnable α ∈ R^{N×N} → k-hot binary adjacency (static parameter)
 2. EncoderBlock – relation-enhanced transformer encoder per agent node
 3. DecoderBlock – auto-regressive decoder conditioned on encoder output
 4. Action head  – projects decoder output to action logits

Key difference from MAGIC: CommFormer's communication graph α is a
*learned model parameter*, not a dynamic per-step output. The adjacency
is the same at every timestep (deterministic k-argmax at inference).
What varies per step are the agent representations flowing through this
fixed graph structure.

Expected outputs from CommFormerMAPPO.act() (via CommFormerPolicyNet):
    outputs["adj_matrices"]   – np/jax array, shape (N, N)
                                 Hard binary k-hot adjacency derived from α.
                                 Same every step (static graph).
    outputs["encoder_out"]    – np/jax array, shape (num_envs, hidden_dim)
                                 Post-encoder-decoder representation per agent.
                                 Varies each step — analogous to aggregated
                                 messages in MAGIC.
    outputs["net_output"]     – logits (B, num_actions) — always present

Usage
-----
from utils.commformer_comm_analysis import CommFormerCommCollector
from utils.commformer_comm_visualizer import save_all_commformer_figures

collector = CommFormerCommCollector(
    num_agents=2,
    hidden_dim=64,
    sparsity=0.4,
)
data = collector.collect(env, commformer_agent, n_episodes=30)
save_all_commformer_figures(data, output_dir="eval_plots/commformer", prefix="cf_blindspot")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class CommFormerStepRecord:
    """All communication-relevant data captured at one environment step."""

    step: int

    # ---------- environment context ------------------------------------------
    actions: dict[str, int] = field(default_factory=dict)
    rewards: dict[str, float] = field(default_factory=dict)
    total_reward: float = 0.0

    # ---------- communication outputs (may be None if unavailable) -----------
    # adj: (N, N) hard binary adjacency from CommGraph — same every step.
    adj: np.ndarray | None = None
    # encoder_out: (N, hidden_dim) post-encoder-decoder representations per agent.
    encoder_out: np.ndarray | None = None
    # logits: (N, num_actions) policy logits per agent.
    logits: np.ndarray | None = None


@dataclass
class CommFormerEpisodeData:
    """Episode-level container for CommFormer analysis data."""

    episode_idx: int
    steps: list[CommFormerStepRecord] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    has_comm_data: bool = False

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def success(self) -> bool:
        """Treat terminated (not truncated) as task success."""
        return self.terminated and not self.truncated

    @property
    def total_rewards(self) -> dict[str, float]:
        all_agents = set()
        for s in self.steps:
            all_agents.update(s.rewards.keys())
        out: dict[str, float] = {a: 0.0 for a in all_agents}
        for s in self.steps:
            for a, r in s.rewards.items():
                out[a] += r
        return out

    @property
    def mean_episode_reward(self) -> float:
        totals = self.total_rewards
        return float(np.mean(list(totals.values()))) if totals else 0.0


@dataclass
class CommFormerAnalysisData:
    """Top-level container returned by CommFormerCommCollector.collect()."""

    num_agents: int
    hidden_dim: int
    sparsity: float
    episodes: list[CommFormerEpisodeData] = field(default_factory=list)
    has_comm_data: bool = False

    # Learned α parameter — read once from model state_dict.
    # Shape: (N, N). None if unavailable.
    alpha: np.ndarray | None = None

    # Hard k-hot adjacency derived from α (first step that has adj data).
    # Shape: (N, N). None if unavailable.
    hard_adj: np.ndarray | None = None

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def success_rate(self) -> float:
        return float(np.mean([e.success for e in self.episodes])) if self.episodes else 0.0

    @property
    def mean_episode_length(self) -> float:
        return float(np.mean([e.length for e in self.episodes])) if self.episodes else 0.0

    @property
    def mean_episode_reward(self) -> float:
        return float(np.mean([e.mean_episode_reward for e in self.episodes])) if self.episodes else 0.0

    def all_steps(self) -> list[CommFormerStepRecord]:
        return [s for e in self.episodes for s in e.steps]

    def comm_steps(self) -> list[CommFormerStepRecord]:
        """Steps that have valid encoder_out data."""
        return [s for s in self.all_steps() if s.encoder_out is not None]

    def encoder_out_stack(self) -> np.ndarray:
        """Stack all encoder outputs → (T*N, hidden_dim)."""
        mats = [s.encoder_out for s in self.comm_steps() if s.encoder_out is not None]
        return np.concatenate(mats, axis=0) if mats else np.empty((0, self.hidden_dim))

    def agent_labels_for_encoder_stack(self) -> np.ndarray:
        """Agent index (0..N-1) for each row in encoder_out_stack()."""
        labels = []
        for s in self.comm_steps():
            if s.encoder_out is not None:
                labels.extend(list(range(s.encoder_out.shape[0])))
        return np.asarray(labels, dtype=np.int32)

    def episode_rewards(self) -> np.ndarray:
        """Mean episode reward per episode → (n_episodes,)."""
        return np.asarray([e.mean_episode_reward for e in self.episodes], dtype=np.float32)

    def episode_lengths(self) -> np.ndarray:
        """Episode length per episode → (n_episodes,)."""
        return np.asarray([e.length for e in self.episodes], dtype=np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# Output-key extraction helper
# ──────────────────────────────────────────────────────────────────────────────

_MISSING_KEYS_WARNED = False


def _np(v: Any) -> np.ndarray:
    try:
        import jax
        return np.asarray(jax.device_get(v))
    except Exception:
        return np.asarray(v)


def _extract_comm_outputs(
    outputs_per_agent: dict[str, Any],
    agents: list[str],
    num_agents: int,
    hidden_dim: int,
) -> tuple[
    np.ndarray | None,   # adj: (N, N) hard binary adjacency
    np.ndarray | None,   # encoder_out: (N, hidden_dim)
    np.ndarray | None,   # logits: (N, num_actions)
]:
    """Extract CommFormer outputs from the per-agent outputs dict.

    CommFormerMAPPO.act() stores:
    - ``adj_matrices`` in every agent's slot (shared, unsliced) → (N, N)
    - ``encoder_out`` per-agent (sliced) → (num_envs, hidden_dim)
    - ``net_output`` per-agent → (num_envs, num_actions)
    """
    global _MISSING_KEYS_WARNED

    uid0 = agents[0]
    agent_outputs_0 = outputs_per_agent.get(uid0, {})

    # ---- adjacency matrix ---------------------------------------------------
    adj: np.ndarray | None = None
    if "adj_matrices" in agent_outputs_0:
        raw = _np(agent_outputs_0["adj_matrices"])
        # Expected shape: (N, N)
        if raw.ndim == 2:
            adj = raw
        elif raw.ndim == 3:
            adj = raw[0]  # take first group
        elif raw.ndim == 4:
            adj = raw[0, 0]

    # ---- encoder outputs per agent ------------------------------------------
    encoder_out: np.ndarray | None = None
    enc_rows = []
    for uid in agents:
        ag_out = outputs_per_agent.get(uid, {})
        if "encoder_out" in ag_out:
            raw = _np(ag_out["encoder_out"])
            # Shape: (num_envs, hidden_dim) → take first env row
            enc_rows.append(raw[0] if raw.ndim == 2 else raw)
    if len(enc_rows) == num_agents:
        encoder_out = np.stack(enc_rows, axis=0)  # (N, hidden_dim)

    # ---- logits per agent ---------------------------------------------------
    logits: np.ndarray | None = None
    logit_rows = []
    for uid in agents:
        ag_out = outputs_per_agent.get(uid, {})
        if "net_output" in ag_out:
            raw = _np(ag_out["net_output"])
            logit_rows.append(raw[0] if raw.ndim == 2 else raw)
    if len(logit_rows) == num_agents:
        logits = np.stack(logit_rows, axis=0)  # (N, num_actions)

    # ---- warn once if nothing found ----------------------------------------
    if adj is None and encoder_out is None and not _MISSING_KEYS_WARNED:
        warnings.warn(
            "\n[CommFormerAnalysis] Communication outputs not found in policy outputs dict.\n"
            "  Expected keys: 'adj_matrices', 'encoder_out'\n"
            "  Make sure CommFormerMAPPO is used (not plain CategoricalMAPPO).\n"
            "  Communication-specific plots will be skipped.\n",
            stacklevel=3,
        )
        _MISSING_KEYS_WARNED = True

    return adj, encoder_out, logits


# ──────────────────────────────────────────────────────────────────────────────
# Alpha parameter extraction
# ──────────────────────────────────────────────────────────────────────────────


def extract_alpha_from_agent(agent: Any) -> np.ndarray | None:
    """Read the learned α parameter from a CommFormerMAPPO agent.

    CommFormerPolicyNet stores α in its Flax parameter tree at:
      ``state_dict.params["params"]["comm_graph"]["alpha"]``

    Returns (N, N) float32 array or None if not found.
    """
    try:
        uid0 = agent.possible_agents[0]
        policy = agent.policies[uid0]
        inner = policy.state_dict.params["params"]
        alpha_raw = inner["comm_graph"]["alpha"]
        return _np(alpha_raw).astype(np.float32)
    except Exception as e:
        warnings.warn(f"[CommFormerAnalysis] Could not extract α: {e}")
        return None


def compute_hard_adj_from_alpha(alpha: np.ndarray, sparsity: float) -> np.ndarray:
    """Compute the deterministic k-hot adjacency from α (CommFormer Eq. 12).

    Parameters
    ----------
    alpha : (N, N) learned parameter.
    sparsity : S ∈ (0, 1] — fraction of neighbors each agent attends to.

    Returns
    -------
    (N, N) binary adjacency.
    """
    n = alpha.shape[0]
    k = max(1, int(round(sparsity * n)))
    adj = np.zeros_like(alpha)
    for i in range(n):
        top_k_idx = np.argsort(alpha[i])[::-1][:k]
        adj[i, top_k_idx] = 1.0
    return adj


# ──────────────────────────────────────────────────────────────────────────────
# Main collector
# ──────────────────────────────────────────────────────────────────────────────


class CommFormerCommCollector:
    """Run evaluation episodes with a CommFormer agent and collect per-step data.

    The collector is environment-agnostic: it captures generic step data
    (actions, rewards) plus CommFormer-specific outputs (adjacency, encoder
    representations).  Environment-specific context (positions, battery, etc.)
    can be added by subclassing or post-processing.

    Parameters
    ----------
    num_agents   : number of agents in the environment.
    hidden_dim   : CommFormer hidden_dim (from config).
    sparsity     : graph sparsity parameter S (from config).
    max_cycles   : episode length cap (used for progress reporting).
    """

    def __init__(
        self,
        num_agents: int = 2,
        hidden_dim: int = 64,
        sparsity: float = 0.4,
        max_cycles: int = 500,
    ) -> None:
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.sparsity = sparsity
        self.max_cycles = max_cycles

    def collect(
        self,
        env: Any,
        agent: Any,
        n_episodes: int = 30,
        max_steps_per_episode: int | None = None,
    ) -> CommFormerAnalysisData:
        """Run n_episodes evaluation episodes, capturing CommFormer data.

        Parameters
        ----------
        env     : skrl-wrapped PettingZoo environment.
        agent   : CommFormerMAPPO instance.
        n_episodes          : number of episodes to collect.
        max_steps_per_episode : overrides max_cycles if provided.
        """
        agent.set_running_mode("eval")
        max_steps = max_steps_per_episode or self.max_cycles
        possible_agents = list(env.possible_agents)
        num_agents = len(possible_agents)

        # Read α once from model parameters.
        alpha = extract_alpha_from_agent(agent)
        hard_adj_from_alpha: np.ndarray | None = None
        if alpha is not None:
            hard_adj_from_alpha = compute_hard_adj_from_alpha(alpha, self.sparsity)

        data = CommFormerAnalysisData(
            num_agents=num_agents,
            hidden_dim=self.hidden_dim,
            sparsity=self.sparsity,
            alpha=alpha,
            hard_adj=hard_adj_from_alpha,
        )

        comm_data_found = False

        for ep_idx in range(n_episodes):
            episode = CommFormerEpisodeData(episode_idx=ep_idx)
            obs, _ = env.reset()

            for t in range(max_steps):
                actions, _, outputs_per_agent = agent.act(
                    obs, timestep=t, timesteps=max_steps
                )

                actions_int = {
                    k: int(np.asarray(v).ravel()[0]) for k, v in actions.items()
                }

                next_obs, rewards, terminated, truncated, _ = env.step(actions)

                rewards_float = {
                    k: float(np.asarray(v).ravel()[0]) for k, v in rewards.items()
                }

                adj, encoder_out, logits = _extract_comm_outputs(
                    outputs_per_agent,
                    possible_agents,
                    num_agents,
                    self.hidden_dim,
                )

                if adj is not None or encoder_out is not None:
                    comm_data_found = True

                record = CommFormerStepRecord(
                    step=t,
                    actions=actions_int,
                    rewards=rewards_float,
                    total_reward=sum(rewards_float.values()),
                    adj=adj,
                    encoder_out=encoder_out,
                    logits=logits,
                )
                episode.steps.append(record)

                done = any(
                    bool(np.asarray(v).ravel()[0])
                    for v in list(terminated.values()) + list(truncated.values())
                )
                if done:
                    episode.terminated = any(
                        bool(np.asarray(v).ravel()[0]) for v in terminated.values()
                    )
                    episode.truncated = any(
                        bool(np.asarray(v).ravel()[0]) for v in truncated.values()
                    )
                    break

                obs = next_obs

            episode.has_comm_data = comm_data_found
            data.episodes.append(episode)
            print(
                f"  Episode {ep_idx + 1:3d}/{n_episodes}  |  "
                f"Len: {episode.length:3d}  |  "
                f"Success: {str(episode.success):5s}  |  "
                f"Reward: {episode.mean_episode_reward:+6.2f}"
            )

        data.has_comm_data = comm_data_found
        return data


# ──────────────────────────────────────────────────────────────────────────────
# Statistical analysis helpers (consumed by the visualizer)
# ──────────────────────────────────────────────────────────────────────────────


def compute_graph_metrics(data: CommFormerAnalysisData) -> dict[str, Any]:
    """Compute structural metrics of the learned communication graph.

    Returns a dict with:
    - ``in_degree``      : (N,) in-degree per agent (how many agents message it)
    - ``out_degree``     : (N,) out-degree per agent (how many agents it messages)
    - ``density``        : scalar fraction of directed edges that are active
    - ``reciprocity``    : fraction of bidirectional edges
    - ``hub_scores``     : (N,) combined in+out degree normalized
    - ``is_symmetric``   : bool — whether adj is symmetric
    - ``self_loops``     : (N,) diagonal of hard_adj
    """
    adj = data.hard_adj
    if adj is None:
        n = data.num_agents
        return {
            "in_degree": np.zeros(n),
            "out_degree": np.zeros(n),
            "density": 0.0,
            "reciprocity": 0.0,
            "hub_scores": np.zeros(n),
            "is_symmetric": True,
            "self_loops": np.zeros(n),
        }

    n = adj.shape[0]
    in_deg = adj.sum(axis=0)   # column sum: how many send TO agent j
    out_deg = adj.sum(axis=1)  # row sum: how many agent i sends TO

    # Density: fraction of off-diagonal edges active
    off_diag = adj.copy()
    np.fill_diagonal(off_diag, 0)
    density = float(off_diag.sum()) / max(n * (n - 1), 1)

    # Reciprocity: fraction of (i→j, j→i) pairs both active
    sym_pairs = (off_diag * off_diag.T).sum()
    total_pairs = float(off_diag.sum())
    reciprocity = float(sym_pairs / max(total_pairs, 1))

    hub_scores = (in_deg + out_deg) / max(float((in_deg + out_deg).sum()), 1.0)

    is_symmetric = bool(np.allclose(adj, adj.T))

    self_loops = np.diag(adj)

    return {
        "in_degree": in_deg,
        "out_degree": out_deg,
        "density": density,
        "reciprocity": reciprocity,
        "hub_scores": hub_scores,
        "is_symmetric": is_symmetric,
        "self_loops": self_loops,
    }


def compute_encoder_pca(
    data: CommFormerAnalysisData,
    n_components: int = 2,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """PCA on all encoder output embeddings.

    Returns
    -------
    projected  : (M, n_components) projected embeddings, or None
    labels     : (M,) agent index for each row, or None
    evr        : explained variance ratio, or None
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return None, None, None

    X = data.encoder_out_stack()
    if X.shape[0] < 2 or X.shape[1] < n_components:
        return None, None, None

    labels = data.agent_labels_for_encoder_stack()
    pca = PCA(n_components=n_components, random_state=0)
    projected = pca.fit_transform(X)
    return projected, labels, pca.explained_variance_ratio_


def compute_encoder_tsne(
    data: CommFormerAnalysisData,
    n_components: int = 2,
    perplexity: float = 30.0,
    max_samples: int = 2000,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """t-SNE on encoder output embeddings.

    Returns (projected (M, 2), agent_labels (M,)) or (None, None).
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return None, None

    X = data.encoder_out_stack()
    labels = data.agent_labels_for_encoder_stack()
    if X.shape[0] < 2:
        return None, None

    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]
        labels = labels[idx]

    perp = min(perplexity, len(X) - 1)
    tsne = TSNE(
        n_components=n_components,
        perplexity=perp,
        random_state=0,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X), labels


def compute_encoder_norm_per_agent(
    data: CommFormerAnalysisData,
) -> dict[int, np.ndarray]:
    """L2-norm of encoder outputs per agent over episode time.

    Returns dict mapping agent_idx → (T,) array of L2 norms (one per step).
    """
    norms: dict[int, list[float]] = {i: [] for i in range(data.num_agents)}
    for s in data.comm_steps():
        if s.encoder_out is not None:
            for i in range(min(data.num_agents, s.encoder_out.shape[0])):
                norms[i].append(float(np.linalg.norm(s.encoder_out[i])))
    return {i: np.asarray(v, dtype=np.float32) for i, v in norms.items()}


def compute_alpha_stats(data: CommFormerAnalysisData) -> dict[str, Any]:
    """Summary statistics of the learned α parameter.

    Returns a dict with mean, std, min, max, and per-agent row/col stats.
    """
    alpha = data.alpha
    if alpha is None:
        return {}

    n = alpha.shape[0]
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_vals = alpha[off_diag_mask]

    return {
        "mean": float(alpha.mean()),
        "std": float(alpha.std()),
        "min": float(alpha.min()),
        "max": float(alpha.max()),
        "off_diag_mean": float(off_vals.mean()),
        "off_diag_std": float(off_vals.std()),
        "row_means": alpha.mean(axis=1),    # (N,) — sender strength
        "col_means": alpha.mean(axis=0),    # (N,) — receiver popularity
        "diag_vals": np.diag(alpha),        # (N,) — self-attention strength
    }


def compute_reward_vs_graph_density(
    data: CommFormerAnalysisData,
) -> dict[str, float]:
    """Correlate episode rewards with the fixed graph density.

    Since the graph is static, this primarily tests whether the learned graph
    density predicts average performance across different episodes (which
    differ in initial conditions, not graph).  Returns Pearson r and the
    density itself for reporting.
    """
    density = compute_graph_metrics(data).get("density", 0.0)
    rewards = data.episode_rewards()
    return {
        "graph_density": density,
        "mean_reward": float(rewards.mean()) if len(rewards) > 0 else 0.0,
        "std_reward": float(rewards.std()) if len(rewards) > 0 else 0.0,
    }


def print_summary(data: CommFormerAnalysisData) -> None:
    """Print a compact analysis summary to stdout."""
    metrics = compute_graph_metrics(data)
    alpha_stats = compute_alpha_stats(data)

    print("\n" + "=" * 64)
    print("  CommFormer Communication Analysis Summary")
    print("=" * 64)
    print(f"  Agents     : {data.num_agents}")
    print(f"  Sparsity S : {data.sparsity}")
    print(f"  Episodes   : {data.n_episodes}")
    print(f"  Success    : {data.success_rate:.1%}")
    print(f"  Mean len   : {data.mean_episode_length:.1f} steps")
    print(f"  Mean reward: {data.mean_episode_reward:+.3f}")
    print(f"  Comm data  : {'available' if data.has_comm_data else 'NOT AVAILABLE'}")

    if data.alpha is not None:
        print(f"\n  α stats    : mean={alpha_stats.get('mean', 0):.3f}  "
              f"std={alpha_stats.get('std', 0):.3f}  "
              f"range=[{alpha_stats.get('min', 0):.3f}, {alpha_stats.get('max', 0):.3f}]")

    if data.hard_adj is not None:
        print(f"\n  Graph density      : {metrics['density']:.3f}")
        print(f"  Graph reciprocity  : {metrics['reciprocity']:.3f}")
        print(f"  Symmetric          : {metrics['is_symmetric']}")
        print(f"  In-degree          : {metrics['in_degree']}")
        print(f"  Out-degree         : {metrics['out_degree']}")

    print("=" * 64 + "\n")
