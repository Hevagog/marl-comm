"""
magic_comm_analysis.py
======================
Data-collection and statistical analysis for the MAGIC communication mechanism
on the BlindSpot Navigation environment.

Architecture recap (MAGIC, Niu et al. AAMAS 2021)
--------------------------------------------------
 1. Scheduler  – GAT encoder → MLP → Gumbel-Softmax → directed adjacency G_t^(l)
 2. Message Processor – GAT over G_t aggregates messages z_i across num_comm_rounds
 3. Policy head – acts on [obs_enc || aggregated_msg]

The MAGIC policy net is expected to expose the following keys in its outputs dict
(returned from policy.act()):

    outputs["adj_matrices"]   – np/jax array, shape (num_comm_rounds, N, N) or (N, N)
                                 Soft Gumbel-Softmax weights BEFORE hard-threshold.
                                 Captures the probability that agent i sends to agent j.
    outputs["hard_adj"]       – binary adjacency after hard-threshold  (N, N), optional
    outputs["messages"]       – pre-aggregation message embeddings (N, message_dim)
    outputs["agg_messages"]   – post-aggregation (N, message_dim), optional
    outputs["net_output"]     – logits (N, num_actions)  — always present

If those keys are absent the collector still runs but communication-specific plots
will be skipped and a warning is printed once.

Usage
-----
from utils.magic_comm_analysis import MAGICCommCollector
from utils.magic_comm_visualizer import save_all_magic_figures

collector = MAGICCommCollector(
    grid_size=9, num_traps=5, max_cycles=100,
    num_comm_rounds=2, message_dim=128,
)
data = collector.collect(env, magic_agent, n_episodes=30)
save_all_magic_figures(data, output_dir="eval_plots/magic", prefix="magic_blindspot_v2")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

AGENT_A = "agent_0"
AGENT_B = "agent_1"
AGENTS = [AGENT_A, AGENT_B]


def _denorm(val: float, grid_size: int) -> int:
    gs_norm = max(grid_size - 1, 1)
    return int(round(float(val) * gs_norm))


def _decode_positions_and_goal(
    obs_a: np.ndarray, grid_size: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    own_x = _denorm(obs_a[0], grid_size)
    own_y = _denorm(obs_a[1], grid_size)
    other_x = _denorm(obs_a[2], grid_size)
    other_y = _denorm(obs_a[3], grid_size)
    goal_x = _denorm(obs_a[4], grid_size)
    goal_y = _denorm(obs_a[5], grid_size)
    return (own_x, own_y), (other_x, other_y), (goal_x, goal_y)


def _decode_traps_from_obs(
    obs_a: np.ndarray, grid_size: int, num_traps: int
) -> list[tuple[int, int]]:
    gs_norm = max(grid_size - 1, 1)
    own_x = _denorm(obs_a[0], grid_size)
    own_y = _denorm(obs_a[1], grid_size)
    traps: list[tuple[int, int]] = []
    trap_start = 8
    for idx in range(num_traps):
        base = trap_start + idx * 3
        rel_x = float(obs_a[base + 0])
        rel_y = float(obs_a[base + 1])
        present = float(obs_a[base + 2])
        if present <= 0.5:
            continue
        tx = int(round(own_x + rel_x * gs_norm))
        ty = int(round(own_y + rel_y * gs_norm))
        traps.append((max(0, min(grid_size - 1, tx)), max(0, min(grid_size - 1, ty))))
    return traps


def _decode_reached_flags(obs_a: np.ndarray, num_traps: int) -> tuple[bool, bool]:
    rs = 8 + num_traps * 3
    return float(obs_a[rs]) > 0.5, float(obs_a[rs + 1]) > 0.5


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _min_trap_dist(pos: tuple[int, int], traps: list[tuple[int, int]]) -> int:
    if not traps:
        return 999
    return min(_manhattan(pos, t) for t in traps)


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class CommStepRecord:
    """All communication-relevant data captured at one environment step."""

    step: int

    # ---------- state context ------------------------------------------------
    positions: dict[str, tuple[int, int]] = field(default_factory=dict)
    goal: tuple[int, int] = (0, 0)
    traps: list[tuple[int, int]] = field(default_factory=list)
    dist_to_goal: dict[str, int] = field(default_factory=dict)
    min_trap_dist: dict[str, int] = field(default_factory=dict)
    reached: dict[str, bool] = field(default_factory=dict)
    actions: dict[str, int] = field(default_factory=dict)
    rewards: dict[str, float] = field(default_factory=dict)

    # ---------- communication internals (may be None if unavailable) ---------
    # adj_per_round[r] → shape (N, N) soft adjacency for comm round r
    adj_per_round: list[np.ndarray] | None = None
    # hard_adj → shape (N, N) binary adjacency (after threshold)
    hard_adj: np.ndarray | None = None
    # messages[i] → shape (message_dim,) raw embedding for agent i
    messages: np.ndarray | None = None  # (N, msg_dim)
    # agg_messages[i] → after GAT aggregation
    agg_messages: np.ndarray | None = None  # (N, msg_dim)
    # policy logits per agent, shape (N, num_actions)
    logits: np.ndarray | None = None

    # ---------- derived scalars (filled during collection) -------------------
    comm_density: float = 0.0  # fraction of directed edges that are active
    total_reward: float = 0.0


@dataclass
class CommEpisodeData:
    episode_idx: int
    steps: list[CommStepRecord] = field(default_factory=list)
    traps: list[tuple[int, int]] = field(default_factory=list)
    goal: tuple[int, int] = (0, 0)
    terminated: bool = False
    truncated: bool = False
    has_comm_data: bool = False

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def success(self) -> bool:
        return self.terminated

    @property
    def total_rewards(self) -> dict[str, float]:
        out: dict[str, float] = {a: 0.0 for a in AGENTS}
        for s in self.steps:
            for a in AGENTS:
                out[a] += s.rewards.get(a, 0.0)
        return out

    @property
    def mean_comm_density(self) -> float:
        vals = [s.comm_density for s in self.steps if s.adj_per_round is not None]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def trap_hit_steps(self) -> list[int]:
        """Steps where at least one agent was ON a trap cell."""
        trap_set = set(self.traps)
        return [
            s.step
            for s in self.steps
            if any(s.positions.get(a) in trap_set for a in AGENTS)
        ]


@dataclass
class MAGICAnalysisData:
    """Top-level container returned by MAGICCommCollector.collect()."""

    grid_size: int
    max_cycles: int
    num_comm_rounds: int
    message_dim: int
    episodes: list[CommEpisodeData] = field(default_factory=list)
    has_comm_data: bool = False

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def success_rate(self) -> float:
        return float(np.mean([e.success for e in self.episodes]))

    @property
    def mean_episode_length(self) -> float:
        return float(np.mean([e.length for e in self.episodes]))

    def all_steps(self) -> list[CommStepRecord]:
        return [s for e in self.episodes for s in e.steps]

    def comm_steps(self) -> list[CommStepRecord]:
        """Steps that contain valid communication data."""
        return [s for s in self.all_steps() if s.adj_per_round is not None]

    def adj_stack(self, round_idx: int = 0) -> np.ndarray:
        """Stack adj matrices for the given communication round → (T, N, N)."""
        mats = [
            s.adj_per_round[round_idx]
            for s in self.comm_steps()
            if len(s.adj_per_round) > round_idx
        ]
        return np.stack(mats, axis=0) if mats else np.empty((0, 2, 2))

    def context_at_comm_steps(self) -> dict[str, np.ndarray]:
        """
        Returns a dict of context feature arrays aligned with comm_steps():
            dist_goal_a / dist_goal_b  – Manhattan distance to goal
            trap_dist_a / trap_dist_b  – nearest trap distance
            comm_density               – edge density in adjacency
            reward_a / reward_b
            step_fraction              – t / max_cycles
            reached_a / reached_b
        """
        csteps = self.comm_steps()
        out: dict[str, list] = {
            k: []
            for k in [
                "dist_goal_a",
                "dist_goal_b",
                "trap_dist_a",
                "trap_dist_b",
                "comm_density",
                "reward_a",
                "reward_b",
                "step_fraction",
                "reached_a",
                "reached_b",
            ]
        }
        for s in csteps:
            out["dist_goal_a"].append(s.dist_to_goal.get(AGENT_A, 0))
            out["dist_goal_b"].append(s.dist_to_goal.get(AGENT_B, 0))
            out["trap_dist_a"].append(s.min_trap_dist.get(AGENT_A, 0))
            out["trap_dist_b"].append(s.min_trap_dist.get(AGENT_B, 0))
            out["comm_density"].append(s.comm_density)
            out["reward_a"].append(s.rewards.get(AGENT_A, 0.0))
            out["reward_b"].append(s.rewards.get(AGENT_B, 0.0))
            out["step_fraction"].append(s.step / max(self.max_cycles - 1, 1))
            out["reached_a"].append(float(s.reached.get(AGENT_A, False)))
            out["reached_b"].append(float(s.reached.get(AGENT_B, False)))
        return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Output-key extraction helper
# ──────────────────────────────────────────────────────────────────────────────

_MISSING_KEYS_WARNED = False


def _extract_comm_outputs(
    outputs: dict[str, Any],
    num_agents: int,
    num_comm_rounds: int,
    message_dim: int,
) -> tuple[
    list[np.ndarray] | None,  # adj_per_round
    np.ndarray | None,  # hard_adj
    np.ndarray | None,  # messages  (N, msg_dim)
    np.ndarray | None,  # agg_messages (N, msg_dim)
    np.ndarray | None,  # logits (N, num_actions)
]:
    """Pull communication tensors out of a policy outputs dict."""
    global _MISSING_KEYS_WARNED

    def _np(v):
        try:
            import jax

            return np.asarray(jax.device_get(v))
        except Exception:
            return np.asarray(v)

    # ---- adjacency matrices -------------------------------------------------
    adj_per_round: list[np.ndarray] | None = None
    hard_adj: np.ndarray | None = None

    if "adj_matrices" in outputs:
        raw = _np(outputs["adj_matrices"])
        if raw.ndim == 4:
            if raw.shape[0] == num_comm_rounds:
                adj_per_round = [raw[r, 0] for r in range(num_comm_rounds)]
            else:
                adj_per_round = [raw[0, r] for r in range(num_comm_rounds)]
        elif raw.ndim == 3:
            adj_per_round = [raw[r] for r in range(min(num_comm_rounds, raw.shape[0]))]
        elif raw.ndim == 2:
            adj_per_round = [raw]
        while adj_per_round and len(adj_per_round) < num_comm_rounds:
            adj_per_round.append(adj_per_round[-1])
    elif "comm_weights" in outputs:
        raw = _np(outputs["comm_weights"])
        if raw.ndim >= 2:
            mat = (
                raw.reshape(num_agents, num_agents)
                if raw.size == num_agents**2
                else raw
            )
            adj_per_round = [mat]

    if "hard_adj" in outputs:
        hard_adj = _np(outputs["hard_adj"])
        if hard_adj.ndim > 2:
            hard_adj = hard_adj[0]

    # ---- messages -----------------------------------------------------------
    messages: np.ndarray | None = None
    agg_messages: np.ndarray | None = None

    if "messages" in outputs:
        raw = _np(outputs["messages"])
        messages = raw[0] if raw.ndim == 3 else raw

    if "agg_messages" in outputs:
        raw = _np(outputs["agg_messages"])
        agg_messages = raw[0] if raw.ndim == 3 else raw

    # ---- logits -------------------------------------------------------------
    logits: np.ndarray | None = None
    if "net_output" in outputs:
        raw = _np(outputs["net_output"])
        if raw.ndim == 2 and raw.shape[0] == num_agents:
            logits = raw
        elif raw.ndim == 2:
            logits = raw[:num_agents]

    # ---- warn once if nothing found ----------------------------------------
    if adj_per_round is None and not _MISSING_KEYS_WARNED:
        warnings.warn(
            "\n[MAGICCommAnalysis] Communication outputs not found in policy outputs dict.\n"
            "  Expected keys: 'adj_matrices', 'messages', 'agg_messages', 'hard_adj'\n"
            "  Communication-specific plots will be skipped.\n"
            "  To enable full analysis, expose these tensors from MAGICPolicyNet.__call__.",
            stacklevel=3,
        )
        _MISSING_KEYS_WARNED = True

    return adj_per_round, hard_adj, messages, agg_messages, logits


# ──────────────────────────────────────────────────────────────────────────────
# Main collector
# ──────────────────────────────────────────────────────────────────────────────


class MAGICCommCollector:
    """
    Run evaluation episodes with a MAGIC agent and collect per-step
    communication data from the policy's output tensors.

    Parameters
    ----------
    grid_size       : environment grid size (default 9)
    num_traps       : number of traps in the environment (default 5)
    max_cycles      : episode length cap (default 100)
    num_comm_rounds : rounds of message passing in MAGIC (from config, default 2)
    message_dim     : dimensionality of message embeddings (from config, default 128)
    """

    def __init__(
        self,
        grid_size: int = 9,
        num_traps: int = 5,
        max_cycles: int = 100,
        num_comm_rounds: int = 2,
        message_dim: int = 128,
    ) -> None:
        self.grid_size = grid_size
        self.num_traps = num_traps
        self.max_cycles = max_cycles
        self.num_comm_rounds = num_comm_rounds
        self.message_dim = message_dim
        self._num_agents = 2

    def collect(
        self,
        env: Any,
        agent: Any,
        n_episodes: int = 30,
        max_steps_per_episode: int | None = None,
    ) -> MAGICAnalysisData:
        """Run n_episodes evaluation episodes, capturing communication data."""
        agent.set_running_mode("eval")
        max_steps = max_steps_per_episode or self.max_cycles
        data = MAGICAnalysisData(
            grid_size=self.grid_size,
            max_cycles=self.max_cycles,
            num_comm_rounds=self.num_comm_rounds,
            message_dim=self.message_dim,
        )

        comm_data_found = False

        for ep_idx in range(n_episodes):
            episode = CommEpisodeData(episode_idx=ep_idx)
            obs, _ = env.reset()

            obs_a = np.asarray(obs[AGENT_A], dtype=np.float32).ravel()
            _, _, goal = _decode_positions_and_goal(obs_a, self.grid_size)
            traps = _decode_traps_from_obs(obs_a, self.grid_size, self.num_traps)
            episode.traps = traps
            episode.goal = goal

            for t in range(max_steps):
                actions, _, outputs_per_agent = agent.act(
                    obs, timestep=t, timesteps=max_steps
                )

                # MAGICMAPPO.act() processes all agents in one forward pass;
                # agent_0's outputs slice contains the shared Scheduler outputs.
                merged_outputs = outputs_per_agent.get(AGENT_A, {})

                actions_int = {
                    k: int(np.asarray(v).ravel()[0]) for k, v in actions.items()
                }

                next_obs, rewards, terminated, truncated, _ = env.step(actions)

                next_obs_a = np.asarray(next_obs[AGENT_A], dtype=np.float32).ravel()
                pos_a, pos_b, goal_t = _decode_positions_and_goal(
                    next_obs_a, self.grid_size
                )
                reached_a, reached_b = _decode_reached_flags(next_obs_a, self.num_traps)
                rewards_float = {
                    k: float(np.asarray(v).ravel()[0]) for k, v in rewards.items()
                }

                adj_per_round, hard_adj, messages, agg_messages, logits = (
                    _extract_comm_outputs(
                        merged_outputs,
                        self._num_agents,
                        self.num_comm_rounds,
                        self.message_dim,
                    )
                )
                if adj_per_round is not None:
                    comm_data_found = True

                d2g_a = _manhattan(pos_a, goal_t)
                d2g_b = _manhattan(pos_b, goal_t)
                mt_a = _min_trap_dist(pos_a, traps)
                mt_b = _min_trap_dist(pos_b, traps)

                comm_density = 0.0
                if adj_per_round:
                    last = adj_per_round[-1]
                    comm_density = float(last.mean())

                record = CommStepRecord(
                    step=t,
                    positions={AGENT_A: pos_a, AGENT_B: pos_b},
                    goal=goal_t,
                    traps=traps,
                    dist_to_goal={AGENT_A: d2g_a, AGENT_B: d2g_b},
                    min_trap_dist={AGENT_A: mt_a, AGENT_B: mt_b},
                    reached={AGENT_A: reached_a, AGENT_B: reached_b},
                    actions=actions_int,
                    rewards=rewards_float,
                    adj_per_round=adj_per_round,
                    hard_adj=hard_adj,
                    messages=messages,
                    agg_messages=agg_messages,
                    logits=logits,
                    comm_density=comm_density,
                    total_reward=sum(rewards_float.values()),
                )
                episode.steps.append(record)

                done = any(
                    bool(np.asarray(v).ravel()[0])
                    for v in list(terminated.values()) + list(truncated.values())
                )
                if done:
                    episode.terminated = all(
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
                f"Comm density: {episode.mean_comm_density:.3f}  |  "
                f"A rew: {episode.total_rewards[AGENT_A]:+6.1f}  "
                f"B rew: {episode.total_rewards[AGENT_B]:+6.1f}"
            )

        data.has_comm_data = comm_data_found
        return data


# ──────────────────────────────────────────────────────────────────────────────
# Statistical analysis helpers (consumed by the visualizer)
# ──────────────────────────────────────────────────────────────────────────────


def compute_mean_adj(data: MAGICAnalysisData, round_idx: int = 0) -> np.ndarray:
    """Average adjacency matrix across all comm steps for the given round."""
    stack = data.adj_stack(round_idx)
    return stack.mean(axis=0) if stack.size > 0 else np.zeros((2, 2))


def compute_conditional_adj(
    data: MAGICAnalysisData,
    condition_fn,
    round_idx: int = 0,
) -> np.ndarray:
    """Mean adj for steps satisfying condition_fn(CommStepRecord) → bool."""
    mats = [
        s.adj_per_round[round_idx]
        for s in data.comm_steps()
        if condition_fn(s) and len(s.adj_per_round) > round_idx
    ]
    return np.stack(mats, axis=0).mean(axis=0) if mats else np.zeros((2, 2))


def compute_comm_vs_context(
    data: MAGICAnalysisData,
    context_key: str,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin communication density by a continuous context variable.
    Returns (bin_centres, mean_density, std_density).
    """
    ctx = data.context_at_comm_steps()
    if context_key not in ctx:
        raise ValueError(f"Unknown context key: {context_key}")
    x, y = ctx[context_key], ctx["comm_density"]
    bins = np.linspace(x.min(), x.max() + 1e-9, n_bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    means, stds = np.zeros(n_bins), np.zeros(n_bins)
    for i in range(n_bins):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.sum() > 0:
            means[i] = y[mask].mean()
            stds[i] = y[mask].std()
    return centres, means, stds


def compute_message_pca(
    data: MAGICAnalysisData, n_components: int = 2, use_agg: bool = False
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """PCA on all message embeddings. Returns (projected, explained_variance_ratio)."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return None, None

    mats = []
    for s in data.comm_steps():
        m = s.agg_messages if use_agg else s.messages
        if m is not None:
            mats.append(m)
    if not mats:
        return None, None

    X = np.concatenate(mats, axis=0)
    if X.shape[1] < n_components:
        return None, None
    pca = PCA(n_components=n_components, random_state=0)
    return pca.fit_transform(X), pca.explained_variance_ratio_


def compute_message_tsne(
    data: MAGICAnalysisData,
    n_components: int = 2,
    perplexity: float = 30.0,
    use_agg: bool = False,
    max_samples: int = 2000,
) -> np.ndarray | None:
    """t-SNE on message embeddings. Returns (M, 2) or None."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return None

    mats = []
    for s in data.comm_steps():
        m = s.agg_messages if use_agg else s.messages
        if m is not None:
            mats.append(m)
    if not mats:
        return None

    X = np.concatenate(mats, axis=0)
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]
    perp = min(perplexity, len(X) - 1)
    tsne = TSNE(
        n_components=n_components,
        perplexity=perp,
        random_state=0,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X)


def compute_directional_asymmetry(
    data: MAGICAnalysisData, round_idx: int = 0
) -> dict[str, float]:
    """
    For a 2-agent system compute communication directionality.
    Returns: a_to_b_mean, b_to_a_mean, asymmetry, self_loop_a, self_loop_b.
    """
    stack = data.adj_stack(round_idx)
    if stack.size == 0:
        return {
            "a_to_b_mean": 0.0,
            "b_to_a_mean": 0.0,
            "asymmetry": 0.0,
            "self_loop_a": 0.0,
            "self_loop_b": 0.0,
        }
    a2b = float(stack[:, 0, 1].mean())
    b2a = float(stack[:, 1, 0].mean())
    sa = float(stack[:, 0, 0].mean())
    sb = float(stack[:, 1, 1].mean())
    asym = abs(a2b - b2a) / (a2b + b2a + 1e-9)
    return {
        "a_to_b_mean": a2b,
        "b_to_a_mean": b2a,
        "asymmetry": asym,
        "self_loop_a": sa,
        "self_loop_b": sb,
    }


def align_around_event(
    data: MAGICAnalysisData,
    event_fn,
    window_before: int = 5,
    window_after: int = 10,
    round_idx: int = 0,
) -> np.ndarray | None:
    """
    Time-align adjacency matrices around events.
    Returns mean adj of shape (window_before + window_after, N, N), or None.
    """
    W = window_before + window_after
    N = 2
    acc = np.zeros((W, N, N))
    cnt_per_pos = np.zeros(W)

    for ep in data.episodes:
        events = event_fn(ep)
        for ev_t in events:
            for dt in range(-window_before, window_after):
                t_idx = ev_t + dt
                pos = dt + window_before
                if 0 <= t_idx < len(ep.steps):
                    s = ep.steps[t_idx]
                    if s.adj_per_round and len(s.adj_per_round) > round_idx:
                        acc[pos] += s.adj_per_round[round_idx]
                        cnt_per_pos[pos] += 1

    if cnt_per_pos.max() == 0:
        return None
    safe = np.where(cnt_per_pos > 0, cnt_per_pos, 1)
    return acc / safe[:, None, None]


def compute_action_entropy(data: MAGICAnalysisData) -> np.ndarray:
    """Per-step mean policy entropy from logits. Returns (T,) array."""
    entropies = []
    for s in data.comm_steps():
        if s.logits is not None:
            logits = s.logits
            mx = logits.max(axis=-1, keepdims=True)
            log_p = (
                logits
                - mx
                - np.log(np.sum(np.exp(logits - mx), axis=-1, keepdims=True) + 1e-9)
            )
            probs = np.exp(log_p)
            H = -(probs * log_p).sum(axis=-1)
            entropies.append(float(H.mean()))
    return np.asarray(entropies, dtype=np.float32)


def print_summary(data: MAGICAnalysisData) -> None:
    """Print a compact analysis summary to stdout."""
    print("\n" + "=" * 60)
    print("  MAGIC Communication Analysis Summary")
    print("=" * 60)
    print(f"  Episodes   : {data.n_episodes}")
    print(f"  Success    : {data.success_rate:.1%}")
    print(f"  Mean length: {data.mean_episode_length:.1f} steps")
    print(f"  Comm data  : {'available' if data.has_comm_data else 'NOT AVAILABLE'}")

    if data.has_comm_data:
        for r in range(data.num_comm_rounds):
            adj = compute_mean_adj(data, r)
            asym = compute_directional_asymmetry(data, r)
            print(f"\n  --- Comm Round {r} ---")
            print(f"  Mean adj:  A→B: {adj[0, 1]:.3f}   B→A: {adj[1, 0]:.3f}")
            print(f"             A self: {adj[0, 0]:.3f}   B self: {adj[1, 1]:.3f}")
            print(f"  Asymmetry: {asym['asymmetry']:.3f}")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Extended graph-theoretic analysis (new)
# ──────────────────────────────────────────────────────────────────────────────


def compute_topology_stability(
    data: MAGICAnalysisData,
    round_idx: int = 0,
    binary_threshold: float = 0.5,
) -> np.ndarray:
    """Per-step Jaccard similarity between consecutive hard-thresholded adjacency matrices.

    Measures how rapidly the communication topology changes from step to step.
    High stability (→ 1.0) = topology is learned and relatively fixed.
    Low stability (→ 0.0) = reactive, state-dependent communication.

    Returns
    -------
    (T-1,) float array of Jaccard similarities.  Empty array if < 2 comm steps.
    """
    csteps = [s for s in data.comm_steps() if len(s.adj_per_round) > round_idx]
    if len(csteps) < 2:
        return np.empty(0, dtype=np.float32)

    def _hard(adj: np.ndarray) -> np.ndarray:
        b = (adj >= binary_threshold).astype(np.float32)
        np.fill_diagonal(b, 0.0)
        return b.ravel()

    similarities = []
    for i in range(len(csteps) - 1):
        a = _hard(csteps[i].adj_per_round[round_idx])
        b = _hard(csteps[i + 1].adj_per_round[round_idx])
        inter = float((a * b).sum())
        union = float(((a + b) > 0).sum())
        similarities.append(inter / max(union, 1.0))
    return np.asarray(similarities, dtype=np.float32)


def compute_spectral_properties(
    data: MAGICAnalysisData,
    round_idx: int = 0,
) -> dict[str, float | np.ndarray]:
    """Spectral analysis of the time-averaged adjacency matrix.

    Returns
    -------
    dict with keys:
      eigenvalues     : (N,) real eigenvalues of mean_adj, descending.
      spectral_radius : largest |eigenvalue| — bounds message propagation speed.
      fiedler_value   : second-smallest eigenvalue of normalised Laplacian
                        (algebraic connectivity; 0 = disconnected).
      spectral_gap    : fiedler_value (alias for readability in figures).
    """
    mean_adj = compute_mean_adj(data, round_idx)
    eigs = np.sort(np.real(np.linalg.eigvals(mean_adj)))[::-1]

    # Normalised Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}
    deg = mean_adj.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-9)), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = np.eye(len(mean_adj)) - D_inv_sqrt @ mean_adj @ D_inv_sqrt
    L_eigs = np.sort(np.real(np.linalg.eigvalsh(L_norm)))

    fiedler = float(L_eigs[1]) if len(L_eigs) > 1 else 0.0

    return {
        "eigenvalues": eigs,
        "spectral_radius": float(np.abs(eigs).max()),
        "fiedler_value": fiedler,
        "spectral_gap": fiedler,
        "laplacian_eigenvalues": L_eigs,
    }


def compute_role_distribution(
    data: MAGICAnalysisData,
    round_idx: int = 0,
    binary_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Classify each agent into communication roles per step.

    Roles (based on off-diagonal adjacency):
      broadcaster : high out-degree, low in-degree  — sends to many, few reply
      receiver    : low out-degree, high in-degree   — listens to many, sends little
      relay       : high both                        — hub node
      isolated    : low both                         — barely connected

    Threshold for "high": above-median degree across the step.

    Returns
    -------
    dict mapping role_name → (N,) float array of fraction of steps in that role.
    """
    csteps = [s for s in data.comm_steps() if len(s.adj_per_round) > round_idx]
    if not csteps:
        return {}

    n = csteps[0].adj_per_round[round_idx].shape[0]
    role_counts: dict[str, np.ndarray] = {
        r: np.zeros(n, dtype=np.int32)
        for r in ("broadcaster", "receiver", "relay", "isolated")
    }

    for s in csteps:
        adj = s.adj_per_round[round_idx].copy()
        np.fill_diagonal(adj, 0.0)
        hard = (adj >= binary_threshold).astype(float)
        out_deg = hard.sum(axis=1)  # sends to how many
        in_deg = hard.sum(axis=0)  # receives from how many
        med = np.median(np.concatenate([out_deg, in_deg]))
        for i in range(n):
            is_out = out_deg[i] > med
            is_in = in_deg[i] > med
            if is_out and is_in:
                role_counts["relay"][i] += 1
            elif is_out:
                role_counts["broadcaster"][i] += 1
            elif is_in:
                role_counts["receiver"][i] += 1
            else:
                role_counts["isolated"][i] += 1

    total = len(csteps)
    return {role: counts / max(total, 1) for role, counts in role_counts.items()}


def compute_edge_bimodality(
    data: MAGICAnalysisData,
    round_idx: int = 0,
) -> dict[str, float]:
    """Measure how binary (0/1) the Gumbel-Softmax edge weights are.

    A learned, well-sharpened Gumbel-Softmax should produce off-diagonal
    weights that concentrate near 0 and near 1 — i.e., a bimodal distribution.

    Returns bimodality coefficient BC = (skewness² + 1) / (kurtosis + 3),
    with BC > 0.555 suggesting bimodality (D'Agostino & Pearson 1973).
    Also returns the fraction of weights above 0.5 (effective density).

    Returns
    -------
    dict with keys: bimodality_coeff, mean_weight, std_weight,
                    frac_above_half, frac_below_tenth.
    """
    stack = data.adj_stack(round_idx)
    if stack.size == 0:
        return {
            k: 0.0
            for k in (
                "bimodality_coeff",
                "mean_weight",
                "std_weight",
                "frac_above_half",
                "frac_below_tenth",
            )
        }

    n = stack.shape[1]
    off_mask = ~np.eye(n, dtype=bool)
    weights = stack[:, off_mask].ravel()

    mean = float(weights.mean())
    std = float(weights.std()) + 1e-9
    skew = float(((weights - mean) / std).mean() ** 3)  # crude skewness
    kurt = float(((weights - mean) / std**4).mean())  # crude kurtosis (excess)
    bc = (skew**2 + 1.0) / (kurt + 3.0) if abs(kurt + 3.0) > 1e-6 else 0.0

    return {
        "bimodality_coeff": float(bc),
        "mean_weight": mean,
        "std_weight": float(std),
        "frac_above_half": float((weights > 0.5).mean()),
        "frac_below_tenth": float((weights < 0.1).mean()),
    }


def compute_round_agreement(
    data: MAGICAnalysisData,
    r1: int = 0,
    r2: int = 1,
    binary_threshold: float = 0.5,
) -> np.ndarray:
    """Per-step Jaccard agreement between two communication rounds.

    Tells us how much the Scheduler's topology changes between rounds r1 and r2.
    Agreement near 1 = rounds learn the same graph (redundant communication).
    Agreement near 0 = rounds refine or contradict the topology.

    Returns (T,) float array, or empty array if data.num_comm_rounds < 2.
    """
    if data.num_comm_rounds < 2:
        return np.empty(0, dtype=np.float32)

    csteps = [s for s in data.comm_steps() if len(s.adj_per_round) > max(r1, r2)]
    if not csteps:
        return np.empty(0, dtype=np.float32)

    def _hard(adj: np.ndarray) -> np.ndarray:
        b = (adj >= binary_threshold).astype(float)
        np.fill_diagonal(b, 0.0)
        return b.ravel()

    agreements = []
    for s in csteps:
        a = _hard(s.adj_per_round[r1])
        b = _hard(s.adj_per_round[r2])
        inter = float((a * b).sum())
        union = float(((a + b) > 0).sum())
        agreements.append(inter / max(union, 1.0))
    return np.asarray(agreements, dtype=np.float32)


def compute_message_information_gain(
    data: MAGICAnalysisData,
) -> dict[str, np.ndarray]:
    """Measure how much messages change after GAT aggregation (per agent, per step).

    Information gain = normalised L2 distance between raw and aggregated messages:
        gain_i = ||agg_i - raw_i|| / (||raw_i|| + eps)

    High gain → GAT significantly transforms the message (rich aggregation).
    Low gain  → GAT barely changes the message (neighbours had little to add).

    Returns
    -------
    dict mapping agent_idx (int) → (T,) float array of per-step gain.
    """
    csteps = [
        s
        for s in data.comm_steps()
        if s.messages is not None and s.agg_messages is not None
    ]
    if not csteps:
        return {}

    n = csteps[0].messages.shape[0]
    gains: dict[int, list[float]] = {i: [] for i in range(n)}

    for s in csteps:
        for i in range(min(n, s.messages.shape[0])):
            raw = s.messages[i].astype(float)
            agg = s.agg_messages[i].astype(float)
            delta = float(np.linalg.norm(agg - raw))
            norm = float(np.linalg.norm(raw)) + 1e-9
            gains[i].append(delta / norm)

    return {i: np.asarray(v, dtype=np.float32) for i, v in gains.items()}


def compute_graph_density_over_time(
    data: MAGICAnalysisData,
    round_idx: int = 0,
    window: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling-mean communication density over episode steps.

    Returns (steps, mean_density_smoothed, raw_density).
    """
    csteps = [s for s in data.comm_steps() if len(s.adj_per_round) > round_idx]
    if not csteps:
        return np.empty(0), np.empty(0), np.empty(0)

    steps = np.asarray([s.step for s in csteps], dtype=np.float32)

    adjs = np.stack([s.adj_per_round[round_idx] for s in csteps], axis=0)  # (T, N, N)
    n = adjs.shape[1]
    off_mask = ~np.eye(n, dtype=bool)
    raw = adjs[:, off_mask].mean(axis=1)  # (T,)

    smoothed = np.convolve(raw, np.ones(window) / window, mode="same")
    return steps, smoothed, raw
