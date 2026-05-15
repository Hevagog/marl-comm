"""
magic_warehouse_comm_analysis.py
=================================
Data collection and statistical analysis of MAGIC's communication mechanism
in the multi-robot warehouse environment (4 agents, 12×16 grid).

Architecture recap (MAGIC, Niu et al. AAMAS 2021)
--------------------------------------------------
 1. Scheduler  – GAT encoder → MLP → Gumbel-Softmax → directed adj G^(l) ∈ {0,1}^(4×4)
 2. Message Processor – multi-head GAT over G^(l) aggregates 128-dim messages
 3. Policy head – acts on [obs_enc || agg_msg]

Expected communication tensor shapes (from MAGICMAPPO.act()):
    outputs["adj_matrices"]   – (num_comm_rounds, groups, N, N)  soft adjacency
    outputs["hard_adj"]       – (groups, N, N)                   binary adjacency
    outputs["messages"]       – (groups, N, message_dim)         pre-aggregation
    outputs["agg_messages"]   – (groups, N, message_dim)         post-GAT

Warehouse-specific context fields (from env info dict):
    phase, battery, speed, capacity, total_deliveries,
    pending_tasks, expired_tasks, stranded, dragging

Usage
-----
from utils.magic_warehouse_comm_analysis import MAGICWarehouseCommCollector
from utils.magic_warehouse_comm_visualizer import save_all_magic_warehouse_figures

collector = MAGICWarehouseCommCollector(
    grid_height=12, grid_width=16, num_agents=4,
    num_comm_rounds=2, message_dim=128,
)
data = collector.collect(env, magic_agent, n_episodes=30)
save_all_magic_warehouse_figures(data, output_dir="eval_plots/warehouse/magic")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ─── Agent labelling ──────────────────────────────────────────────────────────

NUM_AGENTS = 4
AGENT_LABELS = [f"agent_{i}" for i in range(NUM_AGENTS)]
SHORT_LABELS = [f"A{i}" for i in range(NUM_AGENTS)]

# Speed groups used in the warehouse heterogeneous config (speed_options=(1,1,2))
# Agents assigned cyclically: A0→speed=1, A1→speed=1, A2→speed=2, A3→speed=1
# (or whatever was assigned during env init — detected from info dict at runtime)
SPEED_FAST = 2.0


# ─── Output-key extraction (shared with magic_comm_analysis.py pattern) ───────

_MISSING_KEYS_WARNED = False


def _np(v: Any) -> np.ndarray:
    try:
        import jax

        return np.asarray(jax.device_get(v))
    except Exception:
        return np.asarray(v)


def _extract_comm_outputs(
    outputs: dict[str, Any],
    num_agents: int,
    num_comm_rounds: int,
    message_dim: int,
) -> tuple[
    list[np.ndarray] | None,  # adj_per_round: list of (N,N) per round
    np.ndarray | None,  # hard_adj: (N,N) binary
    np.ndarray | None,  # messages: (N, msg_dim)
    np.ndarray | None,  # agg_messages: (N, msg_dim)
    np.ndarray | None,  # logits: (N, num_actions)
]:
    global _MISSING_KEYS_WARNED

    # ── adjacency matrices ────────────────────────────────────────────────────
    adj_per_round: list[np.ndarray] | None = None
    if "adj_matrices" in outputs:
        raw = _np(outputs["adj_matrices"])
        if raw.ndim == 4:
            # Shape (num_rounds, groups, N, N) or (groups, num_rounds, N, N)
            if raw.shape[0] == num_comm_rounds:
                adj_per_round = [raw[r, 0] for r in range(num_comm_rounds)]
            else:
                adj_per_round = [raw[0, r] for r in range(num_comm_rounds)]
        elif raw.ndim == 3:
            adj_per_round = [raw[r] for r in range(min(num_comm_rounds, raw.shape[0]))]
        elif raw.ndim == 2:
            adj_per_round = [raw]
        if adj_per_round is not None:
            while len(adj_per_round) < num_comm_rounds:
                adj_per_round.append(adj_per_round[-1])

    hard_adj: np.ndarray | None = None
    if "hard_adj" in outputs:
        raw = _np(outputs["hard_adj"])
        hard_adj = raw[0] if raw.ndim > 2 else raw

    # ── messages ─────────────────────────────────────────────────────────────
    messages: np.ndarray | None = None
    if "messages" in outputs:
        raw = _np(outputs["messages"])
        messages = raw[0] if raw.ndim == 3 else raw

    agg_messages: np.ndarray | None = None
    if "agg_messages" in outputs:
        raw = _np(outputs["agg_messages"])
        agg_messages = raw[0] if raw.ndim == 3 else raw

    # ── logits ────────────────────────────────────────────────────────────────
    logits: np.ndarray | None = None
    if "net_output" in outputs:
        raw = _np(outputs["net_output"])
        logits = raw[:num_agents] if raw.ndim == 2 else raw

    if adj_per_round is None and not _MISSING_KEYS_WARNED:
        warnings.warn(
            "\n[MAGICWarehouseCommAnalysis] Communication outputs not found.\n"
            "  Expected keys: 'adj_matrices', 'messages', 'agg_messages', 'hard_adj'\n"
            "  Communication-specific plots will be skipped.",
            stacklevel=3,
        )
        _MISSING_KEYS_WARNED = True

    return adj_per_round, hard_adj, messages, agg_messages, logits


# ─── Per-step data container ──────────────────────────────────────────────────


@dataclass
class WarehouseCommStepRecord:
    """Communication + context data for one warehouse environment step."""

    step: int

    # ── Communication internals ───────────────────────────────────────────────
    adj_per_round: list[np.ndarray] | None = None  # list of (N,N) soft adj
    hard_adj: np.ndarray | None = None  # (N,N) binary
    messages: np.ndarray | None = None  # (N, msg_dim) pre-GAT
    agg_messages: np.ndarray | None = None  # (N, msg_dim) post-GAT
    logits: np.ndarray | None = None  # (N, num_actions)

    # ── Warehouse context ─────────────────────────────────────────────────────
    positions: dict[str, tuple[int, int]] = field(default_factory=dict)
    phase: dict[str, int] = field(default_factory=dict)
    battery: dict[str, float] = field(default_factory=dict)
    carrying: dict[str, int] = field(default_factory=dict)
    speed: dict[str, float] = field(default_factory=dict)
    capacity: dict[str, int] = field(default_factory=dict)
    stranded: dict[str, bool] = field(default_factory=dict)
    dragging: dict[str, bool] = field(default_factory=dict)
    active: dict[str, bool] = field(default_factory=dict)

    pending_tasks: int = 0
    expired_tasks_cumulative: int = 0
    delivery_cumulative: dict[str, int] = field(default_factory=dict)

    rewards: dict[str, float] = field(default_factory=dict)
    actions: dict[str, int] = field(default_factory=dict)

    # ── Derived scalars ───────────────────────────────────────────────────────
    comm_density: float = 0.0  # fraction of off-diagonal directed edges active

    @property
    def min_battery(self) -> float:
        vals = [
            v
            for v in self.battery.values()
            if self.active.get(
                k := list(self.battery.keys())[list(self.battery.values()).index(v)],
                True,
            )
        ]
        return float(min(self.battery.values())) if self.battery else 160.0

    @property
    def battery_emergency(self) -> bool:
        return any(v < 30.0 for v in self.battery.values())

    @property
    def any_rescuing(self) -> bool:
        return any(self.dragging.values())

    @property
    def treating_count(self) -> int:
        return sum(1 for p in self.phase.values() if p == 2)

    @property
    def searching_count(self) -> int:
        return sum(1 for p in self.phase.values() if p == 0)

    @property
    def total_deliveries(self) -> int:
        return sum(self.delivery_cumulative.values())


# ─── Episode container ────────────────────────────────────────────────────────


@dataclass
class WarehouseCommEpisodeData:
    episode_idx: int
    steps: list[WarehouseCommStepRecord] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    has_comm_data: bool = False
    # Agent role lookup (set from first step)
    agent_speeds: dict[str, float] = field(default_factory=dict)
    agent_capacities: dict[str, int] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def total_deliveries(self) -> int:
        return self.steps[-1].total_deliveries if self.steps else 0

    @property
    def mean_comm_density(self) -> float:
        vals = [s.comm_density for s in self.steps if s.adj_per_round is not None]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def total_rewards(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in self.steps:
            for a, r in s.rewards.items():
                out[a] = out.get(a, 0.0) + r
        return out

    def rescue_steps(self) -> list[int]:
        return [s.step for s in self.steps if s.any_rescuing]


# ─── Top-level analysis data container ───────────────────────────────────────


@dataclass
class MAGICWarehouseData:
    """Container returned by MAGICWarehouseCommCollector.collect()."""

    grid_height: int
    grid_width: int
    num_agents: int
    num_comm_rounds: int
    message_dim: int
    episodes: list[WarehouseCommEpisodeData] = field(default_factory=list)
    has_comm_data: bool = False

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def mean_deliveries(self) -> float:
        return float(np.mean([ep.total_deliveries for ep in self.episodes]))

    @property
    def mean_episode_length(self) -> float:
        return float(np.mean([ep.length for ep in self.episodes]))

    def all_steps(self) -> list[WarehouseCommStepRecord]:
        return [s for ep in self.episodes for s in ep.steps]

    def comm_steps(self) -> list[WarehouseCommStepRecord]:
        return [s for s in self.all_steps() if s.adj_per_round is not None]

    def adj_stack(self, round_idx: int = 0) -> np.ndarray:
        """Stack soft adjacency matrices for the given round → (T, N, N)."""
        mats = [
            s.adj_per_round[round_idx]
            for s in self.comm_steps()
            if len(s.adj_per_round) > round_idx
        ]
        N = self.actual_num_agents
        return np.stack(mats, axis=0) if mats else np.empty((0, N, N))

    @property
    def actual_num_agents(self) -> int:
        """Actual N inferred from collected adj/message tensors.

        The configured ``num_agents`` may be smaller than the size of the
        adjacency matrices produced by MAGIC (which uses ``possible_agents``
        / ``max_agents``). This property reads the first available tensor to
        return the true N so that visualisation code uses the correct shape.
        """
        for ep in self.episodes:
            for s in ep.steps:
                if s.adj_per_round:
                    return int(s.adj_per_round[0].shape[0])
                if s.messages is not None:
                    return int(s.messages.shape[0])
        return self.num_agents

    # ── Role lookups ───────────────────────────────────────────────────────────

    def agent_roles(self) -> dict[str, dict[str, Any]]:
        """Return per-agent role dict {speed, capacity} from first available step."""
        roles: dict[str, dict] = {}
        for ep in self.episodes:
            if ep.agent_speeds:
                for a, sp in ep.agent_speeds.items():
                    roles[a] = {
                        "speed": sp,
                        "capacity": ep.agent_capacities.get(a, 1),
                        "is_fast": sp >= SPEED_FAST,
                    }
                break
        return roles

    # ── Context array for correlation analysis ─────────────────────────────────

    def context_array(self) -> dict[str, np.ndarray]:
        """
        Returns a dict of float arrays, one value per comm step, for correlating
        with communication density. Keys:
            comm_density, min_battery, mean_battery, pending_tasks,
            treating_count, searching_count, step_fraction,
            any_rescue, total_deliveries_at_step
        """
        csteps = self.comm_steps()
        if not csteps:
            return {}
        max_cycles = max(s.step for s in csteps) + 1 or 1
        ctx: dict[str, list] = {
            k: []
            for k in [
                "comm_density",
                "min_battery",
                "mean_battery",
                "pending_tasks",
                "treating_count",
                "searching_count",
                "step_fraction",
                "any_rescue",
                "total_deliveries",
            ]
        }
        for s in csteps:
            batt_vals = list(s.battery.values())
            ctx["comm_density"].append(s.comm_density)
            ctx["min_battery"].append(float(min(batt_vals)) if batt_vals else 160.0)
            ctx["mean_battery"].append(
                float(np.mean(batt_vals)) if batt_vals else 160.0
            )
            ctx["pending_tasks"].append(float(s.pending_tasks))
            ctx["treating_count"].append(float(s.treating_count))
            ctx["searching_count"].append(float(s.searching_count))
            ctx["step_fraction"].append(s.step / max_cycles)
            ctx["any_rescue"].append(float(s.any_rescuing))
            ctx["total_deliveries"].append(float(s.total_deliveries))
        return {k: np.asarray(v, dtype=np.float32) for k, v in ctx.items()}


# ─── Main collector ───────────────────────────────────────────────────────────


class MAGICWarehouseCommCollector:
    """
    Run evaluation episodes with a MAGIC agent in the warehouse and collect
    per-step communication tensors alongside warehouse-specific context.

    Parameters
    ----------
    grid_height      : warehouse rows (default 12)
    grid_width       : warehouse cols (default 16)
    num_agents       : active agents (default 4)
    num_comm_rounds  : MAGIC communication rounds (default 2)
    message_dim      : message embedding dimension (default 128)
    battery_capacity : full battery level (default 160)
    max_cycles       : episode length cap (default 500)
    """

    def __init__(
        self,
        grid_height: int = 12,
        grid_width: int = 16,
        num_agents: int = 4,
        num_comm_rounds: int = 2,
        message_dim: int = 128,
        battery_capacity: float = 160.0,
        max_cycles: int = 500,
    ) -> None:
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_agents = num_agents
        self.num_comm_rounds = num_comm_rounds
        self.message_dim = message_dim
        self.battery_capacity = battery_capacity
        self.max_cycles = max_cycles
        self._agents = [f"agent_{i}" for i in range(num_agents)]

    def collect(
        self,
        env: Any,
        agent: Any,
        n_episodes: int = 4,
        max_steps_per_episode: int | None = None,
    ) -> MAGICWarehouseData:
        agent.set_running_mode("eval")
        max_steps = max_steps_per_episode or self.max_cycles

        data = MAGICWarehouseData(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            num_agents=self.num_agents,
            num_comm_rounds=self.num_comm_rounds,
            message_dim=self.message_dim,
        )
        comm_found = False

        for ep_idx in range(n_episodes):
            episode = WarehouseCommEpisodeData(episode_idx=ep_idx)
            obs, info = env.reset()
            actual_agents = list(obs.keys())

            # Extract agent roles from initial info dict
            for a in actual_agents:
                a_info = info.get(a, {}) if info else {}
                episode.agent_speeds[a] = float(a_info.get("speed", 1.0))
                episode.agent_capacities[a] = int(a_info.get("capacity", 1))

            prev_deliveries: dict[str, int] = {a: 0 for a in actual_agents}

            for t in range(max_steps):
                actions, _, outputs_per_agent = agent.act(
                    obs, timestep=t, timesteps=max_steps
                )

                # MAGICMAPPO passes shared Scheduler outputs via the first agent's dict
                merged_outputs = outputs_per_agent.get(actual_agents[0], {})

                actions_int = {
                    k: int(np.asarray(v).ravel()[0]) for k, v in actions.items()
                }

                next_obs, rewards, terminated, truncated, next_info = env.step(actions)
                rewards_float = {
                    k: float(np.asarray(v).ravel()[0]) for k, v in rewards.items()
                }

                # Decode comm tensors
                adj_per_round, hard_adj, messages, agg_messages, logits = (
                    _extract_comm_outputs(
                        merged_outputs,
                        self.num_agents,
                        self.num_comm_rounds,
                        self.message_dim,
                    )
                )
                if adj_per_round is not None:
                    comm_found = True

                # Comm density: mean weight of off-diagonal edges in last-round adj
                comm_density = 0.0
                if adj_per_round:
                    last = adj_per_round[-1]
                    N = last.shape[0]
                    if N > 1:
                        off_diag = last[~np.eye(N, dtype=bool)]
                        comm_density = float(off_diag.mean())

                # Build context from info dict
                positions: dict[str, tuple[int, int]] = {}
                phase: dict[str, int] = {}
                battery: dict[str, float] = {}
                carrying: dict[str, int] = {}
                speed: dict[str, float] = {}
                capacity: dict[str, int] = {}
                stranded: dict[str, bool] = {}
                dragging: dict[str, bool] = {}
                active: dict[str, bool] = {}
                delivery_cumulative: dict[str, int] = {}

                for a in actual_agents:
                    a_obs = np.asarray(
                        next_obs.get(a, obs.get(a, np.zeros(227))), dtype=np.float32
                    ).ravel()
                    a_info = next_info.get(a, {}) if next_info else {}

                    row = int(round(float(a_obs[4]) * max(self.grid_height - 1, 1)))
                    col = int(round(float(a_obs[5]) * max(self.grid_width - 1, 1)))
                    positions[a] = (
                        max(0, min(self.grid_height - 1, row)),
                        max(0, min(self.grid_width - 1, col)),
                    )

                    phase[a] = int(a_info.get("phase", int(round(float(a_obs[1]) * 3))))
                    battery[a] = float(
                        a_info.get("battery", float(a_obs[-1]) * self.battery_capacity)
                    )
                    carrying[a] = int(a_info.get("carrying", int(a_obs[0] > 0.5)))
                    speed[a] = float(a_info.get("speed", 1.0))
                    capacity[a] = int(a_info.get("capacity", 1))
                    stranded[a] = bool(a_info.get("stranded", False))
                    dragging[a] = bool(a_info.get("dragging", False))
                    active[a] = bool(a_info.get("active", True))
                    delivery_cumulative[a] = int(
                        a_info.get("total_deliveries", prev_deliveries.get(a, 0))
                    )

                prev_deliveries = dict(delivery_cumulative)

                first_info = next(
                    (
                        next_info.get(a, {})
                        for a in actual_agents
                        if next_info and a in next_info
                    ),
                    {},
                )
                pending_tasks = int(first_info.get("pending_tasks", 0))
                expired_cumulative = int(first_info.get("expired_tasks", 0))

                record = WarehouseCommStepRecord(
                    step=t,
                    adj_per_round=adj_per_round,
                    hard_adj=hard_adj,
                    messages=messages,
                    agg_messages=agg_messages,
                    logits=logits,
                    positions=positions,
                    phase=phase,
                    battery=battery,
                    carrying=carrying,
                    speed=speed,
                    capacity=capacity,
                    stranded=stranded,
                    dragging=dragging,
                    active=active,
                    pending_tasks=pending_tasks,
                    expired_tasks_cumulative=expired_cumulative,
                    delivery_cumulative=delivery_cumulative,
                    rewards=rewards_float,
                    actions=actions_int,
                    comm_density=comm_density,
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

            episode.has_comm_data = comm_found
            data.episodes.append(episode)
            print(
                f"  Episode {ep_idx + 1:3d}/{n_episodes}  |  "
                f"Len: {episode.length:3d}  |  "
                f"Deliveries: {episode.total_deliveries:3d}  |  "
                f"Comm density: {episode.mean_comm_density:.3f}  |  "
                f"Reward: {sum(episode.total_rewards.values()):+7.1f}"
            )

        data.has_comm_data = comm_found
        return data


# ─── Statistical analysis helpers ─────────────────────────────────────────────


def compute_mean_adj(data: MAGICWarehouseData, round_idx: int = 0) -> np.ndarray:
    """Average soft adjacency matrix across all comm steps → (N, N)."""
    stack = data.adj_stack(round_idx)
    N = data.actual_num_agents
    return stack.mean(axis=0) if stack.size > 0 else np.zeros((N, N))


def compute_conditional_adj(
    data: MAGICWarehouseData,
    condition_fn,
    round_idx: int = 0,
) -> np.ndarray:
    """Mean adj for steps satisfying condition_fn(WarehouseCommStepRecord) → bool."""
    N = data.actual_num_agents
    mats = [
        s.adj_per_round[round_idx]
        for s in data.comm_steps()
        if condition_fn(s) and s.adj_per_round and len(s.adj_per_round) > round_idx
    ]
    return np.stack(mats, axis=0).mean(axis=0) if mats else np.zeros((N, N))


def compute_comm_vs_context(
    data: MAGICWarehouseData,
    context_key: str,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin communication density by a continuous context variable.
    Returns (bin_centres, mean_density, std_density).
    """
    ctx = data.context_array()
    if context_key not in ctx or "comm_density" not in ctx:
        return np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    x, y = ctx[context_key], ctx["comm_density"]
    if len(x) == 0:
        return np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    bins = np.linspace(x.min(), x.max() + 1e-9, n_bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    means = np.zeros(n_bins)
    stds = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.sum() > 0:
            means[i] = y[mask].mean()
            stds[i] = y[mask].std()
    return centres, means, stds


def compute_pairwise_weights(
    data: MAGICWarehouseData, round_idx: int = 0
) -> dict[str, float]:
    """
    For each directed pair (i→j, i≠j), return mean soft adjacency weight.
    Keys are like "A0→A1".
    """
    stack = data.adj_stack(round_idx)
    if stack.size == 0:
        return {}
    N = stack.shape[1]
    labels = [f"A{i}" for i in range(N)]
    result = {}
    for i in range(N):
        for j in range(N):
            if i != j:
                result[f"{labels[i]}→{labels[j]}"] = float(stack[:, i, j].mean())
    return result


def compute_hub_scores(
    data: MAGICWarehouseData, round_idx: int = 0, use_hard: bool = True
) -> np.ndarray:
    """
    In-degree of each agent (how many others communicate to it).
    Returns (T, N) array of per-step in-degrees.
    """
    N = data.actual_num_agents
    csteps = data.comm_steps()
    if not csteps:
        return np.empty((0, N))
    scores = []
    for s in csteps:
        if use_hard and s.hard_adj is not None:
            mat = s.hard_adj.copy()
        elif s.adj_per_round and len(s.adj_per_round) > round_idx:
            mat = (s.adj_per_round[round_idx] > 0.5).astype(float)
        else:
            mat = np.zeros((N, N))
        # In-degree = column sum, excluding self
        np.fill_diagonal(mat, 0)
        scores.append(mat.sum(axis=0))  # (N,)
    return np.array(scores)  # (T, N)


def compute_message_pca(
    data: MAGICWarehouseData,
    n_components: int = 2,
    use_agg: bool = False,
    max_samples: int = 4000,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    PCA on message embeddings.
    Returns (projected (M,2), explained_variance_ratio (2,), labels (M,)).
    Labels encode which agent produced the message (0–3).
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return None, None, None

    mats, labels = [], []
    for s in data.comm_steps():
        m = s.agg_messages if use_agg else s.messages
        if m is not None and m.ndim == 2:
            mats.append(m)
            labels.extend(list(range(m.shape[0])))

    if not mats:
        return None, None, None

    X = np.concatenate(mats, axis=0)
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, labels = X[idx], [labels[i] for i in idx]

    if X.shape[1] < n_components:
        return None, None, None
    pca = PCA(n_components=n_components, random_state=0)
    return pca.fit_transform(X), pca.explained_variance_ratio_, np.array(labels)


def compute_message_pca_with_context(
    data: MAGICWarehouseData,
    context_key: str = "phase",
    n_components: int = 2,
    use_agg: bool = False,
    max_samples: int = 4000,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    PCA on message embeddings coloured by a per-agent context variable.
    context_key: 'phase', 'battery', 'speed', 'capacity'
    Returns (projected, explained_variance_ratio, context_values).
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return None, None, None

    mats, context_vals = [], []
    for s in data.comm_steps():
        m = s.agg_messages if use_agg else s.messages
        if m is None or m.ndim != 2:
            continue
        ctx_source = {
            "phase": s.phase,
            "battery": s.battery,
            "speed": s.speed,
            "capacity": s.capacity,
        }.get(context_key, s.phase)
        agents = [f"agent_{i}" for i in range(m.shape[0])]
        for i, a in enumerate(agents):
            mats.append(m[i])
            context_vals.append(float(ctx_source.get(a, 0)))

    if not mats:
        return None, None, None

    X = np.array(mats)
    C = np.array(context_vals)
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, C = X[idx], C[idx]

    if X.shape[1] < n_components:
        return None, None, None
    pca = PCA(n_components=n_components, random_state=0)
    return pca.fit_transform(X), pca.explained_variance_ratio_, C


def compute_round_divergence(
    data: MAGICWarehouseData,
) -> np.ndarray | None:
    """
    Per-step mean L2 distance between Round-1 and Round-2 soft adjacency matrices.
    Returns (T,) array or None if fewer than 2 rounds.
    """
    if data.num_comm_rounds < 2:
        return None
    diffs = []
    for s in data.comm_steps():
        if s.adj_per_round and len(s.adj_per_round) >= 2:
            diff = np.linalg.norm(s.adj_per_round[1] - s.adj_per_round[0])
            diffs.append(diff)
    return np.array(diffs, dtype=np.float32) if diffs else None


def compute_message_norm_over_time(
    data: MAGICWarehouseData, use_agg: bool = False
) -> dict[str, np.ndarray]:
    """
    Per-step mean L2 norm of messages, split by agent.
    Returns dict {agent_name: (T,) array}.
    """
    N = data.actual_num_agents
    agents = [f"agent_{i}" for i in range(N)]
    norms: dict[str, list[float]] = {a: [] for a in agents}
    for ep in data.episodes:
        for s in ep.steps:
            if s.adj_per_round is None:
                continue
            m = s.agg_messages if use_agg else s.messages
            if m is None or m.ndim != 2:
                continue
            for i in range(min(m.shape[0], N)):
                norms[agents[i]].append(float(np.linalg.norm(m[i])))
    return {a: np.array(v, dtype=np.float32) for a, v in norms.items()}


def align_comm_around_rescue(
    data: MAGICWarehouseData,
    window_before: int = 10,
    window_after: int = 15,
    round_idx: int = 0,
) -> np.ndarray | None:
    """
    Time-align adjacency matrices around rescue events (steps where any agent drags).
    Returns mean adj of shape (window_before+window_after, N, N), or None.
    """
    W = window_before + window_after
    N = data.actual_num_agents
    acc = np.zeros((W, N, N))
    cnt = np.zeros(W)

    for ep in data.episodes:
        rescue_ts = ep.rescue_steps()
        steps_by_t = {s.step: s for s in ep.steps}
        for ev_t in rescue_ts:
            for dt in range(-window_before, window_after):
                t_idx = ev_t + dt
                pos = dt + window_before
                if t_idx in steps_by_t:
                    s = steps_by_t[t_idx]
                    if s.adj_per_round and len(s.adj_per_round) > round_idx:
                        acc[pos] += s.adj_per_round[round_idx]
                        cnt[pos] += 1

    if cnt.max() == 0:
        return None
    safe = np.where(cnt > 0, cnt, 1)
    return acc / safe[:, None, None]


def print_summary(data: MAGICWarehouseData) -> None:
    print("\n" + "=" * 65)
    print("  MAGIC Warehouse Communication Analysis Summary")
    print("=" * 65)
    print(f"  Episodes            : {data.n_episodes}")
    print(f"  Mean deliveries/ep  : {data.mean_deliveries:.1f}")
    print(f"  Mean ep. length     : {data.mean_episode_length:.0f}")
    print(f"  Comm data available : {'YES' if data.has_comm_data else 'NO'}")
    if data.has_comm_data:
        for r in range(data.num_comm_rounds):
            adj = compute_mean_adj(data, r)
            N = adj.shape[0]
            agent_labels = [f"A{i}" for i in range(N)]
            print(f"\n  ── Round {r + 1} mean adj (rows=sender, cols=receiver) ──")
            header = "        " + "  ".join(f"{lbl:>5}" for lbl in agent_labels)
            print(header)
            for i in range(N):
                row_str = f"  {agent_labels[i]}  |  " + "  ".join(
                    f"{adj[i, j]:.3f}" for j in range(N)
                )
                print(row_str)
        pw = compute_pairwise_weights(data, round_idx=data.num_comm_rounds - 1)
        if pw:
            print("\n  Top-3 communication pairs (final round):")
            for k, v in sorted(pw.items(), key=lambda x: -x[1])[:3]:
                print(f"    {k}: {v:.3f}")
    print("=" * 65 + "\n")
