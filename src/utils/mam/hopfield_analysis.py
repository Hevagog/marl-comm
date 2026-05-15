"""
mamhm_analysis.py
==================
Data collection and statistical analysis for MAMHM's Hopfield Memory Bank
in any supported environment.

Architecture recap (MAMHM)
--------------------------
 1. BiMamba Encoder — bidirectional Mamba blocks encode all agents' observations
 2. Mamba Decoder — causal Mamba self-attn + CrossMamba + MLP
 3. HopfieldMemoryBank — learnable prototypes xi, attention-based retrieval
 4. Policy head — projects memory-augmented hidden state to action logits

Key explainability surface: the Hopfield attention weights alpha (N, K) reveal
which coordination prototypes each agent retrieves at each timestep.

Expected outputs from MAMHMMAPPO.act():
    outputs["logits"]          — rollout logits (B, num_actions) when available
    outputs["net_output"]      — policy-specific raw output (actions in AR mode)
    (Internal) memory attention — extracted separately via model forward pass

Usage
-----
from utils.mamhm_analysis import MAMHMCollector
from utils.mamhm_visualizer import save_all_mamhm_figures

collector = MAMHMCollector(num_agents=4, d_model=128, num_memories=64)
data = collector.collect(env, agent, n_episodes=30)
save_all_mamhm_figures(data, output_dir="eval_plots/mamhm", prefix="mamhm_warehouse")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class MAMHMStepRecord:
    """All memory/communication-relevant data captured at one environment step."""

    step: int

    # ---------- environment context ------------------------------------------
    actions: dict[str, int] = field(default_factory=dict)
    rewards: dict[str, float] = field(default_factory=dict)
    total_reward: float = 0.0

    # ---------- Hopfield memory outputs --------------------------------------
    # memory_attention: (N, K) attention weights over memory prototypes per agent
    memory_attention: np.ndarray | None = None
    # logits: (N, num_actions) policy logits per agent
    logits: np.ndarray | None = None
    # gate_value: scalar gate value (shared across agents)
    gate_value: float | None = None


@dataclass
class MAMHMEpisodeData:
    """Episode-level container for MAMHM analysis data."""

    episode_idx: int
    steps: list[MAMHMStepRecord] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    has_memory_data: bool = False

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def success(self) -> bool:
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
class MAMHMAnalysisData:
    """Top-level container returned by MAMHMCollector.collect()."""

    num_agents: int
    d_model: int
    num_memories: int
    episodes: list[MAMHMEpisodeData] = field(default_factory=list)
    has_memory_data: bool = False

    # Learned memory prototypes xi — shape (K, d_model). None if unavailable.
    xi_patterns: np.ndarray | None = None
    # Gate logit value (scalar). None if unavailable.
    gate_logit: float | None = None

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def success_rate(self) -> float:
        return (
            float(np.mean([e.success for e in self.episodes])) if self.episodes else 0.0
        )

    @property
    def mean_episode_length(self) -> float:
        return (
            float(np.mean([e.length for e in self.episodes])) if self.episodes else 0.0
        )

    @property
    def mean_episode_reward(self) -> float:
        return (
            float(np.mean([e.mean_episode_reward for e in self.episodes]))
            if self.episodes
            else 0.0
        )

    @property
    def gate_value(self) -> float | None:
        if self.gate_logit is not None:
            return float(1.0 / (1.0 + np.exp(-self.gate_logit)))
        return None

    def all_steps(self) -> list[MAMHMStepRecord]:
        return [s for e in self.episodes for s in e.steps]

    def memory_steps(self) -> list[MAMHMStepRecord]:
        """Steps that have valid memory attention data."""
        return [s for s in self.all_steps() if s.memory_attention is not None]

    def memory_attention_stack(self) -> np.ndarray:
        """Stack all memory attention matrices -> (T, N, K)."""
        mats = [
            s.memory_attention
            for s in self.memory_steps()
            if s.memory_attention is not None
        ]
        return (
            np.stack(mats, axis=0)
            if mats
            else np.empty((0, self.num_agents, self.num_memories))
        )

    def episode_rewards(self) -> np.ndarray:
        return np.asarray(
            [e.mean_episode_reward for e in self.episodes], dtype=np.float32
        )

    def episode_lengths(self) -> np.ndarray:
        return np.asarray([e.length for e in self.episodes], dtype=np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# JAX/numpy conversion helper
# ──────────────────────────────────────────────────────────────────────────────


def _np(v: Any) -> np.ndarray:
    try:
        import jax

        return np.asarray(jax.device_get(v))
    except Exception:
        return np.asarray(v)


# ──────────────────────────────────────────────────────────────────────────────
# Parameter extraction from agent
# ──────────────────────────────────────────────────────────────────────────────


def extract_hopfield_params(agent: Any) -> tuple[np.ndarray | None, float | None]:
    """Read learned Hopfield Memory Bank parameters from a MAMHM agent.

    Returns (xi_patterns, gate_logit) or (None, None) if not found.
    xi_patterns: (K, d_model) learnable memory prototypes.
    gate_logit: scalar float, the gate logit (sigmoid gives the actual gate).
    """
    try:
        uid0 = agent.possible_agents[0]
        policy = agent.policies[uid0]
        params = policy.state_dict.params["params"]

        # Navigate to the decoder's memory bank parameters
        decoder_params = params["_decoder"]
        mb_params = decoder_params["memory_bank"]

        xi = _np(mb_params["xi"]).astype(np.float32)
        gate_logit = float(_np(mb_params["gate_logit"]).ravel()[0])

        return xi, gate_logit
    except Exception as e:
        warnings.warn(f"[MAMHMAnalysis] Could not extract Hopfield params: {e}")
        return None, None


def compute_memory_attention_from_obs(
    agent: Any,
    obs_stacked: Any,
    taken_actions: Any | None = None,
) -> np.ndarray | None:
    """Run a forward pass to extract Hopfield memory attention weights.

    Parameters
    ----------
    agent      : MAMHMMAPPO instance
    obs_stacked  : (num_agents, obs_dim) stacked observations
    taken_actions: optional sampled actions for teacher-forced decoder inputs.
                   Supplying these makes the analysis match the actual rollout
                   prefixes instead of the zero-start fallback.

    Returns
    -------
    attention: (N, K) memory attention weights, or None on failure.
    """
    try:
        import jax
        import jax.numpy as jnp

        uid0 = agent.possible_agents[0]
        policy = agent.policies[uid0]
        params = policy.state_dict.params

        obs = jnp.asarray(obs_stacked, dtype=jnp.float32)
        if obs.ndim == 2:
            obs_grouped = obs[None, :, :]
        elif obs.ndim == 3:
            obs_grouped = obs
        else:
            return None

        groups = obs_grouped.shape[0]
        n = obs_grouped.shape[1]
        act_dim = int(policy.num_actions)

        if taken_actions is None:
            shifted = jnp.zeros((groups, n, act_dim + 1))
            shifted = shifted.at[:, 0, 0].set(1.0)
        else:
            actions = jnp.asarray(taken_actions, dtype=jnp.int32).reshape(groups * n)
            one_hot = jax.nn.one_hot(actions, act_dim).reshape(groups, n, act_dim)
            shifted = jnp.zeros((groups, n, act_dim + 1))
            shifted = shifted.at[:, 0, 0].set(1.0)
            shifted = shifted.at[:, 1:, 1:].set(one_hot[:, :-1, :])

        # Run encoder + decoder blocks with the same shifted actions used by the policy
        def _forward(model, obs_g, shifted_actions):
            obs_rep = model._encoder(obs_g)
            decoder = model._decoder
            x = decoder.ln(decoder.action_encoder(shifted_actions))
            for block in decoder.blocks:
                x = block(x, obs_rep)
            return decoder.memory_bank.get_memory_attention(x)

        attn = policy.apply(params, obs_grouped, shifted, method=_forward)
        return _np(attn[0])  # (N, K)

    except Exception as e:
        warnings.warn(f"[MAMHMAnalysis] Memory attention extraction failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Main collector
# ──────────────────────────────────────────────────────────────────────────────


class MAMHMCollector:
    """Run evaluation episodes with a MAMHM agent and collect per-step data.

    Captures generic step data (actions, rewards) plus MAMHM-specific data
    (Hopfield memory attention weights, gate value, memory prototypes).

    Parameters
    ----------
    num_agents   : number of agents in the environment.
    d_model      : MAMHM hidden dimension (from config).
    num_memories : number of Hopfield memory prototypes K.
    max_cycles   : episode length cap.
    """

    def __init__(
        self,
        num_agents: int = 4,
        d_model: int = 128,
        num_memories: int = 64,
        max_cycles: int = 500,
    ) -> None:
        self.num_agents = num_agents
        self.d_model = d_model
        self.num_memories = num_memories
        self.max_cycles = max_cycles

    def collect(
        self,
        env: Any,
        agent: Any,
        n_episodes: int = 30,
        max_steps_per_episode: int | None = None,
    ) -> MAMHMAnalysisData:
        """Run n_episodes evaluation episodes, capturing MAMHM Hopfield data."""
        import jax.numpy as jnp

        agent.set_running_mode("eval")
        max_steps = max_steps_per_episode or self.max_cycles
        possible_agents = list(env.possible_agents)
        num_agents = len(possible_agents)

        # Read Hopfield params once
        xi_patterns, gate_logit = extract_hopfield_params(agent)

        data = MAMHMAnalysisData(
            num_agents=num_agents,
            d_model=self.d_model,
            num_memories=self.num_memories,
            xi_patterns=xi_patterns,
            gate_logit=gate_logit,
        )

        memory_data_found = False

        for ep_idx in range(n_episodes):
            episode = MAMHMEpisodeData(episode_idx=ep_idx)
            obs, _ = env.reset()

            for t in range(max_steps):
                # Run the standard agent.act() for actions/rewards
                actions, _, outputs_per_agent = agent.act(
                    obs, timestep=t, timesteps=max_steps
                )

                actions_int = {
                    k: int(np.asarray(v).ravel()[0]) for k, v in actions.items()
                }

                # Extract memory attention from the actual sampled joint action prefix.
                try:
                    stacked_obs = jnp.concatenate(
                        [
                            agent._state_preprocessor[uid](obs[uid])
                            for uid in possible_agents
                        ],
                        axis=0,
                    )
                    stacked_actions = np.asarray(
                        [actions_int[uid] for uid in possible_agents],
                        dtype=np.int32,
                    )
                    memory_attn = compute_memory_attention_from_obs(
                        agent, stacked_obs, taken_actions=stacked_actions
                    )
                except Exception:
                    memory_attn = None

                if memory_attn is not None:
                    memory_data_found = True

                # Extract logits per agent
                logits = None
                logit_rows = []
                for uid in possible_agents:
                    ag_out = outputs_per_agent.get(uid, {})
                    raw = ag_out.get("logits")
                    if raw is None and "net_output" in ag_out:
                        candidate = _np(ag_out["net_output"])
                        if candidate.ndim >= 2 and candidate.shape[-1] > 1:
                            raw = ag_out["net_output"]
                    if raw is not None:
                        raw_np = _np(raw)
                        logit_rows.append(raw_np[0] if raw_np.ndim == 2 else raw_np)
                if len(logit_rows) == num_agents:
                    logits = np.stack(logit_rows, axis=0)

                next_obs, rewards, terminated, truncated, _ = env.step(actions)

                rewards_float = {
                    k: float(np.asarray(v).ravel()[0]) for k, v in rewards.items()
                }

                gate_val = None
                if gate_logit is not None:
                    gate_val = float(1.0 / (1.0 + np.exp(-gate_logit)))

                record = MAMHMStepRecord(
                    step=t,
                    actions=actions_int,
                    rewards=rewards_float,
                    total_reward=sum(rewards_float.values()),
                    memory_attention=memory_attn,
                    logits=logits,
                    gate_value=gate_val,
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

            episode.has_memory_data = any(
                s.memory_attention is not None for s in episode.steps
            )
            data.episodes.append(episode)
            print(
                f"  Episode {ep_idx + 1:3d}/{n_episodes}  |  "
                f"Len: {episode.length:3d}  |  "
                f"Success: {str(episode.success):5s}  |  "
                f"Reward: {episode.mean_episode_reward:+6.2f}"
            )

        data.has_memory_data = memory_data_found
        return data


# ──────────────────────────────────────────────────────────────────────────────
# Statistical analysis helpers (consumed by the visualizer)
# ──────────────────────────────────────────────────────────────────────────────


def compute_memory_utilization(data: MAMHMAnalysisData) -> dict[str, Any]:
    """Compute utilization statistics of the Hopfield memory prototypes.

    Returns
    -------
    dict with:
    - mean_attention: (K,) average attention per memory slot across all steps/agents
    - active_memories: int — number of memory slots with mean attention > 1/K
    - utilization_ratio: float — fraction of active memories
    - entropy_per_step: (T,) attention entropy per step (averaged over agents)
    - max_slot_per_agent: (N,) most-attended memory slot per agent
    """
    attn_stack = data.memory_attention_stack()  # (T, N, K)
    if attn_stack.shape[0] == 0:
        K = data.num_memories
        return {
            "mean_attention": np.zeros(K),
            "active_memories": 0,
            "utilization_ratio": 0.0,
            "entropy_per_step": np.array([]),
            "max_slot_per_agent": np.zeros(data.num_agents, dtype=int),
        }

    _, N, K = attn_stack.shape

    # Mean attention per memory slot
    mean_attn = attn_stack.mean(axis=(0, 1))  # (K,)
    uniform_threshold = 1.0 / K
    active_memories = int((mean_attn > uniform_threshold).sum())

    # Attention entropy per step (averaged over agents)
    eps = 1e-10
    entropy = -np.sum(attn_stack * np.log(attn_stack + eps), axis=-1)  # (T, N)
    entropy_per_step = entropy.mean(axis=1)  # (T,)

    # Most-attended memory slot per agent (mode across all steps)
    max_slots = attn_stack.argmax(axis=-1)  # (T, N)
    max_slot_per_agent = np.zeros(N, dtype=int)
    for i in range(N):
        counts = np.bincount(max_slots[:, i], minlength=K)
        max_slot_per_agent[i] = counts.argmax()

    return {
        "mean_attention": mean_attn,
        "active_memories": active_memories,
        "utilization_ratio": float(active_memories / K),
        "entropy_per_step": entropy_per_step,
        "max_slot_per_agent": max_slot_per_agent,
    }


def compute_agent_memory_profiles(data: MAMHMAnalysisData) -> dict[int, np.ndarray]:
    """Compute per-agent average attention profile over memory slots.

    Returns dict mapping agent_idx -> (K,) average attention distribution.
    """
    attn_stack = data.memory_attention_stack()  # (T, N, K)
    if attn_stack.shape[0] == 0:
        return {i: np.zeros(data.num_memories) for i in range(data.num_agents)}

    profiles = {}
    for i in range(attn_stack.shape[1]):
        profiles[i] = attn_stack[:, i, :].mean(axis=0)  # (K,)
    return profiles


def compute_memory_attention_dynamics(
    data: MAMHMAnalysisData,
    episode_idx: int = 0,
) -> np.ndarray | None:
    """Get memory attention over time for a single episode.

    Returns (T_ep, N, K) array or None if unavailable.
    """
    if episode_idx >= len(data.episodes):
        return None

    ep = data.episodes[episode_idx]
    attns = [s.memory_attention for s in ep.steps if s.memory_attention is not None]
    if not attns:
        return None
    return np.stack(attns, axis=0)


def compute_xi_similarity_matrix(data: MAMHMAnalysisData) -> np.ndarray | None:
    """Compute cosine similarity matrix between learned memory prototypes.

    Returns (K, K) cosine similarity matrix, or None.
    """
    xi = data.xi_patterns
    if xi is None:
        return None

    norms = np.linalg.norm(xi, axis=-1, keepdims=True) + 1e-8
    xi_normed = xi / norms
    return xi_normed @ xi_normed.T


def compute_xi_pca(
    data: MAMHMAnalysisData,
    n_components: int = 2,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """PCA on learned memory prototypes xi.

    Returns (projected (K, n_components), explained_variance_ratio) or (None, None).
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return None, None

    xi = data.xi_patterns
    if xi is None or xi.shape[0] < n_components:
        return None, None

    pca = PCA(n_components=n_components, random_state=0)
    projected = pca.fit_transform(xi)
    return projected, pca.explained_variance_ratio_


def compute_memory_reward_correlation(data: MAMHMAnalysisData) -> dict[str, Any]:
    """Correlate memory attention patterns with episode rewards.

    Returns dict with:
    - high_reward_profile: (K,) mean attention for top-quartile reward episodes
    - low_reward_profile: (K,) mean attention for bottom-quartile reward episodes
    - diff_profile: (K,) high - low (which memory slots associate with high reward)
    """
    K = data.num_memories
    empty = {
        "high_reward_profile": np.zeros(K),
        "low_reward_profile": np.zeros(K),
        "diff_profile": np.zeros(K),
    }

    if len(data.episodes) < 4:
        return empty

    # Compute per-episode mean attention profile
    ep_profiles = []
    ep_rewards = []
    for ep in data.episodes:
        attns = [s.memory_attention for s in ep.steps if s.memory_attention is not None]
        if not attns:
            continue
        stacked = np.stack(attns, axis=0)  # (T, N, K)
        ep_profiles.append(stacked.mean(axis=(0, 1)))  # (K,)
        ep_rewards.append(ep.mean_episode_reward)

    if len(ep_profiles) < 4:
        return empty

    ep_profiles = np.stack(ep_profiles, axis=0)  # (n_ep, K)
    ep_rewards = np.array(ep_rewards)

    q25 = np.percentile(ep_rewards, 25)
    q75 = np.percentile(ep_rewards, 75)

    high_mask = ep_rewards >= q75
    low_mask = ep_rewards <= q25

    high_profile = (
        ep_profiles[high_mask].mean(axis=0) if high_mask.any() else np.zeros(K)
    )
    low_profile = ep_profiles[low_mask].mean(axis=0) if low_mask.any() else np.zeros(K)

    return {
        "high_reward_profile": high_profile,
        "low_reward_profile": low_profile,
        "diff_profile": high_profile - low_profile,
    }


def compute_attention_concentration(data: MAMHMAnalysisData) -> dict[str, Any]:
    """Measure how concentrated (peaked) the attention distributions are.

    Uses the effective number of attended patterns: exp(H) where H is entropy.
    Low effective_k = concentrated; high = diffuse.

    Returns dict with:
    - effective_k_per_agent: (N,) mean effective number of attended patterns
    - effective_k_overall: float mean across all agents
    - concentration_ratio: float — how much more concentrated than uniform
    """
    attn_stack = data.memory_attention_stack()
    K = data.num_memories
    N = data.num_agents

    if attn_stack.shape[0] == 0:
        return {
            "effective_k_per_agent": np.full(N, K, dtype=np.float32),
            "effective_k_overall": float(K),
            "concentration_ratio": 1.0,
        }

    eps = 1e-10
    entropy = -np.sum(attn_stack * np.log(attn_stack + eps), axis=-1)  # (T, N)
    effective_k = np.exp(entropy)  # (T, N)

    effective_k_per_agent = effective_k.mean(axis=0)  # (N,)
    effective_k_overall = float(effective_k.mean())

    # Concentration ratio: uniform would give effective_k = K
    concentration_ratio = float(effective_k_overall / K)

    return {
        "effective_k_per_agent": effective_k_per_agent,
        "effective_k_overall": effective_k_overall,
        "concentration_ratio": concentration_ratio,
    }


def print_summary(data: MAMHMAnalysisData) -> None:
    """Print a compact MAMHM analysis summary to stdout."""
    util = compute_memory_utilization(data)
    conc = compute_attention_concentration(data)

    print("\n" + "=" * 64)
    print("  MAMHM Hopfield Memory Analysis Summary")
    print("=" * 64)
    print(f"  Agents        : {data.num_agents}")
    print(f"  d_model       : {data.d_model}")
    print(f"  Num memories K: {data.num_memories}")
    print(f"  Episodes      : {data.n_episodes}")
    print(f"  Success       : {data.success_rate:.1%}")
    print(f"  Mean length   : {data.mean_episode_length:.1f} steps")
    print(f"  Mean reward   : {data.mean_episode_reward:+.3f}")
    print(
        f"  Memory data   : {'available' if data.has_memory_data else 'NOT AVAILABLE'}"
    )

    if data.gate_logit is not None:
        gate_val = float(data.gate_value or 0.0)
        print(f"\n  Gate logit    : {data.gate_logit:.3f}")
        print(f"  Gate value    : {gate_val:.4f}  (sigmoid of logit)")
        print(f"  Effective γ   : {gate_val * 0.1:.5f}  (gate * gamma=0.1)")

    if data.xi_patterns is not None:
        xi = data.xi_patterns
        print(f"\n  xi patterns   : shape {xi.shape}")
        print(
            f"  xi norm range : [{np.linalg.norm(xi, axis=-1).min():.3f}, "
            f"{np.linalg.norm(xi, axis=-1).max():.3f}]"
        )

    print(
        f"\n  Active memories   : {util['active_memories']}/{data.num_memories} "
        f"({util['utilization_ratio']:.1%})"
    )
    print(
        f"  Effective K       : {conc['effective_k_overall']:.1f}/{data.num_memories} "
        f"({conc['concentration_ratio']:.1%} of uniform)"
    )
    if len(util["entropy_per_step"]) > 0:
        print(f"  Mean attention H  : {util['entropy_per_step'].mean():.3f} nats")

    print("=" * 64 + "\n")
