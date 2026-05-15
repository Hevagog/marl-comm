"""CoinGame altruism-persistence analysis.

This module collects event-level behavioural metrics and Hopfield-prototype
diagnostics for the reward-shift experiment:

Phase 1: train with social_welfare_alpha > 0.
Phase 2: resume in the standard environment with social_welfare_alpha = 0.

The important distinction is that cooperation is measured from coin-pickup
events, while rewards are decomposed from the environment's raw reward
components rather than reconstructed from aggregate returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


_REWARD_COMPONENTS = (
    "own_coin",
    "steal_gain",
    "steal_penalty_received",
    "raw_reward",
    "mixed_reward",
    "social_welfare_transfer",
)


@dataclass
class PrototypeProbe:
    """One probe of the HSC prototype-attention distribution."""

    entropy: float
    max_prob: float
    perplexity: float
    usage: np.ndarray


@dataclass
class AltruismEpisodeData:
    episode_idx: int
    agent_names: list[str] = field(default_factory=list)

    own_picks_per_step: dict[str, list[int]] = field(default_factory=dict)
    steal_picks_per_step: dict[str, list[int]] = field(default_factory=dict)
    victimizations_per_step: dict[str, list[int]] = field(default_factory=dict)
    rewards_per_step: dict[str, list[float]] = field(default_factory=dict)
    raw_rewards_per_step: dict[str, list[float]] = field(default_factory=dict)

    own_picks_total: dict[str, int] = field(default_factory=dict)
    steal_picks_total: dict[str, int] = field(default_factory=dict)
    victimizations_total: dict[str, int] = field(default_factory=dict)
    total_reward: dict[str, float] = field(default_factory=dict)
    raw_total_reward: dict[str, float] = field(default_factory=dict)
    reward_components_total: dict[str, dict[str, float]] = field(default_factory=dict)

    pickup_events: list[dict[str, Any]] = field(default_factory=list)
    episode_len: int = 0

    def cooperation_rate_of(self, agent: str) -> float:
        own = self.own_picks_total.get(agent, 0)
        steal = self.steal_picks_total.get(agent, 0)
        total = own + steal
        return float(own) / total if total > 0 else float("nan")

    def steal_rate_of(self, agent: str) -> float:
        own = self.own_picks_total.get(agent, 0)
        steal = self.steal_picks_total.get(agent, 0)
        total = own + steal
        return float(steal) / total if total > 0 else float("nan")


@dataclass
class AltruismAnalysisData:
    """Aggregated results across evaluation episodes."""

    episodes: list[AltruismEpisodeData] = field(default_factory=list)
    prototype_cosine_drift: list[float | None] = field(default_factory=list)
    prototype_entropy: list[float | None] = field(default_factory=list)
    prototype_max_prob: list[float | None] = field(default_factory=list)
    prototype_perplexity: list[float | None] = field(default_factory=list)
    prototype_usage: list[np.ndarray | None] = field(default_factory=list)
    reference_prototypes: np.ndarray | None = None
    checkpoint_label: str | None = None

    @property
    def agent_names(self) -> list[str]:
        return self.episodes[0].agent_names if self.episodes else []

    @property
    def cooperation_rates(self) -> dict[str, np.ndarray]:
        return {
            a: np.array(
                [ep.cooperation_rate_of(a) for ep in self.episodes], dtype=float
            )
            for a in self.agent_names
        }

    @property
    def steal_rates(self) -> dict[str, np.ndarray]:
        return {
            a: np.array([ep.steal_rate_of(a) for ep in self.episodes], dtype=float)
            for a in self.agent_names
        }

    @property
    def steal_counts(self) -> dict[str, np.ndarray]:
        return {
            a: np.array([ep.steal_picks_total.get(a, 0) for ep in self.episodes])
            for a in self.agent_names
        }

    @property
    def victimization_counts(self) -> dict[str, np.ndarray]:
        return {
            a: np.array([ep.victimizations_total.get(a, 0) for ep in self.episodes])
            for a in self.agent_names
        }

    @property
    def episode_rewards(self) -> dict[str, np.ndarray]:
        return {
            a: np.array([ep.total_reward.get(a, 0.0) for ep in self.episodes])
            for a in self.agent_names
        }

    @property
    def raw_episode_rewards(self) -> dict[str, np.ndarray]:
        return {
            a: np.array([ep.raw_total_reward.get(a, 0.0) for ep in self.episodes])
            for a in self.agent_names
        }

    @property
    def pickup_counts(self) -> dict[str, np.ndarray]:
        return {
            a: np.array(
                [
                    ep.own_picks_total.get(a, 0) + ep.steal_picks_total.get(a, 0)
                    for ep in self.episodes
                ]
            )
            for a in self.agent_names
        }

    @property
    def welfare(self) -> np.ndarray:
        """Per-episode total mixed reward across agents."""
        return np.array(
            [sum(ep.total_reward.values()) for ep in self.episodes], dtype=float
        )

    @property
    def raw_welfare(self) -> np.ndarray:
        return np.array(
            [sum(ep.raw_total_reward.values()) for ep in self.episodes], dtype=float
        )

    @property
    def reward_inequality(self) -> np.ndarray:
        """Per-episode absolute reward gap between the two agents."""
        vals = []
        for ep in self.episodes:
            rewards = [ep.total_reward.get(a, 0.0) for a in ep.agent_names]
            vals.append(
                abs(rewards[0] - rewards[1]) if len(rewards) == 2 else np.std(rewards)
            )
        return np.array(vals, dtype=float)

    @property
    def mean_cooperation_rate(self) -> float:
        rates = _mean_agent_series(self.cooperation_rates)
        return _nanmean_or_zero(rates)

    @property
    def persistence_auc(self) -> float:
        """Mean area above the defection boundary, clipped at zero."""
        rates = _mean_agent_series(self.cooperation_rates)
        if rates.size == 0:
            return 0.0
        return float(np.maximum(np.nan_to_num(rates, nan=0.0) - 0.5, 0.0).mean())

    @property
    def time_to_defection(self) -> int | None:
        rates = _mean_agent_series(self.cooperation_rates)
        if rates.size == 0:
            return None
        rates = np.nan_to_num(rates, nan=0.0)
        window = 5
        for i in range(window, len(rates) + 1):
            if rates[max(0, i - window) : i].mean() < 0.5:
                return i - window
        return None

    def reward_component_series(self) -> dict[str, np.ndarray]:
        """Per-episode mean component value, averaged across agents."""
        out: dict[str, list[float]] = {k: [] for k in _REWARD_COMPONENTS}
        for ep in self.episodes:
            for comp in _REWARD_COMPONENTS:
                vals = [
                    ep.reward_components_total.get(a, {}).get(comp, 0.0)
                    for a in ep.agent_names
                ]
                out[comp].append(float(np.mean(vals)) if vals else 0.0)
        return {k: np.array(v, dtype=float) for k, v in out.items()}

    def summary_row(self, label: str) -> dict[str, float | int | str | None]:
        steal_rates = _mean_agent_series(self.steal_rates)
        pickups = _mean_agent_series(self.pickup_counts)
        return {
            "condition": label,
            "episodes": len(self.episodes),
            "mean_cooperation_rate": self.mean_cooperation_rate,
            "mean_steal_rate": _nanmean_or_zero(steal_rates),
            "mean_pickups_per_agent": _nanmean_or_zero(pickups),
            "mean_welfare": _nanmean_or_zero(self.welfare),
            "mean_raw_welfare": _nanmean_or_zero(self.raw_welfare),
            "mean_reward_inequality": _nanmean_or_zero(self.reward_inequality),
            "time_to_defection": self.time_to_defection,
            "persistence_auc": self.persistence_auc,
            "mean_prototype_entropy": _nanmean_or_zero(
                np.array(
                    [x for x in self.prototype_entropy if x is not None], dtype=float
                )
            ),
            "mean_prototype_max_prob": _nanmean_or_zero(
                np.array(
                    [x for x in self.prototype_max_prob if x is not None], dtype=float
                )
            ),
        }


def _nanmean_or_zero(arr: np.ndarray) -> float:
    if arr.size == 0 or np.all(np.isnan(arr)):
        return 0.0
    return float(np.nanmean(arr))


def _mean_agent_series(series: dict[str, np.ndarray]) -> np.ndarray:
    if not series:
        return np.array([], dtype=float)
    stacked = np.stack(list(series.values()), axis=0).astype(float)
    valid = ~np.isnan(stacked)
    counts = valid.sum(axis=0)
    sums = np.where(valid, stacked, 0.0).sum(axis=0)
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def save_summary_csv(data_dict: dict[str, AltruismAnalysisData], path: str) -> None:
    """Write a compact condition-level summary table."""
    import csv
    import os

    rows = [data.summary_row(label) for label, data in data_dict.items()]
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_prototypes(agent: Any) -> np.ndarray | None:
    """Extract the HSC prototype bank from a loaded MAGIC agent."""
    try:
        uid0 = list(agent.policies.keys())[0]
        policy = agent.policies[uid0]
        params = policy.state_dict.params["params"]
        return np.asarray(params["recurrent"]["hopfield_state"]["prototypes"])
    except (KeyError, AttributeError, IndexError):
        return None


def prototype_cosine_drift(protos_a: np.ndarray, protos_b: np.ndarray) -> float:
    """Mean paired cosine distance between two prototype banks."""
    if protos_a.shape != protos_b.shape:
        return float("nan")
    a_norm = protos_a / (np.linalg.norm(protos_a, axis=-1, keepdims=True) + 1e-8)
    b_norm = protos_b / (np.linalg.norm(protos_b, axis=-1, keepdims=True) + 1e-8)
    return float((1.0 - (a_norm * b_norm).sum(axis=-1)).mean())


def prototype_attention_probe(
    agent: Any, obs_batch: np.ndarray
) -> PrototypeProbe | None:
    """Probe HSC prototype attention using the actual policy input path.

    The previous analysis fed raw observations directly into the Hopfield cell.
    This mirrors the policy path instead: state preprocessor, obs_encoder, tanh,
    HSC input projection, shared retrieval LayerNorm, then prototype attention.
    """
    try:
        import jax
        import jax.numpy as jnp

        uid0 = list(agent.policies.keys())[0]
        policy = agent.policies[uid0]
        params = policy.state_dict.params["params"]
        hsc = params["recurrent"]["hopfield_state"]

        x = jnp.asarray(obs_batch, dtype=jnp.float32)
        preproc = getattr(agent, "_state_preprocessor", {}).get(uid0)
        if preproc is not None:
            try:
                x = preproc(x, train=False)
            except TypeError:
                x = preproc(x)

        obs_params = params["obs_encoder"]
        x = jnp.tanh(x @ obs_params["kernel"] + obs_params["bias"])

        H = int(hsc["prototypes"].shape[-1])
        if x.shape[-1] != H:
            proj = hsc["input_proj"]
            x = x @ proj["kernel"] + proj["bias"]

        prototypes = jnp.asarray(hsc["prototypes"])
        beta = jnp.asarray(hsc["beta"])
        ln_scale = jnp.asarray(hsc["retrieval_ln"]["scale"])
        ln_bias = jnp.asarray(hsc["retrieval_ln"]["bias"])

        def _layer_norm(v):
            mean = v.mean(axis=-1, keepdims=True)
            var = v.var(axis=-1, keepdims=True)
            return ln_scale * (v - mean) / jnp.sqrt(var + 1e-6) + ln_bias

        scores = (
            beta * (_layer_norm(x) @ _layer_norm(prototypes).T) / jnp.sqrt(float(H))
        )
        attn = jax.nn.softmax(scores, axis=-1)
        usage = np.asarray(attn.mean(axis=0))
        entropy_per_sample = -(attn * jnp.log(attn + 1e-9)).sum(axis=-1)
        entropy = float(entropy_per_sample.mean())
        max_prob = float(attn.max(axis=-1).mean())
        return PrototypeProbe(
            entropy=entropy,
            max_prob=max_prob,
            perplexity=float(np.exp(entropy)),
            usage=usage,
        )
    except Exception:
        return None


class CoinGameAltruismCollector:
    """Run evaluation episodes and collect altruism-persistence metrics."""

    def __init__(
        self,
        protos_reference: np.ndarray | None = None,
        max_cycles: int = 100,
        checkpoint_label: str | None = None,
    ):
        self.protos_reference = protos_reference
        self.max_cycles = max_cycles
        self.checkpoint_label = checkpoint_label

    def collect(
        self,
        env: Any,
        agent: Any,
        n_episodes: int = 50,
    ) -> AltruismAnalysisData:
        agent.set_running_mode("eval")
        data = AltruismAnalysisData(
            reference_prototypes=self.protos_reference,
            checkpoint_label=self.checkpoint_label,
        )

        for ep_idx in range(n_episodes):
            obs, _ = env.reset()
            possible_agents = list(obs.keys())
            ep = AltruismEpisodeData(episode_idx=ep_idx, agent_names=possible_agents)

            for a in possible_agents:
                ep.own_picks_per_step[a] = []
                ep.steal_picks_per_step[a] = []
                ep.victimizations_per_step[a] = []
                ep.rewards_per_step[a] = []
                ep.raw_rewards_per_step[a] = []
                ep.own_picks_total[a] = 0
                ep.steal_picks_total[a] = 0
                ep.victimizations_total[a] = 0
                ep.total_reward[a] = 0.0
                ep.raw_total_reward[a] = 0.0
                ep.reward_components_total[a] = {k: 0.0 for k in _REWARD_COMPONENTS}

            obs_probe: list[np.ndarray] = []
            t = 0
            while True:
                for a in possible_agents:
                    obs_probe.append(np.asarray(obs[a], dtype=np.float32).reshape(-1))

                actions, _, _ = agent.act(obs, timestep=t, timesteps=self.max_cycles)
                next_obs, rewards, terminated, truncated, infos = env.step(actions)

                step_events = infos[possible_agents[0]].get("pickup_events", [])
                ep.pickup_events.extend(dict(e, step=t) for e in step_events)

                for a in possible_agents:
                    info = infos.get(a, {})
                    own = int(info.get("own_picks", 0))
                    steal = int(info.get("steal_picks", 0))
                    victimized = int(info.get("victimizations", 0))
                    r = float(np.asarray(rewards[a]).ravel()[0])

                    comps = info.get("reward_components", {})
                    raw_r = float(comps.get("raw_reward", info.get("raw_reward", r)))

                    ep.own_picks_per_step[a].append(own)
                    ep.steal_picks_per_step[a].append(steal)
                    ep.victimizations_per_step[a].append(victimized)
                    ep.rewards_per_step[a].append(r)
                    ep.raw_rewards_per_step[a].append(raw_r)
                    ep.own_picks_total[a] += own
                    ep.steal_picks_total[a] += steal
                    ep.victimizations_total[a] += victimized
                    ep.total_reward[a] += r
                    ep.raw_total_reward[a] += raw_r
                    for comp in _REWARD_COMPONENTS:
                        ep.reward_components_total[a][comp] += float(
                            comps.get(comp, 0.0)
                        )

                t += 1
                obs = next_obs
                done = all(
                    truncated.get(a, False) or terminated.get(a, False)
                    for a in possible_agents
                )
                if done or t >= self.max_cycles:
                    break

            ep.episode_len = t
            data.episodes.append(ep)

            current_protos = load_prototypes(agent)
            if self.protos_reference is not None and current_protos is not None:
                data.prototype_cosine_drift.append(
                    prototype_cosine_drift(self.protos_reference, current_protos)
                )
            else:
                data.prototype_cosine_drift.append(None)

            if obs_probe:
                probe = prototype_attention_probe(agent, np.stack(obs_probe, axis=0))
            else:
                probe = None
            data.prototype_entropy.append(None if probe is None else probe.entropy)
            data.prototype_max_prob.append(None if probe is None else probe.max_prob)
            data.prototype_perplexity.append(
                None if probe is None else probe.perplexity
            )
            data.prototype_usage.append(None if probe is None else probe.usage)

            if (ep_idx + 1) % 10 == 0:
                print(
                    f"  Episode {ep_idx + 1}/{n_episodes}: "
                    f"coop_rate={data.mean_cooperation_rate:.3f}, "
                    f"welfare={_nanmean_or_zero(data.welfare):.2f}"
                )

        return data
