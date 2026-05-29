"""
continuous_coord_comm_analysis.py
=================================
MAGIC pre-/post-GAT message collection for the Continuous Coordination
(blind-scenario) environment.

This is the continuous-coordination analogue of
``MAGICWarehouseCommCollector`` (``utils.magic.warehouse_analysis``).  It rolls
out evaluation episodes and captures the **shared** communication tensors that
``MAGICMAPPO.act()`` returns in its per-agent ``outputs`` dict:

    outputs[uid]["messages"]      – (groups, N, message_dim)  pre-GAT
    outputs[uid]["agg_messages"]  – (groups, N, message_dim)  post-GAT
    outputs[uid]["adj_matrices"]  – (R, groups, N, N)         soft adjacency

Each emitted message row (one agent at one step) is paired with two
blind-scenario labels:

  * **agent type** — taken from the env config ``agent_types`` (e.g. ``[0,0,1,1]``);
    robust regardless of obs layout.
  * **target visibility** — whether the agent currently observes ≥1 target.
    Targets are vision-gated *per agent* in the env (``_build_obs`` only writes a
    target's slot into agent *i*'s observation when it lies within
    ``target_vision_range``), so an agent's own active target slots are exactly
    "this agent sees a target now" — the relay-relevant signal of the scenario.

The companion ``utils.magic.continuous_coord_visualizer.save_all_magic_cc_figures``
turns the collected arrays into the S2-style pre-/post-GAT message PCA.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.magic.warehouse_analysis import _extract_comm_outputs
from utils.mappo.continuous_coord_analysis import _decode_agent_state, _decode_targets


# ─── Collected-data container ─────────────────────────────────────────────────


@dataclass
class MAGICCCData:
    """Row-aligned MAGIC message embeddings + blind-scenario labels.

    Every list holds one entry per (agent, step); index ``k`` of every list
    refers to the same emitted message, so ``pre_messages[k]`` is coloured by
    ``agent_type[k]`` / ``visible[k]``.
    """

    num_agents: int
    message_dim: int
    num_comm_rounds: int
    type_dim: int
    n_episodes: int = 0
    has_comm_data: bool = False
    has_agg: bool = False

    pre_messages: list[np.ndarray] = field(default_factory=list)   # each (D,)
    post_messages: list[np.ndarray] = field(default_factory=list)  # each (D,) or NaN
    agent_idx: list[int] = field(default_factory=list)
    agent_type: list[int] = field(default_factory=list)
    visible: list[int] = field(default_factory=list)        # 1 if sees ≥1 target
    n_visible: list[int] = field(default_factory=list)
    nearest_dist: list[float] = field(default_factory=list)  # NaN if none visible
    step_frac: list[float] = field(default_factory=list)

    # ── array accessors ────────────────────────────────────────────────────────
    def pre_array(self) -> np.ndarray:
        return (
            np.vstack(self.pre_messages)
            if self.pre_messages
            else np.empty((0, self.message_dim))
        )

    def post_array(self) -> np.ndarray:
        return (
            np.vstack(self.post_messages)
            if self.post_messages
            else np.empty((0, self.message_dim))
        )

    def labels(self) -> dict[str, np.ndarray]:
        return {
            "agent_idx": np.asarray(self.agent_idx, dtype=int),
            "agent_type": np.asarray(self.agent_type, dtype=int),
            "visible": np.asarray(self.visible, dtype=int),
            "n_visible": np.asarray(self.n_visible, dtype=int),
            "nearest_dist": np.asarray(self.nearest_dist, dtype=float),
            "step_frac": np.asarray(self.step_frac, dtype=float),
        }


# ─── Collector ────────────────────────────────────────────────────────────────


class MAGICContinuousCoordCommCollector:
    """Collect MAGIC pre-/post-GAT messages over continuous-coord episodes."""

    def __init__(
        self,
        num_agents: int = 4,
        num_comm_rounds: int = 2,
        message_dim: int = 32,
        max_cycles: int = 200,
        max_targets: int = 3,
        type_dim: int = 0,
        agent_types: list[int] | None = None,
        deadline_avg: float = 70.0,
    ) -> None:
        self.num_agents = num_agents
        self.num_comm_rounds = num_comm_rounds
        self.message_dim = message_dim
        self.max_cycles = max_cycles
        self.max_targets = max_targets
        self.type_dim = type_dim
        self.deadline_avg = deadline_avg
        self.agent_types = (
            list(agent_types) if agent_types else [0] * num_agents
        )

    def _agent_order(self, agent: Any, obs_keys: list[str]) -> list[str]:
        """Return the canonical agent order that message axis-1 follows.

        ``MAGICMAPPO`` stacks observations in ``possible_agents`` order and the
        message tensors share that ordering, so labels must be built in the same
        order.  The continuous-coord env builds its obs dict in that order too,
        but we assert it to catch any divergence before it silently mis-colours
        the PCA.
        """
        order = list(getattr(agent, "possible_agents", []) or [])
        if not order:
            return obs_keys
        if set(order) != set(obs_keys):
            warnings.warn(
                "[MAGIC-CC] agent.possible_agents does not match obs keys; "
                "falling back to obs-key order.",
                stacklevel=2,
            )
            return obs_keys
        if order != obs_keys:
            warnings.warn(
                "[MAGIC-CC] obs-key order differs from possible_agents; using "
                "possible_agents order to match the message tensor layout.",
                stacklevel=2,
            )
        return order

    def collect(
        self,
        env: Any,
        agent: Any,
        n_episodes: int = 20,
        max_steps_per_episode: int | None = None,
    ) -> MAGICCCData:
        agent.set_running_mode("eval")
        max_steps = max_steps_per_episode or self.max_cycles

        data = MAGICCCData(
            num_agents=self.num_agents,
            message_dim=self.message_dim,
            num_comm_rounds=self.num_comm_rounds,
            type_dim=self.type_dim,
        )
        comm_found = False
        agg_found = False

        for ep_idx in range(n_episodes):
            obs, _ = env.reset()
            order = self._agent_order(agent, list(obs.keys()))

            for t in range(max_steps):
                actions, _, outputs_per_agent = agent.act(
                    obs, timestep=t, timesteps=max_steps
                )
                # Shared comm tensors live (unsliced) under every agent's slot.
                merged = (
                    outputs_per_agent.get(order[0], {})
                    if isinstance(outputs_per_agent, dict)
                    else {}
                )
                _adj, _hard, msgs, agg, _logits = _extract_comm_outputs(
                    merged, self.num_agents, self.num_comm_rounds, self.message_dim
                )

                if msgs is not None:
                    comm_found = True
                    n = int(msgs.shape[0])
                    if agg is not None:
                        agg_found = True
                    for i in range(min(n, len(order))):
                        a = order[i]
                        a_obs = np.asarray(obs[a], dtype=np.float32).ravel()
                        a_state = _decode_agent_state(a_obs)
                        tgts = _decode_targets(
                            a_obs,
                            self.num_agents,
                            self.max_targets,
                            self.type_dim,
                            self.deadline_avg,
                        )
                        vis = [tt for tt in tgts if tt.active]
                        if vis:
                            dists = [
                                float(
                                    np.hypot(tt.abs_x - a_state.px, tt.abs_y - a_state.py)
                                )
                                for tt in vis
                            ]
                            nearest = float(min(dists))
                            n_vis = len(vis)
                            v = 1
                        else:
                            nearest = float("nan")
                            n_vis = 0
                            v = 0

                        data.pre_messages.append(np.asarray(msgs[i], dtype=np.float32))
                        data.post_messages.append(
                            np.asarray(agg[i], dtype=np.float32)
                            if agg is not None
                            else np.full(self.message_dim, np.nan, dtype=np.float32)
                        )
                        data.agent_idx.append(i)
                        data.agent_type.append(
                            int(self.agent_types[i])
                            if i < len(self.agent_types)
                            else 0
                        )
                        data.visible.append(v)
                        data.n_visible.append(n_vis)
                        data.nearest_dist.append(nearest)
                        data.step_frac.append(t / max(max_steps, 1))

                next_obs, _rewards, terminated, truncated, _ = env.step(actions)
                obs = next_obs
                done = any(
                    bool(np.asarray(x).ravel()[0])
                    for x in list(terminated.values()) + list(truncated.values())
                )
                if done:
                    break

            data.n_episodes += 1
            print(
                f"  [MAGIC-CC] episode {ep_idx + 1:3d}/{n_episodes}  |  "
                f"messages so far: {len(data.pre_messages)}"
            )

        data.has_comm_data = comm_found
        data.has_agg = agg_found
        if not comm_found:
            warnings.warn(
                "[MAGIC-CC] No 'messages' tensor found in act() outputs — the "
                "message PCA will be skipped. Is this a MAGIC agent?",
                stacklevel=2,
            )
        elif not agg_found:
            warnings.warn(
                "[MAGIC-CC] 'agg_messages' (post-GAT) not found — only the "
                "pre-GAT PCA will be produced.",
                stacklevel=2,
            )
        return data
