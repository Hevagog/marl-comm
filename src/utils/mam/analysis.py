"""Performance analysis for the base MAM agent (BiMamba / CrossMamba paper variant).

The base MAM policy (Daniel et al. 2024) runs a BiMamba encoder + CrossMamba
decoder but does not expose internal hidden states through the outputs dict.
Analysis is therefore performance-focused:

  MAMPerfCollector — runs warehouse (or coingame) rollouts and records per-step
      per-agent rewards, actions, and deliveries.  Results feed into:
        - aggregate_metrics() from shared.stats for IQM + bootstrap CIs
        - warehouse_eval_visualizer figures for visual comparison

Usage
-----
    from utils.mam.analysis import MAMPerfCollector
    from utils.shared.stats import aggregate_metrics, print_aggregate_table
    from utils.mappo.warehouse_visualizer import save_all_warehouse_figures

    collector = MAMPerfCollector(num_agents=4, max_cycles=500)
    data = collector.collect(env, agent, n_episodes=30)
    # data is a WarehouseEvalData — use the standard warehouse visualizer
    save_all_warehouse_figures(data, out_dir="eval_plots/mam", prefix="mam_warehouse")

    # rliable statistics
    episode_returns = [ep.mean_episode_reward for ep in data.episodes]
    table = aggregate_metrics({"MAM": episode_returns})
    print_aggregate_table(table)
"""

from __future__ import annotations

from typing import Any

import numpy as np


def collect_episode_returns(
    env: Any,
    agent: Any,
    n_episodes: int = 30,
    max_cycles: int = 500,
) -> list[float]:
    """Lightweight collector: returns list of mean episode returns.

    For full warehouse metrics use WarehouseEvalCollector from mappo/.
    This helper is useful for quick IQM/CI computation without the overhead
    of a full WarehouseEvalData object.
    """
    returns: list[float] = []
    for ep_idx in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        step = 0
        done = False
        while not done and step < max_cycles:
            actions_out, _, _ = agent.act(obs, role="policy")
            try:
                import jax

                actions_np = np.asarray(jax.device_get(actions_out))
            except Exception:
                actions_np = np.asarray(actions_out)

            n_agents = len(env.possible_agents)
            if actions_np.ndim == 0:
                actions_np = actions_np.reshape(1)
            actions_np = actions_np.flatten()[:n_agents]
            actions_dict = {
                agent_id: int(actions_np[i])
                for i, agent_id in enumerate(env.possible_agents)
            }

            obs, rewards, terminations, truncations, _ = env.step(actions_dict)
            ep_reward += float(np.mean(list(rewards.values())))
            done = all(terminations.values()) or all(truncations.values())
            step += 1

        returns.append(ep_reward)

    return returns
