"""
mamhm_runner_integration.py
============================
Drop-in addition to BaseRunner for MAMHM analysis mode.

Dispatches to the MAMHM Hopfield memory analysis pipeline, then also
runs environment-specific performance analysis.

Usage
-----
In BaseRunner.analyze(), add:

    if agent_type == "mamhm":
        from utils.mamhm_runner_integration import run_mamhm_analysis
        run_mamhm_analysis(self, checkpoint_path, n_episodes,
                           output_dir=f"{output_dir}/mamhm")
        return
"""

from __future__ import annotations

from typing import Any


def run_mamhm_analysis(
    runner: Any,
    checkpoint_path: str | None = None,
    n_episodes: int = 30,
    output_dir: str = "eval_plots/mamhm",
) -> None:
    """Full MAMHM Hopfield memory analysis pipeline.

    Parameters
    ----------
    runner          : a BaseRunner (or subclass) instance holding a
                      MAMHMMAPPO agent and the environment.
    checkpoint_path : path to a saved checkpoint; if None uses the path
                      from runner._cfg["eval"]["checkpoint_path"].
    n_episodes      : number of evaluation episodes to collect.
    output_dir      : directory where figures (PDF + PNG) are written.
    """
    from utils.mam.hopfield_analysis import MAMHMCollector
    from utils.mam.hopfield_visualizer import save_all_mamhm_figures

    # ── Resolve checkpoint ───────────────────────────────────────────────────
    path = checkpoint_path or runner._cfg.get("eval", {}).get("checkpoint_path")
    if path:
        runner._load_checkpoint(path)

    # ── Pull configuration ───────────────────────────────────────────────────
    cfg = runner._cfg
    env_cfg = cfg.get("env", {})
    mamhm_cfg = cfg.get("mamhm", {})
    exp_name = cfg.get("experiment", {}).get("name", "mamhm_experiment")
    env_id = env_cfg.get("id", "coingame")

    num_agents = len(runner._env.possible_agents)
    d_model = int(mamhm_cfg.get("n_embd", 128))
    num_memories = int(mamhm_cfg.get("num_memories", 64))
    max_cycles = int(env_cfg.get("max_cycles", env_cfg.get("horizon", 500)))

    # ── MAMHM Hopfield memory analysis (all envs) ───────────────────────────
    collector = MAMHMCollector(
        num_agents=num_agents,
        d_model=d_model,
        num_memories=num_memories,
        max_cycles=max_cycles,
    )

    print(f"\n[MAMHM Analysis] Collecting {n_episodes} episodes ...")
    mamhm_data = collector.collect(
        env=runner._env,
        agent=runner._agent,
        n_episodes=n_episodes,
    )

    save_all_mamhm_figures(
        mamhm_data,
        output_dir=f"{output_dir}/hopfield",
        prefix=exp_name,
    )

    # ── Environment-specific performance analysis ────────────────────────────
    if env_id in ("warehouse", "warehouse-jax"):
        _run_warehouse_perf(runner, n_episodes, output_dir, exp_name, env_cfg)
    elif env_id in ("coingame", "coingame-partialobs"):
        _run_coingame_perf(runner, n_episodes, output_dir, exp_name, env_cfg)
    elif env_id == "continuous_coord":
        _run_cc_perf(runner, n_episodes, output_dir, exp_name, env_cfg)
    else:
        _run_warehouse_perf(runner, n_episodes, output_dir, exp_name, env_cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Per-environment performance analysis helpers
# ──────────────────────────────────────────────────────────────────────────────


def _run_warehouse_perf(runner, n_episodes, output_dir, exp_name, env_cfg):
    from utils.mappo.warehouse_analysis import WarehouseEvalCollector
    from utils.mappo.warehouse_visualizer import save_all_warehouse_figures

    collector = WarehouseEvalCollector(
        num_agents=env_cfg.get("num_agents", 4),
        battery_capacity=float(env_cfg.get("battery_capacity", 160)),
    )
    print(f"\n[MAM Warehouse Perf] Collecting {n_episodes} episodes ...")
    data = collector.collect(
        env=runner._env, agent=runner._agent, n_episodes=n_episodes
    )
    save_all_warehouse_figures(data, output_dir=f"{output_dir}/perf", prefix=exp_name)


def _run_coingame_perf(runner, n_episodes, output_dir, exp_name, env_cfg):
    from utils.mappo.coingame_analysis import EvalCollector
    from utils.mappo.coingame_visualizer import save_all_figures

    collector = EvalCollector(
        grid_size=env_cfg.get("grid_size", 7),
        pick_reward=env_cfg.get("pick_reward", 1.0),
        steal_penalty=env_cfg.get("steal_penalty", -2.0),
    )
    print(f"\n[MAM CoinGame Perf] Collecting {n_episodes} episodes ...")
    data = collector.collect(
        env=runner._env, agent=runner._agent, n_episodes=n_episodes
    )
    save_all_figures(data, output_dir=f"{output_dir}/perf", prefix=exp_name)


def _run_cc_perf(runner, n_episodes, output_dir, exp_name, env_cfg):
    from utils.mappo.continuous_coord_analysis import ContinuousCoordEvalCollector
    from utils.mappo.continuous_coord_visualizer import save_all_cc_figures

    collector = ContinuousCoordEvalCollector(
        num_agents=env_cfg.get("num_agents", 4),
        max_targets=env_cfg.get("max_targets", 3),
        max_cycles=env_cfg.get("max_cycles", 200),
        capture_reward=float(env_cfg.get("capture_reward", 10.0)),
        collision_radius=float(env_cfg.get("collision_radius", 0.03)),
        deadline_avg=float(env_cfg.get("deadline_avg", 55.0)),
        type_dim=int(env_cfg.get("type_dim", 0)),
    )
    print(f"\n[MAM ContinuousCoord Perf] Collecting {n_episodes} episodes ...")
    data = collector.collect(
        env=runner._env, agent=runner._agent, n_episodes=n_episodes
    )
    save_all_cc_figures(data, output_dir=f"{output_dir}/perf", prefix=exp_name)
