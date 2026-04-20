"""
commformer_runner_integration.py
==================================
Drop-in addition to BaseRunner for CommFormer analysis mode.

Dispatches to the correct CommFormer analysis pipeline based on env_id.
CommFormer's graph is environment-agnostic (learned α parameter), so the
same CommFormerCommCollector is used across all environments, but
environment-specific performance analyses are also run in parallel.

Usage
-----
In BaseRunner.analyze(), add:

    if agent_type == "commformer":
        from utils.commformer_runner_integration import run_commformer_analysis
        run_commformer_analysis(self, checkpoint_path, n_episodes,
                                output_dir=f"{output_dir}/commformer")
        return
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def run_commformer_analysis(
    runner: Any,
    checkpoint_path: str | None = None,
    n_episodes: int = 30,
    output_dir: str = "eval_plots/commformer",
) -> None:
    """Full CommFormer communication analysis pipeline.

    Parameters
    ----------
    runner          : a BaseRunner (or subclass) instance holding a
                      CommFormerMAPPO agent and the environment.
    checkpoint_path : path to a saved checkpoint; if None uses the path
                      from runner._cfg["eval"]["checkpoint_path"].
    n_episodes      : number of evaluation episodes to collect.
    output_dir      : directory where figures (PDF + PNG) are written.
    """
    from utils.commformer_comm_analysis import CommFormerCommCollector
    from utils.commformer_comm_visualizer import save_all_commformer_figures

    # ── Resolve checkpoint ───────────────────────────────────────────────────
    path = checkpoint_path or runner._cfg.get("eval", {}).get("checkpoint_path")
    if path:
        runner._load_checkpoint(path)

    # ── Pull configuration ───────────────────────────────────────────────────
    cfg = runner._cfg
    env_cfg = cfg.get("env", {})
    cf_cfg = cfg.get("commformer", {})
    exp_name = cfg.get("experiment", {}).get("name", "commformer_experiment")
    env_id = env_cfg.get("id", "coingame")

    num_agents = len(runner._env.possible_agents)
    hidden_dim = int(cf_cfg.get("hidden_dim", 64))
    sparsity = float(cf_cfg.get("sparsity", 0.4))
    max_cycles = int(env_cfg.get("max_cycles", env_cfg.get("horizon", 500)))

    # ── CommFormer graph analysis (all envs) ─────────────────────────────────
    cf_collector = CommFormerCommCollector(
        num_agents=num_agents,
        hidden_dim=hidden_dim,
        sparsity=sparsity,
        max_cycles=max_cycles,
    )

    print(f"\n[CommFormer Analysis] Collecting {n_episodes} episodes …")
    cf_data = cf_collector.collect(
        env=runner._env,
        agent=runner._agent,
        n_episodes=n_episodes,
    )

    save_all_commformer_figures(
        cf_data,
        output_dir=f"{output_dir}/graph",
        prefix=exp_name,
    )

    # ── Environment-specific performance analysis ─────────────────────────────
    if env_id == "blindspot":
        _run_blindspot_perf(runner, n_episodes, output_dir, exp_name, env_cfg)

    elif env_id == "warehouse":
        _run_warehouse_perf(runner, n_episodes, output_dir, exp_name, env_cfg)

    elif env_id == "simple_adversary":
        _run_simple_adversary_perf(runner, n_episodes, output_dir, exp_name)

    elif env_id == "overcooked":
        _run_overcooked_perf(runner, n_episodes, output_dir, exp_name, env_cfg)

    elif env_id == "intersection":
        _run_highway_perf(runner, n_episodes, output_dir, exp_name, env_cfg)

    elif env_id == "flatland":
        _run_flatland_perf(runner, n_episodes, output_dir, exp_name, env_cfg)

    else:
        _run_generic_perf(runner, n_episodes, output_dir, exp_name, env_cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Per-environment performance analysis helpers
# ──────────────────────────────────────────────────────────────────────────────


def _run_blindspot_perf(runner, n_episodes, output_dir, exp_name, env_cfg):
    from utils.blindspot_eval_analysis import BlindSpotEvalCollector
    from utils.blindspot_eval_visualizer import save_all_blindspot_figures

    collector = BlindSpotEvalCollector(
        grid_size=env_cfg.get("grid_size", 9),
        num_traps=env_cfg.get("num_traps", 5),
        max_cycles=env_cfg.get("max_cycles", 100),
        use_communication=env_cfg.get("use_communication", False),
        num_message_tokens=env_cfg.get("num_message_tokens", 4),
    )
    print(f"\n[CommFormer BlindSpot Perf] Collecting {n_episodes} episodes …")
    data = collector.collect(env=runner._env, agent=runner._agent, n_episodes=n_episodes)
    save_all_blindspot_figures(data, output_dir=f"{output_dir}/perf", prefix=exp_name)


def _run_warehouse_perf(runner, n_episodes, output_dir, exp_name, env_cfg):
    from utils.warehouse_eval_analysis import WarehouseEvalCollector
    from utils.warehouse_eval_visualizer import save_all_warehouse_figures

    collector = WarehouseEvalCollector(
        grid_height=env_cfg.get("grid_height", 12),
        grid_width=env_cfg.get("grid_width", 16),
        num_agents=env_cfg.get("num_agents", 4),
        max_cycles=env_cfg.get("max_cycles", 500),
        battery_capacity=float(env_cfg.get("battery_capacity", 160)),
    )
    print(f"\n[CommFormer Warehouse Perf] Collecting {n_episodes} episodes …")
    data = collector.collect(env=runner._env, agent=runner._agent, n_episodes=n_episodes)
    save_all_warehouse_figures(data, output_dir=f"{output_dir}/perf", prefix=exp_name)


def _run_simple_adversary_perf(runner, n_episodes, output_dir, exp_name):
    from utils.simple_adversary_eval_analysis import SimpleAdversaryEvalCollector
    from utils.simple_adversary_eval_visualizer import save_all_sa_figures

    collector = SimpleAdversaryEvalCollector()
    print(f"\n[CommFormer SA Perf] Collecting {n_episodes} episodes …")
    data = collector.collect(env=runner._env, agent=runner._agent, n_episodes=n_episodes)
    save_all_sa_figures(data, output_dir=f"{output_dir}/perf", prefix=exp_name)


def _run_overcooked_perf(runner, n_episodes, output_dir, exp_name, env_cfg):
    from utils.overcooked_eval_analysis import OvercookedEvalCollector
    from utils.overcooked_eval_visualizer import save_all_overcooked_figures

    collector = OvercookedEvalCollector()
    print(f"\n[CommFormer Overcooked Perf] Collecting {n_episodes} episodes …")
    data = collector.collect(
        env=runner._env, agent=runner._agent, n_episodes=n_episodes,
        max_steps_per_episode=env_cfg.get("horizon", env_cfg.get("max_cycles", 200)),
    )
    save_all_overcooked_figures(data, output_dir=f"{output_dir}/perf", prefix=exp_name)


def _run_highway_perf(runner, n_episodes, output_dir, exp_name, env_cfg):
    from utils.highway_eval_analysis import HighwayIntersectionEvalCollector
    from utils.highway_eval_visualizer import save_all_highway_figures

    collector = HighwayIntersectionEvalCollector(
        num_agents=env_cfg.get("num_agents", 4),
        duration=env_cfg.get("duration", 13),
        collision_reward=env_cfg.get("collision_reward", -5.0),
        arrived_reward=env_cfg.get("arrived_reward", 1.0),
    )
    print(f"\n[CommFormer Highway Perf] Collecting {n_episodes} episodes …")
    data = collector.collect(env=runner._env, agent=runner._agent, n_episodes=n_episodes)
    save_all_highway_figures(data, output_dir=f"{output_dir}/perf", prefix=exp_name)


def _run_flatland_perf(runner, n_episodes, output_dir, exp_name, env_cfg):
    from utils.flatland_eval_analysis import FlatlandEvalCollector
    from utils.flatland_eval_visualizer import save_all_flatland_figures

    collector = FlatlandEvalCollector(
        num_agents=env_cfg.get("num_agents", len(runner._env.possible_agents)),
        max_steps_per_episode=env_cfg.get("max_episode_steps"),
    )
    print(f"\n[CommFormer Flatland Perf] Collecting {n_episodes} episodes …")
    data = collector.collect(env=runner._env, agent=runner._agent, n_episodes=n_episodes)
    save_all_flatland_figures(data, output_dir=f"{output_dir}/perf", prefix=exp_name)


def _run_generic_perf(runner, n_episodes, output_dir, exp_name, env_cfg):
    from utils.eval_analysis import EvalCollector
    from utils.eval_visualizer import save_all_figures

    collector = EvalCollector(
        grid_size=env_cfg.get("grid_size", 7),
        pick_reward=env_cfg.get("pick_reward", 1.0),
        steal_penalty=env_cfg.get("steal_penalty", -2.0),
    )
    print(f"\n[CommFormer Perf] Collecting {n_episodes} episodes …")
    data = collector.collect(env=runner._env, agent=runner._agent, n_episodes=n_episodes)
    save_all_figures(data, output_dir=f"{output_dir}/perf", prefix=exp_name)
