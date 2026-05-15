"""MAM analysis dispatcher for BaseRunner.analyze().

Handles agent_type in {"mam", "mam_enc_only", "mamhm",
                       "mam_hopfield_pooling", "mam_et_encoder"}.

For non-HM variants (base BiMamba MAM) the analysis is warehouse performance.
For mamhm, an additional Hopfield memory analysis is run.

Usage in BaseRunner.analyze():

    agent_type = self._cfg.get("experiment", {}).get("agent_type", "mappo")
    if agent_type in ("mam", "mam_enc_only", "mamhm", ...):
        from utils.mam.runner import run_mam_analysis
        run_mam_analysis(self, checkpoint_path, n_episodes,
                         output_dir=f"{output_dir}/mam")
        return
"""

from __future__ import annotations

from typing import Any

_MAM_HM_TYPES = {"mamhm"}
_MAM_BASE_TYPES = {
    "mam",
    "mam_enc_only",
    "mam_hopfield_pooling",
    "mam_hopfield_layer",
    "mam_et_encoder",
}


def run_mam_analysis(
    runner: Any,
    checkpoint_path: str | None = None,
    n_episodes: int = 30,
    output_dir: str = "eval_plots/mam",
) -> None:
    """Run MAM performance (+ optional Hopfield memory) analysis.

    Parameters
    ----------
    runner          : BaseRunner holding a MAM* agent + environment.
    checkpoint_path : checkpoint to load; None → use runner._cfg["eval"]["checkpoint_path"].
    n_episodes      : evaluation episodes.
    output_dir      : output directory for figures.
    """
    path = checkpoint_path or runner._cfg.get("eval", {}).get("checkpoint_path")
    if path:
        runner._load_checkpoint(path)

    cfg = runner._cfg
    env_cfg = cfg.get("env", {})
    agent_type = cfg.get("experiment", {}).get("agent_type", "mam")
    exp_name = cfg.get("experiment", {}).get("name", "mam_experiment")
    env_id = env_cfg.get("id", "warehouse")

    # ── Performance analysis (all MAM variants) ──────────────────────────────
    _run_perf(runner, n_episodes, output_dir, exp_name, env_id, env_cfg)

    # ── Hopfield memory analysis (mamhm only) ────────────────────────────────
    if agent_type in _MAM_HM_TYPES:
        from utils.mam.hopfield_runner import run_mamhm_analysis

        run_mamhm_analysis(
            runner,
            checkpoint_path=None,  # already loaded above
            n_episodes=n_episodes,
            output_dir=f"{output_dir}/hopfield",
        )


def _run_perf(runner, n_episodes, output_dir, exp_name, env_id, env_cfg):
    """Dispatch to the appropriate environment performance collector."""
    if env_id in ("warehouse", "warehouse-jax"):
        from utils.mappo.warehouse_analysis import WarehouseEvalCollector
        from utils.mappo.warehouse_visualizer import save_all_warehouse_figures

        collector = WarehouseEvalCollector(
            num_agents=env_cfg.get("num_agents", 4),
            battery_capacity=float(env_cfg.get("battery_capacity", 160)),
        )
        print(f"\n[MAM Warehouse Perf] Collecting {n_episodes} episodes …")
        data = collector.collect(
            env=runner._env,
            agent=runner._agent,
            n_episodes=n_episodes,
        )
        save_all_warehouse_figures(data, output_dir=output_dir, prefix=exp_name)

    elif env_id in ("coingame", "coingame-partialobs"):
        from utils.mappo.coingame_analysis import EvalCollector
        from utils.mappo.coingame_visualizer import save_all_figures

        collector = EvalCollector(
            grid_size=env_cfg.get("grid_size", 7),
            pick_reward=env_cfg.get("pick_reward", 1.0),
            steal_penalty=env_cfg.get("steal_penalty", -2.0),
        )
        print(f"\n[MAM CoinGame Perf] Collecting {n_episodes} episodes …")
        data = collector.collect(
            env=runner._env,
            agent=runner._agent,
            n_episodes=n_episodes,
        )
        save_all_figures(data, output_dir=output_dir, prefix=exp_name)

    elif env_id == "continuous_coord":
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
        print(f"\n[MAM ContinuousCoord Perf] Collecting {n_episodes} episodes …")
        data = collector.collect(
            env=runner._env,
            agent=runner._agent,
            n_episodes=n_episodes,
        )
        save_all_cc_figures(data, output_dir=output_dir, prefix=exp_name)
