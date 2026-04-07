from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Literal

from skrl.envs.wrappers.jax import wrap_env


from utils import load_config


def _get_runner(agent_type: str):
    if agent_type == "mappo":
        from agents.train import MAPPORunner

        return MAPPORunner
    if agent_type == "magic":
        from agents.magic.train import MAGICRunner

        return MAGICRunner
    if agent_type == "commformer":
        from agents.commformer.train import CommFormerRunner

        return CommFormerRunner
    if agent_type == "mam":
        from agents.mam.train import MAMRunner

        return MAMRunner
    if agent_type == "etmat":
        from agents.etmat.train import ETMATRunner

        return ETMATRunner
    if agent_type == "mamhm":
        from agents import MAMHMRunner

        return MAMHMRunner
    raise ValueError(
        f"Unknown agent_type '{agent_type}'. Registered types: {['mappo', 'magic', 'commformer', 'mam', 'etmat']}"
    )


def _create_env_factory(
    env_id: str, env_cfg: dict, render_mode: Literal["human", "rgb_array"] | None
):
    """Create a factory function for the specified environment.

    This function returns a callable that creates a single environment instance.
    Used by the vectorized environment wrapper.
    """
    match env_id:
        case "coingame":
            from environments import make_coin_game_env, CoinGameConfig

            config = CoinGameConfig(
                grid_size=env_cfg.get("grid_size", 7),
                max_cycles=env_cfg.get("max_cycles", 50),
                pick_reward=env_cfg.get("pick_reward", 1.0),
                steal_penalty=env_cfg.get("steal_penalty", -2.0),
            )
            return partial(make_coin_game_env, config=config, render_mode=render_mode)

        case "blindspot":
            from environments import make_blind_spot_env, BlindSpotConfig

            config = BlindSpotConfig(
                grid_size=env_cfg.get("grid_size", 9),
                max_cycles=env_cfg.get("max_cycles", 100),
                num_traps=env_cfg.get("num_traps", 5),
                use_communication=env_cfg.get("use_communication", False),
                random_goal=env_cfg.get("random_goal", True),
                min_goal_start_distance=env_cfg.get("min_goal_start_distance", 4),
            )
            return partial(make_blind_spot_env, config=config, render_mode=render_mode)

        case "simple_adversary":
            from pettingzoo.mpe import simple_adversary_v3

            return partial(
                simple_adversary_v3.parallel_env,
                N=env_cfg.get("N", env_cfg.get("num_good_agents", 2)),
                max_cycles=env_cfg.get("max_cycles", 25),
                continuous_actions=env_cfg.get("continuous_actions", False),
                dynamic_rescaling=env_cfg.get("dynamic_rescaling", False),
                render_mode=render_mode,
            )

        case "overcooked":
            from environments import OvercookedConfig, make_overcooked_env

            config = OvercookedConfig(
                layout_name=env_cfg.get("layout_name", "cramped_room"),
                horizon=env_cfg.get("horizon", env_cfg.get("max_cycles", 200)),
                use_dense_obs=env_cfg.get("use_dense_obs", False),
                reward_shaping=env_cfg.get("reward_shaping", True),
                reward_shaping_factor=env_cfg.get("reward_shaping_factor", 1.0),
            )
            return partial(make_overcooked_env, config=config, render_mode=render_mode)

        case "intersection":
            from environments import IntersectionConfig, make_intersection_env

            config = IntersectionConfig(
                num_agents=env_cfg.get("num_agents", 4),
                duration=env_cfg.get("duration", 13),
                vehicles_count=env_cfg.get("vehicles_count", 10),
                initial_vehicle_count=env_cfg.get("initial_vehicle_count", 10),
            )
            return partial(
                make_intersection_env, config=config, render_mode=render_mode
            )

        case "warehouse":
            from environments import make_warehouse_env, WarehouseConfig

            config_kwargs = {
                key: value
                for key, value in env_cfg.items()
                if key in WarehouseConfig.__dataclass_fields__
            }
            config = WarehouseConfig(**config_kwargs)
            return partial(make_warehouse_env, config=config, render_mode=render_mode)

        case _:
            raise ValueError(
                f"Unknown env.id '{env_id}'. Choices: ['coingame', 'blindspot', 'simple_adversary', 'overcooked', 'intersection', 'warehouse']"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mappo_default.py",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["train", "eval", "record", "analyze"],
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file.  Overrides cfg[eval/record][checkpoint_path].",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Resume training from CHECKPOINT.  Loads weights and infers the "
            "starting timestep from the filename (e.g. agent_1000000.pickle → "
            "1000000).  Only valid with --task train."
        ),
    )

    args = parser.parse_args()

    if args.resume is not None and args.task != "train":
        parser.error("--resume is only valid with --task train")

    config_path = Path(args.config)
    sys.path.insert(0, str(config_path.parent))
    cfg = load_config(config_path)

    if args.task in ["eval", "record", "analyze"]:
        num_envs = 1
        cfg["experiment"]["wandb"] = False
    else:
        num_envs = cfg.get("env", {}).get("num_envs", 1)

    if "env" not in cfg:
        cfg["env"] = {}
    cfg["env"]["num_envs"] = num_envs

    mode: Literal["human", "rgb_array"] | None = (
        "human" if args.task == "eval" else "rgb_array"
    )
    if args.task == "train" and num_envs > 1:
        mode = None

    env_cfg = cfg.get("env", {})
    env_id = env_cfg.get("id", "coingame")

    env_factory = _create_env_factory(env_id, env_cfg, mode)
    if num_envs > 1:
        from environments import make_vectorized_env

        raw_env = make_vectorized_env(env_fn=env_factory, num_envs=num_envs)
        print(f"Created vectorized environment with {num_envs} parallel instances")
    else:
        raw_env = env_factory()

    env = wrap_env(raw_env, wrapper="pettingzoo")

    agent_type: str = cfg["experiment"].get("agent_type", "mappo")
    RunnerClass = _get_runner(agent_type)
    runner = RunnerClass(env=env, cfg=cfg)

    task: str = args.task
    if task == "train":
        runner.train(resume_from=args.resume)
    elif task == "eval":
        runner.eval(checkpoint_path=args.checkpoint)
    elif task == "record":
        runner.record(checkpoint_path=args.checkpoint)
    elif task == "analyze":
        runner.analyze(checkpoint_path=args.checkpoint)
    else:
        parser.error(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
