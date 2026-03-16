from __future__ import annotations

import argparse
import sys
from pathlib import Path

from skrl.envs.wrappers.jax import wrap_env


from utils import load_config


def _get_runner(agent_type: str):
    if agent_type == "mappo":
        from agents.train import MAPPORunner

        return MAPPORunner
    if agent_type == "magic":
        from agents.magic.train import MAGICRunner

        return MAGICRunner
    raise ValueError(
        f"Unknown agent_type '{agent_type}'. Registered types: {['mappo', 'magic']}"
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

    args = parser.parse_args()

    config_path = Path(args.config)
    sys.path.insert(0, str(config_path.parent))
    cfg = load_config(config_path)

    if args.task in ["eval", "record"]:
        cfg["experiment"]["wandb"] = False

    mode = "human" if args.task == "eval" else "rgb_array"

    env_cfg = cfg.get("env", {})
    env_id = env_cfg.get("id", "coingame")

    if env_id == "coingame":
        from environments.gridworld.coingame.config import CoinGameConfig
        from environments import make_coin_game_env

        coin_config = CoinGameConfig(
            grid_size=env_cfg.get("grid_size", 7),
            max_cycles=env_cfg.get("max_cycles", 50),
            pick_reward=env_cfg.get("pick_reward", 1.0),
            steal_penalty=env_cfg.get("steal_penalty", -2.0),
        )
        raw_env = make_coin_game_env(config=coin_config, render_mode=mode)
    elif env_id == "blindspot":
        from environments.gridworld.blindspot.config import BlindSpotConfig
        from environments.gridworld.blindspot.blindspot import make_blind_spot_env

        bs_config = BlindSpotConfig(
            grid_size=env_cfg.get("grid_size", 9),
            max_cycles=env_cfg.get("max_cycles", 100),
            num_traps=env_cfg.get("num_traps", 5),
            use_communication=env_cfg.get("use_communication", False),
        )
        raw_env = make_blind_spot_env(config=bs_config, render_mode=mode)
    elif env_id == "simple_adversary":
        from pettingzoo.mpe import simple_adversary_v3

        raw_env = simple_adversary_v3.parallel_env(render_mode=mode)

    else:
        raise ValueError(
            f"Unknown env.id '{env_id}'. Choices: ['coingame', 'blindspot', 'simple_adversary']"
        )

    env = wrap_env(raw_env, wrapper="pettingzoo")

    agent_type: str = cfg["experiment"].get("agent_type", "mappo")
    RunnerClass = _get_runner(agent_type)
    runner = RunnerClass(env=env, cfg=cfg)

    task: str = args.task
    if task == "train":
        runner.train()
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
