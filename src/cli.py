from __future__ import annotations

import argparse
import sys
from pathlib import Path


from environments.registry import make_wrapped_env
from utils import load_config


def _get_runner(agent_type: str):
    if agent_type == "mappo":
        from agents.train import MAPPORunner

        return MAPPORunner
    raise ValueError(
        f"Unknown agent_type '{agent_type}'. Registered types: {['mappo']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["train", "eval", "record"],
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

    env = make_wrapped_env(cfg)

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
    else:
        parser.error(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
