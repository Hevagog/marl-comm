"""Environment registry — single look-up table for all supported envs.

Adding a new environment:
  1. Implement it under ``environments/``.
  2. Add a factory entry to ``_ENV_FACTORIES`` below.
  3. Add its ``id`` string to the config validator in ``configs/loader.py``.

No other files need to change.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from skrl.envs.wrappers.jax import wrap_env


# ---------------------------------------------------------------------------
# Protocol — what every raw env must expose
# ---------------------------------------------------------------------------


@runtime_checkable
class ParallelEnv(Protocol):
    """Minimal duck-type interface for PettingZoo-style parallel envs."""

    @property
    def possible_agents(self) -> list[str]: ...

    @property
    def observation_spaces(self) -> Any: ...

    @property
    def action_spaces(self) -> Any: ...

    @property
    def state_spaces(self) -> Any: ...

    def reset(self, seed: int | None = None, **kwargs: Any) -> tuple[Any, Any]: ...

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, Any]: ...

    def state(self) -> Any: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Private registry
# ---------------------------------------------------------------------------

_ENV_FACTORIES: dict[str, Callable[[dict[str, Any]], ParallelEnv]] = {}


def _register(env_id: str):
    """Decorator that registers an env factory under *env_id*."""

    def decorator(fn: Callable[[dict[str, Any]], ParallelEnv]):
        _ENV_FACTORIES[env_id] = fn
        return fn

    return decorator


@_register("coingame")
def _make_coingame(env_cfg: dict[str, Any]) -> ParallelEnv:
    from environments.gridworld.coingame import make_coin_game_env
    from environments.gridworld.coingame.config import CoinGameConfig

    coin_cfg = CoinGameConfig(
        grid_size=env_cfg["grid_size"],
        max_cycles=env_cfg["max_cycles"],
        pick_reward=env_cfg["pick_reward"],
        steal_penalty=env_cfg["steal_penalty"],
    )
    return make_coin_game_env(config=coin_cfg)


@_register("blindspot")
def _make_blindspot(env_cfg: dict[str, Any]) -> ParallelEnv:
    from environments.gridworld.blindspot.blindspot import BlindSpotEnv, BlindSpotConfig

    bs_cfg = BlindSpotConfig(
        grid_size=env_cfg.get("grid_size", 9),
        max_cycles=env_cfg.get("max_cycles", 100),
        num_traps=env_cfg.get("num_traps", 5),
        trap_penalty=env_cfg.get("trap_penalty", -5.0),
        goal_reward=env_cfg.get("goal_reward", 10.0),
        vision_range=env_cfg.get("vision_range", 1),
        step_penalty=env_cfg.get("step_penalty", -0.01),
        use_distance_shaping=env_cfg.get("use_distance_shaping", True),
        distance_shaping_scale=env_cfg.get("distance_shaping_scale", 0.5),
        use_communication=env_cfg.get("use_communication", False),
    )
    return BlindSpotEnv(config=bs_cfg)


def make_env(cfg: dict[str, Any]) -> ParallelEnv:
    """Create a raw (un-wrapped) environment from the config ``env`` section.

    Parameters
    ----------
    cfg:
        The full run config dict.  The ``cfg["env"]["id"]`` key selects
        which environment to build.

    Returns
    -------
    ParallelEnv
        A PettingZoo-compatible parallel environment instance.

    Raises
    ------
    KeyError
        If ``cfg["env"]["id"]`` is not registered.
    """
    env_cfg = cfg["env"]
    env_id: str = env_cfg["id"]
    if env_id not in _ENV_FACTORIES:
        raise KeyError(
            f"Unknown env id '{env_id}'. Registered envs: {sorted(_ENV_FACTORIES)}"
        )
    return _ENV_FACTORIES[env_id](env_cfg)


def make_wrapped_env(cfg: dict[str, Any]):
    """Create and wrap the environment for use with skrl.

    Parameters
    ----------
    cfg:
        The full run config dict.

    Returns
    -------
    skrl.envs.wrappers.jax.MultiAgentEnvWrapper
        The wrapped environment, ready for ``SequentialTrainer``.
    """
    raw_env = make_env(cfg)
    return wrap_env(raw_env, wrapper="pettingzoo")
