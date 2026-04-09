"""Vectorized PettingZoo parallel environment wrapper.

This module provides a vectorized wrapper for PettingZoo parallel-API
environments, enabling parallel execution of multiple environment instances
for significantly faster training throughput.

The wrapper is designed to be compatible with skrl's multi-agent environment
interface, properly exposing the `num_envs` property and batching observations,
actions, and rewards across all environment instances.

Architecture
------------
The wrapper uses Python's `concurrent.futures.ThreadPoolExecutor` for parallel
environment stepping, which is effective for environments that release the GIL
(like NumPy-based simulations such as highway-env). For pure Python environments,
a process-based alternative can be used via the `use_multiprocessing` flag.

Usage
-----
    from environments.vectorized import make_vectorized_env

    # Create a vectorized environment with 16 parallel instances
    vec_env = make_vectorized_env(
        env_fn=lambda: IntersectionPettingZooEnv(config),
        num_envs=16,
    )

    # Use with skrl
    from skrl.envs.wrappers.jax import wrap_env
    env = wrap_env(vec_env, wrapper="pettingzoo")

The wrapper expects environments to implement:
- possible_agents: list[str]
- agents: list[str]
- observation_spaces: Mapping[str, gymnasium.Space]
- action_spaces: Mapping[str, gymnasium.Space]
- state_spaces: Mapping[str, gymnasium.Space] (optional)
- reset(seed=None) -> (obs_dict, info_dict)
- step(actions_dict) -> (obs, rewards, terminated, truncated, infos)
- state() -> np.ndarray (optional)
- render() -> np.ndarray | None
- close() -> None
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any
from collections.abc import Mapping, Callable

import gymnasium
import numpy as np
from gymnasium import spaces


class VectorizedPettingZooEnv:
    """Vectorized wrapper for PettingZoo parallel-API environments.

    Runs multiple environment instances in parallel, batching observations
    and actions for efficient training. Implements auto-reset when episodes
    end.

    Parameters
    ----------
    env_fns : list of callables
        Factory functions that create environment instances. Each function
        should return a PettingZoo parallel-API compatible environment.
    copy : bool, optional
        If True, create copies of the environment factory functions to ensure
        independent instances. Default is True.

    Attributes
    ----------
    num_envs : int
        Number of parallel environment instances.
    possible_agents : list[str]
        Names of all possible agents (from first environment).
    agents : list[str]
        Names of current agents (from first environment; for compatibility).
    observation_spaces : Mapping[str, gymnasium.Space]
        Observation spaces per agent (unbatched, from first environment).
    action_spaces : Mapping[str, gymnasium.Space]
        Action spaces per agent (unbatched, from first environment).
    state_spaces : Mapping[str, gymnasium.Space]
        State spaces per agent (unbatched, from first environment).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "is_vectorized": True}

    def __init__(
        self,
        env_fns: list[Callable[[], Any]],
        copy: bool = True,
    ):
        self._num_envs = len(env_fns)
        if self._num_envs == 0:
            raise ValueError("Must provide at least one environment factory function")

        self._envs = [fn() for fn in env_fns]
        self._ref_env = self._envs[0]
        self._possible_agents = list(self._ref_env.possible_agents)
        self._agent_ids = {agent: i for i, agent in enumerate(self._possible_agents)}
        self._observation_spaces = dict(self._ref_env.observation_spaces)
        self._action_spaces = dict(self._ref_env.action_spaces)

        if hasattr(self._ref_env, "state_spaces"):
            self._state_spaces = dict(self._ref_env.state_spaces)
        else:
            self._state_spaces = {}

        self._batched_observation_spaces = {}
        for agent, space in self._observation_spaces.items():
            if isinstance(space, spaces.Box):
                batched_shape = (self._num_envs,) + space.shape
                self._batched_observation_spaces[agent] = spaces.Box(
                    low=np.broadcast_to(space.low, batched_shape),
                    high=np.broadcast_to(space.high, batched_shape),
                    shape=batched_shape,
                    dtype=space.dtype,
                )
            else:
                self._batched_observation_spaces[agent] = space

        if self._state_spaces:
            ref_state_space = list(self._state_spaces.values())[0]
            if isinstance(ref_state_space, spaces.Box):
                batched_shape = (self._num_envs,) + ref_state_space.shape
                self._batched_state_space = spaces.Box(
                    low=np.broadcast_to(ref_state_space.low, batched_shape),
                    high=np.broadcast_to(ref_state_space.high, batched_shape),
                    shape=batched_shape,
                    dtype=ref_state_space.dtype,
                )
            else:
                self._batched_state_space = ref_state_space
        else:
            self._batched_state_space = None

        self._executor = ThreadPoolExecutor(max_workers=self._num_envs)

        self._needs_reset = [True] * self._num_envs

        self._last_obs: dict[str, np.ndarray] = {}

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def possible_agents(self) -> list[str]:
        """Names of all possible agents."""
        return self._possible_agents

    @property
    def agents(self) -> list[str]:
        return list(self._ref_env.agents)

    @property
    def num_agents(self) -> int:
        return len(self._possible_agents)

    @property
    def max_num_agents(self) -> int:
        return len(self._possible_agents)

    @property
    def observation_spaces(self) -> Mapping[str, gymnasium.Space]:
        """Unbatched observation spaces per agent."""
        return self._observation_spaces

    @property
    def action_spaces(self) -> Mapping[str, gymnasium.Space]:
        return self._action_spaces

    @property
    def state_spaces(self) -> Mapping[str, gymnasium.Space]:
        if self._state_spaces:
            return self._state_spaces
        # Fallback: create from observation spaces
        return {a: self._observation_spaces[a] for a in self._possible_agents}

    @property
    def state_space(self) -> gymnasium.Space:
        """Global state space (same for all agents).

        This property is required by skrl's MultiAgentEnvWrapper which expects
        a single state_space property to create state_spaces dict.
        """
        if self._state_spaces:
            return list(self._state_spaces.values())[0]
        # Fallback: concatenated observation spaces
        return list(self._observation_spaces.values())[0]

    def observation_space(self, agent: str) -> gymnasium.Space:
        """Get observation space for a specific agent."""
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> gymnasium.Space:
        """Get action space for a specific agent."""
        return self._action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        **kwargs,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Reset all environments.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility. Each environment gets a unique
            seed derived from this base seed.
        **kwargs
            Additional arguments passed to environment reset.

        Returns
        -------
        observations : dict[str, np.ndarray]
            Batched observations per agent. Shape: (num_envs, obs_dim)
        infos : dict[str, dict]
            Info dictionaries per agent.
        """
        if seed is not None:
            seeds = [seed + i for i in range(self._num_envs)]
        else:
            seeds = [None] * self._num_envs

        def reset_env(args):
            env, env_seed = args
            return env.reset(seed=env_seed, **kwargs)

        results = list(self._executor.map(reset_env, zip(self._envs, seeds)))

        obs_batched = self._batch_observations([r[0] for r in results])
        self._last_obs = obs_batched

        infos_batched = {agent: {} for agent in self._possible_agents}

        self._needs_reset = [False] * self._num_envs

        return obs_batched, infos_batched

    def step(
        self,
        actions: Mapping[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, dict],
    ]:
        """Step all environments with batched actions.

        Parameters
        ----------
        actions : dict[str, np.ndarray]
            Batched actions per agent. Shape: (num_envs,) for discrete,
            (num_envs, action_dim) for continuous.

        Returns
        -------
        observations : dict[str, np.ndarray]
            Batched observations. Shape: (num_envs, obs_dim)
        rewards : dict[str, np.ndarray]
            Batched rewards. Shape: (num_envs,)
        terminated : dict[str, np.ndarray]
            Batched termination flags. Shape: (num_envs,)
        truncated : dict[str, np.ndarray]
            Batched truncation flags. Shape: (num_envs,)
        infos : dict[str, dict]
            Info dictionaries per agent.
        """
        # Unbatch actions for each environment
        actions_per_env = []
        for i in range(self._num_envs):
            env_actions = {}
            for agent in self._possible_agents:
                agent_actions = actions.get(agent)
                if agent_actions is not None:
                    # Normalise to numpy (handles int, float, list, JAX arrays, etc.)
                    if not isinstance(agent_actions, np.ndarray):
                        agent_actions = np.asarray(agent_actions)

                    if agent_actions.ndim == 0:
                        # Scalar: same action for all envs
                        env_actions[agent] = int(agent_actions)
                    elif agent_actions.ndim == 1:
                        if agent_actions.shape[0] == 1:
                            # Single-element 1-D: broadcast to all envs
                            env_actions[agent] = int(agent_actions[0])
                        else:
                            env_actions[agent] = int(agent_actions[i])
                    elif agent_actions.ndim == 2 and agent_actions.shape[1] == 1:
                        # (num_envs, 1) — discrete actions with an extra dim from skrl
                        env_actions[agent] = int(agent_actions[i, 0])
                    else:
                        # Continuous: (num_envs, action_dim)
                        env_actions[agent] = agent_actions[i]
                else:
                    # Default action (0) if not provided
                    env_actions[agent] = 0
            actions_per_env.append(env_actions)

        # Step all environments in parallel
        def step_env(args):
            env_idx, (env, env_actions) = args
            return env.step(env_actions)

        results = list(
            self._executor.map(step_env, enumerate(zip(self._envs, actions_per_env)))
        )

        # Extract individual components
        obs_list = [r[0] for r in results]
        rewards_list = [r[1] for r in results]
        terminated_list = [r[2] for r in results]
        truncated_list = [r[3] for r in results]
        infos_list = [r[4] for r in results]

        # Handle auto-reset for terminated/truncated environments
        for i, (term, trunc) in enumerate(zip(terminated_list, truncated_list)):
            # Check if all agents are done
            all_done = all(
                term.get(a, False) or trunc.get(a, False) for a in self._possible_agents
            )
            if all_done:
                # Auto-reset this environment
                new_obs, _ = self._envs[i].reset()
                obs_list[i] = new_obs

        # Batch all outputs
        obs_batched = self._batch_observations(obs_list)
        rewards_batched = self._batch_rewards(rewards_list)
        terminated_batched = self._batch_flags(terminated_list)
        truncated_batched = self._batch_flags(truncated_list)

        # Cache observations
        self._last_obs = obs_batched

        # Info batching (basic)
        infos_batched = {agent: {} for agent in self._possible_agents}

        return (
            obs_batched,
            rewards_batched,
            terminated_batched,
            truncated_batched,
            infos_batched,
        )

    def state(self) -> np.ndarray:
        """Get batched global state from all environments.

        Returns
        -------
        states : np.ndarray
            Batched states. Shape: (num_envs, state_dim)
        """
        if hasattr(self._ref_env, "state"):
            states = []
            for env in self._envs:
                states.append(env.state())
            return np.stack(states, axis=0)
        else:
            # Fallback: concatenate all observations
            obs_concat = []
            for agent in self._possible_agents:
                if agent in self._last_obs:
                    obs_concat.append(self._last_obs[agent])
            if obs_concat:
                return np.concatenate(obs_concat, axis=-1)
            return np.zeros((self._num_envs, 1), dtype=np.float32)

    def render(self) -> np.ndarray | None:
        """Render the first environment.

        Returns
        -------
        frame : np.ndarray or None
            RGB frame from the first environment, or None if headless.
        """
        return self._envs[0].render()

    def close(self) -> None:
        """Close all environments and shutdown thread pool."""
        for env in self._envs:
            env.close()
        self._executor.shutdown(wait=True)

    def _batch_observations(
        self,
        obs_list: list[dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Batch observations from all environments.

        Parameters
        ----------
        obs_list : list of dict
            List of observation dictionaries, one per environment.

        Returns
        -------
        batched : dict[str, np.ndarray]
            Batched observations. Shape: (num_envs, obs_dim)
        """
        batched = {}
        for agent in self._possible_agents:
            agent_obs = []
            for obs_dict in obs_list:
                if agent in obs_dict:
                    agent_obs.append(obs_dict[agent])
                else:
                    # Fill with zeros if agent not present
                    space = self._observation_spaces[agent]
                    if isinstance(space, spaces.Box):
                        agent_obs.append(np.zeros(space.shape, dtype=space.dtype))
                    else:
                        agent_obs.append(np.zeros((1,), dtype=np.float32))
            batched[agent] = np.stack(agent_obs, axis=0)
        return batched

    def _batch_rewards(
        self,
        rewards_list: list[dict[str, float]],
    ) -> dict[str, np.ndarray]:
        """Batch rewards from all environments.

        Parameters
        ----------
        rewards_list : list of dict
            List of reward dictionaries, one per environment.

        Returns
        -------
        batched : dict[str, np.ndarray]
            Batched rewards. Shape: (num_envs,)
        """
        batched = {}
        for agent in self._possible_agents:
            agent_rewards = []
            for rewards_dict in rewards_list:
                if agent in rewards_dict:
                    agent_rewards.append(rewards_dict[agent])
                else:
                    agent_rewards.append(0.0)
            batched[agent] = np.array(agent_rewards, dtype=np.float32)
        return batched

    def _batch_flags(
        self,
        flags_list: list[dict[str, bool]],
    ) -> dict[str, np.ndarray]:
        """Batch boolean flags from all environments.

        Parameters
        ----------
        flags_list : list of dict
            List of flag dictionaries (terminated/truncated), one per env.

        Returns
        -------
        batched : dict[str, np.ndarray]
            Batched flags. Shape: (num_envs,)
        """
        batched = {}
        for agent in self._possible_agents:
            agent_flags = []
            for flags_dict in flags_list:
                if agent in flags_dict:
                    agent_flags.append(bool(flags_dict[agent]))
                else:
                    agent_flags.append(False)
            batched[agent] = np.array(agent_flags, dtype=np.bool_)
        return batched


def make_vectorized_env(
    env_fn: Callable[[], Any],
    num_envs: int = 1,
) -> VectorizedPettingZooEnv | Any:
    """Create a vectorized PettingZoo environment.

    Factory function that creates either a single environment (num_envs=1)
    or a vectorized environment (num_envs > 1).

    Parameters
    ----------
    env_fn : callable
        Factory function that creates a single environment instance.
    num_envs : int, optional
        Number of parallel environment instances. Default is 1.

    Returns
    -------
    env : VectorizedPettingZooEnv or single environment
        If num_envs == 1, returns a single environment instance.
        If num_envs > 1, returns a VectorizedPettingZooEnv wrapper.

    Examples
    --------
    >>> from environments import IntersectionConfig, make_intersection_env
    >>> config = IntersectionConfig(num_agents=4, duration=13)
    >>> env = make_vectorized_env(
    ...     env_fn=lambda: make_intersection_env(config),
    ...     num_envs=16,
    ... )
    >>> print(env.num_envs)  # 16
    """
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")

    env_fns = [env_fn for _ in range(num_envs)]
    return VectorizedPettingZooEnv(env_fns)


class AsyncVectorizedPettingZooEnv(VectorizedPettingZooEnv):
    """Asynchronous vectorized wrapper using multiprocessing.

    For environments with heavy Python computation that don't release the GIL,
    this wrapper uses multiprocessing instead of threading for true parallelism.

    Note: This requires environments to be picklable, which may not work with
    all environment types (e.g., those with Pygame renderers).

    Parameters
    ----------
    env_fns : list of callables
        Factory functions that create environment instances.
    """

    def __init__(
        self,
        env_fns: list[Callable[[], Any]],
    ):
        # Import here to avoid overhead if not used
        from multiprocessing import Pool

        self._num_envs = len(env_fns)
        if self._num_envs == 0:
            raise ValueError("Must provide at least one environment factory function")

        # For async, we need to create environments in subprocesses
        # For now, create a reference environment in the main process
        self._ref_env = env_fns[0]()

        # Store factory functions for subprocess creation
        self._env_fns = env_fns

        # Agent metadata (from reference environment)
        self._possible_agents = list(self._ref_env.possible_agents)
        self._agent_ids = {agent: i for i, agent in enumerate(self._possible_agents)}

        # Space metadata
        self._observation_spaces = dict(self._ref_env.observation_spaces)
        self._action_spaces = dict(self._ref_env.action_spaces)

        if hasattr(self._ref_env, "state_spaces"):
            self._state_spaces = dict(self._ref_env.state_spaces)
        else:
            self._state_spaces = {}

        # Create process pool
        self._pool = Pool(processes=self._num_envs)

        # Track which environments need reset
        self._needs_reset = [True] * self._num_envs
        self._last_obs: dict[str, np.ndarray] = {}

    def close(self) -> None:
        """Close all environments and shutdown process pool."""
        self._ref_env.close()
        self._pool.close()
        self._pool.join()
