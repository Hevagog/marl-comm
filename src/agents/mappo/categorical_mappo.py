from __future__ import annotations
import functools
import pickle

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from skrl import config
from skrl.multi_agents.jax.base import MultiAgent
from skrl.multi_agents.jax.mappo import MAPPO

from agents.mappo.adamw import AdamW


# ---------------------------------------------------------------------------
# Clipped value loss with optional Huber penalty.
#
# The stock skrl ``_update_value`` uses plain MSE on the (optionally-clipped)
# value prediction error.  MSE is vulnerable to reward-distribution shifts
# and large initial advantage variance: a single outlier (pred − target)²
# term dominates the batch gradient and destabilises the critic in the
# first ~1M transitions of a run.  Huber (smooth-L1) caps per-sample
# gradient at |δ|, which keeps the critic learnable under noisy targets
# without changing its fixed-point behaviour once targets stabilise.
#
# We keep the clipped-vs-unclipped PPO pessimistic-max formulation from
# skrl's implementation to preserve the value trust-region semantics.
# ---------------------------------------------------------------------------


@functools.partial(
    jax.jit,
    static_argnames=("value_act", "clip_predicted_values", "use_huber"),
)
def _update_value_huber(
    value_act,
    value_state_dict,
    sampled_states,
    sampled_values,
    sampled_returns,
    value_loss_scale,
    clip_predicted_values,
    value_clip,
    use_huber,
    huber_delta,
):
    def _per_sample(err):
        # Huber: 0.5*e² for |e| ≤ δ, δ*(|e| − 0.5*δ) otherwise.
        # Branchless via ``jnp.where`` so the function stays jit-friendly.
        abs_err = jnp.abs(err)
        quad = jnp.minimum(abs_err, huber_delta)
        lin = abs_err - quad
        huber = 0.5 * quad * quad + huber_delta * lin
        mse = 0.5 * err * err
        return jnp.where(use_huber, huber, mse)

    def _value_loss(params):
        predicted_values, _, _ = value_act({"states": sampled_states}, "value", params)
        if clip_predicted_values:
            # Value-function trust region: clip the *change* in prediction
            # relative to the rollout-time value estimate.  Identical in
            # spirit to the PPO policy clip.
            predicted_values = sampled_values + jnp.clip(
                predicted_values - sampled_values, -value_clip, value_clip
            )
            loss_clipped = _per_sample(sampled_returns - predicted_values)
            # Pessimistic upper bound — but when ``clip_predicted_values`` is
            # True the skrl reference only evaluates the clipped branch, so
            # we match that behaviour exactly.
            return value_loss_scale * loss_clipped.mean()
        loss = _per_sample(sampled_returns - predicted_values)
        return value_loss_scale * loss.mean()

    value_loss, grad = jax.value_and_grad(_value_loss, has_aux=False)(
        value_state_dict.params
    )
    return grad, value_loss


def _categorical_entropy(logits: jax.Array) -> jax.Array:
    """Compute categorical entropy directly from unnormalised logits.

    This avoids relying on the fragile ``outputs["stddev"]`` relay
    through ``CategoricalMixin`` and is mathematically equivalent to
    ``CategoricalMixin._entropy``.
    """
    log_probs = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    return -(jax.nn.softmax(logits, axis=-1) * log_probs).sum(axis=-1)


# ---------------------------------------------------------------------------
# Custom GAE without buffer-level advantage normalisation.
# skrl's _compute_gae normalises advantages at the buffer level.  We then
# normalise within each mini-batch in _update_policy_fixed.  Applying both
# flips the sign of ~2-5% of small advantages (a sample with buffer-norm
# advantage +0.03 in a mini-batch with mean +0.05 becomes -0.02), injecting
# gradient noise.  The MAPPO reference normalises once at buffer level only;
# we choose to normalise at mini-batch level only, which is mathematically
# sounder for stochastic mini-batch optimisation.
# ---------------------------------------------------------------------------


def _compute_gae_no_norm(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE without buffer-level advantage normalisation (numpy fallback)."""
    advantage = 0
    advantages = np.zeros_like(rewards)
    not_dones = np.logical_not(dones)
    memory_size = rewards.shape[0]
    for i in reversed(range(memory_size)):
        _next_values = values[i + 1] if i < memory_size - 1 else next_values
        advantage = (
            rewards[i]
            - values[i]
            + discount_factor
            * not_dones[i]
            * (_next_values + lambda_coefficient * advantage)
        )
        advantages[i] = advantage
    returns = advantages + values
    # No normalisation here — done at mini-batch level in _update_policy_fixed
    return returns, advantages


@jax.jit
def _jit_compute_gae_no_norm(
    rewards: jax.Array,
    dones: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
) -> tuple[jax.Array, jax.Array]:
    """GAE without buffer-level advantage normalisation (JIT version).

    Uses jax.lax.scan instead of an unrolled Python loop so that JIT
    compilation cost is O(1) in the buffer length rather than O(T).
    """
    not_dones = jnp.logical_not(dones).astype(rewards.dtype)

    # next-step bootstrap values: [values[1], ..., values[T-1], next_values]
    # Shape matches values: (T, feature_dim)
    all_next_v = jnp.concatenate(
        [values[1:], next_values.reshape(1, *values.shape[1:])], axis=0
    )

    def _step(carry_adv, x):
        r, nd, v, nv = x
        adv = r - v + discount_factor * nd * (nv + lambda_coefficient * carry_adv)
        return adv, adv

    # Reverse the time axis so scan processes T-1 → 0
    xs = jax.tree.map(
        lambda a: jnp.flip(a, axis=0), (rewards, not_dones, values, all_next_v)
    )
    _, adv_rev = jax.lax.scan(_step, jnp.zeros_like(values[0]), xs)
    advantages = jnp.flip(adv_rev, axis=0)
    returns = advantages + values
    return returns, advantages


@functools.partial(
    jax.jit,
    static_argnames=(
        "policy_act",
        "debug_entropy",
        "comm_reg_scale",
    ),
)
def _update_policy_fixed(
    policy_act,
    policy_state_dict,
    sampled_states,
    sampled_actions,
    sampled_log_prob,
    sampled_advantages,
    ratio_clip,
    entropy_loss_scale,
    debug_entropy,
    comm_reg_scale=0.0,
    sampled_recurrent_state=None,
):
    """Like skrl's ``_update_policy`` but with two critical fixes:

    1. **Direct entropy from logits** — computes categorical entropy
       directly from ``outputs["net_output"]`` (the policy logits) instead
       of the fragile ``get_entropy(outputs["stddev"])`` relay.
    2. **Mini-batch advantage normalisation** — normalises advantages
       *within each mini-batch* (MAPPO paper, Section 5.1) rather than
       relying on the buffer-level normalisation in ``compute_gae``.
    3. **Communication regularisation** (MAGIC) — when ``comm_reg_scale > 0``,
       adds a binary-entropy loss on the off-diagonal adjacency density.
       Pushes the Gumbel-Softmax topology away from the all-zero (no-comm)
       and all-one (full-comm) collapse modes toward informative sparse graphs.
       Gradient flows via the straight-through Gumbel-Softmax estimator.
    """

    raw_adv_mean = sampled_advantages.mean()
    raw_adv_std = sampled_advantages.std()
    raw_adv_min = sampled_advantages.min()
    raw_adv_max = sampled_advantages.max()

    sampled_advantages = (sampled_advantages - raw_adv_mean) / (raw_adv_std + 1e-8)

    # Resolve at Python level so the conditional is not traced by JAX.
    use_comm_reg = comm_reg_scale > 0.0

    def _policy_loss(params):
        _inputs = {"states": sampled_states, "taken_actions": sampled_actions}
        #  Matches skrl-torch ppo_rnn. Patch to the skrl since it does not support
        # LSTM and GRU for JAX
        if sampled_recurrent_state is not None:
            _inputs["recurrent_state"] = jax.lax.stop_gradient(sampled_recurrent_state)
        _, next_log_prob, outputs = policy_act(
            _inputs,
            "policy",
            params,
        )

        # Approximate KL divergence
        ratio = next_log_prob - sampled_log_prob
        kl_divergence = ((jnp.exp(ratio) - 1) - ratio).mean()

        # Clipped surrogate objective
        ratio = jnp.exp(next_log_prob - sampled_log_prob)
        surrogate = sampled_advantages * ratio
        surrogate_clipped = sampled_advantages * jnp.clip(
            ratio, 1.0 - ratio_clip, 1.0 + ratio_clip
        )

        ratio_mean = ratio.mean()
        ratio_std = ratio.std()
        ratio_max_abs_dev = jnp.abs(ratio - 1.0).max()
        ratio_clipped_frac = (
            ((ratio < 1.0 - ratio_clip) | (ratio > 1.0 + ratio_clip))
            .mean()
            .astype(jnp.float32)
        )
        surrogate_raw_mean = surrogate.mean()

        policy_loss = -jnp.minimum(surrogate, surrogate_clipped).mean()

        logits = outputs["net_output"]
        entropy = _categorical_entropy(logits)
        if debug_entropy:
            jax.debug.print("entropy mean: {}", entropy.mean())
        entropy_loss = -entropy_loss_scale * entropy.mean()

        total_loss = policy_loss + entropy_loss

        # Communication-topology regularisation (MAGIC §4.2).
        # Binary entropy H(p) is maximised at p=0.5 (balanced comm graph).
        # Minimising -H(p) penalises adjacency collapse to all-zero or all-one.
        # Gradient flows through the straight-through Gumbel-Softmax estimator.
        if use_comm_reg:
            adj = outputs.get("adj_matrices", None)
            if adj is not None:
                n = adj.shape[-1]
                off_diag = jnp.ones((n, n), dtype=jnp.float32) - jnp.eye(
                    n, dtype=jnp.float32
                )
                # Total number of off-diagonal slots across all rounds/groups.
                # adj.size / N^2 == R * groups, so total = (R*groups) * (N^2-N).
                total_off_diag = jnp.float32(adj.size) * (
                    jnp.float32(n * n - n) / jnp.float32(n * n)
                )
                density = jnp.sum(adj * off_diag) / total_off_diag
                # Clip into [eps, 1-eps] to keep both log(density) and
                # log(1-density) finite even at exact 0/1 saturation.
                eps = jnp.float32(1e-6)
                density = jnp.clip(density, eps, 1.0 - eps)
                h = -(
                    density * jnp.log(density)
                    + (1.0 - density) * jnp.log(1.0 - density)
                )
                total_loss = total_loss - jnp.float32(comm_reg_scale) * h

        diag = {
            "ratio_mean": ratio_mean,
            "ratio_std": ratio_std,
            "ratio_max_abs_dev": ratio_max_abs_dev,
            "ratio_clipped_frac": ratio_clipped_frac,
            "surrogate_raw_mean": surrogate_raw_mean,
        }
        return total_loss, (entropy_loss, kl_divergence, entropy.mean(), diag)

    (total_loss, (entropy_loss, kl_divergence, mean_entropy, diag)), grad = (
        jax.value_and_grad(_policy_loss, has_aux=True)(policy_state_dict.params)
    )

    leaves = jax.tree_util.tree_leaves(grad)
    grad_global_norm = jnp.sqrt(sum(jnp.vdot(leaf, leaf).real for leaf in leaves))

    # NaN/Inf grad guard: if any leaf is non-finite, zero the entire grad tree.
    # Without this an upstream NaN propagates through optax → params → all
    # subsequent logits NaN → categorical sampler degenerates to argmax-of-NaN
    # = action 0 ("stay") and the agent appears to have learned a trivial
    # no-op policy (warehouse_scaled v1/v5/v5b reward ≈315 = no-move floor).
    grad_is_finite = jnp.isfinite(grad_global_norm)
    grad = jax.tree_util.tree_map(
        lambda g: jnp.where(grad_is_finite, g, jnp.zeros_like(g)), grad
    )

    diag = {
        **diag,
        "adv_raw_mean": raw_adv_mean,
        "adv_raw_std": raw_adv_std,
        "adv_raw_min": raw_adv_min,
        "adv_raw_max": raw_adv_max,
        "grad_global_norm": grad_global_norm,
        "grad_nonfinite_skipped": jnp.float32(1.0) - grad_is_finite.astype(jnp.float32),
    }

    policy_loss = total_loss - entropy_loss
    return grad, policy_loss, entropy_loss, kl_divergence, mean_entropy, diag


def _is_shared_policy(models) -> bool:
    agents = list(models.keys())
    if len(agents) <= 1:
        return True
    first_policy = models[agents[0]].get("policy")
    return all(models[uid].get("policy") is first_policy for uid in agents[1:])


class CategoricalMAPPO(MAPPO):
    """MAPPO variant that correctly applies entropy regularisation for
    categorical (discrete-action) policies.

    Additional fixes over stock skrl MAPPO:
    - Direct categorical entropy from logits (no fragile stddev relay)
    - Mini-batch advantage normalisation (MAPPO paper §5.1)
    - Agent-ID one-hot appended to shared states for the value function
      (MAPPO paper §5.2 — allows the shared critic to distinguish agents)
    - Support for heterogeneous observation spaces: when agents have different
      observation sizes, per-agent policy networks are used instead of
      parameter sharing.  The shared critic always remains shared.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Detect whether a single shared policy is used across all agents,
        # or each agent has its own separate policy network.
        self._shared_policy: bool = _is_shared_policy(self.models)

        # Huber-value-loss flags.  Resolved to Python scalars here so they
        # can be passed as jit-static args into ``_update_value_huber``
        # without forcing a recompile on every minibatch.
        self._use_huber_value_loss: bool = bool(
            self.cfg.get("use_huber_value_loss", False)
        )
        self._huber_delta: float = float(self.cfg.get("huber_delta", 1.0))

        # Build a one-hot LUT: agent uid → one-hot vector (num_agents,).
        n = len(self.possible_agents)
        self._agent_onehot = {
            uid: jnp.array(
                [[1.0 if j == i else 0.0 for j in range(n)]],
                dtype=jnp.float32,
            )
            for i, uid in enumerate(self.possible_agents)
        }

        # Save initial learning rate for linear decay.
        uid0 = self.possible_agents[0]
        self._initial_learning_rate = float(self._learning_rate[uid0])

        # share preprocessors across all agents.
        # Only uid0's scalers are trained (via train=True), so all agents
        # must reference the same scaler instances to get consistent
        # preprocessing during both rollout (act/record_transition) and
        # training (_update).
        for uid in self.possible_agents:
            self._shared_state_preprocessor[uid] = self._shared_state_preprocessor[uid0]
            self._value_preprocessor[uid] = self._value_preprocessor[uid0]

            self.checkpoint_modules[uid]["shared_state_preprocessor"] = (
                self._shared_state_preprocessor[uid0]
            )
            self.checkpoint_modules[uid]["value_preprocessor"] = (
                self._value_preprocessor[uid0]
            )

        # For homogeneous agents (shared policy), also share the
        # _state_preprocessor so all agents pool their obs into a single
        # RunningStandardScaler instead of 4 separate scalers that each see
        # only 1/N of the data.  Heterogeneous agents keep separate scalers
        # since their obs spaces may differ in semantics.
        if self._shared_policy:
            for uid in self.possible_agents:
                self._state_preprocessor[uid] = self._state_preprocessor[uid0]
                self.checkpoint_modules[uid]["state_preprocessor"] = (
                    self._state_preprocessor[uid0]
                )

        # Create a SINGLE set of AdamW optimisers for the shared models.
        # With parameter sharing, all agents' data is pooled and
        # the shared model is updated once per rollout — a single optimiser
        # maintains consistent Adam moment estimates.

        # With heterogeneous agents (per-agent policies), each agent gets its
        # own separate optimizer.
        weight_decay = self.cfg.get("weight_decay", 0.0)
        value = self.values[uid0]

        if self._shared_policy:
            # Homogeneous: one shared policy optimizer for all agents.
            policy = self.policies[uid0]
            if policy is not None and value is not None:
                shared_policy_opt = AdamW(
                    model=policy,
                    lr=self._learning_rate[uid0],
                    weight_decay=weight_decay,
                    grad_norm_clip=self._grad_norm_clip[uid0],
                    scale=False,
                )
                shared_value_opt = AdamW(
                    model=value,
                    lr=self._learning_rate[uid0],
                    weight_decay=weight_decay,
                    grad_norm_clip=self._grad_norm_clip[uid0],
                    scale=False,
                )
                # Point all per-agent optimizer slots to the shared instances.
                for uid in self.possible_agents:
                    self.policy_optimizer[uid] = shared_policy_opt
                    self.value_optimizer[uid] = shared_value_opt
        else:
            # Heterogeneous: per-agent policy optimizer; shared value optimizer.
            shared_value_opt = AdamW(
                model=value,
                lr=self._learning_rate[uid0],
                weight_decay=weight_decay,
                grad_norm_clip=self._grad_norm_clip[uid0],
                scale=False,
            )
            for uid in self.possible_agents:
                policy = self.policies[uid]
                if policy is not None and value is not None:
                    per_agent_policy_opt = AdamW(
                        model=policy,
                        lr=self._learning_rate[uid],
                        weight_decay=weight_decay,
                        grad_norm_clip=self._grad_norm_clip[uid],
                        scale=False,
                    )
                    self.policy_optimizer[uid] = per_agent_policy_opt
                    self.value_optimizer[uid] = shared_value_opt

        for uid in self.possible_agents:
            self.checkpoint_modules[uid].pop("policy_optimizer", None)
            self.checkpoint_modules[uid].pop("value_optimizer", None)

    def load(self, path: str) -> None:
        """Load checkpoint, fixing skrl JAX multi-agent ``load()`` bug.

        **ROOT-001 fix**: skrl 1.4.3 ``MultiAgent.load()`` (JAX backend)
        checks ``hasattr(module, "load_state_dict")`` but no JAX module
        defines that method (it's a PyTorch-ism).  The correct check is
        ``hasattr(module, "state_dict")``, matching the single-agent
        ``Agent.load()`` at ``agents/jax/base.py:444``.

        Without this fix, ``load()`` silently restores nothing — the agent
        operates with random-init weights and default preprocessor stats.
        """
        with open(path, "rb") as f:
            modules = pickle.load(f)

        if not isinstance(modules, dict):
            print("Checkpoint is not a dict — skipping load.")
            return

        for uid in self.possible_agents:
            if uid not in modules:
                print(
                    f"Cannot load modules for {uid}. "
                    "The agent doesn't have such an instance"
                )
                continue
            for name, data in modules[uid].items():
                module = self.checkpoint_modules[uid].get(name, None)
                if module is not None:
                    # FIX ROOT-001: check "state_dict" (correct) instead of
                    # "load_state_dict" (PyTorch-only, never present in JAX).
                    if hasattr(module, "state_dict"):
                        params = flax.serialization.from_bytes(
                            module.state_dict.params, data
                        )
                        module.state_dict = module.state_dict.replace(params=params)
                    else:
                        print(f"Module {uid}:{name} has no state_dict — skipped.")
                else:
                    print(
                        f"Cannot load the {uid}:{name} module. "
                        "The agent doesn't have such an instance"
                    )

    @staticmethod
    def _append_agent_id(shared_states: jax.Array, one_hot: jax.Array) -> jax.Array:
        """Concatenate a one-hot agent-ID vector to shared states.

        ``one_hot`` has shape ``(1, num_agents)`` and is broadcast along the
        batch dimension of ``shared_states`` ``(B, state_dim)``.
        Handles both 2-D ``(B, state_dim)`` and 3-D ``(B, 1, state_dim)``
        inputs that skrl memory tensors may produce.
        """
        if shared_states.ndim == 3 and shared_states.shape[1] == 1:
            shared_states = shared_states[:, 0, :]
        batch = shared_states.shape[0]
        return jnp.concatenate(
            [shared_states, jnp.broadcast_to(one_hot, (batch, one_hot.shape[1]))],
            axis=-1,
        )

    def record_transition(
        self,
        states,
        actions,
        rewards,
        next_states,
        terminated,
        truncated,
        infos,
        timestep,
        timesteps,
    ):
        """Override to append agent-ID before the value-function forward pass.

        ``MAPPO.record_transition`` calls ``value.act(shared_states)`` to
        store V(s) in the rollout buffer.  The value net now expects the
        extra agent-ID dimensions, so we must expand shared states first.
        """

        MultiAgent.record_transition(
            self,
            states,
            actions,
            rewards,
            next_states,
            terminated,
            truncated,
            infos,
            timestep,
            timesteps,
        )

        if self.memories:
            shared_states = infos["shared_states"]
            self._current_shared_next_states = infos["shared_next_states"]

            for uid in self.possible_agents:
                # reward shaping
                if self._rewards_shaper is not None:
                    rewards[uid] = self._rewards_shaper(
                        rewards[uid], timestep, timesteps
                    )

                # compute values (with agent-ID)
                preprocessed = self._shared_state_preprocessor[uid](shared_states)
                expanded = self._append_agent_id(preprocessed, self._agent_onehot[uid])
                values, _, _ = self.values[uid].act({"states": expanded}, role="value")
                if not self._jax:
                    values = jax.device_get(values)
                values = self._value_preprocessor[uid](values, inverse=True)

                # time-limit (truncation) bootstrapping: V(s_{t+1}), not V(s_t).
                # The truncation reward augmentation must use the value of the
                # *next* state to correctly approximate the discounted return
                # beyond the time-limit boundary.
                if self._time_limit_bootstrap[uid]:
                    next_preprocessed = self._shared_state_preprocessor[uid](
                        infos["shared_next_states"]
                    )
                    next_expanded = self._append_agent_id(
                        next_preprocessed, self._agent_onehot[uid]
                    )
                    next_values, _, _ = self.values[uid].act(
                        {"states": next_expanded}, role="value"
                    )
                    if not self._jax:
                        next_values = jax.device_get(next_values)
                    next_values = self._value_preprocessor[uid](
                        next_values, inverse=True
                    )
                    rewards[uid] += (
                        self._discount_factor[uid] * next_values * truncated[uid]
                    )

                # storage transition in memory
                self.memories[uid].add_samples(
                    states=states[uid],
                    actions=actions[uid],
                    rewards=rewards[uid],
                    next_states=next_states[uid],
                    terminated=terminated[uid],
                    truncated=truncated[uid],
                    log_prob=self._current_log_prob[uid],
                    values=values,
                    shared_states=shared_states,
                )

    def _shuffle_buffer_indices(self, buffer_size: int) -> np.ndarray:
        """Create shuffled indices for one training epoch.

        Default: random permutation of all samples.  MAGICMAPPO overrides
        this to shuffle at the timestep level, preserving agent pairing
        within each communication group.
        """
        return np.random.permutation(buffer_size)

    def _shuffle_grouped_buffer_indices(self, buffer_size: int) -> np.ndarray:
        n = len(self.possible_agents)
        timesteps_per_agent = buffer_size // n
        timestep_perm = np.random.permutation(timesteps_per_agent)
        offsets = np.arange(n, dtype=np.intp)[:, None] * timesteps_per_agent
        return (timestep_perm[None, :] + offsets).T.reshape(-1)

    def _update(self, timestep: int, timesteps: int) -> None:  # noqa: C901
        uid0 = self.possible_agents[0]
        value = self.values[uid0]  # always shared value

        # Per-agent GAE and preprocessing
        per_agent_tensors: dict[str, dict[str, jax.Array]] = {}

        for uid in self.possible_agents:
            memory = self.memories[uid]

            value.training = False
            bootstrap_shared = self._shared_state_preprocessor[uid](
                self._current_shared_next_states
            )
            bootstrap_shared = self._append_agent_id(
                bootstrap_shared, self._agent_onehot[uid]
            )
            last_values, _, _ = value.act({"states": bootstrap_shared}, role="value")
            value.training = True
            if not self._jax:
                last_values = jax.device_get(last_values)
            last_values = self._value_preprocessor[uid](last_values, inverse=True)

            raw_values = memory.get_tensor_by_name("values")
            returns, advantages = (
                _jit_compute_gae_no_norm if self._jax else _compute_gae_no_norm
            )(
                rewards=memory.get_tensor_by_name("rewards"),
                dones=memory.get_tensor_by_name("terminated")
                | memory.get_tensor_by_name("truncated"),
                values=raw_values,
                next_values=last_values,
                discount_factor=self._discount_factor[uid],
                lambda_coefficient=self._lambda[uid],
            )

            memory.set_tensor_by_name(
                "values", self._value_preprocessor[uid](raw_values, train=True)
            )
            memory.set_tensor_by_name(
                "returns", self._value_preprocessor[uid](returns, train=True)
            )
            memory.set_tensor_by_name("advantages", advantages)

            # Extract all named tensors from this agent's memory.
            tensors = {}
            for name in self._tensors_names:
                t = memory.get_tensor_by_name(name)
                tensors[name] = t.reshape(-1, t.shape[-1])
            # Preprocess states per-agent (each has its own scaler with correct size)
            # train=True only on first call to update running stats once
            update_state_preproc = bool(
                self.cfg.get("update_state_preprocessor_in_update", False)
            )
            update_shared_state_preproc = bool(
                self.cfg.get("update_shared_state_preprocessor_in_update", False)
            )
            tensors["states"] = self._state_preprocessor[uid](
                tensors["states"], train=update_state_preproc
            )

            tensors["shared_states"] = self._append_agent_id(
                self._shared_state_preprocessor[uid](
                    tensors["shared_states"],
                    train=(uid == uid0) and update_shared_state_preproc,
                ),
                self._agent_onehot[uid],
            )
            per_agent_tensors[uid] = tensors

        #  Entropy annealing
        effective_entropy_loss_scale = float(self._entropy_loss_scale[uid0])
        if self.cfg.get("entropy_annealing", False) and timesteps > 0:
            e_start = float(self.cfg.get("entropy_loss_scale_start", 0.05))
            e_end = float(self.cfg.get("entropy_loss_scale_end", 0.005))
            frac = max(0.0, 1.0 - timestep / timesteps)
            effective_entropy_loss_scale = e_end + frac * (e_start - e_end)
        for uid in self.possible_agents:
            self._entropy_loss_scale[uid] = effective_entropy_loss_scale

        kl_warmup_fraction = float(self.cfg.get("kl_warmup_fraction", 0.0))
        apply_kl_stop = True
        if timesteps > 0 and kl_warmup_fraction > 0.0:
            apply_kl_stop = timestep > (timesteps * kl_warmup_fraction)
        # ratio_max_threshold: stop the epoch loop when any mini-batch's
        # ratio_max_abs_dev exceeds this value.  Mean KL (kl_threshold) can
        # remain below 0.05 even at ratio_mad=14 because the high-ratio
        # outlier transitions are diluted across the full buffer.  Checking
        # ratio_max_abs_dev directly catches per-transition drift that mean
        # KL misses.  None means disabled (backward-compatible default).
        _ratio_max_threshold: float | None = (
            float(self.cfg["ratio_max_threshold"])
            if self.cfg.get("ratio_max_threshold") is not None
            else None
        )

        #  Mini-batch training
        if self._shared_policy:
            # Homogeneous path: pool all agents' data and update a single
            # shared policy.  This is the standard MAPPO parameter-sharing
            # approach (Yu et al. 2021, §5).
            self._update_shared_policy(
                per_agent_tensors,
                uid0,
                value,
                effective_entropy_loss_scale,
                apply_kl_stop,
                timestep,
                timesteps,
                _ratio_max_threshold,
            )
        else:
            # Heterogeneous path: update each agent's policy independently
            # on its own data.  The value network is still updated jointly
            # (pooled) since it uses the shared global state.
            self._update_per_agent_policies(
                per_agent_tensors,
                uid0,
                value,
                effective_entropy_loss_scale,
                apply_kl_stop,
                timestep,
                timesteps,
                _ratio_max_threshold,
            )

    def _update_shared_policy(
        self,
        per_agent_tensors,
        uid0,
        value,
        effective_entropy_loss_scale,
        apply_kl_stop,
        timestep,
        timesteps,
        ratio_max_threshold: float | None = None,
    ) -> None:  # noqa: C901
        """Training update for homogeneous agents (shared policy network).

        All agents' transitions are pooled into a single buffer and the
        shared policy + shared value are updated jointly, exactly as in
        the MAPPO paper (Yu et al. 2021, §5).
        """
        policy = self.policies[uid0]

        pooled_list = [
            jnp.concatenate(
                [per_agent_tensors[uid][name] for uid in self.possible_agents],
                axis=0,
            )
            for name in self._tensors_names
        ]
        buffer_size = pooled_list[0].shape[0]
        batch_size = buffer_size // self._mini_batches[uid0]

        cumulative_policy_loss = 0.0
        cumulative_entropy_loss = 0.0
        cumulative_value_loss = 0.0
        cumulative_diag: dict[str, float] = {}
        actual_batch_count = 0
        kl_exceeded = False
        mean_entropy = jnp.float32(0.0)
        new_lr = self._learning_rate[uid0]

        for _ in range(self._learning_epochs[uid0]):
            if kl_exceeded:
                break

            _indices = self._shuffle_buffer_indices(buffer_size)
            for i in range(0, buffer_size, batch_size):
                idx = _indices[i : i + batch_size]
                (
                    sampled_states,
                    sampled_shared_states,
                    sampled_actions,
                    sampled_log_prob,
                    sampled_values,
                    sampled_returns,
                    sampled_advantages,
                ) = (t[idx] for t in pooled_list)
                # Optional recurrent-state minibatch (MAGICMAPPO LSTM/GRU
                # path). Subclasses set `_sample_recurrent_minibatch` to
                # return the stored carry sliced by `idx`; default Dense
                # path leaves it None and behavior is unchanged.
                sampled_recurrent_state = None
                _sample_rec = getattr(self, "_sample_recurrent_minibatch", None)
                if _sample_rec is not None:
                    sampled_recurrent_state = _sample_rec(idx)
                # --- Policy update ---
                (
                    grad,
                    policy_loss,
                    entropy_loss,
                    kl_divergence,
                    mean_entropy,
                    diag,
                ) = _update_policy_fixed(
                    policy.act,
                    policy.state_dict,
                    sampled_states,
                    sampled_actions,
                    sampled_log_prob,
                    sampled_advantages,
                    self._ratio_clip[uid0],
                    effective_entropy_loss_scale,
                    self.cfg.get("debug_entropy_stats", False),
                    self.cfg.get("comm_reg_scale", 0.0),
                    sampled_recurrent_state=sampled_recurrent_state,
                )
                for k, v in diag.items():
                    cumulative_diag[k] = cumulative_diag.get(k, 0.0) + float(v)

                if apply_kl_stop and (
                    (
                        self._kl_threshold[uid0]
                        and kl_divergence > self._kl_threshold[uid0]
                    )
                    or (
                        ratio_max_threshold is not None
                        and float(diag["ratio_max_abs_dev"]) > ratio_max_threshold
                    )
                ):
                    kl_exceeded = True
                    break

                if self.cfg.get("debug_kl_stats", False):
                    self.track_data(
                        "Diagnostics / Mean KL (shared)",
                        kl_divergence.item(),
                    )

                if config.jax.is_distributed:
                    grad = policy.reduce_parameters(grad)
                self.policy_optimizer[uid0] = self.policy_optimizer[uid0].step(
                    grad, policy, self._learning_rate[uid0]
                )
                for uid in self.possible_agents:
                    self.policy_optimizer[uid] = self.policy_optimizer[uid0]

                # --- Value update ---
                # Huber is selected once at __init__ from the cfg and plumbed
                # as a jit-static flag; the MSE branch inside
                # ``_update_value_huber`` preserves the exact skrl semantics.
                grad, value_loss = _update_value_huber(
                    value.act,
                    value.state_dict,
                    sampled_shared_states,
                    sampled_values,
                    sampled_returns,
                    self._value_loss_scale[uid0],
                    self._clip_predicted_values[uid0],
                    self._value_clip[uid0],
                    self._use_huber_value_loss,
                    self._huber_delta,
                )

                if config.jax.is_distributed:
                    grad = value.reduce_parameters(grad)
                self.value_optimizer[uid0] = self.value_optimizer[uid0].step(
                    grad, value, self._learning_rate[uid0]
                )
                for uid in self.possible_agents:
                    self.value_optimizer[uid] = self.value_optimizer[uid0]

                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if effective_entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                actual_batch_count += 1

        new_lr = self._apply_lr_decay(timestep, timesteps)
        self._log_training_stats(
            uid0,
            actual_batch_count,
            cumulative_policy_loss,
            cumulative_value_loss,
            cumulative_entropy_loss,
            effective_entropy_loss_scale,
            mean_entropy,
            new_lr,
            per_agent_tensors,
            cumulative_diag=cumulative_diag,
        )

    def _update_per_agent_policies(
        self,
        per_agent_tensors,
        uid0,
        value,
        effective_entropy_loss_scale,
        apply_kl_stop,
        timestep,
        timesteps,
        ratio_max_threshold: float | None = None,
    ) -> None:  # noqa: C901
        """Training update for heterogeneous agents (per-agent policy networks).

        Each agent's policy is updated on its own rollout data independently.
        The value network (shared critic) is updated using the pooled shared
        states from all agents — the value target is the same global-state-
        conditioned V(s) regardless of which agent's local obs differs.

        This follows the CTDE (Centralized Training, Decentralized Execution)
        paradigm: each agent has its own actor, but a single centralized
        critic is trained on the global state.
        """
        #  Value update data: pool shared states from all agents
        all_shared_states = jnp.concatenate(
            [per_agent_tensors[uid]["shared_states"] for uid in self.possible_agents],
            axis=0,
        )
        all_values = jnp.concatenate(
            [per_agent_tensors[uid]["values"] for uid in self.possible_agents], axis=0
        )
        all_returns = jnp.concatenate(
            [per_agent_tensors[uid]["returns"] for uid in self.possible_agents], axis=0
        )

        pooled_buffer_size = all_shared_states.shape[0]
        value_batch_size = max(pooled_buffer_size // self._mini_batches[uid0], 1)

        cumulative_policy_loss = 0.0
        cumulative_entropy_loss = 0.0
        cumulative_value_loss = 0.0
        cumulative_diag: dict[str, float] = {}
        actual_batch_count = 0
        mean_entropy = jnp.float32(0.0)
        new_lr = self._learning_rate[uid0]

        kl_exceeded = False
        for epoch in range(self._learning_epochs[uid0]):
            if kl_exceeded:
                break

            # Per-agent policy updates (each on its own data)
            for uid in self.possible_agents:
                policy = self.policies[uid]
                tensors = per_agent_tensors[uid]
                buf_size = tensors["states"].shape[0]
                batch_size = max(buf_size // self._mini_batches[uid], 1)

                uid_indices = self._shuffle_buffer_indices(buf_size)

                for i in range(0, buf_size, batch_size):
                    idx = uid_indices[i : i + batch_size]
                    sampled_states = tensors["states"][idx]
                    sampled_actions = tensors["actions"][idx]
                    sampled_log_prob = tensors["log_prob"][idx]
                    sampled_advantages = tensors["advantages"][idx]

                    (
                        grad,
                        policy_loss,
                        entropy_loss,
                        kl_divergence,
                        mean_entropy,
                        diag,
                    ) = _update_policy_fixed(
                        policy.act,
                        policy.state_dict,
                        sampled_states,
                        sampled_actions,
                        sampled_log_prob,
                        sampled_advantages,
                        self._ratio_clip[uid],
                        effective_entropy_loss_scale,
                        self.cfg.get("debug_entropy_stats", False),
                        self.cfg.get("comm_reg_scale", 0.0),
                    )
                    for k, v in diag.items():
                        cumulative_diag[k] = cumulative_diag.get(k, 0.0) + float(v)

                    if apply_kl_stop and (
                        (
                            self._kl_threshold[uid]
                            and kl_divergence > self._kl_threshold[uid]
                        )
                        or (
                            ratio_max_threshold is not None
                            and float(diag["ratio_max_abs_dev"]) > ratio_max_threshold
                        )
                    ):
                        kl_exceeded = True
                        break

                    if config.jax.is_distributed:
                        grad = policy.reduce_parameters(grad)
                    self.policy_optimizer[uid] = self.policy_optimizer[uid].step(
                        grad, policy, self._learning_rate[uid]
                    )

                    cumulative_policy_loss += policy_loss.item()
                    if effective_entropy_loss_scale:
                        cumulative_entropy_loss += entropy_loss.item()
                    actual_batch_count += 1

                if kl_exceeded:
                    break

            if kl_exceeded:
                break

            # Shared value update on pooled data from all agents
            value_indices = np.random.permutation(pooled_buffer_size)
            for i in range(0, pooled_buffer_size, value_batch_size):
                idx = value_indices[i : i + value_batch_size]
                grad, value_loss = _update_value_huber(
                    value.act,
                    value.state_dict,
                    all_shared_states[idx],
                    all_values[idx],
                    all_returns[idx],
                    self._value_loss_scale[uid0],
                    self._clip_predicted_values[uid0],
                    self._value_clip[uid0],
                    self._use_huber_value_loss,
                    self._huber_delta,
                )

                if config.jax.is_distributed:
                    grad = value.reduce_parameters(grad)
                self.value_optimizer[uid0] = self.value_optimizer[uid0].step(
                    grad, value, self._learning_rate[uid0]
                )
                # Sync all value optimiser slots
                for uid in self.possible_agents:
                    self.value_optimizer[uid] = self.value_optimizer[uid0]

                cumulative_value_loss += value_loss.item()

        new_lr = self._apply_lr_decay(timestep, timesteps)
        self._log_training_stats(
            uid0,
            actual_batch_count,
            cumulative_policy_loss,
            cumulative_value_loss,
            cumulative_entropy_loss,
            effective_entropy_loss_scale,
            mean_entropy,
            new_lr,
            per_agent_tensors,
            cumulative_diag=cumulative_diag,
        )

    def _apply_lr_decay(self, timestep: int, timesteps: int) -> float:
        """Apply linear learning-rate decay and return the new LR."""
        uid0 = self.possible_agents[0]
        new_lr = float(self._learning_rate[uid0])

        if self.cfg.get("linear_lr_decay", False) and timesteps > 0:
            delay_frac = float(self.cfg.get("lr_decay_start_fraction", 0.3))
            min_lr_frac = float(self.cfg.get("min_lr_fraction", 0.1))

            decay_start = timesteps * delay_frac
            if timestep <= decay_start:
                new_lr = self._initial_learning_rate
            else:
                remaining = timesteps - decay_start
                elapsed_since_start = timestep - decay_start
                frac = max(1.0 - elapsed_since_start / remaining, min_lr_frac)
                new_lr = self._initial_learning_rate * frac

            for uid in self.possible_agents:
                self._learning_rate[uid] = new_lr

        return new_lr

    def _log_training_stats(
        self,
        uid0,
        actual_batch_count,
        cumulative_policy_loss,
        cumulative_value_loss,
        cumulative_entropy_loss,
        effective_entropy_loss_scale,
        mean_entropy,
        new_lr,
        per_agent_tensors,
        cumulative_diag: dict[str, float] | None = None,
    ) -> None:
        """Log training statistics to the experiment tracker."""
        n_batches = max(actual_batch_count, 1)

        for uid in self.possible_agents:
            self.track_data(
                f"Loss / Policy loss ({uid})",
                cumulative_policy_loss / n_batches,
            )
            self.track_data(
                f"Loss / Value loss ({uid})",
                cumulative_value_loss / n_batches,
            )
            if effective_entropy_loss_scale:
                self.track_data(
                    f"Loss / Entropy loss ({uid})",
                    cumulative_entropy_loss / n_batches,
                )
            self.track_data(f"Policy / Mean entropy ({uid})", mean_entropy.item())
            self.track_data(
                f"Learning / Learning rate ({uid})",
                self._learning_rate[uid0],
            )
        self.track_data("Learning / Learning rate", float(new_lr))
        self.track_data("Learning / Entropy scale", effective_entropy_loss_scale)

        if cumulative_diag:
            for key, total in cumulative_diag.items():
                self.track_data(f"Diagnostics / {key}", total / n_batches)
