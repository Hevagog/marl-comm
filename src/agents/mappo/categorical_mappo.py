from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from skrl import config
from skrl.multi_agents.jax.base import MultiAgent
from skrl.multi_agents.jax.mappo import MAPPO
from skrl.multi_agents.jax.mappo.mappo import (
    _compute_gae,
    _update_value,
    compute_gae,
)
from skrl.resources.schedulers.jax import KLAdaptiveLR

from agents.mappo.adamw import AdamW


def _categorical_entropy(logits: jax.Array) -> jax.Array:
    """Compute categorical entropy directly from unnormalised logits.

    This avoids relying on the fragile ``outputs["stddev"]`` relay
    through ``CategoricalMixin`` and is mathematically equivalent to
    ``CategoricalMixin._entropy``.
    """
    log_probs = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    return -(jax.nn.softmax(logits, axis=-1) * log_probs).sum(axis=-1)


@functools.partial(
    jax.jit,
    static_argnames=("policy_act", "entropy_loss_scale"),
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
):
    """Like skrl's ``_update_policy`` but with two critical fixes:

    1. **Direct entropy from logits** — computes categorical entropy
       directly from ``outputs["net_output"]`` (the policy logits) instead
       of the fragile ``get_entropy(outputs["stddev"])`` relay.
    2. **Mini-batch advantage normalisation** — normalises advantages
       *within each mini-batch* (MAPPO paper, Section 5.1) rather than
       relying on the buffer-level normalisation in ``compute_gae``.
    """

    sampled_advantages = (sampled_advantages - sampled_advantages.mean()) / (
        sampled_advantages.std() + 1e-8
    )

    def _policy_loss(params):
        _, next_log_prob, outputs = policy_act(
            {"states": sampled_states, "taken_actions": sampled_actions},
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

        policy_loss = -jnp.minimum(surrogate, surrogate_clipped).mean()

        logits = outputs["net_output"]
        entropy = _categorical_entropy(logits)
        entropy_loss = jnp.float32(0.0)
        if entropy_loss_scale:
            entropy_loss = -entropy_loss_scale * entropy.mean()

        total_loss = policy_loss + entropy_loss

        return total_loss, (entropy_loss, kl_divergence, entropy.mean())

    (total_loss, (entropy_loss, kl_divergence, mean_entropy)), grad = (
        jax.value_and_grad(_policy_loss, has_aux=True)(policy_state_dict.params)
    )

    policy_loss = total_loss - entropy_loss
    return grad, policy_loss, entropy_loss, kl_divergence, mean_entropy


class CategoricalMAPPO(MAPPO):
    """MAPPO variant that correctly applies entropy regularisation for
    categorical (discrete-action) policies.

    Additional fixes over stock skrl MAPPO:
    - Direct categorical entropy from logits (no fragile stddev relay)
    - Mini-batch advantage normalisation (MAPPO paper §5.1)
    - Agent-ID one-hot appended to shared states for the value function
      (MAPPO paper §5.2 — allows the shared critic to distinguish agents)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Build a one-hot LUT: agent uid → one-hot vector (num_agents,).
        n = len(self.possible_agents)
        self._agent_onehot = {
            uid: jnp.array(
                [[1.0 if j == i else 0.0 for j in range(n)]],
                dtype=jnp.float32,
            )
            for i, uid in enumerate(self.possible_agents)
        }

        weight_decay = self.cfg.get("weight_decay", 0.0)
        if weight_decay > 0:

            for uid in self.possible_agents:
                policy = self.policies[uid]
                value = self.values[uid]
                if policy is not None and value is not None:
                    self.policy_optimizer[uid] = AdamW(
                        model=policy,
                        lr=self._learning_rate[uid],
                        weight_decay=weight_decay,
                        grad_norm_clip=self._grad_norm_clip[uid],
                        scale=not self._learning_rate_scheduler[uid],
                    )
                    self.value_optimizer[uid] = AdamW(
                        model=value,
                        lr=self._learning_rate[uid],
                        weight_decay=weight_decay,
                        grad_norm_clip=self._grad_norm_clip[uid],
                        scale=not self._learning_rate_scheduler[uid],
                    )
                    self.checkpoint_modules[uid]["policy_optimizer"] = (
                        self.policy_optimizer[uid]
                    )
                    self.checkpoint_modules[uid]["value_optimizer"] = (
                        self.value_optimizer[uid]
                    )

    @staticmethod
    def _append_agent_id(shared_states: jax.Array, one_hot: jax.Array) -> jax.Array:
        """Concatenate a one-hot agent-ID vector to shared states.

        ``one_hot`` has shape ``(1, num_agents)`` and is broadcast along the
        batch dimension of ``shared_states`` ``(B, state_dim)``.
        """
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

                # time-limit (truncation) bootstrapping
                if self._time_limit_bootstrap[uid]:
                    rewards[uid] += self._discount_factor[uid] * values * truncated[uid]

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

    def _update(self, timestep: int, timesteps: int) -> None:  # noqa: C901
        """Copy of ``MAPPO._update`` except it calls
        ``_update_policy_fixed`` instead of ``_update_policy``.
        """

        for uid in self.possible_agents:
            policy = self.policies[uid]
            value = self.values[uid]
            memory = self.memories[uid]

            # compute returns and advantages
            value.training = False
            bootstrap_shared = self._shared_state_preprocessor[uid](
                self._current_shared_next_states
            )
            bootstrap_shared = self._append_agent_id(
                bootstrap_shared, self._agent_onehot[uid]
            )
            last_values, _, _ = value.act(
                {"states": bootstrap_shared},
                role="value",
            )  # TODO: .float()
            value.training = True
            if not self._jax:  # numpy backend
                last_values = jax.device_get(last_values)
            last_values = self._value_preprocessor[uid](last_values, inverse=True)

            values = memory.get_tensor_by_name("values")
            returns, advantages = (_compute_gae if self._jax else compute_gae)(
                rewards=memory.get_tensor_by_name("rewards"),
                dones=memory.get_tensor_by_name("terminated")
                | memory.get_tensor_by_name("truncated"),
                values=values,
                next_values=last_values,
                discount_factor=self._discount_factor[uid],
                lambda_coefficient=self._lambda[uid],
            )

            memory.set_tensor_by_name(
                "values", self._value_preprocessor[uid](values, train=True)
            )
            memory.set_tensor_by_name(
                "returns", self._value_preprocessor[uid](returns, train=True)
            )
            memory.set_tensor_by_name("advantages", advantages)

            # sample mini-batches from memory
            sampled_batches = memory.sample_all(
                names=self._tensors_names, mini_batches=self._mini_batches[uid]
            )

            cumulative_policy_loss = 0
            cumulative_entropy_loss = 0
            cumulative_value_loss = 0

            # learning epochs
            for epoch in range(self._learning_epochs[uid]):
                kl_divergences = []

                # mini-batches loop
                for (
                    sampled_states,
                    sampled_shared_states,
                    sampled_actions,
                    sampled_log_prob,
                    sampled_values,
                    sampled_returns,
                    sampled_advantages,
                ) in sampled_batches:

                    sampled_states = self._state_preprocessor[uid](
                        sampled_states, train=not epoch
                    )
                    sampled_shared_states = self._shared_state_preprocessor[uid](
                        sampled_shared_states, train=not epoch
                    )
                    # Append one-hot agent ID to shared states (Change 8)
                    sampled_shared_states = self._append_agent_id(
                        sampled_shared_states, self._agent_onehot[uid]
                    )

                    # compute policy loss
                    grad, policy_loss, entropy_loss, kl_divergence, mean_entropy = (
                        _update_policy_fixed(
                            policy.act,
                            policy.state_dict,
                            sampled_states,
                            sampled_actions,
                            sampled_log_prob,
                            sampled_advantages,
                            self._ratio_clip[uid],
                            self._entropy_loss_scale[uid],
                        )
                    )

                    kl_divergences.append(kl_divergence.item())

                    # early stopping with KL divergence
                    if (
                        self._kl_threshold[uid]
                        and kl_divergence > self._kl_threshold[uid]
                    ):
                        break

                    # optimization step (policy)
                    if config.jax.is_distributed:
                        grad = policy.reduce_parameters(grad)
                    self.policy_optimizer[uid] = self.policy_optimizer[uid].step(
                        grad,
                        policy,
                        (
                            self._learning_rate[uid]
                            if self._learning_rate_scheduler[uid]
                            else None
                        ),
                    )

                    # compute value loss
                    grad, value_loss = _update_value(
                        value.act,
                        value.state_dict,
                        sampled_shared_states,
                        sampled_values,
                        sampled_returns,
                        self._value_loss_scale[uid],
                        self._clip_predicted_values[uid],
                        self._value_clip[uid],
                    )

                    # optimization step (value)
                    if config.jax.is_distributed:
                        grad = value.reduce_parameters(grad)
                    self.value_optimizer[uid] = self.value_optimizer[uid].step(
                        grad,
                        value,
                        (
                            self._learning_rate[uid]
                            if self._learning_rate_scheduler[uid]
                            else None
                        ),
                    )

                    # update cumulative losses
                    cumulative_policy_loss += policy_loss.item()
                    cumulative_value_loss += value_loss.item()
                    if self._entropy_loss_scale[uid]:
                        cumulative_entropy_loss += entropy_loss.item()

                # update learning rate
                if self._learning_rate_scheduler[uid]:
                    if self._learning_rate_scheduler[uid] is KLAdaptiveLR:
                        kl = np.mean(kl_divergences)
                        # reduce (collect from all workers/processes) KL in distributed runs
                        if config.jax.is_distributed:
                            kl = jax.pmap(
                                lambda x: jax.lax.psum(x, "i"), axis_name="i"
                            )(kl.reshape(1)).item()
                            kl /= config.jax.world_size
                        self._learning_rate[uid] = self.schedulers[uid](
                            timestep, self._learning_rate[uid], kl
                        )
                    else:
                        self._learning_rate[uid] *= self.schedulers[uid](timestep)

            # record data
            self.track_data(
                f"Loss / Policy loss ({uid})",
                cumulative_policy_loss
                / (self._learning_epochs[uid] * self._mini_batches[uid]),
            )
            self.track_data(
                f"Loss / Value loss ({uid})",
                cumulative_value_loss
                / (self._learning_epochs[uid] * self._mini_batches[uid]),
            )
            if self._entropy_loss_scale:
                self.track_data(
                    f"Loss / Entropy loss ({uid})",
                    cumulative_entropy_loss
                    / (self._learning_epochs[uid] * self._mini_batches[uid]),
                )

            self.track_data(f"Policy / Mean entropy ({uid})", mean_entropy.item())

            if self._learning_rate_scheduler[uid]:
                self.track_data(
                    f"Learning / Learning rate ({uid})", self._learning_rate[uid]
                )
