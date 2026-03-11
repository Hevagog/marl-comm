from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from skrl.models.jax import CategoricalMixin, Model

from agents.magic.models.comm_layers import MessageProcessor, Scheduler

# Orthogonal init gains (following MAPPO paper, Yu et al. 2021)
_HIDDEN_GAIN = jnp.sqrt(2.0)
_OUTPUT_GAIN = 0.01


class _CommunicateBlock(nn.Module):
    message_dim: int
    num_comm_rounds: int
    gumbel_temperature: float
    num_heads: int

    @nn.compact
    def __call__(self, msg_group: jax.Array) -> jax.Array:
        adjs = Scheduler(
            hidden_dim=self.message_dim,
            num_rounds=self.num_comm_rounds,
            temperature=self.gumbel_temperature,
            name="scheduler",
        )(msg_group, rng=None, hard=True)

        processed = MessageProcessor(
            hidden_dim=self.message_dim,
            num_heads=self.num_heads,
            num_rounds=self.num_comm_rounds,
            name="msg_processor",
        )(msg_group, adjs)
        return processed  # (N, message_dim)


class MAGICPolicyNet(CategoricalMixin, Model):
    """Categorical policy with MAGIC communication protocol.

    Architecture
    ------------
    1. **Observation encoder** ``e(o_i)`` — FC + tanh  (MAGIC §4.1, Eq. 3)
    2. **Message encoder** ``e_m(·)`` — FC → message ``m_i^{t(0)}``
    3. **Scheduler** — GAT encoder + Gumbel-Softmax hard attention  (§4.2)
    4. **Message Processor** — multi-head GAT with dynamic graphs  (§4.3)
    5. **Message decoder** ``e'_m(·)`` — FC  (Algorithm 1, Step 14)
    6. **Action head** — hidden layers + logit projection

    The processed message is concatenated with the agent's encoded observation
    before the action head, giving the policy access to both local information
    and communicated information.
    """

    hidden_sizes: tuple = (128, 128)
    message_dim: int = 64
    num_comm_rounds: int = 1
    num_heads: int = 1
    gumbel_temperature: float = 1.0
    num_agents: int = 2

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes: Sequence[int] = (128, 128),
        message_dim: int = 64,
        num_comm_rounds: int = 1,
        num_heads: int = 1,
        gumbel_temperature: float = 1.0,
        num_agents: int = 2,
        unnormalized_log_prob: bool = True,
        device=None,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        CategoricalMixin.__init__(self, unnormalized_log_prob)
        object.__setattr__(self, "hidden_sizes", tuple(int(h) for h in hidden_sizes))
        object.__setattr__(self, "message_dim", int(message_dim))
        object.__setattr__(self, "num_comm_rounds", int(num_comm_rounds))
        object.__setattr__(self, "num_heads", int(num_heads))
        object.__setattr__(self, "gumbel_temperature", float(gumbel_temperature))
        object.__setattr__(self, "num_agents", int(num_agents))

    @nn.compact
    def __call__(
        self,
        inputs: Mapping[str, Any],
        role: str = "",
    ):
        """
        ``inputs["states"]`` has shape ``(B, obs_dim)`` where B may be a
        multiple of ``num_agents`` (stacked agent observations).  For the
        communication protocol we reshape to ``(B // N, N, obs_dim)`` so each
        group of N consecutive rows represents one timestep's observations
        for all agents.
        """
        x = inputs["states"]  # (B, obs_dim)
        n = self.num_agents
        b = x.shape[0]

        #  Observation encoder  (MAGIC §4.1, Eq. 3 — FC part) ---
        obs_enc = nn.Dense(
            self.hidden_sizes[0],
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            bias_init=nn.initializers.constant(0.0),
            name="obs_encoder",
        )(x)
        obs_enc = nn.tanh(obs_enc)  # (B, H)

        #  Message encoder  e_m(·)  (MAGIC §4.1) ---
        messages = nn.Dense(
            self.message_dim,
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            bias_init=nn.initializers.constant(0.0),
            name="msg_encoder",
        )(obs_enc)  # (B, message_dim)

        # Reshape for communication: (B//N, N, message_dim) ---
        # If B is not divisible by N, fall back to no communication
        if b >= n and b % n == 0:
            groups = b // n
            msg_grouped = messages.reshape(groups, n, self.message_dim)

            # Vectorize communication over groups (shared params)
            VmappedComm = nn.vmap(
                _CommunicateBlock,
                variable_axes={"params": None},
                split_rngs={"params": False},
                in_axes=0,
                out_axes=0,
            )
            processed = VmappedComm(
                message_dim=self.message_dim,
                num_comm_rounds=self.num_comm_rounds,
                gumbel_temperature=self.gumbel_temperature,
                num_heads=self.num_heads,
                name="comm_block",
            )(msg_grouped)  # (groups, N, message_dim)
            processed = processed.reshape(b, self.message_dim)
        else:
            # Fallback: no communication (single agent or misaligned batch)
            processed = messages

        #  Message decoder  e'_m(·)  (MAGIC Algorithm 1, Step 14) ---
        msg_decoded = nn.Dense(
            self.hidden_sizes[0],
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            bias_init=nn.initializers.constant(0.0),
            name="msg_decoder",
        )(processed)
        msg_decoded = nn.tanh(msg_decoded)  # (B, H)

        # Combine obs encoding + processed message → action head ---
        combined = jnp.concatenate([obs_enc, msg_decoded], axis=-1)  # (B, 2H)

        h = combined
        for i, size in enumerate(self.hidden_sizes):
            h = nn.Dense(
                int(size),
                kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
                bias_init=nn.initializers.constant(0.0),
                name=f"action_fc_{i}",
            )(h)
            h = nn.tanh(h)

        logits = nn.Dense(
            int(self.num_actions),
            kernel_init=nn.initializers.orthogonal(scale=_OUTPUT_GAIN),
            bias_init=nn.initializers.constant(0.0),
            name="action_logits",
        )(h)

        return logits, {}

    def init_state_dict(
        self,
        role: str,
        inputs=None,
        key=None,
    ) -> None:
        """Override to ensure batch size = num_agents so communication params init."""
        if inputs is None:
            obs_sample = self.observation_space.sample()
            obs_batch = jnp.tile(
                jnp.asarray(obs_sample, dtype=jnp.float32)[None, :],
                (self.num_agents, 1),
            )  # (N, obs_dim)
            inputs = {
                "states": obs_batch,
                "taken_actions": jnp.zeros((self.num_agents, 1)),
            }
        super().init_state_dict(role, inputs, key)

    def act(
        self,
        inputs: Mapping[str, Union[Union[np.ndarray, jax.Array], Any]],
        role: str = "",
        params: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, Union[jax.Array, None], Mapping[str, Union[jax.Array, Any]]]:
        actions, log_prob, outputs = super().act(inputs, role, params)
        # Store logits for entropy computation (same pattern as MAPPO)
        outputs["stddev"] = outputs["net_output"]
        return actions, log_prob, outputs

    @property
    def _modules(self):
        return {}
