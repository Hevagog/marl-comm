from __future__ import annotations

from typing import Any
from collections.abc import Sequence, Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from skrl.models.jax import CategoricalMixin, Model
from skrl.models.jax.categorical import _categorical

from agents.magic.models.comm_layers import MessageProcessor, Scheduler

# Orthogonal init gains (following MAPPO paper, Yu et al. 2021)
_HIDDEN_GAIN = jnp.sqrt(2.0)
_OUTPUT_GAIN = 0.01


class _CommunicateBlock(nn.Module):
    """One Scheduler + MessageProcessor pass for a group of N agents.

    Returns
    -------
    processed : jax.Array, shape (N, message_dim)
        Aggregated messages after the GAT Message Processor.
    adjs : jax.Array, shape (num_comm_rounds, N, N)
        Soft Gumbel-Softmax adjacency matrices produced by the Scheduler,
        one per communication round.  Exposed for analysis.
    """

    message_dim: int
    num_comm_rounds: int
    gumbel_temperature: float
    num_heads: int

    @nn.compact
    def __call__(
        self,
        msg_group: jax.Array,  # (N, message_dim)
        rng: jax.Array | None = None,
        temperature_override: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:  # (processed, adjs)
        adjs_raw = Scheduler(
            hidden_dim=self.message_dim,
            num_rounds=self.num_comm_rounds,
            temperature=self.gumbel_temperature,
            name="scheduler",
        )(msg_group, rng=rng, hard=True, temperature_override=temperature_override)

        # Scheduler may return either:
        #   (a) a stacked JAX array of shape (num_rounds, N, N), or
        #   (b) a Python list of R arrays each (N, N).
        # Normalise to (num_rounds, N, N) so vmap can lift it correctly.
        if isinstance(adjs_raw, (list, tuple)):
            adjs = jnp.stack(adjs_raw, axis=0)  # list[R × (N,N)] → (R, N, N)
        else:
            adjs = jnp.asarray(adjs_raw)  # already an array — ensure JAX type

        processed = MessageProcessor(
            hidden_dim=self.message_dim,
            num_heads=self.num_heads,
            num_rounds=self.num_comm_rounds,
            name="msg_processor",
        )(msg_group, adjs)  # (N, message_dim)

        return processed, adjs  # (N, msg_dim), (R, N, N)


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

    Heterogeneous-agent communication
    ----------------------------------
    When agents have different observation shapes, ``__call__`` can be invoked
    with ``inputs["external_messages"]`` (shape ``(B, message_dim)``).  In
    this mode the obs/msg encoders and the communication block are bypassed:
    only the message decoder + action head are applied.  The caller (e.g.
    ``MAGICMAPPO.act``) is responsible for collecting messages from all
    agents, running the Scheduler+MessageProcessor externally (using the first
    agent's communication block), and distributing the aggregated messages.

    Communication tensors forwarded to ``act()`` outputs
    ----------------------------------------------------
    ``adj_matrices``  shape ``(num_comm_rounds, groups, N, N)``
        Soft Gumbel-Softmax adjacency weights from the Scheduler.
    ``hard_adj``      shape ``(groups, N, N)``
        Binary threshold of the last-round adjacency.
    ``messages``      shape ``(groups, N, message_dim)``
        Pre-aggregation message embeddings (output of the message encoder).
    ``agg_messages``  shape ``(groups, N, message_dim)``
        Post-aggregation embeddings (output of the Message Processor).
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
    ) -> tuple[jax.Array, dict]:
        """
        Parameters
        ----------
        inputs["states"] : (B, obs_dim) or (B, 1, obs_dim)
            B must be a multiple of ``num_agents`` for communication to
            activate (in the standard homogeneous case).
        inputs["gumbel_rng"] : jax.Array, optional
            PRNG key for Gumbel-Softmax topology exploration.  Injected by
            ``act()``; falls back to a fixed seed during init.
        inputs["external_messages"] : (B, message_dim), optional
            Pre-aggregated messages from the shared communication block.
            When present, the obs/message encoders and the communication block
            are still used (obs_enc is always computed) but the message
            encoder output is replaced by these external messages.  The comm
            block is skipped entirely.  Used in the heterogeneous-agent path
            where ``MAGICMAPPO.act`` orchestrates cross-agent communication.
        role == "encode_only" : special mode
            When ``role`` is set to ``"encode_only"``, only run the obs
            encoder + message encoder and return
            ``(messages, {"obs_enc": obs_enc})``. Used by MAGICMAPPO.act to
            collect per-agent messages before running the comm block.

        Returns
        -------
        logits : (B, num_actions)
        outputs : dict with keys
            ``adj_matrices``  (num_comm_rounds, groups, N, N) — or zeros if no comm
            ``hard_adj``      (groups, N, N)
            ``messages``      (groups, N, message_dim)  raw message embeddings
            ``agg_messages``  (groups, N, message_dim)  post-GAT embeddings
        """
        x = inputs["states"]
        # skrl RandomMemory may preserve a spurious num_envs=1 dimension.
        if x.ndim == 3:
            x = x.squeeze(1)  # (B, 1, obs_dim) → (B, obs_dim)
        n = self.num_agents
        b = x.shape[0]

        gumbel_rng = inputs.get("gumbel_rng", jax.random.PRNGKey(0))
        temperature_override = inputs.get("gumbel_temperature_override", None)
        _temp_ov = (
            jnp.asarray(temperature_override, dtype=jnp.float32)
            if temperature_override is not None
            else jnp.asarray(self.gumbel_temperature, dtype=jnp.float32)
        )

        # Observation encoder
        obs_enc = nn.Dense(
            self.hidden_sizes[0],
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            bias_init=nn.initializers.constant(0.0),
            name="obs_encoder",
        )(x)
        obs_enc = nn.tanh(obs_enc)  # (B, H)

        # Message encoder
        messages = nn.Dense(
            self.message_dim,
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            bias_init=nn.initializers.constant(0.0),
            name="msg_encoder",
        )(obs_enc)  # (B, message_dim)

        # encode_only mode: return messages without running the comm block.
        # IMPORTANT: key off `role` (a static Python string), not an entry in
        # `inputs`, to avoid boolean conversion of traced arrays under JIT.
        # Used in heterogeneous MAGIC act() to collect per-agent messages.
        if role == "encode_only":
            # We still need to materialise the comm_block parameters so that
            # Flax can initialise them on the first call.  Run a dummy path.
            _dummy_msg = jnp.zeros((n, self.message_dim))
            _dummy_key = jax.random.PRNGKey(0)
            # Touch the comm_block parameters.
            VmappedComm = nn.vmap(
                _CommunicateBlock,
                variable_axes={"params": None},
                split_rngs={"params": False},
                in_axes=0,
                out_axes=0,
            )
            VmappedComm(
                message_dim=self.message_dim,
                num_comm_rounds=self.num_comm_rounds,
                gumbel_temperature=self.gumbel_temperature,
                num_heads=self.num_heads,
                name="comm_block",
            )(_dummy_msg[None, :, :], _dummy_key[None], jnp.full((1,), _temp_ov))
            # Also touch msg_decoder and action_fc layers.
            _dummy_proc = jnp.zeros((b, self.message_dim))
            _dummy_dec = nn.Dense(
                self.hidden_sizes[0],
                kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
                bias_init=nn.initializers.constant(0.0),
                name="msg_decoder",
            )(_dummy_proc)
            _dummy_combined = jnp.concatenate([obs_enc, _dummy_dec], axis=-1)
            _h = _dummy_combined
            for i, size in enumerate(self.hidden_sizes):
                _h = nn.Dense(
                    int(size),
                    kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
                    bias_init=nn.initializers.constant(0.0),
                    name=f"action_fc_{i}",
                )(_h)
            nn.Dense(
                int(self.num_actions),
                kernel_init=nn.initializers.orthogonal(scale=_OUTPUT_GAIN),
                bias_init=nn.initializers.constant(0.0),
                name="action_logits",
            )(_h)
            # Return the real message embeddings.
            return messages, {"obs_enc": obs_enc}

        # Scheduler + Message Processor (vmapped over groups)
        external_messages = inputs.get("external_messages", None)
        R = self.num_comm_rounds

        if external_messages is not None:
            # Heterogeneous path: use externally-computed aggregated messages.
            # The comm block is still initialised below via a dummy run.
            processed = external_messages  # (B, message_dim)
            groups = b // n if (b >= n and b % n == 0) else 1
            adj_matrices = jnp.zeros((R, groups, n, n))
            hard_adj = jnp.zeros((groups, n, n))
            # Reshape messages for output tensors
            msg_b = min(b, n * groups)
            raw_msg = (
                messages[:msg_b].reshape(groups, -1, self.message_dim)
                if msg_b >= n
                else messages.reshape(1, b, self.message_dim)
            )
            agg_messages = (
                processed.reshape(groups, -1, self.message_dim)
                if msg_b >= n
                else processed.reshape(1, b, self.message_dim)
            )
            # Touch comm_block params so they get initialised.
            _dummy_msg = jnp.zeros((n, self.message_dim))
            _dummy_key = jax.random.PRNGKey(0)
            VmappedComm = nn.vmap(
                _CommunicateBlock,
                variable_axes={"params": None},
                split_rngs={"params": False},
                in_axes=0,
                out_axes=0,
            )
            VmappedComm(
                message_dim=self.message_dim,
                num_comm_rounds=self.num_comm_rounds,
                gumbel_temperature=self.gumbel_temperature,
                num_heads=self.num_heads,
                name="comm_block",
            )(_dummy_msg[None, :, :], _dummy_key[None], jnp.full((1,), _temp_ov))
        else:
            comm_active = (b >= n) and (b % n == 0)
            groups = b // n if comm_active else 1

            if comm_active:
                msg_grouped = messages.reshape(groups, n, self.message_dim)
                # (groups, N, message_dim)

                group_keys = jax.random.split(gumbel_rng, groups)

                # vmap _CommunicateBlock over the groups axis.
                # Returns (processed, adjs):
                #   processed : (groups, N, message_dim)
                #   adjs      : (groups, num_rounds, N, N)
                VmappedComm = nn.vmap(
                    _CommunicateBlock,
                    variable_axes={"params": None},
                    split_rngs={"params": False},
                    in_axes=0,
                    out_axes=0,
                )
                processed_grouped, adjs_grouped = VmappedComm(
                    message_dim=self.message_dim,
                    num_comm_rounds=R,
                    gumbel_temperature=self.gumbel_temperature,
                    num_heads=self.num_heads,
                    name="comm_block",
                )(msg_grouped, group_keys, jnp.full((groups,), _temp_ov))
                # processed_grouped : (groups, N, message_dim)
                # adjs_grouped      : (groups, num_rounds, N, N)

                # Reorder adjs to (num_rounds, groups, N, N) for the analysis layer.
                adj_matrices = jnp.transpose(jnp.asarray(adjs_grouped), (1, 0, 2, 3))
                # (num_rounds, groups, N, N)

                hard_adj = (adj_matrices[-1] > 0.5).astype(jnp.float32)
                # (groups, N, N)

                agg_messages = processed_grouped  # (groups, N, message_dim)
                raw_msg = msg_grouped  # (groups, N, message_dim)

                processed = processed_grouped.reshape(b, self.message_dim)
                # (B, message_dim) — flattened back for the action head
            else:
                # No communication: fall back to identity pass-through.
                processed = messages
                adj_matrices = jnp.zeros((R, 1, n, n))
                hard_adj = jnp.zeros((1, n, n))
                raw_msg = messages.reshape(1, b, self.message_dim)
                agg_messages = messages.reshape(1, b, self.message_dim)

        # Message decoder
        msg_decoded = nn.Dense(
            self.hidden_sizes[0],
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            bias_init=nn.initializers.constant(0.0),
            name="msg_decoder",
        )(processed)
        msg_decoded = nn.tanh(msg_decoded)  # (B, H)

        # Action head
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
        )(h)  # (B, num_actions)

        # Pack all communication tensors into the outputs dict so that
        # act() can forward them to callers (e.g. the analysis collector).
        comm_outputs = {
            "adj_matrices": adj_matrices,  # (R, groups, N, N)
            "hard_adj": hard_adj,  # (groups, N, N)
            "messages": raw_msg,  # (groups, N, message_dim)
            "agg_messages": agg_messages,  # (groups, N, message_dim)
        }

        return logits, comm_outputs

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
                "gumbel_rng": jax.random.PRNGKey(0),
            }
        super().init_state_dict(role, inputs, key)

    def act(
        self,
        inputs: Mapping[str, np.ndarray | jax.Array | Any],
        role: str = "",
        params: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array | None, Mapping[str, jax.Array | Any]]:
        """Override CategoricalMixin.act to inject Gumbel RNG and forward
        communication tensors to callers.

        All communication outputs (``adj_matrices``, ``hard_adj``,
        ``messages``, ``agg_messages``) are returned in the third element of
        the tuple and consumed by ``MAGICCommCollector`` for analysis.
        """
        with jax.default_device(self.device):
            self._c_i += 1
            subkey = jax.random.fold_in(self._c_key, self._c_i)
            inputs["key"] = subkey
            # Separate key for Gumbel-Softmax to avoid correlation with the
            # action-sampling key.  Large offset prevents accidental collision.
            inputs["gumbel_rng"] = jax.random.fold_in(
                self._c_key, self._c_i + 1_000_000_000
            )

        # __call__ now returns (logits, comm_outputs_dict)
        net_output, outputs = self.apply(
            self.state_dict.params if params is None else params,
            inputs,
            role,
        )

        actions, log_prob = _categorical(
            net_output,
            self._c_unnormalized_log_prob,
            inputs.get("taken_actions", None),
            subkey,
        )

        # Always-present entries used by the base MAPPO training code.
        outputs["net_output"] = net_output
        outputs["stddev"] = net_output  # entropy relay expected by CategoricalMixin

        # adj_matrices, hard_adj, messages, agg_messages are already in
        # `outputs` — populated by __call__ above.  Nothing more to add.

        return actions, log_prob, outputs

    def encode_messages(
        self,
        obs: jax.Array,
        gumbel_rng: jax.Array | None = None,
        params: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Encode observations to message embeddings without running the comm block.

        Returns
        -------
        messages : (B, message_dim)
        obs_enc : (B, hidden_sizes[0])
        """
        with jax.default_device(self.device):
            self._c_i += 1
            subkey = jax.random.fold_in(self._c_key, self._c_i)
            gumbel_key = (
                gumbel_rng
                if gumbel_rng is not None
                else jax.random.fold_in(self._c_key, self._c_i + 1_000_000_000)
            )

        inputs = {
            "states": obs,
            "key": subkey,
            "gumbel_rng": gumbel_key,
        }
        messages, extra = self.apply(
            self.state_dict.params if params is None else params,
            inputs,
            "encode_only",
        )
        return messages, extra["obs_enc"]

    def act_with_messages(
        self,
        obs: jax.Array,
        agg_messages: jax.Array,
        adj_matrices: jax.Array,
        hard_adj: jax.Array,
        raw_messages: jax.Array,
        params: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, dict]:
        """Generate actions using pre-computed aggregated messages.

        Parameters
        ----------
        obs : (B, obs_dim) — this agent's observations.
        agg_messages : (B, message_dim) — output of the shared comm block.
        adj_matrices, hard_adj, raw_messages : comm tensors for logging.

        Returns
        -------
        actions, log_prob, outputs (same format as ``act``).
        """
        with jax.default_device(self.device):
            self._c_i += 1
            subkey = jax.random.fold_in(self._c_key, self._c_i)
            gumbel_key = jax.random.fold_in(self._c_key, self._c_i + 1_000_000_000)

        inputs = {
            "states": obs,
            "key": subkey,
            "gumbel_rng": gumbel_key,
            "external_messages": agg_messages,
        }

        net_output, outputs = self.apply(
            self.state_dict.params if params is None else params,
            inputs,
            "policy",
        )

        # Override the placeholder comm tensors with the real ones from the
        # shared comm block.
        outputs["adj_matrices"] = adj_matrices
        outputs["hard_adj"] = hard_adj
        outputs["messages"] = raw_messages
        outputs["net_output"] = net_output
        outputs["stddev"] = net_output

        actions, log_prob = _categorical(
            net_output,
            self._c_unnormalized_log_prob,
            None,
            subkey,
        )
        return actions, log_prob, outputs

    @property
    def _modules(self):
        return {}
