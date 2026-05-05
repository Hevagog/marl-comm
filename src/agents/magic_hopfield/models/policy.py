"""MAGICHopfieldPolicyNet: MAGIC with per-agent Hopfield self-context.

Architecture
------------
1. Observation encoder  : obs → FC+tanh → obs_enc  (H-dim)
2. Message encoder      : obs_enc → FC → messages  (D-dim)
3. HopfieldSelfContext  : messages → prototype retrieval → messages_aug (D-dim)
4. Scheduler            : messages_aug → GAT+Gumbel-STE → adj (inter-agent only)
5. Message Processor    : messages_aug + adj → safe multi-head GAT → processed (D-dim)
6. Message decoder      : processed → FC+tanh → msg_decoded (H-dim)
7. Action head          : [obs_enc || msg_decoded] → MLP → logits

Key differences from base MAGIC
---------------------------------
- Self-loop replaced: no forced +eye(n) in adjacency.  Each agent's self-context
  comes from HopfieldSelfContext (learned coordination prototypes).
- Safe GAT: dead rows (zero incoming edges) yield zero vector, preserved by the
  residual connection in MessageProcessor.
- No HopfieldMsgPooling: output dim stays D (not 2D), so msg_decoder fan-in
  is unchanged vs base MAGIC.

References
----------
- Niu et al. 2021 "MAGIC" (AAMAS 2021)
- Ramsauer et al. 2021 "Hopfield Networks is All You Need" §3.3
- Bengio et al. 2015 "Scheduled Sampling"
"""

from __future__ import annotations

from typing import Any
from collections.abc import Sequence, Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp

from skrl.models.jax import CategoricalMixin, Model
from skrl.models.jax.categorical import _categorical

from agents.magic_hopfield.models.hopfield_msg_pooling import (
    HopfieldMsgPooling,
    HopfieldSelfContext,
)
from agents.magic_hopfield.models.utils import gumbel_softmax_scaled

_HIDDEN_GAIN = jnp.sqrt(2.0)
_OUTPUT_GAIN = 0.01


class _GATLayer(nn.Module):
    """Single-head Graph Attention (Velickovic et al. 2018, MAGIC Eq. 5/8)."""

    out_features: int
    negative_slope: float = 0.2
    use_mask: bool = False
    safe_mask: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        n = x.shape[0]
        Wh = nn.Dense(self.out_features, use_bias=False, name="W")(x)
        Wh_i = jnp.repeat(Wh[:, None, :], n, axis=1)
        Wh_j = jnp.repeat(Wh[None, :, :], n, axis=0)
        concat = jnp.concatenate([Wh_i, Wh_j], axis=-1)
        attn = nn.Dense(1, use_bias=False, name="a")(concat).squeeze(-1)
        attn = nn.leaky_relu(attn, negative_slope=self.negative_slope)

        if self.use_mask and mask is not None:
            attn = jnp.where(mask > 0, attn, jnp.finfo(jnp.float32).min)
            if self.safe_mask:
                # Dead rows: all mask==0 → softmax would NaN.
                # Replace with 0.0 → uniform softmax → mask zeroes it → zero vector.
                # MessageProcessor's residual then preserves the agent's own message.
                row_is_dead = jnp.all(mask == 0, axis=-1, keepdims=True)
                attn = jnp.where(row_is_dead, 0.0, attn)

        alpha = jax.nn.softmax(attn, axis=-1)

        if self.use_mask and mask is not None:
            alpha = alpha * mask

        return jnp.matmul(alpha, Wh)


class _MultiHeadGATLayer(nn.Module):
    out_features: int
    num_heads: int = 1
    negative_slope: float = 0.2
    use_mask: bool = False
    safe_mask: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        heads = [
            _GATLayer(
                out_features=self.out_features,
                negative_slope=self.negative_slope,
                use_mask=self.use_mask,
                safe_mask=self.safe_mask,
                name=f"head_{h}",
            )(x, mask)
            for h in range(self.num_heads)
        ]
        return nn.Dense(self.out_features, name="proj")(jnp.concatenate(heads, axis=-1))


class _SubSchedulerScaled(nn.Module):
    """Sub-scheduler with gumbel_scale controlling noise magnitude.

    gumbel_scale=1.0 → stochastic topology (rollout).
    gumbel_scale=0.0 → deterministic argmax (training).
    force_self_loop=False → pure inter-agent graph (no +eye(n)).
    """

    hidden_dim: int
    use_gat_encoder: bool = True
    temperature: float = 1.0
    force_self_loop: bool = True

    @nn.compact
    def __call__(
        self,
        messages: jax.Array,
        rng: jax.Array,
        gumbel_scale: jax.Array,
        temperature_override: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        n = messages.shape[0]

        if self.use_gat_encoder:
            node_feat = nn.elu(
                _GATLayer(out_features=self.hidden_dim, use_mask=False, name="gat_enc")(
                    messages
                )
            )
        else:
            node_feat = messages

        e_i = jnp.repeat(node_feat[:, None, :], n, axis=1)
        e_j = jnp.repeat(node_feat[None, :, :], n, axis=0)
        pair_feat = jnp.concatenate([e_i, e_j], axis=-1)

        h = nn.relu(nn.Dense(self.hidden_dim, name="mlp1")(pair_feat))
        logits = nn.Dense(2, name="mlp2")(h)  # (N, N, 2)

        adj = gumbel_softmax_scaled(
            logits,
            rng=rng,
            gumbel_scale=gumbel_scale,
            temperature=temperature_override,
            hard=True,
        )
        adj = adj[..., 1]  # take "edge present" channel

        if self.force_self_loop:
            adj = jnp.clip(adj + jnp.eye(n), 0.0, 1.0)
        else:
            adj = jnp.clip(adj, 0.0, 1.0)
        return adj, node_feat


class _SchedulerScaled(nn.Module):
    hidden_dim: int
    num_rounds: int = 1
    temperature: float = 1.0
    force_self_loop: bool = True

    @nn.compact
    def __call__(
        self,
        messages: jax.Array,
        rng: jax.Array,
        gumbel_scale: jax.Array,
        temperature_override: jax.Array,
    ) -> list[jax.Array]:
        adjs = []
        for round_idx in range(self.num_rounds):
            sub = _SubSchedulerScaled(
                hidden_dim=self.hidden_dim,
                use_gat_encoder=(round_idx == 0),
                temperature=self.temperature,
                force_self_loop=self.force_self_loop,
                name=f"sub_sched_{round_idx}",
            )
            rng, sub_rng = jax.random.split(rng)
            adj, _ = sub(messages, sub_rng, gumbel_scale, temperature_override)
            adjs.append(adj)
        return adjs


class _MessageProcessorLocal(nn.Module):
    """Message Processor (MAGIC §4.3, Eqs. 7–9)."""

    hidden_dim: int
    num_heads: int = 1
    num_rounds: int = 1
    safe_mask: bool = False

    @nn.compact
    def __call__(self, messages: jax.Array, adjs: jax.Array) -> jax.Array:
        m = messages
        for round_idx in range(self.num_rounds):
            m_new = nn.elu(
                _MultiHeadGATLayer(
                    out_features=self.hidden_dim,
                    num_heads=self.num_heads,
                    use_mask=True,
                    safe_mask=self.safe_mask,
                    name=f"sub_proc_{round_idx}",
                )(m, adjs[round_idx])
            )
            bias = self.param(
                f"bias_{round_idx}",
                nn.initializers.zeros,
                (self.hidden_dim,),
            )
            m = m_new + m + bias
        return m


class _CommBlockHopfield(nn.Module):
    """Original MAGIC+HopfieldMsgPooling comm block (kept for reference).

    Not used by MAGICHopfieldPolicyNet; retained so existing imports in
    magic_hopfield_mappo.py do not break if referenced.
    Output dim is 2D.
    """

    message_dim: int
    num_comm_rounds: int
    gumbel_temperature: float
    num_heads: int
    hopfield_num_queries: int
    hopfield_beta: float

    @nn.compact
    def __call__(
        self,
        msg_group: jax.Array,
        rng: jax.Array,
        gumbel_scale: jax.Array,
        temperature_override: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        adjs_raw = _SchedulerScaled(
            hidden_dim=self.message_dim,
            num_rounds=self.num_comm_rounds,
            temperature=self.gumbel_temperature,
            force_self_loop=True,
            name="scheduler",
        )(msg_group, rng, gumbel_scale, temperature_override)
        adjs = jnp.stack(adjs_raw, axis=0)

        processed = _MessageProcessorLocal(
            hidden_dim=self.message_dim,
            num_heads=self.num_heads,
            num_rounds=self.num_comm_rounds,
            safe_mask=False,
            name="msg_processor",
        )(msg_group, adjs)

        processed_3d = processed[None, :, :]
        augmented_3d = HopfieldMsgPooling(
            d_model=self.message_dim,
            num_queries=self.hopfield_num_queries,
            beta=self.hopfield_beta,
            name="hopfield_pool",
        )(processed_3d)
        return augmented_3d[0], adjs


class _CommBlockHopfieldSelf(nn.Module):
    """MAGIC comm block with Hopfield self-context replacing the forced self-loop.

    The Gumbel-Softmax scheduler produces a purely inter-agent adjacency
    (force_self_loop=False).  HopfieldSelfContext provides per-agent
    prototype-based self-context before the GAT runs.  The GAT uses a safe
    masked softmax so dead rows (no incoming edges) produce zero, preserved
    by the residual connection in MessageProcessor.

    Output dim is D (same as message_dim), not 2D.
    """

    message_dim: int
    num_comm_rounds: int
    gumbel_temperature: float
    num_heads: int
    hopfield_num_prototypes: int
    hopfield_beta: float
    hopfield_gate_init: float = 0.0

    @nn.compact
    def __call__(
        self,
        msg_group: jax.Array,  # (N, D)
        rng: jax.Array,
        gumbel_scale: jax.Array,
        temperature_override: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Returns processed (N, D), adjs (R, N, N)."""

        # Step 1: per-agent Hopfield self-context (replaces forced +eye(n))
        msg_aug = HopfieldSelfContext(
            d_model=self.message_dim,
            num_prototypes=self.hopfield_num_prototypes,
            beta=self.hopfield_beta,
            gate_init=self.hopfield_gate_init,
            name="hopfield_self",
        )(msg_group)  # (N, D)

        # Step 2: purely inter-agent Gumbel graph — no self-loops
        adjs_raw = _SchedulerScaled(
            hidden_dim=self.message_dim,
            num_rounds=self.num_comm_rounds,
            temperature=self.gumbel_temperature,
            force_self_loop=False,
            name="scheduler",
        )(msg_aug, rng, gumbel_scale, temperature_override)
        adjs = jnp.stack(adjs_raw, axis=0)  # (R, N, N)

        # Step 3: safe GAT — dead rows → zero → residual handles it
        processed = _MessageProcessorLocal(
            hidden_dim=self.message_dim,
            num_heads=self.num_heads,
            num_rounds=self.num_comm_rounds,
            safe_mask=True,
            name="msg_processor",
        )(msg_aug, adjs)  # (N, D)

        return processed, adjs


class MAGICHopfieldPolicyNet(CategoricalMixin, Model):
    """MAGIC policy with per-agent Hopfield self-context replacing the self-loop.

    Parameters
    ----------
    hopfield_num_prototypes : int
        K learned prototypes in HopfieldSelfContext (default 8).
    hopfield_beta : float
        Hopfield inverse temperature (default 1.0).
    """

    hidden_sizes: tuple = (128, 128)
    message_dim: int = 64
    num_comm_rounds: int = 1
    num_heads: int = 1
    gumbel_temperature: float = 1.0
    num_agents: int = 2
    hopfield_num_prototypes: int = 8
    hopfield_beta: float = 1.0
    hopfield_gate_init: float = 0.0

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
        hopfield_num_prototypes: int = 8,
        hopfield_beta: float = 1.0,
        hopfield_gate_init: float = 0.0,
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
        object.__setattr__(
            self, "hopfield_num_prototypes", int(hopfield_num_prototypes)
        )
        object.__setattr__(self, "hopfield_beta", float(hopfield_beta))
        object.__setattr__(self, "hopfield_gate_init", float(hopfield_gate_init))

    @nn.compact
    def __call__(
        self,
        inputs: Mapping[str, Any],
        role: str = "",
    ) -> tuple[jax.Array, dict]:
        x = inputs["states"]
        if x.ndim == 3:
            x = x.squeeze(1)
        n = self.num_agents
        b = x.shape[0]

        gumbel_rng = inputs.get("gumbel_rng", jax.random.PRNGKey(0))
        temperature_override = inputs.get("gumbel_temperature_override", None)
        _temp_ov = (
            jnp.asarray(temperature_override, dtype=jnp.float32)
            if temperature_override is not None
            else jnp.asarray(self.gumbel_temperature, dtype=jnp.float32)
        )

        training_mode = inputs.get("taken_actions", None) is not None
        _gumbel_scale = jnp.asarray(0.0 if training_mode else 1.0, dtype=jnp.float32)

        # Observation encoder
        obs_enc = nn.tanh(
            nn.Dense(
                self.hidden_sizes[0],
                kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
                bias_init=nn.initializers.constant(0.0),
                name="obs_encoder",
            )(x)
        )  # (B, H)

        # Message encoder
        messages = nn.Dense(
            self.message_dim,
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            bias_init=nn.initializers.constant(0.0),
            name="msg_encoder",
        )(obs_enc)  # (B, D)

        # encode_only mode: materialise comm_block params and return messages.
        if role == "encode_only":
            _dummy_msg = jnp.zeros((n, self.message_dim))
            _dummy_key = jax.random.PRNGKey(0)
            VmappedComm = nn.vmap(
                _CommBlockHopfieldSelf,
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
                hopfield_num_prototypes=self.hopfield_num_prototypes,
                hopfield_beta=self.hopfield_beta,
                hopfield_gate_init=self.hopfield_gate_init,
                name="comm_block",
            )(
                _dummy_msg[None, :, :],
                _dummy_key[None],
                jnp.ones((1,), dtype=jnp.float32),
                jnp.full((1,), _temp_ov),
            )
            _dummy_proc = jnp.zeros((b, self.message_dim))
            _dummy_dec = nn.tanh(
                nn.Dense(
                    self.hidden_sizes[0],
                    kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
                    bias_init=nn.initializers.constant(0.0),
                    name="msg_decoder",
                )(_dummy_proc)
            )
            _h = jnp.concatenate([obs_enc, _dummy_dec], axis=-1)
            for i, size in enumerate(self.hidden_sizes):
                _h = nn.tanh(
                    nn.Dense(
                        int(size),
                        kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
                        bias_init=nn.initializers.constant(0.0),
                        name=f"action_fc_{i}",
                    )(_h)
                )
            nn.Dense(
                int(self.num_actions),  # type: ignore[arg-type]
                kernel_init=nn.initializers.orthogonal(scale=_OUTPUT_GAIN),
                bias_init=nn.initializers.constant(0.0),
                name="action_logits",
            )(_h)
            return messages, {"obs_enc": obs_enc}

        # --- Communication block (vmapped over groups) ---
        external_messages = inputs.get("external_messages", None)
        R = self.num_comm_rounds

        if external_messages is not None:
            # Heterogeneous path: pre-computed processed messages (B, D).
            processed = external_messages
            groups = b // n if (b >= n and b % n == 0) else 1
            adj_matrices = jnp.zeros((R, groups, n, n))
            hard_adj = jnp.zeros((groups, n, n))
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
            # Touch comm_block params for init consistency.
            _dummy_msg = jnp.zeros((n, self.message_dim))
            _dummy_key = jax.random.PRNGKey(0)
            VmappedComm = nn.vmap(
                _CommBlockHopfieldSelf,
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
                hopfield_num_prototypes=self.hopfield_num_prototypes,
                hopfield_beta=self.hopfield_beta,
                hopfield_gate_init=self.hopfield_gate_init,
                name="comm_block",
            )(
                _dummy_msg[None, :, :],
                _dummy_key[None],
                jnp.ones((1,), dtype=jnp.float32),
                jnp.full((1,), _temp_ov),
            )
        else:
            comm_active = (b >= n) and (b % n == 0)
            groups = b // n if comm_active else 1

            if comm_active:
                msg_grouped = messages.reshape(groups, n, self.message_dim)
                group_keys = jax.random.split(gumbel_rng, groups)

                VmappedComm = nn.vmap(
                    _CommBlockHopfieldSelf,
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
                    hopfield_num_prototypes=self.hopfield_num_prototypes,
                    hopfield_beta=self.hopfield_beta,
                    hopfield_gate_init=self.hopfield_gate_init,
                    name="comm_block",
                )(
                    msg_grouped,
                    group_keys,
                    jnp.full((groups,), _gumbel_scale),
                    jnp.full((groups,), _temp_ov),
                )
                # processed_grouped: (groups, N, D)
                # adjs_grouped:      (groups, R, N, N)

                adj_matrices = jnp.transpose(jnp.asarray(adjs_grouped), (1, 0, 2, 3))
                hard_adj = (adj_matrices[-1] > 0.5).astype(jnp.float32)
                agg_messages = processed_grouped  # (groups, N, D)
                raw_msg = msg_grouped  # (groups, N, D)
                processed = processed_grouped.reshape(b, self.message_dim)
            else:
                # No-comm fallback: pass messages directly
                processed = messages  # (B, D)
                adj_matrices = jnp.zeros((R, 1, n, n))
                hard_adj = jnp.zeros((1, n, n))
                raw_msg = messages.reshape(1, b, self.message_dim)
                agg_messages = processed.reshape(1, b, self.message_dim)

        # Message decoder (D → H)
        msg_decoded = nn.tanh(
            nn.Dense(
                self.hidden_sizes[0],
                kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
                bias_init=nn.initializers.constant(0.0),
                name="msg_decoder",
            )(processed)
        )  # (B, H)

        # Action head
        h = jnp.concatenate([obs_enc, msg_decoded], axis=-1)  # (B, 2H)
        for i, size in enumerate(self.hidden_sizes):
            h = nn.tanh(
                nn.Dense(
                    int(size),
                    kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
                    bias_init=nn.initializers.constant(0.0),
                    name=f"action_fc_{i}",
                )(h)
            )

        logits = nn.Dense(
            int(self.num_actions),  # type: ignore[arg-type]
            kernel_init=nn.initializers.orthogonal(scale=_OUTPUT_GAIN),
            bias_init=nn.initializers.constant(0.0),
            name="action_logits",
        )(h)

        return logits, {
            "adj_matrices": adj_matrices,
            "hard_adj": hard_adj,
            "messages": raw_msg,
            "agg_messages": agg_messages,
        }

    def init_state_dict(self, role: str, inputs=None, key=None) -> None:
        if inputs is None:
            obs_sample = self.observation_space.sample()
            obs_batch = jnp.tile(
                jnp.asarray(obs_sample, dtype=jnp.float32)[None, :],
                (self.num_agents, 1),
            )
            inputs = {
                "states": obs_batch,
                "taken_actions": jnp.zeros((self.num_agents, 1)),
                "gumbel_rng": jax.random.PRNGKey(0),
            }
        super().init_state_dict(role, inputs, key)

    def act(
        self,
        inputs: Mapping[str, Any],
        role: str = "",
        params: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array | None, Mapping[str, Any]]:
        with jax.default_device(self.device):
            self._c_i += 1
            subkey = jax.random.fold_in(self._c_key, self._c_i)
            inputs = dict(inputs)
            inputs["key"] = subkey
            inputs["gumbel_rng"] = jax.random.fold_in(
                self._c_key, self._c_i + 1_000_000_000
            )

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

        outputs["net_output"] = net_output
        outputs["stddev"] = net_output
        return actions, log_prob, outputs

    def encode_messages(
        self,
        obs: jax.Array,
        gumbel_rng: jax.Array | None = None,
        params: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        with jax.default_device(self.device):
            self._c_i += 1
            subkey = jax.random.fold_in(self._c_key, self._c_i)
            gumbel_key = (
                gumbel_rng
                if gumbel_rng is not None
                else jax.random.fold_in(self._c_key, self._c_i + 1_000_000_000)
            )
        inputs = {"states": obs, "key": subkey, "gumbel_rng": gumbel_key}
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
