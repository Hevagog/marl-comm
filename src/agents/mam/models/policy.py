"""MAM policy network: BiMamba encoder + Mamba/CrossMamba decoder.

Implements the Multi-Agent Mamba architecture (Daniel et al. 2024):
- Encoder: observation embedding → BiMamba blocks → observation representation
- Decoder: shifted-action embedding → Mamba self-attn + CrossMamba → action logits
- Parallel mode (training): teacher-forced evaluation of all agents simultaneously
- Autoregressive mode (rollout): sequential per-agent action generation

References
----------
- Daniel et al. 2024, §3: MAM architecture (Encoder §3.1, Decoder §3.2)
- Wen et al. 2022: MAT encoder-decoder framework
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import orthogonal

from skrl.models.jax import CategoricalMixin, Model

from agents.shared.mamba_blocks import BiMamba, CrossMamba, FIFOBuffer, Mamba

_HIDDEN_GAIN = jnp.sqrt(2.0)
_OUTPUT_GAIN = 0.01


class EncodeBlock(nn.Module):
    """Pre-norm BiMamba + residual → pre-norm MLP + residual."""

    n_embd: int
    n_agent: int
    d_state: int
    d_conv: int
    delta_rank: int

    def setup(self) -> None:
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.bimamba = BiMamba(
            self.n_agent,
            self.n_embd,
            self.d_state,
            self.d_conv,
            self.delta_rank,
        )
        self.mlp = nn.Sequential(
            [
                nn.Dense(self.n_embd, kernel_init=orthogonal(_HIDDEN_GAIN)),
                nn.gelu,
                nn.Dense(self.n_embd, kernel_init=orthogonal(_OUTPUT_GAIN)),
            ]
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.bimamba(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Encoder(nn.Module):
    """BiMamba encoder: obs → obs_rep."""

    obs_dim: int
    action_dim: int
    n_block: int
    n_embd: int
    n_agent: int
    d_state: int
    d_conv: int
    delta_rank: int

    def setup(self) -> None:
        self.obs_encoder = nn.Sequential(
            [
                nn.Dense(self.n_embd, kernel_init=orthogonal(_HIDDEN_GAIN)),
                nn.gelu,
            ]
        )
        self.ln = nn.LayerNorm()
        self.blocks = nn.Sequential(
            [
                EncodeBlock(
                    self.n_embd,
                    self.n_agent,
                    self.d_state,
                    self.d_conv,
                    self.delta_rank,
                )
                for _ in range(self.n_block)
            ]
        )

    def __call__(self, obs: jax.Array) -> jax.Array:
        """obs: (batch, n_agent, obs_dim) → obs_rep: (batch, n_agent, n_embd)."""
        emb = self.obs_encoder(obs)
        rep = self.blocks(self.ln(emb))
        return rep


class DecodeBlock(nn.Module):
    """Pre-norm Mamba self → CrossMamba → MLP (with residual connections)."""

    n_embd: int
    n_agent: int
    d_state: int
    d_conv: int
    delta_rank: int

    def setup(self) -> None:
        kw = dict(
            num_agents=self.n_agent,
            d_model=self.n_embd,
            d_state=self.d_state,
            d_conv=self.d_conv,
            delta_rank=self.delta_rank,
        )
        self.mamba_self = Mamba(**kw)
        self.mamba_cross = CrossMamba(**kw)
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.ln3 = nn.LayerNorm()
        self.mlp = nn.Sequential(
            [
                nn.Dense(self.n_embd, kernel_init=orthogonal(_HIDDEN_GAIN)),
                nn.gelu,
                nn.Dense(self.n_embd, kernel_init=orthogonal(_OUTPUT_GAIN)),
            ]
        )

    def __call__(self, x: jax.Array, obs_rep: jax.Array) -> jax.Array:
        """
        x       : (batch, n_agent, n_embd)  — action embeddings
        obs_rep : (batch, n_agent, n_embd)  — encoder output
        """
        x = x + self.mamba_self(self.ln1(x))
        x = x + self.mamba_cross((self.ln2(x), obs_rep))
        x = x + self.mlp(self.ln3(x))
        return x

    def recurrent(
        self,
        x: jax.Array,
        obs_rep: jax.Array,
        self_hs: jax.Array,
        self_buf: jax.Array,
        cross_hs: jax.Array,
        cross_buf: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Single-step recurrent forward.

        Returns (x, self_hs, self_buf, cross_hs, cross_buf).
        """
        x_new, self_hs, self_buf = self.mamba_self.recurrent(
            self.ln1(x), self_hs, self_buf
        )
        x = x + x_new
        x_cross, cross_hs, cross_buf = self.mamba_cross.recurrent(
            (self.ln2(x), obs_rep),
            cross_hs,
            cross_buf,
        )
        x = x + x_cross
        x = x + self.mlp(self.ln3(x))
        return x, self_hs, self_buf, cross_hs, cross_buf


class Decoder(nn.Module):
    """Mamba decoder with parallel and recurrent modes."""

    obs_dim: int
    action_dim: int
    n_block: int
    n_embd: int
    n_agent: int
    d_state: int
    d_conv: int
    delta_rank: int

    def setup(self) -> None:
        self.action_encoder = nn.Sequential(
            [
                nn.Dense(
                    self.n_embd, use_bias=False, kernel_init=orthogonal(_HIDDEN_GAIN)
                ),
                nn.gelu,
            ]
        )
        self.ln = nn.LayerNorm()
        self.blocks = [
            DecodeBlock(
                self.n_embd,
                self.n_agent,
                self.d_state,
                self.d_conv,
                self.delta_rank,
                name=f"decode_block_{i}",
            )
            for i in range(self.n_block)
        ]
        self.head = nn.Sequential(
            [
                nn.Dense(self.n_embd, kernel_init=orthogonal(_HIDDEN_GAIN)),
                nn.gelu,
                nn.LayerNorm(),
                nn.Dense(self.action_dim, kernel_init=orthogonal(_OUTPUT_GAIN)),
            ]
        )

    def __call__(
        self, shifted_action: jax.Array, obs_rep: jax.Array, obs: jax.Array
    ) -> jax.Array:
        """Parallel (teacher-forced) forward.

        shifted_action : (batch, n_agent, action_dim+1)
        obs_rep        : (batch, n_agent, n_embd)
        Returns logits : (batch, n_agent, action_dim)
        """
        x = self.ln(self.action_encoder(shifted_action))
        for block in self.blocks:
            x = block(x, obs_rep)
        return self.head(x)

    def recurrent_step(
        self,
        shifted_action: jax.Array,
        obs_rep: jax.Array,
        self_hs_all: jax.Array,
        self_buf_all: jax.Array,
        cross_hs_all: jax.Array,
        cross_buf_all: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Single-agent recurrent decode step.

        shifted_action : (batch, 1, action_dim+1)
        obs_rep        : (batch, 1, n_embd)
        *_hs_all       : (batch, n_block, 1, d_inner, d_state)
        *_buf_all      : (batch, n_block, d_conv, d_inner)

        Returns (logits, self_hs_all, self_buf_all, cross_hs_all, cross_buf_all)
        logits : (batch, 1, action_dim)
        """
        x = self.ln(self.action_encoder(shifted_action))

        for i, block in enumerate(self.blocks):
            s_hs = self_hs_all[:, i, :, :, :]
            s_buf = self_buf_all[:, i, :, :]
            c_hs = cross_hs_all[:, i, :, :, :]
            c_buf = cross_buf_all[:, i, :, :]

            x, s_hs_new, s_buf_new, c_hs_new, c_buf_new = block.recurrent(
                x,
                obs_rep,
                s_hs,
                s_buf,
                c_hs,
                c_buf,
            )

            self_hs_all = self_hs_all.at[:, i, :, :, :].set(s_hs_new)
            self_buf_all = self_buf_all.at[:, i, :, :].set(s_buf_new)
            cross_hs_all = cross_hs_all.at[:, i, :, :, :].set(c_hs_new)
            cross_buf_all = cross_buf_all.at[:, i, :, :].set(c_buf_new)

        logits = self.head(x)
        return logits, self_hs_all, self_buf_all, cross_hs_all, cross_buf_all


class MAMPolicyNet(CategoricalMixin, Model):
    """Multi-Agent Mamba policy network.

    Wraps the MAM Encoder + Decoder into a skrl-compatible Model with
    CategoricalMixin for discrete action spaces.

    Two modes of operation:
    - **Parallel** (training): ``taken_actions`` present in inputs → teacher forcing.
    - **Autoregressive** (rollout): ``ar_key`` present → sequential per-agent decode.
    - **Fallback**: zero start tokens → independent per-agent logits.

    Ablation 3 parameters
    ---------------------
    sort_agents_by_type : bool
        When True, agents are reordered by type (agent_idx % type_cycle_len) before
        being passed to the encoder and decoder.  This groups same-type agents
        consecutively so the AR chain conditions on within-type ordering first.
        Logits are unsorting back to original agent order before returning.
    type_cycle_len : int
        Number of distinct agent types.  Agent i has type ``i % type_cycle_len``.
    """

    n_embd: int = 128
    n_block: int = 1
    num_agents: int = 2
    d_state: int = 32
    d_conv: int = 4
    delta_rank: int = 128

    def __init__(
        self,
        observation_space,
        action_space,
        n_embd: int = 128,
        n_block: int = 1,
        num_agents: int = 2,
        d_state: int = 32,
        d_conv: int = 4,
        delta_rank: int = 128,
        unnormalized_log_prob: bool = True,
        sort_agents_by_type: bool = False,
        type_cycle_len: int = 3,
        device=None,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        CategoricalMixin.__init__(self, unnormalized_log_prob)
        object.__setattr__(self, "n_embd", int(n_embd))
        object.__setattr__(self, "n_block", int(n_block))
        object.__setattr__(self, "num_agents", int(num_agents))
        object.__setattr__(self, "d_state", int(d_state))
        object.__setattr__(self, "d_conv", int(d_conv))
        object.__setattr__(self, "delta_rank", int(delta_rank))
        object.__setattr__(self, "sort_agents_by_type", bool(sort_agents_by_type))
        object.__setattr__(self, "type_cycle_len", int(type_cycle_len))

        # Precompute sort / inverse-sort permutations (static, no trainable params)
        if sort_agents_by_type:
            n = int(num_agents)
            cycle = int(type_cycle_len)
            types = np.array([i % cycle for i in range(n)])
            sort_perm = np.argsort(types, kind="stable")
            inv_perm = np.empty_like(sort_perm)
            inv_perm[sort_perm] = np.arange(n)
            object.__setattr__(self, "_sort_perm", sort_perm)
            object.__setattr__(self, "_inv_perm", inv_perm)
        else:
            object.__setattr__(self, "_sort_perm", None)
            object.__setattr__(self, "_inv_perm", None)

    def setup(self) -> None:
        obs_dim = self.observation_space.shape[0]
        act_dim = int(self.num_actions)
        kw = dict(
            obs_dim=obs_dim,
            action_dim=act_dim,
            n_block=self.n_block,
            n_embd=self.n_embd,
            n_agent=self.num_agents,
            d_state=self.d_state,
            d_conv=self.d_conv,
            delta_rank=self.delta_rank,
        )
        self._encoder = Encoder(**kw)
        self._decoder = Decoder(**kw)

    def __call__(self, inputs: Mapping[str, Any], role: str = ""):
        x = inputs["states"]  # (B, obs_dim)
        taken_actions = inputs.get("taken_actions", None)
        ar_key = inputs.get("ar_key", None)

        n = self.num_agents
        b = x.shape[0]
        act_dim = int(self.num_actions)

        assert b >= n and b % n == 0, (
            f"MAMPolicyNet requires batch_size ({b}) to be >= num_agents ({n}) "
            f"and divisible by num_agents. This is a config error — check that "
            f"(rollouts * num_envs * num_agents) / mini_batches is divisible by num_agents."
        )

        groups = b // n
        obs_grouped = x.reshape(groups, n, -1)

        # Ablation 3: sort agents by type before encoding
        sort_perm = self._sort_perm
        inv_perm = self._inv_perm
        if sort_perm is not None:
            obs_grouped = obs_grouped[:, sort_perm, :]

        obs_rep = self._encoder(obs_grouped)

        # Decoder: parallel (teacher forcing) or zero-start
        if taken_actions is not None:
            # Training path: build shifted one-hot actions
            actions_int = taken_actions.reshape(b).astype(jnp.int32)
            one_hot = jax.nn.one_hot(actions_int, act_dim)
            oh_grouped = one_hot.reshape(groups, n, act_dim)
            if sort_perm is not None:
                # Sort one-hot actions to match sorted obs order, then rebuild shift
                oh_grouped = oh_grouped[:, sort_perm, :]
            shifted = jnp.zeros((groups, n, act_dim + 1))
            shifted = shifted.at[:, 0, 0].set(1)  # start token
            shifted = shifted.at[:, 1:, 1:].set(oh_grouped[:, :-1, :])
        elif ar_key is not None:
            # Autoregressive rollout (obs_rep already sorted if sort enabled)
            return self._autoregressive_decode(
                obs_rep, ar_key, groups, n, act_dim, inv_perm
            )
        else:
            # Zero start token (no taken_actions, no ar_key)
            shifted = jnp.zeros((groups, n, act_dim + 1))
            shifted = shifted.at[:, 0, 0].set(1)

        logits = self._decoder(shifted, obs_rep, obs_grouped)  # (groups, n, act_dim)

        # Unsort logits back to original agent order
        if inv_perm is not None:
            logits = logits[:, inv_perm, :]

        return logits.reshape(b, act_dim), {}

    def _autoregressive_decode(
        self,
        obs_rep: jax.Array,
        key: jax.Array,
        groups: int,
        n: int,
        act_dim: int,
        inv_perm: np.ndarray | None = None,
    ):
        """Autoregressive per-agent action generation.

        obs_rep is assumed to already be sorted (if sort_agents_by_type is set).
        inv_perm maps sorted positions back to original agent indices for output.

        Returns (actions_flat, {"log_probs": ..., "autoregressive": True}).
        """
        d_inner = self.n_embd * 2

        # Init shifted actions
        shifted = jnp.zeros((groups, n, act_dim + 1))
        shifted = shifted.at[:, 0, 0].set(1)

        output_actions = jnp.zeros((groups, n), dtype=jnp.int32)
        output_log_probs = jnp.zeros((groups, n))

        # Init recurrent state for decoder
        hs_shape = (groups, self.n_block, 1, d_inner, self.d_state)
        buf_shape = (groups, self.n_block, self.d_conv, d_inner)
        self_hs = jnp.zeros(hs_shape)
        self_buf = FIFOBuffer.init(buf_shape)
        cross_hs = jnp.zeros(hs_shape)
        cross_buf = FIFOBuffer.init(buf_shape)

        for i in range(n):
            logits_i, self_hs, self_buf, cross_hs, cross_buf = (
                self._decoder.recurrent_step(
                    shifted[:, i : i + 1, :],
                    obs_rep[:, i : i + 1, :],
                    self_hs,
                    self_buf,
                    cross_hs,
                    cross_buf,
                )
            )
            logits_i = logits_i.squeeze(1)  # (groups, act_dim)

            key, subkey = jax.random.split(key)
            action_i = jax.random.categorical(subkey, logits_i)  # (groups,)
            log_probs_i = jax.nn.log_softmax(logits_i)
            log_prob_i = jnp.take_along_axis(
                log_probs_i, action_i[:, None], axis=-1
            ).squeeze(-1)

            # Ablation 3: store at original agent index if sorted
            out_idx = int(inv_perm[i]) if inv_perm is not None else i
            output_actions = output_actions.at[:, out_idx].set(action_i)
            output_log_probs = output_log_probs.at[:, out_idx].set(log_prob_i)

            # Update shifted action for next sorted agent
            if i + 1 < n:
                shifted = shifted.at[:, i + 1, 1:].set(
                    jax.nn.one_hot(action_i, act_dim)
                )

        # Flatten: (groups, n) → (groups*n,) = (B,)
        # Reorder so output is in original agent order: agent 0 rows first, then 1, ...
        # output_actions[g, a] = action for original agent a in group g
        # We need flat order: [ag0_g0, ag0_g1, ..., ag1_g0, ag1_g1, ...] = agent-major
        # The buffer layout (how MAMMAPPO.act slices) expects:
        #   actions_all[i*num_envs : (i+1)*num_envs] = agent i's actions
        # output_actions is (groups, n); transposing and flattening gives agent-major
        actions_flat = output_actions.T.reshape(-1)[:, None]  # (B, 1), agent-major
        log_probs_flat = output_log_probs.T.reshape(-1)[:, None]  # (B, 1)

        return actions_flat, {"log_probs": log_probs_flat, "autoregressive": True}

    def encode(self, obs_grouped: jax.Array) -> jax.Array:
        """Run encoder only.  obs_grouped: (groups, n_agent, obs_dim) → obs_rep."""
        return self._encoder(obs_grouped)

    def init_state_dict(self, role: str, inputs=None, key=None) -> None:
        """Ensure batch = num_agents so all encoder/decoder params init."""
        if inputs is None:
            obs_sample = self.observation_space.sample()
            obs_batch = jnp.tile(
                jnp.asarray(obs_sample, dtype=jnp.float32)[None, :],
                (self.num_agents, 1),
            )
            inputs = {
                "states": obs_batch,
                "taken_actions": jnp.zeros((self.num_agents, 1)),
            }
        super().init_state_dict(role, inputs, key)

    def act(
        self,
        inputs: Mapping[str, np.ndarray | jax.Array | Any],
        role: str = "",
        params: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array | None, Mapping[str, jax.Array | Any]]:
        """Override to handle autoregressive mode.

        When ``ar_key`` is in inputs, __call__ returns pre-sampled actions
        instead of logits, so we bypass CategoricalMixin's sampling.
        """
        if "ar_key" in inputs:
            with jax.default_device(self.device):
                p = self.state_dict.params if params is None else params
                net_output, extra = self.apply(p, inputs, role)
                # net_output = actions (B, 1), extra has log_probs
                actions = net_output
                log_probs = extra["log_probs"]
                outputs = {"net_output": net_output}
                outputs.update(extra)
                return actions, log_probs, outputs

        # Standard CategoricalMixin path (training or zero-start)
        actions, log_prob, outputs = super().act(inputs, role, params)
        outputs["stddev"] = outputs.get("net_output", outputs.get("stddev"))
        return actions, log_prob, outputs

    @property
    def _modules(self):
        return {}
