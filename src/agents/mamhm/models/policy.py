"""MAMHM policy network: MAM + Hopfield Memory Bank.

Extends the Multi-Agent Mamba architecture with a lightweight Hopfield Memory
Bank after the CrossMamba decoder.

Architecture:
    Observation o ──────────────────────────────────────────┐
                                                            │
    Action prefix a_{<i} ─► Mamba Encoder (BiMamba) ─► h_enc
                                                            │
              ┌─────────────────────────────────────────────┤
              │                                             │
              ▼                                             ▼
         ┌──────────────────────────────────────────────────────┐
         │            MAM DecodeBlock (unchanged)               │
         │  Mamba self-attn → CrossMamba → MLP                  │
         └──────────────────────────────────────────────────────┘
                               │
                               ▼
         ┌──────────────────────────────────────────────────────┐
         │         HopfieldMemoryBank (NEW - single pass)       │
         │                                                      │
         │    q = W_q · h                                       │
         │    α = softmax(β · q^T · Ξ)      ← ONE step          │
         │    m = Σ α_μ · ξ_μ                                   │
         │    h' = h + γ · m                ← Residual          │
         │                                                      │
         └──────────────────────────────────────────────────────┘
                               │
                               ▼
                         Policy Head ─► π(a|o)

This differs from ETMAT which replaces CrossMamba entirely with iterative
energy minimization. By keeping MAM's working components and adding memory
as a lightweight residual, we get:
1. Stable gradients (no energy iteration loops blocking backprop)
2. Explicit associative memory (coordination pattern retrieval)
3. Minimal risk (proven MAM base + small additive module)

References
----------
- Daniel et al. 2024 "Multi-Agent RL with Selective State-Space Models"
- Ramsauer et al. 2021 "Hopfield Networks is All You Need"
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
from agents.mamhm.models.hopfield_memory import HopfieldMemoryBank
from agents.mamhm.models.task_hopfield import (
    EntityHopfieldPooling,
    TaskHopfieldPooling,
)

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
    """BiMamba encoder with optional upstream Hopfield pooling.

    When use_task_hopfield=True, applies TaskHopfieldPooling to the task
    queue features BEFORE the BiMamba blocks so the encoder reasons over
    task-conditioned agent representations.
    """

    obs_dim: int
    action_dim: int
    n_block: int
    n_embd: int
    n_agent: int
    d_state: int
    d_conv: int
    delta_rank: int

    # Upstream Hopfield pooling flags
    use_task_hopfield: bool = False
    use_entity_hopfield: bool = False
    task_hopfield_num_heads: int = 4
    task_hopfield_beta: float = 2.0
    task_hopfield_gate_init: float = -3.0
    entity_hopfield_num_heads: int = 4
    entity_hopfield_beta: float = 2.0
    entity_hopfield_gate_init: float = -3.0

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

        # Upstream Hopfield modules (applied before BiMamba)
        if self.use_task_hopfield:
            self._task_hopfield = TaskHopfieldPooling(
                d_model=self.n_embd,
                num_query_heads=self.task_hopfield_num_heads,
                beta=self.task_hopfield_beta,
                gate_init=self.task_hopfield_gate_init,
            )
        if self.use_entity_hopfield:
            self._entity_hopfield = EntityHopfieldPooling(
                d_model=self.n_embd,
                num_query_heads=self.entity_hopfield_num_heads,
                beta=self.entity_hopfield_beta,
                gate_init=self.entity_hopfield_gate_init,
            )

    def __call__(self, obs: jax.Array) -> jax.Array:
        """obs: (batch, n_agent, obs_dim) → obs_rep: (batch, n_agent, n_embd)."""
        emb = self.obs_encoder(obs)

        # Apply upstream Hopfield pooling to condition embeddings
        # on task/entity context BEFORE the BiMamba encoder blocks.
        if self.use_task_hopfield:
            emb = self._task_hopfield(obs, emb)
        if self.use_entity_hopfield:
            emb = self._entity_hopfield(obs, emb)

        rep = self.blocks(self.ln(emb))
        return rep


class DecodeBlock(nn.Module):
    """Pre-norm Mamba self → CrossMamba → MLP (with residual connections). Identical to MAM's DecodeBlock."""

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
        """Parallel forward.

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
    """Mamba decoder with optional Hopfield Memory Bank.

    The post-decoder HopfieldMemoryBank is now optional (default: off).
    When use_post_decoder_hopfield=False, the decoder outputs go directly
    to the action head.  This allows upstream TaskHopfieldPooling (in the
    encoder) to serve as the sole memory mechanism.
    """

    obs_dim: int
    action_dim: int
    n_block: int
    n_embd: int
    n_agent: int
    d_state: int
    d_conv: int
    delta_rank: int

    # Post-decoder Hopfield Memory Bank parameters (legacy, default off)
    use_post_decoder_hopfield: bool = False
    num_memories: int = 16
    memory_beta: float = 1.5
    memory_gamma: float = 0.25
    memory_gate_init: float = -2.0
    memory_activation: str = "softmax"
    memory_use_pre_ln: bool = True
    memory_diversity_loss_scale: float = 0.01

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

        # Post-decoder Hopfield Memory Bank (legacy path, off by default)
        if self.use_post_decoder_hopfield:
            self.memory_bank = HopfieldMemoryBank(
                d_model=self.n_embd,
                num_memories=self.num_memories,
                beta=self.memory_beta,
                gamma=self.memory_gamma,
                gate_init=self.memory_gate_init,
                activation=self.memory_activation,
                use_pre_ln=self.memory_use_pre_ln,
                diversity_loss_scale=self.memory_diversity_loss_scale,
            )

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

        # Apply post-decoder Hopfield Memory Bank (legacy, optional)
        if self.use_post_decoder_hopfield:
            x = self.memory_bank(x)

        return self.head(x)

    def prepare_memory_bank(self) -> tuple[jax.Array, jax.Array] | None:
        if self.use_post_decoder_hopfield:
            return self.memory_bank.prepare_memory()
        return None

    def memory_diversity_loss(self) -> jax.Array:
        if self.use_post_decoder_hopfield:
            return self.memory_bank.diversity_loss()
        return jnp.array(0.0)

    def recurrent_step(
        self,
        shifted_action: jax.Array,
        obs_rep: jax.Array,
        self_hs_all: jax.Array,
        self_buf_all: jax.Array,
        cross_hs_all: jax.Array,
        cross_buf_all: jax.Array,
        memory_cache: tuple[jax.Array, jax.Array] | None = None,
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

        if self.n_block == 1:
            x, s_hs, s_buf, c_hs, c_buf = self.blocks[0].recurrent(
                x,
                obs_rep,
                self_hs_all[:, 0],
                self_buf_all[:, 0],
                cross_hs_all[:, 0],
                cross_buf_all[:, 0],
            )
            self_hs_all = s_hs[:, None]
            self_buf_all = s_buf[:, None]
            cross_hs_all = c_hs[:, None]
            cross_buf_all = c_buf[:, None]
        else:
            for i, block in enumerate(self.blocks):
                x, s_hs_new, s_buf_new, c_hs_new, c_buf_new = block.recurrent(
                    x,
                    obs_rep,
                    self_hs_all[:, i],
                    self_buf_all[:, i],
                    cross_hs_all[:, i],
                    cross_buf_all[:, i],
                )
                self_hs_all = self_hs_all.at[:, i].set(s_hs_new)
                self_buf_all = self_buf_all.at[:, i].set(s_buf_new)
                cross_hs_all = cross_hs_all.at[:, i].set(c_hs_new)
                cross_buf_all = cross_buf_all.at[:, i].set(c_buf_new)

        if self.use_post_decoder_hopfield:
            if memory_cache is None:
                x = self.memory_bank(x)
            else:
                x = self.memory_bank.apply_memory(x, *memory_cache)

        logits = self.head(x)
        return logits, self_hs_all, self_buf_all, cross_hs_all, cross_buf_all


class MAMHMPolicyNet(CategoricalMixin, Model):
    """Multi-Agent Mamba + Hopfield Memory policy network.

    Wraps the MAM Encoder + Memory-augmented Decoder into a skrl-compatible
    Model with CategoricalMixin for discrete action spaces.

    Two modes of operation:
    - **Parallel** (training): ``taken_actions`` present in inputs → teacher forcing.
    - **Autoregressive** (rollout): ``ar_key`` present → sequential per-agent decode.
    - **Fallback**: zero start tokens → independent per-agent logits.
    """

    # MAM architecture params
    n_embd: int = 128
    n_block: int = 1
    num_agents: int = 2
    d_state: int = 32
    d_conv: int = 4
    delta_rank: int = 128

    # Upstream Hopfield pooling params
    use_task_hopfield: bool = False
    use_entity_hopfield: bool = False
    task_hopfield_num_heads: int = 4
    task_hopfield_beta: float = 2.0
    task_hopfield_gate_init: float = -3.0

    # Post-decoder Hopfield Memory Bank params (legacy)
    use_post_decoder_hopfield: bool = False
    num_memories: int = 16
    memory_beta: float = 1.5
    memory_gamma: float = 0.25
    memory_gate_init: float = -2.0
    memory_activation: str = "softmax"
    memory_use_pre_ln: bool = True
    memory_diversity_loss_scale: float = 0.01

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
        # Upstream Hopfield
        use_task_hopfield: bool = False,
        use_entity_hopfield: bool = False,
        task_hopfield_num_heads: int = 4,
        task_hopfield_beta: float = 2.0,
        task_hopfield_gate_init: float = -3.0,
        # Post-decoder Hopfield (legacy)
        use_post_decoder_hopfield: bool = False,
        num_memories: int = 16,
        memory_beta: float = 1.5,
        memory_gamma: float = 0.25,
        memory_gate_init: float = -2.0,
        memory_activation: str = "softmax",
        memory_use_pre_ln: bool = True,
        memory_diversity_loss_scale: float = 0.01,
        unnormalized_log_prob: bool = True,
        device=None,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        # MAM params
        object.__setattr__(self, "n_embd", int(n_embd))
        object.__setattr__(self, "n_block", int(n_block))
        object.__setattr__(self, "num_agents", int(num_agents))
        object.__setattr__(self, "d_state", int(d_state))
        object.__setattr__(self, "d_conv", int(d_conv))
        object.__setattr__(self, "delta_rank", int(delta_rank))

        # Upstream Hopfield params
        object.__setattr__(self, "use_task_hopfield", bool(use_task_hopfield))
        object.__setattr__(self, "use_entity_hopfield", bool(use_entity_hopfield))
        object.__setattr__(self, "task_hopfield_num_heads", int(task_hopfield_num_heads))
        object.__setattr__(self, "task_hopfield_beta", float(task_hopfield_beta))
        object.__setattr__(self, "task_hopfield_gate_init", float(task_hopfield_gate_init))

        # Post-decoder Hopfield Memory params (legacy)
        object.__setattr__(self, "use_post_decoder_hopfield", bool(use_post_decoder_hopfield))
        object.__setattr__(self, "num_memories", int(num_memories))
        object.__setattr__(self, "memory_beta", float(memory_beta))
        object.__setattr__(self, "memory_gamma", float(memory_gamma))
        object.__setattr__(self, "memory_gate_init", float(memory_gate_init))
        object.__setattr__(self, "memory_activation", str(memory_activation))
        object.__setattr__(self, "memory_use_pre_ln", bool(memory_use_pre_ln))
        object.__setattr__(self, "memory_diversity_loss_scale", float(memory_diversity_loss_scale))

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
        self._encoder = Encoder(
            **kw,
            use_task_hopfield=self.use_task_hopfield,
            use_entity_hopfield=self.use_entity_hopfield,
            task_hopfield_num_heads=self.task_hopfield_num_heads,
            task_hopfield_beta=self.task_hopfield_beta,
            task_hopfield_gate_init=self.task_hopfield_gate_init,
        )
        self._decoder = Decoder(
            **kw,
            use_post_decoder_hopfield=self.use_post_decoder_hopfield,
            num_memories=self.num_memories,
            memory_beta=self.memory_beta,
            memory_gamma=self.memory_gamma,
            memory_gate_init=self.memory_gate_init,
            memory_activation=self.memory_activation,
            memory_use_pre_ln=self.memory_use_pre_ln,
            memory_diversity_loss_scale=self.memory_diversity_loss_scale,
        )

    def __call__(self, inputs: Mapping[str, Any], role: str = ""):
        x = inputs["states"]  # (B, obs_dim)
        taken_actions = inputs.get("taken_actions", None)
        ar_key = inputs.get("ar_key", None)

        n = self.num_agents
        b = x.shape[0]
        act_dim = int(self.num_actions)

        assert b >= n and b % n == 0, (
            f"MAMHMPolicyNet requires batch_size ({b}) to be >= num_agents ({n}) "
            f"and divisible by num_agents. This is a config error — check that "
            f"(rollouts * num_envs * num_agents) / mini_batches is divisible by num_agents."
        )

        groups = b // n
        obs_grouped = x.reshape(groups, n, -1)

        # Encoder
        obs_rep = self._encoder(obs_grouped)

        # Decoder: parallel (teacher forcing) or zero-start
        if taken_actions is not None:
            # Training path: build shifted one-hot actions
            actions_int = taken_actions.reshape(b).astype(jnp.int32)
            one_hot = jax.nn.one_hot(actions_int, act_dim)
            oh_grouped = one_hot.reshape(groups, n, act_dim)
            shifted = jnp.zeros((groups, n, act_dim + 1))
            shifted = shifted.at[:, 0, 0].set(1)  # start token
            shifted = shifted.at[:, 1:, 1:].set(oh_grouped[:, :-1, :])
        elif ar_key is not None:
            # Autoregressive rollout
            return self._autoregressive_decode(obs_rep, ar_key, groups, n, act_dim)
        else:
            # Zero start token (no taken_actions, no ar_key)
            shifted = jnp.zeros((groups, n, act_dim + 1))
            shifted = shifted.at[:, 0, 0].set(1)

        logits = self._decoder(shifted, obs_rep, obs_grouped)  # (groups, n, act_dim)
        return logits.reshape(b, act_dim), {}

    def _autoregressive_decode(
        self,
        obs_rep: jax.Array,
        key: jax.Array,
        groups: int,
        n: int,
        act_dim: int,
    ):
        """Autoregressive per-agent action generation using jax.lax.scan.

        Uses scan instead of a Python for-loop to avoid tracing n separate
        copies of the loop body, reducing compilation time and enabling
        XLA loop optimizations.

        Returns (actions_flat, {"log_probs": ..., "logits": ..., "autoregressive": True}).
        """
        d_inner = self.n_embd * 2  # expand=2
        decoder = self._decoder
        memory_cache = decoder.prepare_memory_bank()
        obs_rep_per_agent = jnp.swapaxes(obs_rep, 0, 1)[:, :, None, :]

        # Init recurrent state for decoder
        hs_shape = (groups, self.n_block, 1, d_inner, self.d_state)
        buf_shape = (groups, self.n_block, self.d_conv, d_inner)

        # Start token for agent 0: [1, 0, ..., 0]
        start_shifted = jnp.zeros((groups, 1, act_dim + 1))
        start_shifted = start_shifted.at[:, 0, 0].set(1.0)

        init_carry = (
            jnp.zeros(hs_shape),  # self_hs
            FIFOBuffer.init(buf_shape),  # self_buf
            jnp.zeros(hs_shape),  # cross_hs
            FIFOBuffer.init(buf_shape),  # cross_buf
            key,  # PRNG
            start_shifted,  # cur_shifted: (groups, 1, act_dim+1)
        )

        def _scan_body(carry, obs_rep_i):
            self_hs, self_buf, cross_hs, cross_buf, rng, cur_shifted = carry

            logits_i, self_hs, self_buf, cross_hs, cross_buf = decoder.recurrent_step(
                cur_shifted,
                obs_rep_i,
                self_hs,
                self_buf,
                cross_hs,
                cross_buf,
                memory_cache,
            )
            logits_i = logits_i.squeeze(1)  # (groups, act_dim)

            rng, subkey = jax.random.split(rng)
            action_i = jax.random.categorical(subkey, logits_i)  # (groups,)
            log_probs_i = jax.nn.log_softmax(logits_i)
            log_prob_i = jnp.take_along_axis(
                log_probs_i, action_i[:, None], axis=-1
            ).squeeze(-1)

            # Build next agent's shifted input: [0, one_hot(action_i)]
            next_shifted = jnp.zeros((groups, 1, act_dim + 1))
            next_shifted = next_shifted.at[:, 0, 1:].set(
                jax.nn.one_hot(action_i, act_dim)
            )

            new_carry = (self_hs, self_buf, cross_hs, cross_buf, rng, next_shifted)
            return new_carry, (action_i, log_prob_i, logits_i)

        _, (all_actions, all_log_probs, all_logits) = jax.lax.scan(
            _scan_body, init_carry, obs_rep_per_agent
        )
        # all_actions: (n, groups), all_log_probs: (n, groups), all_logits: (n, groups, act_dim)

        # Transpose and flatten: (n, groups) → (groups, n) → (groups*n,)
        actions_flat = all_actions.T.reshape(-1)[:, None]  # (B, 1)
        log_probs_flat = all_log_probs.T.reshape(-1)[:, None]  # (B, 1)
        logits_flat = all_logits.transpose(1, 0, 2).reshape(-1, act_dim)  # (B, act_dim)

        return actions_flat, {
            "log_probs": log_probs_flat,
            "logits": logits_flat,
            "autoregressive": True,
        }

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
        """Handle autoregressive mode.

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
