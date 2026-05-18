"""Mamba selective state-space model blocks for MAM.

Implements three Mamba block variants used in the Multi-Agent Mamba architecture:
- MambaBlock / Mamba: vanilla causal Mamba (decoder self-attention replacement)
- BiMambaBlock / BiMamba: bidirectional Mamba (encoder self-attention replacement)
- CrossMambaBlock / CrossMamba: cross-attentional Mamba (decoder cross-attention replacement)

Each variant follows the hierarchy: Block → ResidualBlock (norm + residual) → Module (outer norm).

References
----------
- Daniel et al. 2024 "Multi-Agent RL with Selective State-Space Models"
- Gu & Dao 2024 "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp


import wandb

# Global step counter for debug logging inside JAX callbacks
_mamba_debug_step = 0


def _log_mamba_stats(d, a, bx, hh, bb, cc, xx, prefix="mamba"):
    global _mamba_debug_step
    _mamba_debug_step += 1
    if _mamba_debug_step % 500 == 0 and wandb.run is not None:
        wandb.log(
            {
                f"debug/{prefix}_delta_max": float(jnp.max(d)),
                f"debug/{prefix}_delta_mean": float(jnp.mean(d)),
                f"debug/{prefix}_A_bar_min": float(jnp.min(a)),
                f"debug/{prefix}_Bx_absmax": float(jnp.max(jnp.abs(bx))),
                f"debug/{prefix}_h_absmax": float(jnp.max(jnp.abs(hh))),
                f"debug/{prefix}_B_absmax": float(jnp.max(jnp.abs(bb))),
                f"debug/{prefix}_C_absmax": float(jnp.max(jnp.abs(cc))),
                f"debug/{prefix}_x_absmax": float(jnp.max(jnp.abs(xx))),
            }
        )


# Type aliases for recurrent state
HiddenState = jax.Array  # (batch, 1, d_inner, d_state)
Buffer = jax.Array  # (batch, d_conv, d_inner)


@dataclass
class MambaArgs:
    num_agents: int
    d_model: int  # embedding dimension (n_embd)
    d_state: int  # latent state dimension N
    d_conv: int  # 1-D causal convolution kernel size
    delta_rank: int  # rank of Δ projection
    expand: int = 2  # expansion factor → d_inner = d_model * expand
    delta_min: float = 0.001
    delta_max: float = 0.1  # initialisation upper bound (Gu & Dao 2024 §3.6)
    delta_init: str = "random"
    delta_scale: float = 1.0
    delta_init_floor: float = 1e-4
    # Forward-pass clamp on softplus(Δ) to prevent runaway SSM state after
    # weight drift during training.  Default 20.0 is ~200× the init upper
    # bound — loose enough to leave healthy training dynamics untouched,
    # tight enough to prevent Bx = Δ·B·x overflow that produced the NaN
    # observed in mam_warehouse_nocomm_v4/v5/v6 at step 50k–100k.
    delta_forward_clamp: float = 20.0
    debug_stats: bool = False

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand


# FIFO circular buffer helper (for recurrent conv padding)
class FIFOBuffer:
    @staticmethod
    def init(shape: tuple) -> Buffer:
        return jnp.zeros(shape)

    @staticmethod
    def add(buffer: Buffer, value: jax.Array) -> Buffer:
        """Shift left and insert *value* at the rightmost position.

        buffer : (batch, d_conv, d_inner)
        value  : (batch, 1, d_inner)
        """
        buffer = jnp.roll(buffer, shift=-1, axis=1)
        buffer = buffer.at[:, -1, :].set(value.squeeze(1))
        return buffer


# Vanilla Mamba (causal) — replaces causal self-attention in decoder
class MambaBlock(nn.Module):
    """Core selective SSM block (Gu & Dao 2024, Fig. 3)."""

    args: MambaArgs

    def setup(self) -> None:
        a = self.args
        # Input / output projections
        self.in_proj = nn.Dense(a.d_inner * 2, use_bias=False)
        self.out_proj = nn.Dense(a.d_model, use_bias=False)

        # 1-D depthwise causal convolution (valid padding; we pad manually)
        self.conv1d = nn.Conv(
            features=a.d_inner,
            kernel_size=(a.d_conv,),
            feature_group_count=a.d_inner,
            padding="VALID",
        )

        # SSM parameters
        # A — learnable, time-invariant (S4D-Real init)
        self.A_log = self.param(
            "A_log",
            lambda _key, shape: jnp.log(
                jnp.tile(jnp.arange(1, shape[1] + 1, dtype=jnp.float32), (shape[0], 1))
            ),
            (a.d_inner, a.d_state),
        )
        # D — learnable skip connection
        self.D = self.param("D", lambda _key: jnp.ones(a.d_inner))

        # Δ projection (delta_rank → d_inner) with special init
        def _init_delta_kernel(key, shape, _dtype):
            std = a.delta_rank**-0.5 * a.delta_scale
            if a.delta_init == "random":
                return jax.random.uniform(key, shape, minval=-std, maxval=std)
            return jnp.full(shape, std)

        def _init_delta_bias(key, shape, _dtype):
            dt = jnp.exp(
                jax.random.uniform(key, shape)
                * (jnp.log(a.delta_max) - jnp.log(a.delta_min))
                + jnp.log(a.delta_min)
            )
            dt = jnp.clip(dt, a_min=a.delta_init_floor)
            return dt + jnp.log(-jnp.expm1(-dt))  # inv-softplus

        self.delta_proj = nn.Dense(
            a.d_inner,
            use_bias=True,
            kernel_init=_init_delta_kernel,
            bias_init=_init_delta_bias,
        )

        # x_proj: input → (Δ_raw, B, C)
        self.x_proj = nn.Dense(a.delta_rank + a.d_state * 2, use_bias=False)

    # ---- parallel forward (training) ----
    def __call__(self, x: jax.Array) -> jax.Array:
        """x: (batch, seq_len, d_model) → (batch, seq_len, d_model)."""
        a = self.args
        x_and_res = self.in_proj(x)
        x, residual = jnp.split(x_and_res, 2, axis=-1)

        # Causal pad + conv
        pad = jnp.zeros((x.shape[0], a.d_conv - 1, a.d_inner))
        x = self.conv1d(jnp.concatenate([pad, x], axis=1))
        x = nn.silu(x)

        x = self._ssm(x)
        x = x * nn.silu(residual)
        return self.out_proj(x)

    # ---- recurrent forward (autoregressive rollout) ----
    def recurrent(
        self, x: jax.Array, hidden_state: HiddenState, buffer: Buffer
    ) -> tuple[jax.Array, HiddenState, Buffer]:
        """Single-step recurrent pass.

        x : (batch, 1, d_model)
        """
        x_and_res = self.in_proj(x)
        x, residual = jnp.split(x_and_res, 2, axis=-1)

        buffer = FIFOBuffer.add(buffer, x)
        x = self.conv1d(buffer)

        x = nn.silu(x)
        x, hidden_state = self._ssm_recurrent(x, hidden_state)
        x = x * nn.silu(residual)
        return self.out_proj(x), hidden_state, buffer

    # ---- SSM internals ----
    def _ssm(self, x: jax.Array) -> jax.Array:
        A = -jnp.exp(self.A_log)
        dbc = self.x_proj(x)
        delta_raw, B, C = jnp.split(
            dbc,
            [self.args.delta_rank, self.args.delta_rank + self.args.d_state],
            axis=-1,
        )
        delta = nn.softplus(self.delta_proj(delta_raw))
        delta = jnp.clip(delta, a_max=self.args.delta_forward_clamp)
        return self._selective_scan(x, delta, A, B, C)

    def _ssm_recurrent(
        self, x: jax.Array, hidden_state: HiddenState
    ) -> tuple[jax.Array, HiddenState]:
        A = -jnp.exp(self.A_log)
        dbc = self.x_proj(x)
        delta_raw, B, C = jnp.split(
            dbc,
            [self.args.delta_rank, self.args.delta_rank + self.args.d_state],
            axis=-1,
        )
        delta = nn.softplus(self.delta_proj(delta_raw))
        delta = jnp.clip(delta, a_max=self.args.delta_forward_clamp)
        return self._recurrent_scan(x, delta, A, B, C, hidden_state)

    def _selective_scan(
        self,
        x: jax.Array,
        delta: jax.Array,
        A: jax.Array,
        B: jax.Array,
        C: jax.Array,
    ) -> jax.Array:
        """Parallel selective scan via associative scan."""

        @jax.vmap
        def _assoc_op(prev, now):
            a_prev, b_prev = prev
            a_now, b_now = now
            return (a_now * a_prev, a_now * b_prev + b_now)

        # Discretise: A_bar = exp(Δ·A),  B_bar·x = Δ·B·x  (Euler approx for B)
        A_bar = jnp.exp(jnp.einsum("b l d, d n -> b l d n", delta, A))
        Bx = jnp.einsum("b l d, b l n, b l d -> b l d n", delta, B, x)

        _, h = jax.lax.associative_scan(_assoc_op, (A_bar, Bx), axis=1)
        y = jnp.einsum("b l d n, b l n -> b l d", h, C)

        if getattr(self.args, "debug_stats", False):
            jax.debug.callback(
                _log_mamba_stats, delta, A_bar, Bx, h, B, C, x, self.name or "mamba"
            )

        return y + x * self.D

    def _recurrent_scan(
        self,
        x: jax.Array,
        delta: jax.Array,
        A: jax.Array,
        B: jax.Array,
        C: jax.Array,
        hidden_state: HiddenState,
    ) -> tuple[jax.Array, HiddenState]:
        A_bar = jnp.exp(jnp.einsum("b l d, d n -> b l d n", delta, A))
        Bx = jnp.einsum("b l d, b l n, b l d -> b l d n", delta, B, x)
        hidden_state = A_bar * hidden_state + Bx
        y = jnp.einsum("b l d n, b l n -> b l d", hidden_state, C)
        return y + x * self.D, hidden_state


class ResidualBlock(nn.Module):
    """RMSNorm → MambaBlock → residual."""

    args: MambaArgs

    def setup(self) -> None:
        self.block = MambaBlock(self.args)
        self.norm = nn.RMSNorm()

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.block(self.norm(x))

    def recurrent(
        self, x: jax.Array, hidden_state: HiddenState, buffer: Buffer
    ) -> tuple[jax.Array, HiddenState, Buffer]:
        out, hidden_state, buffer = self.block.recurrent(
            self.norm(x), hidden_state, buffer
        )
        return x + out, hidden_state, buffer


class Mamba(nn.Module):
    """Reference-faithful Mamba wrapper.

    Matches InstaDeep ``mamba_selfattention_block.Mamba``:
    ``outer_norm(x + MambaBlock(RMSNorm(x)))``.
    The inner RMSNorm bounds input magnitudes; the residual + outer
    LayerNorm bound output magnitudes — without these, weight drift
    during PPO produces 1e6+ activation explosions (verified on the
    BiMamba mult-gate hazard but applies to all wrappers).
    """

    num_agents: int
    d_model: int
    d_state: int
    d_conv: int
    delta_rank: int

    def setup(self) -> None:
        self._args = MambaArgs(
            self.num_agents, self.d_model, self.d_state, self.d_conv, self.delta_rank
        )
        self.block = ResidualBlock(self._args)
        self.outer_norm = nn.LayerNorm()

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.outer_norm(self.block(x))

    def recurrent(
        self, x: jax.Array, hidden_state: HiddenState, buffer: Buffer
    ) -> tuple[jax.Array, HiddenState, Buffer]:
        x, hidden_state, buffer = self.block.recurrent(x, hidden_state, buffer)
        return self.outer_norm(x), hidden_state, buffer


# Bidirectional Mamba — replaces non-causal self-attention in encoder
class BiMambaBlock(nn.Module):
    """Bidirectional Mamba: forward then backward with SHARED parameters.

    Matches the InstaDeep MAM reference implementation (BiMambaBlock in
    mava/networks/mamba_bidirectional_block.py):
    1. Forward pass on input x  →  h_fwd
    2. Backward pass on h_fwd   →  h_bwd  (cascaded, not parallel)
    3. Output: h_fwd * flip(h_bwd)  (multiplicative gate)

    Shared parameters halve the BiMamba parameter count while still
    providing bidirectional context (Daniel et al. 2024, §3.1).
    The cascaded design lets the backward pass refine the forward
    representation; the multiplicative combination gates each position
    by consensus between both scan directions.
    """

    args: MambaArgs

    def setup(self) -> None:
        self.block = MambaBlock(self.args)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Forward scan on original input
        h_fwd = self._one_dir(self.block, x)
        # Backward scan on the forward output (cascaded, per reference)
        h_bwd = self._one_dir(self.block, jnp.flip(h_fwd, axis=1))
        # Multiplicative gate: both directions must activate
        return h_fwd * jnp.flip(h_bwd, axis=1)

    @staticmethod
    def _one_dir(block: MambaBlock, x: jax.Array) -> jax.Array:
        a = block.args
        x_and_res = block.in_proj(x)
        x, residual = jnp.split(x_and_res, 2, axis=-1)
        pad = jnp.zeros((x.shape[0], a.d_conv - 1, a.d_inner))
        x = block.conv1d(jnp.concatenate([pad, x], axis=1))
        x = nn.silu(x)
        x = block._ssm(x)
        x = x * nn.silu(residual)
        return block.out_proj(x)


class BiResidualBlock(nn.Module):
    args: MambaArgs

    def setup(self) -> None:
        self.block = BiMambaBlock(self.args)
        self.norm = nn.RMSNorm()

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.block(self.norm(x))


class BiMamba(nn.Module):
    """Reference-faithful BiMamba wrapper.

    Matches InstaDeep ``mamba_bidirectional_block.BiMamba``:
    ``outer_norm(x + BiMambaBlock(RMSNorm(x)))``. Without these wrappers
    the multiplicative gate ``h_fwd * flip(h_bwd)`` is unbounded — under
    PPO weight drift it produces 1e16+ outputs at ×1.5 weight scaling
    and NaNs by ×4 (verified empirically on warehouse_scaled_v1).
    """

    num_agents: int
    d_model: int
    d_state: int
    d_conv: int
    delta_rank: int

    def setup(self) -> None:
        self._args = MambaArgs(
            self.num_agents, self.d_model, self.d_state, self.d_conv, self.delta_rank
        )
        self.block = BiResidualBlock(self._args)
        self.outer_norm = nn.LayerNorm()

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.outer_norm(self.block(x))


# Cross-attentional Mamba — replaces cross-attention in decoder
class CrossMambaBlock(MambaBlock):
    """Mamba with cross-attention: Δ,B from x1 (actions); C from x2 (obs_rep).

    Matches Daniel et al. 2024 §3.2 / Fig. 5 and the InstaDeep reference
    (assets/mam-code/mava/networks/mamba_crossattention_block.py):

        - Δ and B depend on the target (source-being-scanned) sequence x1.
        - Only C (the readout) depends on the second input x2.

    The discretised SSM is h_t = Ā_t h_{t-1} + B̄_t · x1_t with
    Ā_t = exp(Δ_t(x1) · A), B̄_t = Δ_t(x1) · B(x1).  The readout
    y_t = C_t(x2) · h_t mixes obs information position-wise.  This
    preserves the causal action-AR chain required by the multi-agent
    advantage-decomposition theorem (Kuba et al. 2022).
    """

    def setup(self) -> None:
        a = self.args
        # Reuse parent's setup for projections, conv, A, D, delta_proj
        self.in_proj = nn.Dense(a.d_inner * 2, use_bias=False)
        self.out_proj = nn.Dense(a.d_model, use_bias=False)
        self.conv1d = nn.Conv(
            features=a.d_inner,
            kernel_size=(a.d_conv,),
            feature_group_count=a.d_inner,
            padding="VALID",
        )
        self.A_log = self.param(
            "A_log",
            lambda _key, shape: jnp.log(
                jnp.tile(jnp.arange(1, shape[1] + 1, dtype=jnp.float32), (shape[0], 1))
            ),
            (a.d_inner, a.d_state),
        )
        self.D = self.param("D", lambda _key: jnp.ones(a.d_inner))

        def _init_delta_kernel(key, shape, _dtype):
            std = a.delta_rank**-0.5 * a.delta_scale
            if a.delta_init == "random":
                return jax.random.uniform(key, shape, minval=-std, maxval=std)
            return jnp.full(shape, std)

        def _init_delta_bias(key, shape, _dtype):
            dt = jnp.exp(
                jax.random.uniform(key, shape)
                * (jnp.log(a.delta_max) - jnp.log(a.delta_min))
                + jnp.log(a.delta_min)
            )
            dt = jnp.clip(dt, a_min=a.delta_init_floor)
            return dt + jnp.log(-jnp.expm1(-dt))

        self.delta_proj = nn.Dense(
            a.d_inner,
            use_bias=True,
            kernel_init=_init_delta_kernel,
            bias_init=_init_delta_bias,
        )

        # Cross-attention (matches reference mamba_crossattention_block.py:131-133):
        #   x_proj(x1) → (Δ, B); C_proj(x2) → C.
        self.x_proj = nn.Dense(a.delta_rank + a.d_state, use_bias=False)
        self.C_proj = nn.Dense(a.d_state, use_bias=False)

    # ---- parallel ----
    def __call__(self, x1_x2: tuple[jax.Array, jax.Array]) -> jax.Array:
        """x1: target (actions), x2: source (observations)."""
        x1, x2 = x1_x2
        a = self.args
        x_and_res = self.in_proj(x1)
        x1, residual = jnp.split(x_and_res, 2, axis=-1)
        pad = jnp.zeros((x1.shape[0], a.d_conv - 1, a.d_inner))
        x1 = self.conv1d(jnp.concatenate([pad, x1], axis=1))
        x1 = nn.silu(x1)
        x1 = self._cross_ssm(x1, x2)
        x1 = x1 * nn.silu(residual)
        return self.out_proj(x1)

    # ---- recurrent ----
    def recurrent(
        self,
        x1_x2: tuple[jax.Array, jax.Array],
        hidden_state: HiddenState,
        buffer: Buffer,
    ) -> tuple[jax.Array, HiddenState, Buffer]:
        x1, x2 = x1_x2
        x_and_res = self.in_proj(x1)
        x1, residual = jnp.split(x_and_res, 2, axis=-1)
        buffer = FIFOBuffer.add(buffer, x1)
        x1 = self.conv1d(buffer)
        x1 = nn.silu(x1)
        x1, hidden_state = self._cross_ssm_recurrent(x1, x2, hidden_state)
        x1 = x1 * nn.silu(residual)
        return self.out_proj(x1), hidden_state, buffer

    # ---- cross-SSM internals ----
    def _cross_ssm(self, x1: jax.Array, x2: jax.Array) -> jax.Array:
        A = -jnp.exp(self.A_log)
        delta_B = self.x_proj(x1)
        delta_raw, B = jnp.split(delta_B, [self.args.delta_rank], axis=-1)
        C = self.C_proj(x2)
        delta = nn.softplus(self.delta_proj(delta_raw))
        delta = jnp.clip(delta, a_max=self.args.delta_forward_clamp)
        return self._selective_scan(x1, delta, A, B, C)

    def _cross_ssm_recurrent(
        self, x1: jax.Array, x2: jax.Array, hidden_state: HiddenState
    ) -> tuple[jax.Array, HiddenState]:
        A = -jnp.exp(self.A_log)
        delta_B = self.x_proj(x1)
        delta_raw, B = jnp.split(delta_B, [self.args.delta_rank], axis=-1)
        C = self.C_proj(x2)
        delta = nn.softplus(self.delta_proj(delta_raw))
        delta = jnp.clip(delta, a_max=self.args.delta_forward_clamp)
        return self._recurrent_scan(x1, delta, A, B, C, hidden_state)


class CrossResidualBlock(nn.Module):
    """RMSNorm (both inputs) → CrossMambaBlock → residual on x1."""

    args: MambaArgs

    def setup(self) -> None:
        self.block = CrossMambaBlock(self.args)
        self.norm1 = nn.RMSNorm()
        self.norm2 = nn.RMSNorm()

    def __call__(
        self, x1_x2: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        x1, x2 = x1_x2
        x1_out = self.block((self.norm1(x1), self.norm2(x2)))
        return (x1 + x1_out, x2)

    def recurrent(
        self,
        x1_x2: tuple[jax.Array, jax.Array],
        hidden_state: HiddenState,
        buffer: Buffer,
    ) -> tuple[tuple[jax.Array, jax.Array], HiddenState, Buffer]:
        x1, x2 = x1_x2
        x1_out, hidden_state, buffer = self.block.recurrent(
            (self.norm1(x1), self.norm2(x2)), hidden_state, buffer
        )
        return (x1 + x1_out, x2), hidden_state, buffer


class CrossMamba(nn.Module):
    """Reference-faithful CrossMamba wrapper.

    Matches InstaDeep ``mamba_crossattention_block.CrossMamba``:
    ``outer_norm(x1 + CrossMambaBlock(RMSNorm(x1), RMSNorm(x2)))``.
    Returns only the transformed x1.
    """

    num_agents: int
    d_model: int
    d_state: int
    d_conv: int
    delta_rank: int

    def setup(self) -> None:
        self._args = MambaArgs(
            self.num_agents, self.d_model, self.d_state, self.d_conv, self.delta_rank
        )
        self.block = CrossResidualBlock(self._args)
        self.outer_norm = nn.LayerNorm()

    def __call__(self, x1_x2: tuple[jax.Array, jax.Array]) -> jax.Array:
        x1, _ = self.block(x1_x2)
        return self.outer_norm(x1)

    def recurrent(
        self,
        x1_x2: tuple[jax.Array, jax.Array],
        hidden_state: HiddenState,
        buffer: Buffer,
    ) -> tuple[jax.Array, HiddenState, Buffer]:
        x1_x2_out, hidden_state, buffer = self.block.recurrent(
            x1_x2, hidden_state, buffer
        )
        x1, _ = x1_x2_out
        return self.outer_norm(x1), hidden_state, buffer
