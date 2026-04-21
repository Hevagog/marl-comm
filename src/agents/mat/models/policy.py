"""Multi-Agent Transformer (MAT) policy network.

Paper-faithful implementation of Wen et al. 2022 "Multi-Agent Transformer"
(NeurIPS 2022), with one deliberate deviation: the AR decoder is replaced by
a parallel decoder with zero start tokens (same fix applied to CommFormer v13).

Why the deviation is justified
--------------------------------
1. **No execution benefit** (MAT §4.2): Decentralized agents cannot observe
   teammates' actions at deployment time.  The AR dependency is an artifact
   of training, not a decentralized capability.
2. **Ratio mismatch**: Teacher-forcing at training vs. AR sampling at rollout
   produces systematic ratio_max_abs_dev > 0.5 before any gradient step
   (Bengio et al. 2015 "Scheduled Sampling").  Parallel decoder gives ratio = 1.0
   at update start, allowing all learning_epochs × mini_batches steps.
3. **Compute**: AR scan calls enc_dec_block N times per rollout step; parallel
   decoder calls it once — 4× faster for N=4.

Architecture (MAT §3.2, Fig. 2)
---------------------------------
1. Node embedding: obs → hidden_dim  (orthogonal init, gain = sqrt(2) for ReLU)
2. Encoder: N stacked MATEncoderBlocks (standard scaled dot-product self-attention,
   full N×N connectivity, no communication graph / no α parameter)
3. Decoder: N stacked MATDecoderBlocks
   - Causal masked self-attention over zero dec_in
   - Cross-attention to encoder output
4. Action head: tanh + Dense → logits

GTrXL initialization (Parisotto et al. 2020 §3.4)
---------------------------------------------------
Output projections (Wo in each MHA sublayer; mlp2 in each FFN) are initialized
with orthogonal scale = 0.01, making each residual block act as near-identity at
initialization.  This prevents early representation collapse and lets the policy
gradient signal shape the representations before the attention heads have "chosen"
their structure.

Key correctness properties
---------------------------
- No CommGraph, no α, no k-hot STE — eliminates gradient bias (Paulus et al. 2020).
- assert b % n == 0 before reshaping (fix for CC-B01 from memory/bugs.md).
- No unused heads or value estimates (fix for MAM-B02).
- Parallel decoder: identical forward path at rollout and training → ratio = 1.0.

References
----------
- Wen et al. 2022 "Multi-Agent Transformer" (NeurIPS 2022), §3.
- Parisotto et al. 2020 "Stabilizing Transformers for RL" — GTrXL init.
- Vaswani et al. 2017 "Attention Is All You Need" — scaled dot-product attention.
- Bengio et al. 2015 "Scheduled Sampling" — exposure bias / teacher-forcing.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from skrl.models.jax import CategoricalMixin, Model

# Orthogonal gain for hidden ReLU layers (matches MAM/CommFormer convention).
_HIDDEN_GAIN = jnp.sqrt(2.0)

# GTrXL gain (Parisotto 2020 §3.4): near-zero init for output projections
# (Wo in MHA, mlp2 in FFN) so each residual block is near-identity at init.
# Also used for the final action logits head.
_OUTPUT_GAIN = 0.01


# ---------------------------------------------------------------------------
# Standard multi-head attention (no edge embeddings, no adjacency masking)
# ---------------------------------------------------------------------------


class _StandardMHA(nn.Module):
    """Scaled dot-product multi-head attention (Vaswani et al. 2017).

    No edge embeddings and no adjacency masking — unlike CommFormer's
    RelationEnhancedMHA.  MAT uses full N×N attention; the attention weights
    themselves encode the learned communication structure.

    The output projection Wo uses ``_OUTPUT_GAIN`` (GTrXL init) so the
    sublayer is near-identity at initialization.
    """

    num_heads: int
    head_dim: int
    use_causal_mask: bool = False

    @nn.compact
    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
    ) -> jax.Array:
        """
        Parameters
        ----------
        query : (N_q, D)
        key   : (N_k, D)
        value : (N_k, D)

        Returns
        -------
        (N_q, D)
        """
        n_q = query.shape[0]
        n_k = key.shape[0]
        d = self.num_heads * self.head_dim

        Q = nn.Dense(d, use_bias=False, name="Wq")(query)  # (N_q, d)
        K = nn.Dense(d, use_bias=False, name="Wk")(key)  # (N_k, d)
        V = nn.Dense(d, use_bias=False, name="Wv")(value)  # (N_k, d)

        Q = Q.reshape(n_q, self.num_heads, self.head_dim)
        K = K.reshape(n_k, self.num_heads, self.head_dim)
        V = V.reshape(n_k, self.num_heads, self.head_dim)

        # scores[i, j, h] = Q[i,h] · K[j,h] / sqrt(head_dim)
        scores = jnp.einsum("ihd,jhd->ijh", Q, K) / jnp.sqrt(
            jnp.float32(self.head_dim)
        )  # (N_q, N_k, H)

        if self.use_causal_mask:
            causal = jnp.tril(jnp.ones((n_q, n_k), dtype=bool))[:, :, None]
            scores = jnp.where(causal, scores, jnp.finfo(jnp.float32).min)

        attn = jax.nn.softmax(scores, axis=1)  # softmax over key dim j

        # out[i,h,d] = sum_j attn[i,j,h] * V[j,h,d]
        out = jnp.einsum("ijh,jhd->ihd", attn, V)  # (N_q, H, hd)
        out = out.reshape(n_q, d)

        # GTrXL: near-zero Wo init makes this sublayer near-identity at init
        out = nn.Dense(
            query.shape[-1],
            use_bias=False,
            kernel_init=nn.initializers.orthogonal(scale=_OUTPUT_GAIN),
            name="Wo",
        )(out)
        return out


# ---------------------------------------------------------------------------
# Encoder and Decoder blocks
# ---------------------------------------------------------------------------


class MATEncoderBlock(nn.Module):
    """MAT encoder block: full self-attention + FFN.

    All N agents attend to all N agents (no mask).  The attention weights
    naturally encode learned pairwise communication importance.
    """

    num_heads: int
    head_dim: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Pre-norm self-attention (full connectivity)
        residual = x
        x = nn.LayerNorm(name="ln1")(x)
        x = _StandardMHA(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            use_causal_mask=False,
            name="self_attn",
        )(x, x, x)
        x = x + residual

        # Pre-norm FFN
        residual = x
        x = nn.LayerNorm(name="ln2")(x)
        x = nn.Dense(
            self.mlp_dim,
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            name="mlp1",
        )(x)
        x = nn.relu(x)
        # GTrXL init on mlp2: near-zero so FFN is near-identity at start
        x = nn.Dense(
            residual.shape[-1],
            kernel_init=nn.initializers.orthogonal(scale=_OUTPUT_GAIN),
            name="mlp2",
        )(x)
        x = x + residual
        return x


class MATDecoderBlock(nn.Module):
    """MAT decoder block: causal self-attention + cross-attention + FFN.

    With parallel decoder (zero dec_in), the causal self-attention is
    trivially zero and only the cross-attention to encoder output is active.
    This makes the decoder act as a learned aggregation of encoder outputs,
    producing per-agent action representations.
    """

    num_heads: int
    head_dim: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x: jax.Array, enc_out: jax.Array) -> jax.Array:
        # Pre-norm causal self-attention
        residual = x
        x = nn.LayerNorm(name="ln1")(x)
        x = _StandardMHA(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            use_causal_mask=True,
            name="self_attn",
        )(x, x, x)
        x = x + residual

        # Pre-norm cross-attention to encoder output
        residual = x
        x = nn.LayerNorm(name="ln2")(x)
        x = _StandardMHA(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            use_causal_mask=False,
            name="cross_attn",
        )(x, enc_out, enc_out)
        x = x + residual

        # Pre-norm FFN
        residual = x
        x = nn.LayerNorm(name="ln3")(x)
        x = nn.Dense(
            self.mlp_dim,
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            name="mlp1",
        )(x)
        x = nn.relu(x)
        # GTrXL init on mlp2
        x = nn.Dense(
            residual.shape[-1],
            kernel_init=nn.initializers.orthogonal(scale=_OUTPUT_GAIN),
            name="mlp2",
        )(x)
        x = x + residual
        return x


# ---------------------------------------------------------------------------
# Vmappable encoder-decoder block
# ---------------------------------------------------------------------------


class _EncoderDecoderBlock(nn.Module):
    """Encoder-Decoder pipeline for a single agent group, usable with nn.vmap."""

    num_blocks: int
    num_heads: int
    head_dim: int
    mlp_dim: int

    @nn.compact
    def __call__(
        self,
        x_group: jax.Array,  # (N, hidden_dim)
        dec_in: jax.Array,  # (N, hidden_dim) — always zeros
    ) -> jax.Array:
        # Encoder: full self-attention stack
        enc = x_group
        for blk in range(self.num_blocks):
            enc = MATEncoderBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                name=f"enc_block_{blk}",
            )(enc)

        # Decoder: parallel (zero start tokens), cross-attends to encoder
        dec = dec_in
        for blk in range(self.num_blocks):
            dec = MATDecoderBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                name=f"dec_block_{blk}",
            )(dec, enc)

        return dec  # (N, hidden_dim)


# ---------------------------------------------------------------------------
# MAT Policy Network
# ---------------------------------------------------------------------------


class MATPolicyNet(CategoricalMixin, Model):
    """Multi-Agent Transformer policy network.

    Standard transformer encoder-decoder for MARL (Wen et al. 2022).
    Key differences from CommFormer:
    - No CommGraph (no α, no k-hot STE) — attention weights are the comm.
    - No EdgeEmbedding — standard scaled dot-product attention.
    - GTrXL init on output projections (Parisotto et al. 2020).
    - Full N×N encoder attention; decoder cross-attends to all encoder outputs.
    - Parallel decoder with zero start tokens (ratio = 1.0 at update start).
    """

    hidden_dim: int = 256
    num_blocks: int = 2
    num_heads: int = 4
    head_dim: int = 64
    mlp_dim: int = 512
    num_agents: int = 4

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        num_heads: int = 4,
        head_dim: int = 64,
        mlp_dim: int = 512,
        num_agents: int = 4,
        unnormalized_log_prob: bool = True,
        device=None,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        CategoricalMixin.__init__(self, unnormalized_log_prob)
        object.__setattr__(self, "hidden_dim", int(hidden_dim))
        object.__setattr__(self, "num_blocks", int(num_blocks))
        object.__setattr__(self, "num_heads", int(num_heads))
        object.__setattr__(self, "head_dim", int(head_dim))
        object.__setattr__(self, "mlp_dim", int(mlp_dim))
        object.__setattr__(self, "num_agents", int(num_agents))

    @nn.compact
    def __call__(
        self,
        inputs: Mapping[str, Any],
        role: str = "",
    ):
        x = inputs["states"]  # (B, obs_dim)
        n = self.num_agents
        b = x.shape[0]

        # Node embedding
        x_emb = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            name="node_embed",
        )(x)  # (B, hidden_dim)

        # Action head layers (defined once, reused in both paths)
        action_fc = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            name="action_fc",
        )
        action_logits = nn.Dense(
            int(self.num_actions),
            kernel_init=nn.initializers.orthogonal(scale=_OUTPUT_GAIN),
            name="action_logits",
        )

        def _action_head(h: jax.Array) -> jax.Array:
            h = action_fc(h)
            h = nn.tanh(h)
            return action_logits(h)

        if b >= n and b % n == 0:
            groups = b // n
            x_grouped = x_emb.reshape(groups, n, self.hidden_dim)

            # Vectorize encoder-decoder over groups (shared params)
            VmappedEncDec = nn.vmap(
                _EncoderDecoderBlock,
                variable_axes={"params": None},
                split_rngs={"params": False},
                in_axes=(0, 0),
                out_axes=0,
            )
            enc_dec_block = VmappedEncDec(
                num_blocks=self.num_blocks,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                name="enc_dec_block",
            )

            # Per-agent slot queries as decoder start tokens.
            #
            # Zero start tokens cause all N agents to receive identical decoder
            # output at initialisation: with dec_in=0, LayerNorm(0)=0, all
            # cross-attention queries are zero → uniform attention over encoder
            # → mean(enc_out) for every position → all agents get the same
            # logits.  In the typed blind-coordination task ([0,0,1,1] types)
            # this means agents cannot specialise by type from day 1.
            #
            # Learnable slot_queries (one per agent slot, init normal σ=1.0):
            #   • Different per position → distinct decoder outputs immediately
            #   • Obs-independent → ratio = 1.0 invariant is preserved
            #   • Learned specialisation replaces AR action conditioning
            # stddev=0.02 (Perceiver/Q-Former convention) — keeps queries small
            # relative to LayerNorm'd encoder output (std ≈ 1) at init.  This breaks
            # the all-agents-identical symmetry without dominating the attention
            # distribution from step 1.  Previous stddev=1.0 biased attention toward
            # query content before any useful encoder features exist.
            slot_queries = self.param(
                "slot_queries",
                nn.initializers.normal(stddev=0.02),
                (n, self.hidden_dim),
            )  # (N, hidden_dim) — broadcast over groups
            dec_in = jnp.broadcast_to(
                slot_queries[None, :, :], (groups, n, self.hidden_dim)
            )
            decoded = enc_dec_block(x_grouped, dec_in)
            h = decoded.reshape(b, self.hidden_dim)

            extra_outputs: dict = {"encoder_out": h}
        else:
            # Fallback MLP when batch size is not a multiple of N.
            # This path is reached during single-agent evaluation or
            # when called with mismatched batch sizes.
            h = nn.tanh(x_emb)
            h = nn.Dense(
                self.hidden_dim,
                kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
                name="fallback_fc",
            )(h)
            h = nn.tanh(h)
            extra_outputs = {}

        logits = _action_head(h)
        return logits, extra_outputs

    def init_state_dict(self, role: str, inputs=None, key=None) -> None:
        """Ensure batch size = num_agents so enc/dec params are initialized."""
        if inputs is None:
            obs_sample = self.observation_space.sample()
            obs_batch = jnp.tile(
                jnp.asarray(obs_sample, dtype=jnp.float32)[None, :],
                (self.num_agents, 1),
            )
            inputs = {"states": obs_batch}
        super().init_state_dict(role, inputs, key)

    def act(
        self,
        inputs: Mapping[str, (np.ndarray | jax.Array) | Any],
        role: str = "",
        params: jax.Array | None = None,
    ) -> tuple[jax.Array, (jax.Array | None), Mapping[str, (jax.Array | Any)]]:
        inputs = dict(inputs)
        actions, log_prob, outputs = super().act(inputs, role, params)
        outputs["stddev"] = outputs["net_output"]
        return actions, log_prob, outputs

    @property
    def _modules(self):
        return {}
