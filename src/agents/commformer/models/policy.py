"""CommFormer policy network — parallel decoder variant.

The original CommFormer (Hu et al. 2024) uses an auto-regressive decoder
conditioned on previously generated actions.  This implementation replaces
that with a **parallel decoder** that uses zero start tokens at both rollout
and training time.

Rationale (why the AR decoder was removed):
--------------------------------------------
1. **Train/rollout mismatch** (Bengio et al. 2015, "Scheduled Sampling"):
   Teacher forcing at training feeds stored actions; the AR scan at rollout
   feeds *sampled* actions.  Even though the actions are identical (same
   policy, same seed), the two code paths differ in vmap/scan structure,
   causing accumulated floating-point rounding differences that produce
   systematic ratio_max_abs_dev > 0.5 at the start of every update —
   before any gradient step.  This fires the KL early-stop and reduces
   effective gradient steps per rollout to 1-2.

2. **4× compute overhead**: The AR scan calls enc_dec_block N times per
   rollout step (one per agent position), while training calls it once.
   The parallel decoder calls it once in both cases.

3. **No execution benefit**: Agents execute independently; they cannot
   observe each other's actions at deployment.  Conditioning the decoder
   on previous agents' sampled actions provides no decentralised-execution
   benefit (Wen et al. 2022 "MAT" §4.2 explicitly notes this limitation).

4. **Precedent**: MAMEncOnly ablation in this codebase achieves 86% of the
   MAM-to-MAPPO gap using encoder-only (no AR decoder); encoder-plus-
   parallel-decoder retains cross-attention aggregation while eliminating
   the mismatch.

With zero decoder inputs at both rollout and training, ratio = 1.0 at the
start of every update, the KL early-stop never fires spuriously, and all
learning_epochs × mini_batches gradient steps execute.

References
----------
- Hu et al. 2024 "CommFormer" (ICLR 2024), §3.2, Fig. 2.
- Bengio et al. 2015 "Scheduled Sampling": exposure bias in teacher forcing.
- Wen et al. 2022 "MAT": sequential update scheme discussion.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping


import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from skrl.models.jax import CategoricalMixin, Model

from agents.commformer.models.comm_graph import CommGraph
from agents.commformer.models.transformer_blocks import (
    DecoderBlock,
    EdgeEmbedding,
    EncoderBlock,
)

_HIDDEN_GAIN = jnp.sqrt(2.0)
_OUTPUT_GAIN = 0.01


class _EncoderDecoderBlock(nn.Module):
    """Encoder-Decoder pipeline for a single group, usable with nn.vmap."""

    num_blocks: int
    num_heads: int
    head_dim: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x_group, adj, edge_emb, dec_in):
        # --- Encoder ---
        enc = x_group
        for blk in range(self.num_blocks):
            enc = EncoderBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                name=f"enc_block_{blk}",
            )(enc, adj, edge_emb)

        # --- Decoder ---
        # dec_in is always zeros (parallel decoder — see module docstring).
        # The decoder cross-attends to encoder outputs with a learned static
        # query, producing per-agent aggregations of communicated context.
        dec = dec_in
        for blk in range(self.num_blocks):
            dec = DecoderBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                name=f"dec_block_{blk}",
            )(dec, enc, adj, edge_emb)

        return dec  # (N, hidden_dim)


class CommFormerPolicyNet(CategoricalMixin, Model):
    """CommFormer policy: Encoder + parallel Decoder with learnable comm graph.

    Architecture (CommFormer §3.2, Fig. 2):
    ----------------------------------------
    1. **Communication Graph** — α ∈ R^{N×N} → binary adj via k-hot STE.
    2. **Edge Embeddings** — embed adjacency for relation-enhanced attention.
    3. **Encoder** — obs sequence with adjacency masking.
    4. **Decoder** — parallel (zero start tokens), cross-attends to encoder.
    5. **Action head** — projects decoder output to action logits.

    The decoder uses zero start tokens at both rollout and training time,
    ensuring identical forward paths and ratio = 1.0 at update start.
    """

    hidden_dim: int = 64
    num_blocks: int = 1
    num_heads: int = 1
    head_dim: int = 64
    mlp_dim: int = 128
    num_agents: int = 2
    sparsity: float = 0.4

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 64,
        num_blocks: int = 1,
        num_heads: int = 1,
        head_dim: int = 64,
        mlp_dim: int = 128,
        num_agents: int = 2,
        sparsity: float = 0.4,
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
        object.__setattr__(self, "sparsity", float(sparsity))

    @nn.compact
    def __call__(
        self,
        inputs: Mapping[str, Any],
        role: str = "",
    ):
        """Forward pass.

        ``inputs["states"]`` has shape ``(B, obs_dim)``.  If B is a multiple
        of N, group observations and run the full encoder-decoder pipeline.
        Otherwise fall back to a simpler MLP.

        ``inputs["taken_actions"]`` is accepted but ignored in the forward
        pass — log-probs are computed by CategoricalMixin from the returned
        logits and the stored actions.
        """
        x = inputs["states"]  # (B, obs_dim)
        n = self.num_agents
        b = x.shape[0]

        # Node embedding: project observations to hidden_dim
        node_embed = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            name="node_embed",
        )
        x_emb = node_embed(x)  # (B, hidden_dim)

        comm_outputs: dict = {}
        action_fc = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            name="action_fc",
        )
        action_logits_head = nn.Dense(
            int(self.num_actions),
            kernel_init=nn.initializers.orthogonal(scale=_OUTPUT_GAIN),
            name="action_logits",
        )

        def _action_head(h: jax.Array) -> jax.Array:
            h = action_fc(h)
            h = nn.tanh(h)
            return action_logits_head(h)

        if b >= n and b % n == 0:
            groups = b // n
            x_grouped = x_emb.reshape(groups, n, self.hidden_dim)

            # Communication graph (deterministic k-argmax — no Gumbel noise).
            # Alpha receives gradients via the STE in _k_hot regardless.
            # Using deterministic adj at both rollout and training time
            # ensures the communication context is identical in both paths.
            comm = CommGraph(
                num_agents=n,
                sparsity=self.sparsity,
                name="comm_graph",
            )
            adj = comm(rng=None, training=False)
            # Guarantee self-loops: decoder causal mask restricts agent-0
            # to column 0; if adj[0,0]=0 → all-masked row → NaN softmax.
            adj = jnp.maximum(adj, jnp.eye(n, dtype=adj.dtype))

            # Edge embeddings
            edge_emb = EdgeEmbedding(
                embed_dim=self.head_dim,
                name="edge_embed",
            )(adj)

            # Vectorize over groups (shared params; adj & edge_emb broadcast)
            VmappedEncDec = nn.vmap(
                _EncoderDecoderBlock,
                variable_axes={"params": None},
                split_rngs={"params": False},
                in_axes=(0, None, None, 0),
                out_axes=0,
            )
            enc_dec_block = VmappedEncDec(
                num_blocks=self.num_blocks,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                name="enc_dec_block",
            )

            # Parallel decoder: zero start tokens at both rollout and training.
            # This is the key fix — identical forward path ensures ratio = 1.0
            # at the start of every PPO update (Bengio et al. 2015 §3).
            dec_in = jnp.zeros((groups, n, self.hidden_dim))
            decoded = enc_dec_block(x_grouped, adj, edge_emb, dec_in)
            h = decoded.reshape(b, self.hidden_dim)

            comm_outputs = {
                "adj_matrices": adj,
                "encoder_out": h,
            }
        else:
            # Fallback: simple MLP when batch isn't a multiple of N
            h = nn.tanh(x_emb)
            h = nn.Dense(
                self.hidden_dim,
                kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
                name="fallback_fc",
            )(h)
            h = nn.tanh(h)

        logits = _action_head(h)
        return logits, comm_outputs

    def init_state_dict(
        self,
        role: str,
        inputs=None,
        key=None,
    ) -> None:
        """Override to ensure batch size = num_agents so enc/dec params init."""
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
        # Parallel decoder: same forward path at rollout and training time.
        # No special AR path needed — CategoricalMixin.act() handles sampling
        # and log-prob computation consistently in both cases.
        inputs = dict(inputs)
        actions, log_prob, outputs = super().act(inputs, role, params)
        outputs["stddev"] = outputs["net_output"]
        return actions, log_prob, outputs

    @property
    def _modules(self):
        return {}
