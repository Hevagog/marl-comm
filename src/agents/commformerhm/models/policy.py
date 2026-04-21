"""CommFormerHM policy network: CommFormer backbone + Task Hopfield Memory.

Combines the CommFormer communication graph architecture with upstream
Task Hopfield pooling from the MAMHM improvements.  The key hypothesis
is that communication is more informative when agent representations
already encode task context.

Architecture pipeline:
  1. obs → node_embed → x_emb
  2. Extract task slice, TaskHopfieldPooling → task_ctx
  3. Gated residual: x_emb += gate * task_ctx  (before comm graph)
  4. CommGraph(α) → adj → EdgeEmbedding → edge_emb
  5. VmappedEncDec (CommFormer encoder + decoder) → decoded
  6. Action head → logits

CTDE: During training the comm graph operates over ALL agents' embeddings.
In 'local_only' mode, the comm graph is replaced with an identity adjacency
for decentralized execution without messages.

References
----------
- Hu et al. 2024 "CommFormer" (ICLR 2024)
- Widrich et al. 2020 "Modern Hopfield Networks for Immune Repertoire"
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
from agents.mamhm.models.task_hopfield import TaskHopfieldPooling

_HIDDEN_GAIN = jnp.sqrt(2.0)
_OUTPUT_GAIN = 0.01


class _EncoderDecoderBlock(nn.Module):
    """Encoder-Decoder pipeline for a single group, usable with nn.vmap.

    Identical to CommFormer's _EncoderDecoderBlock — reused here so that the
    CommFormerHM policy is self-contained and doesn't import private classes.
    """

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
        dec = dec_in
        for blk in range(self.num_blocks):
            dec = DecoderBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                name=f"dec_block_{blk}",
            )(dec, enc, adj, edge_emb)

        return dec  # (N, hidden_dim)


class CommFormerHMPolicyNet(CategoricalMixin, Model):
    """CommFormer + Task Hopfield Memory policy network.

    Augments CommFormer with upstream TaskHopfieldPooling so that each agent's
    representation encodes task context before entering the communication graph.
    """

    hidden_dim: int = 256
    num_blocks: int = 2
    num_heads: int = 4
    head_dim: int = 64
    mlp_dim: int = 512
    num_agents: int = 4
    sparsity: float = 0.5

    # Task Hopfield params
    use_task_hopfield: bool = True
    task_hopfield_num_heads: int = 4
    task_hopfield_beta: float = 2.0
    task_hopfield_gate_init: float = -3.0

    # Execution mode: 'ctde' (full graph) or 'local_only' (skip comm)
    execution_mode: str = "ctde"

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
        sparsity: float = 0.5,
        use_task_hopfield: bool = True,
        task_hopfield_num_heads: int = 4,
        task_hopfield_beta: float = 2.0,
        task_hopfield_gate_init: float = -3.0,
        execution_mode: str = "ctde",
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
        object.__setattr__(self, "use_task_hopfield", bool(use_task_hopfield))
        object.__setattr__(
            self, "task_hopfield_num_heads", int(task_hopfield_num_heads)
        )
        object.__setattr__(self, "task_hopfield_beta", float(task_hopfield_beta))
        object.__setattr__(
            self, "task_hopfield_gate_init", float(task_hopfield_gate_init)
        )
        object.__setattr__(self, "execution_mode", str(execution_mode))

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
            obs_grouped = x.reshape(groups, n, -1)

            # --- Task Hopfield pooling (before comm graph) ---
            if self.use_task_hopfield:
                task_hopfield = TaskHopfieldPooling(
                    d_model=self.hidden_dim,
                    num_query_heads=self.task_hopfield_num_heads,
                    beta=self.task_hopfield_beta,
                    gate_init=self.task_hopfield_gate_init,
                    name="task_hopfield",
                )
                x_grouped = task_hopfield(obs_grouped, x_grouped)

            # --- Communication graph ---
            if self.execution_mode == "local_only":
                # No communication: identity adjacency (each agent attends to self only)
                adj = jnp.eye(n)
                # No learnable α in local_only mode — embed the identity itself
                # so edge_emb is well-defined in the branch that follows.
                alpha_raw = adj
            else:
                # Full CTDE communication graph.
                # Always use deterministic k-argmax (CommFormer Eq. 12):
                # same adj at rollout and training ensures consistent log-probs.
                # Alpha receives gradients via the straight-through estimator.
                comm = CommGraph(
                    num_agents=n,
                    sparsity=self.sparsity,
                    name="comm_graph",
                )
                # BUG-HM-001 fix: never inject gumbel_rng at training time.
                # Rollout uses ar_key path with deterministic adj (training=False).
                # Training must also use deterministic adj so that the recomputed
                # log-probs match the stored rollout log-probs → ratio ≈ 1 → KL stable.
                # Alpha still receives gradients via the STE in _k_hot.
                adj, alpha_raw = comm(rng=None, training=False)
                # BUG-HM-003 fix: guarantee self-loops so that the decoder causal
                # mask (agent-0 restricted to column 0 only) never produces an
                # all-masked row.  Without this, adj[0,0]=0 → all -inf → NaN softmax.
                adj = jnp.maximum(adj, jnp.eye(n, dtype=adj.dtype))

            # Edge embeddings consume continuous α (CommFormer Eq. 2) — using
            # binary adj would collapse to 2 distinct vectors.
            edge_emb = EdgeEmbedding(
                embed_dim=self.head_dim,
                name="edge_embed",
            )(alpha_raw)

            # Vectorize over groups
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
            dec_in = jnp.zeros((groups, n, self.hidden_dim))
            decoded = enc_dec_block(x_grouped, adj, edge_emb, dec_in)
            h = decoded.reshape(b, self.hidden_dim)

            comm_outputs = {
                "adj_matrices": adj,
                "encoder_out": h,
            }
        else:
            # Fallback: simple MLP when batch isn't grouped
            h = nn.tanh(x_emb)
            h = nn.Dense(
                self.hidden_dim,
                kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
                name="fallback_fc",
            )(h)
            h = nn.tanh(h)

        # --- Action head ---
        logits = _action_head(h)

        return logits, comm_outputs

    def init_state_dict(self, role: str, inputs=None, key=None) -> None:
        """Ensure batch = num_agents so encoder/decoder params are initialised."""
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
        inputs = dict(inputs)
        actions, log_prob, outputs = super().act(inputs, role, params)
        outputs["stddev"] = outputs["net_output"]
        return actions, log_prob, outputs

    @property
    def _modules(self):
        return {}
