"""CommFormer policy network (Decoder).

Implements the auto-regressive decoder that generates actions for
each agent sequentially, conditioned on the encoded observations
and previously generated actions.

References
----------
- Hu et al. 2024 "CommFormer" (ICLR 2024), §3.2, Eq. 5:
  π_θ^m(a^m | ô_{1:n}, a_{1:m-1})  — auto-regressive action generation.
- Wen et al. 2022 "MAT": sequential update scheme.
"""

from __future__ import annotations

from typing import Any, Mapping


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

        # --- Decoder (encoder output as context) ---
        # dec_in is the right-shifted action embedding sequence (teacher forcing
        # during training) or zero start tokens during rollout.  CommFormer §3.2,
        # Eq. 5 / Algorithm 1 Step 11: "Input o¹,...,oⁿ and a¹,...,aⁿ⁻¹".
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
    """CommFormer policy: Encoder + Decoder with learnable communication graph.

    Architecture (CommFormer §3.2, Fig. 2)
    ---------------------------------------
    1. **Communication Graph** — learnable ``α ∈ R^{N×N}`` → binary adj.
    2. **Edge Embeddings** — embed adjacency for relation-enhanced attention.
    3. **Encoder** — processes observation sequence with adjacency masking.
    4. **Decoder** — auto-regressively generates actions per agent.
    5. **Action head** — projects decoder output to action logits.

    For integration with skrl's rollout collection (which queries one agent
    at a time), we handle both single-agent and multi-agent batch shapes.
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
        of N (num_agents), we group observations and run the full encoder-
        decoder pipeline.  Otherwise, we fall back to a simpler MLP policy.
        """
        x = inputs["states"]  # (B, obs_dim)
        n = self.num_agents
        b = x.shape[0]

        # Node embedding: project observations to hidden_dim
        x_emb = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            name="node_embed",
        )(x)  # (B, hidden_dim)

        comm_outputs: dict = {}

        if b >= n and b % n == 0:
            groups = b // n
            x_grouped = x_emb.reshape(groups, n, self.hidden_dim)

            # --- Decoder input: action embeddings (teacher forcing) or zeros ---
            # During training _update_policy_fixed passes "taken_actions" (B, 1).
            # We embed them, shift right (prepend a zero start token, drop last),
            # so dec_in[group, m, :] = embed(a_{m-1}) for m > 0 and zero for m=0.
            # This implements CommFormer §3.2 Eq. 5 / Algorithm 1 Step 11.
            # During rollout no actions are available yet → zero start tokens.
            taken_acts = inputs.get("taken_actions", None)
            if taken_acts is not None:
                act_idx = taken_acts.reshape(b).astype(jnp.int32)  # (B,)
                act_emb = nn.Embed(
                    num_embeddings=int(self.num_actions),
                    features=self.hidden_dim,
                    embedding_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
                    name="action_embed",
                )(act_idx)  # (B, hidden_dim)
                act_emb_grouped = act_emb.reshape(groups, n, self.hidden_dim)
                start_tok = jnp.zeros((groups, 1, self.hidden_dim))
                dec_in = jnp.concatenate(
                    [start_tok, act_emb_grouped[:, :-1, :]], axis=1
                )  # (groups, N, hidden_dim)
            else:
                # initialise on the first call
                _dummy_idx = jnp.zeros(n, dtype=jnp.int32)
                nn.Embed(
                    num_embeddings=int(self.num_actions),
                    features=self.hidden_dim,
                    embedding_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
                    name="action_embed",
                )(_dummy_idx)
                dec_in = jnp.zeros((groups, n, self.hidden_dim))

            # --- Communication graph (learned α) ---
            # During training (taken_actions present) sample with Gumbel noise so
            # that α receives a richer gradient signal (CommFormer Eq. 11).
            # During rollout/eval use the deterministic k-argmax (Eq. 12).
            comm = CommGraph(
                num_agents=n,
                sparsity=self.sparsity,
                name="comm_graph",
            )
            gumbel_rng = inputs.get("gumbel_rng", None)
            adj = comm(rng=gumbel_rng, training=(gumbel_rng is not None))  # (N, N)

            # Edge embeddings
            edge_emb = EdgeEmbedding(
                embed_dim=self.head_dim,
                name="edge_embed",
            )(adj)  # (N, N, head_dim)

            # Vectorize over groups (shared params; adj & edge_emb broadcast)
            VmappedEncDec = nn.vmap(
                _EncoderDecoderBlock,
                variable_axes={"params": None},
                split_rngs={"params": False},
                in_axes=(0, None, None, 0),  # dec_in is group-specific
                out_axes=0,
            )
            decoded = VmappedEncDec(
                num_blocks=self.num_blocks,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                name="enc_dec_block",
            )(x_grouped, adj, edge_emb, dec_in)  # (groups, N, hidden_dim)
            decoded_flat = decoded.reshape(b, self.hidden_dim)  # (B, hidden_dim)
            h = decoded_flat

            # Expose graph and per-agent representations for analysis.
            # adj_matrices: (N, N) hard binary k-hot adjacency (static graph).
            # encoder_out: (B, hidden_dim) post-encoder-decoder representations.
            comm_outputs = {
                "adj_matrices": adj,
                "encoder_out": decoded_flat,
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
        h = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            name="action_fc",
        )(h)
        h = nn.tanh(h)

        logits = nn.Dense(
            int(self.num_actions),
            kernel_init=nn.initializers.orthogonal(scale=_OUTPUT_GAIN),
            name="action_logits",
        )(h)

        return logits, comm_outputs

    def init_state_dict(
        self,
        role: str,
        inputs=None,
        key=None,
    ) -> None:
        """Override to ensure batch size = num_agents so encoder/decoder params init.

        Includes ``taken_actions`` and ``gumbel_rng`` so that ``action_embed``
        and the stochastic graph path are initialised on the first call.
        """
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
        inputs: Mapping[str, (np.ndarray | jax.Array) | Any],
        role: str = "",
        params: jax.Array | None = None,
    ) -> tuple[jax.Array, (jax.Array | None), Mapping[str, (jax.Array | Any)]]:
        # Inject a per-call Gumbel RNG so CommGraph samples stochastically
        # during training (CommFormer Eq. 11).  The same _c_key / _c_i pattern
        # used by MAGICPolicyNet ensures a fresh key each call without
        # requiring changes to the skrl training infrastructure.
        with jax.default_device(self.device):
            self._c_i += 1
            gumbel_rng = jax.random.fold_in(self._c_key, self._c_i + 1_000_000_000)

        # Build a new dict so we don't mutate the caller's Mapping.
        inputs = dict(inputs)
        inputs["gumbel_rng"] = gumbel_rng

        actions, log_prob, outputs = super().act(inputs, role, params)
        outputs["stddev"] = outputs["net_output"]
        return actions, log_prob, outputs

    @property
    def _modules(self):
        return {}
