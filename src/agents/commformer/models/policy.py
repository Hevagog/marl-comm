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
        node_embed = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            name="node_embed",
        )
        x_emb = node_embed(x)  # (B, hidden_dim)

        comm_outputs: dict = {}
        action_embed = nn.Embed(
            num_embeddings=int(self.num_actions),
            features=self.hidden_dim,
            embedding_init=nn.initializers.orthogonal(scale=_HIDDEN_GAIN),
            name="action_embed",
        )
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
            ar_key = inputs.get("ar_key", None)

            # --- Decoder input: action embeddings (teacher forcing) or zeros ---
            # During training _update_policy_fixed passes "taken_actions" (B, 1).
            # We embed them, shift right (prepend a zero start token, drop last),
            # so dec_in[group, m, :] = embed(a_{m-1}) for m > 0 and zero for m=0.
            # This implements CommFormer §3.2 Eq. 5 / Algorithm 1 Step 11.
            taken_acts = inputs.get("taken_actions", None)
            if taken_acts is not None:
                act_idx = taken_acts.reshape(b).astype(jnp.int32)  # (B,)
                act_emb = action_embed(act_idx)  # (B, hidden_dim)
                act_emb_grouped = act_emb.reshape(groups, n, self.hidden_dim)
                start_tok = jnp.zeros((groups, 1, self.hidden_dim))
                dec_in = jnp.concatenate(
                    [start_tok, act_emb_grouped[:, :-1, :]], axis=1
                )  # (groups, N, hidden_dim)
            else:
                # Initialise the embedding layer even when no actions are given
                _dummy_idx = jnp.zeros(n, dtype=jnp.int32)
                action_embed(_dummy_idx)

            # --- Communication graph (learned α) ---
            # Use stochastic Gumbel sampling only for PPO training updates
            # (teacher-forced ``taken_actions`` path). During rollout/eval we use
            # deterministic k-argmax, matching CommFormer Eq. 12.
            comm = CommGraph(
                num_agents=n,
                sparsity=self.sparsity,
                name="comm_graph",
            )
            gumbel_rng = inputs.get("gumbel_rng", None)
            graph_training = taken_acts is not None and gumbel_rng is not None
            adj = comm(rng=gumbel_rng, training=graph_training)  # (N, N)
            # BUG-C-002 fix: guarantee self-loops so that the decoder causal mask
            # (which restricts agent-0 to column 0 only) never produces an
            # all-masked row.  Without this, adj[0,0]=0 makes every score -inf
            # → softmax NaN.  Self-loops are assumed in the CommFormer paper.
            adj = jnp.maximum(adj, jnp.eye(n, dtype=adj.dtype))

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
            enc_dec_block = VmappedEncDec(
                num_blocks=self.num_blocks,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                name="enc_dec_block",
            )

            if ar_key is not None:
                # CommFormer paper uses autoregressive decoding at execution time:
                # sample a_1, feed it back, then decode a_2, etc.  Recompute the
                # decoder from the growing prefix at each step. With only 4 agents,
                # this paper-faithful path is cheap and avoids the severe train/test
                # mismatch of using all-zero decoder inputs during rollout.
                init_dec_in = jnp.zeros((groups, n, self.hidden_dim))
                positions = jnp.arange(n, dtype=jnp.int32)
                act_dim = int(self.num_actions)

                def _scan_body(carry, pos):
                    rng, dec_prefix = carry
                    decoded = enc_dec_block(x_grouped, adj, edge_emb, dec_prefix)
                    logits_i = _action_head(decoded[:, pos, :])  # (groups, act_dim)

                    rng, subkey = jax.random.split(rng)
                    action_i = jax.random.categorical(subkey, logits_i)  # (groups,)
                    log_probs_i = jax.nn.log_softmax(logits_i)
                    log_prob_i = jnp.take_along_axis(
                        log_probs_i, action_i[:, None], axis=-1
                    ).squeeze(-1)

                    next_emb = action_embed(action_i.astype(jnp.int32))
                    dec_prefix = jax.lax.cond(
                        pos + 1 < n,
                        lambda d: d.at[:, pos + 1, :].set(next_emb),
                        lambda d: d,
                        dec_prefix,
                    )
                    return (rng, dec_prefix), (action_i, log_prob_i, logits_i)

                _, (all_actions, all_log_probs, all_logits) = jax.lax.scan(
                    _scan_body,
                    (ar_key, init_dec_in),
                    positions,
                )

                # BUG-C-003 fix: use agent-major output order.
                # all_actions shape (N, groups): all_actions[i, g] = agent i, env g.
                # .reshape(-1) → [a0_g0, a0_g1, ..., a0_g{K-1}, a1_g0, ...]
                # CommFormerMAPPO.act() slices [i*K : (i+1)*K] to extract agent i.
                # The previous .T.reshape(-1) produced env-major order, making
                # each slice a mix of different agents → wrong stored log-probs.
                actions_flat = all_actions.reshape(-1)[:, None]
                log_probs_flat = all_log_probs.reshape(-1)[:, None]
                logits_flat = all_logits.reshape(-1, act_dim)
                return actions_flat, {
                    "log_probs": log_probs_flat,
                    "logits": logits_flat,
                    "adj_matrices": adj,
                    "autoregressive": True,
                }

            if taken_acts is not None:
                decoded = enc_dec_block(
                    x_grouped, adj, edge_emb, dec_in
                )  # (groups, N, hidden_dim)
                decoded_flat = decoded.reshape(b, self.hidden_dim)  # (B, hidden_dim)
                h = decoded_flat
            else:
                # Zero-start fallback for analysis / non-AR direct forward
                dec_in = jnp.zeros((groups, n, self.hidden_dim))
                decoded = enc_dec_block(x_grouped, adj, edge_emb, dec_in)
                h = decoded.reshape(b, self.hidden_dim)

            # Expose graph and per-agent representations for analysis.
            # adj_matrices: (N, N) hard binary k-hot adjacency (static graph).
            # encoder_out: (B, hidden_dim) post-encoder-decoder representations.
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
        # Rollout path: __call__ already returns sampled actions and log-probs.
        if "ar_key" in inputs:
            with jax.default_device(self.device):
                p = self.state_dict.params if params is None else params
                net_output, extra = self.apply(p, inputs, role)
                actions = net_output
                log_probs = extra["log_probs"]
                outputs = {"net_output": net_output}
                outputs.update(extra)
                return actions, log_probs, outputs

        # Training path (teacher-forced PPO update).
        #
        # IMPORTANT: do NOT inject gumbel_rng here.
        #
        # The PPO importance ratio r_t^m = π_θ / π_{θ_old} (paper Eq. 5) is only
        # valid when both numerator (recomputed here) and denominator (stored during
        # rollout) use the SAME adjacency matrix.  During rollout, act() takes the
        # ar_key path which calls CommGraph with training=False → deterministic
        # k-argmax adj (paper Eq. 12, no Gumbel noise).  If we inject gumbel_rng
        # here, CommGraph produces a different Gumbel-sampled adj (paper Eq. 11) →
        # different encoder/decoder context → different logits → log-prob mismatch
        # → ratio_max_abs_dev ≈ 2.0, ratio_clipped_frac ≈ 0.53 → KL threshold
        # fires on the first mini-batch → zero gradient steps the entire run.
        #
        # Without gumbel_rng, CommGraph uses deterministic k-argmax (training=False)
        # matching rollout.  Alpha still receives gradients via the straight-through
        # estimator (STE) in _k_hot: hard − stop_grad(softmax(α)) + softmax(α),
        # so d(loss)/d(α) flows through softmax(α).  No Gumbel needed for that.
        inputs = dict(inputs)

        actions, log_prob, outputs = super().act(inputs, role, params)
        outputs["stddev"] = outputs["net_output"]
        return actions, log_prob, outputs

    @property
    def _modules(self):
        return {}
