"""HopfieldMsgPooling: global coordination bottleneck for MAGIC+Hopfield.

K learned query vectors attend over N processed message embeddings and
produce K pooled coordination-context vectors.  These are averaged and
broadcast-concatenated to every agent's processed message, giving the
message decoder access to a permutation-invariant global summary that
complements the local pairwise GAT aggregation.

Motivation
----------
MAGIC's MessageProcessor is local: each agent aggregates only from its
direct graph-neighbors.  With sparse Gumbel-Softmax edges some agents
may receive few or zero messages.  HopfieldMsgPooling provides a parallel
*global* path — it sees all N processed messages regardless of topology —
so even structurally isolated agents benefit from team-wide context.

Architecture
------------
    processed : (groups, N, D)
        ─►  K queries attend → weights (groups, K, N)
        ─►  pooled          : (groups, K, D)
        ─►  mean over K     : (groups, 1, D)
        ─►  broadcast       : (groups, N, D)
        ─►  concat(processed, ctx) : (groups, N, 2D)

References
----------
- Ramsauer et al. 2021 "Hopfield Networks is All You Need" §3.3: HopfieldPooling
- Widrich et al. 2020 "DeepRC": Hopfield attention as set pooling
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class HopfieldSelfContext(nn.Module):
    """Per-agent Hopfield prototype retrieval as a learned self-loop replacement.

    Each agent queries a shared prototype bank and retrieves the nearest
    coordination attractor as a gated residual on its own message.  This
    replaces the forced ``+jnp.eye(N)`` self-loop in the MAGIC adjacency so
    the Gumbel-Softmax scheduler can be a pure inter-agent graph.

    Parameters
    ----------
    d_model : int
        Message embedding dimension D.
    num_prototypes : int
        Prototype bank size K (default 8).
    beta : float
        Inverse temperature for retrieval softmax.
    """

    d_model: int
    num_prototypes: int = 8
    beta: float = 1.0
    gate_init: float = 0.0

    def setup(self) -> None:
        self.prototypes = self.param(
            "prototypes",
            nn.initializers.lecun_normal(),
            (self.num_prototypes, self.d_model),
        )
        # gate_init configurable: 0.0 (sigmoid 0.5) is the v2 default after
        # the gate-starvation fix; -1.0 (sigmoid 0.27) damps random-prototype
        # noise at init when β > 1 sharpens retrieval magnitudes.
        self.gate = self.param(
            "gate",
            nn.initializers.constant(self.gate_init),
            (),
        )

    def __call__(self, messages: jax.Array) -> jax.Array:
        """Retrieve self-context for each agent via prototype attention.

        Parameters
        ----------
        messages : (N, d_model)

        Returns
        -------
        augmented : (N, d_model) — messages + sigmoid(gate) * retrieved
        """
        scores = self.beta * jnp.einsum("nd,kd->nk", messages, self.prototypes)
        weights = jax.nn.softmax(scores, axis=-1)  # (N, K)
        retrieved = jnp.einsum("nk,kd->nd", weights, self.prototypes)  # (N, D)
        return messages + jax.nn.sigmoid(self.gate) * retrieved


class HopfieldMsgPooling(nn.Module):
    """Global coordination bottleneck via Hopfield pooling over agent messages.

    Parameters
    ----------
    d_model : int
        Message embedding dimension D.
    num_queries : int
        Number of learned pooling query prototypes K (default 4).
    beta : float
        Inverse temperature for Hopfield softmax.  Higher → sharper
        winner-take-all retrieval; lower → broader averaging.
    """

    d_model: int
    num_queries: int = 4
    beta: float = 1.0

    def setup(self) -> None:
        # K learned query prototypes: (K, d_model)
        # lecun_normal gives unit-variance projections suitable for
        # attending over unit-scaled processed messages.
        self.queries = self.param(
            "queries",
            nn.initializers.lecun_normal(),
            (self.num_queries, self.d_model),
        )

    def __call__(self, processed: jax.Array) -> jax.Array:
        """Pool agent messages and produce a per-agent global context vector.

        Parameters
        ----------
        processed : (groups, N, d_model)
            Post-GAT agent message embeddings.

        Returns
        -------
        augmented : (groups, N, 2 * d_model)
            Concatenation of [processed_i || global_ctx] per agent i.
            global_ctx is the same for all agents (permutation-invariant).
        """
        # Attention scores: (groups, K, N)
        # scores[g, k, n] = beta * queries[k] · processed[g, n]
        scores = self.beta * jnp.einsum("kd,gnd->gkn", self.queries, processed)

        # Normalize over agents (axis=-1): each query attends over all N agents
        weights = jax.nn.softmax(scores, axis=-1)  # (groups, K, N)

        # Pooled context per query: (groups, K, d_model)
        pooled = jnp.einsum("gkn,gnd->gkd", weights, processed)

        # Average over K queries → single global context per group
        # Shape: (groups, d_model)
        global_ctx = jnp.mean(pooled, axis=1)

        # Broadcast to all agents: (groups, N, d_model)
        n = processed.shape[1]
        global_ctx_broadcast = jnp.broadcast_to(
            global_ctx[:, None, :], (processed.shape[0], n, self.d_model)
        )

        # Concatenate along feature axis: (groups, N, 2*d_model)
        return jnp.concatenate([processed, global_ctx_broadcast], axis=-1)
