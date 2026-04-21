"""Learnable communication graph for CommFormer.

Implements the adjacency matrix parameterization with Gumbel-Max
k-hot sampling for training and deterministic k-argmax at inference.

References
----------
- Hu et al. 2024 "CommFormer" (ICLR 2024), §3.2: Communication Graph.
- Equations 11 (training) and 12 (execution).
- Jang et al. 2016: Gumbel-Max trick.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class CommGraph(nn.Module):
    """Learnable communication graph parameterized by α ∈ R^{N×N}.

    During **training**, uses the Gumbel-Max trick extended to k-hot
    (Eq. 11) to produce a differentiable binary adjacency matrix.

    During **inference**, uses deterministic k-argmax (Eq. 12).

    Parameters
    ----------
    num_agents : int
        Number of agents (N).
    sparsity : float
        Sparsity parameter S ∈ (0, 1].  Each agent sends messages to
        at most ``k = round(S * N)`` other agents (CommFormer §3.1).
    """

    num_agents: int
    sparsity: float = 0.4

    def setup(self):
        # stddev=0.1 matches the CommFormer reference implementation
        # (github.com/charleshsc/CommFormer, graph.py).  With stddev=1.0,
        # initial α differences dominate gradient updates: the STE path
        # produces gradients of order ~1e-7 (measured), so top-k never
        # flips from its random init and the learnable graph is
        # effectively frozen.  With stddev=0.1, initial entries are on
        # the same order as accumulated gradients over a few updates,
        # allowing the top-k selection to reflect learned signal.
        self.alpha = self.param(
            "alpha",
            nn.initializers.normal(stddev=0.1),
            (self.num_agents, self.num_agents),
        )

    def __call__(
        self,
        rng: jax.Array | None = None,
        training: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Returns
        -------
        adj : jax.Array, shape ``(N, N)``
            Binary adjacency matrix where ``adj[i, j] = 1`` means agent j
            sends a message to agent i.
        alpha : jax.Array, shape ``(N, N)``
            Raw continuous parameter (pre-top-k).  Consumers that embed
            relational magnitude (CommFormer §3.2, Eq. 2) should use this
            rather than the binary adj.
        """
        n = self.num_agents
        k = max(1, int(round(self.sparsity * n)))

        if training and rng is not None:
            # Gumbel-Max k-hot (CommFormer Eq. 11)
            gumbels = jax.random.gumbel(rng, shape=(n, n))
            perturbed = self.alpha + gumbels
        else:
            # Deterministic k-argmax (CommFormer Eq. 12)
            perturbed = self.alpha

        # k-hot: select top-k per row
        adj = _k_hot(perturbed, k)
        return adj, self.alpha


def _k_hot(logits: jax.Array, k: int) -> jax.Array:
    """Produce a k-hot binary matrix with straight-through gradients.

    For each row, selects the top-k values and sets them to 1.
    Uses straight-through estimator for gradient flow.

    Parameters
    ----------
    logits : shape ``(N, N)``
    k : int, number of 1s per row.

    Returns
    -------
    shape ``(N, N)`` binary matrix.
    """
    # Soft approximation: softmax over logits, scaled
    soft = jax.nn.softmax(logits, axis=-1)

    # Hard: top-k binary
    _, top_indices = jax.lax.top_k(logits, k)

    # Create one-hot for each top-k index and sum
    hard = jnp.zeros_like(logits)
    rows = jnp.arange(logits.shape[0])[:, None]
    hard = hard.at[rows, top_indices].set(1.0)

    # Straight-through gradient estimator
    return hard - jax.lax.stop_gradient(soft) + soft
