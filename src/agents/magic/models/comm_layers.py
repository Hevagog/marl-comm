from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from .utils import gumbel_softmax


class GATLayer(nn.Module):
    """Single-head Graph Attention layer.

    Implements the attention mechanism from Velickovic et al. 2018 as used
    in the MAGIC paper (Niu et al. 2021, §4.2 Eq. 5 and §4.3 Eq. 8).

    Parameters
    ----------
    out_features : int
        Output feature dimension per node (D' in the paper).
    negative_slope : float
        Negative slope for LeakyReLU (default 0.2).
    use_mask : bool
        If True, multiply attention by adjacency ``mask`` (Message Processor,
        Eq. 8).  If False, attend over all nodes (Scheduler GAT encoder).
    """

    out_features: int
    negative_slope: float = 0.2
    use_mask: bool = False

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        n = x.shape[0]

        # W_S · m  or  W_P · m  —  linear projection  (MAGIC Eq. 5 / Eq. 8)
        Wh = nn.Dense(self.out_features, use_bias=False, name="W")(x)  # (N, D')

        # Pairwise concatenation [Wh_i || Wh_j] for all (i, j)
        Wh_i = jnp.repeat(Wh[:, None, :], n, axis=1)  # (N, N, D')
        Wh_j = jnp.repeat(Wh[None, :, :], n, axis=0)  # (N, N, D')
        concat = jnp.concatenate([Wh_i, Wh_j], axis=-1)  # (N, N, 2D')

        # a^T [Wh_i || Wh_j]  —  attention vector
        attn = nn.Dense(1, use_bias=False, name="a")(concat).squeeze(-1)  # (N, N)
        attn = nn.leaky_relu(attn, negative_slope=self.negative_slope)

        if self.use_mask and mask is not None:
            # MAGIC Eq. 8: multiply by g_{ij} before softmax.
            # Where mask == 0, set attention to -inf so softmax → 0.
            attn = jnp.where(mask > 0, attn, jnp.finfo(jnp.float32).min)

        alpha = jax.nn.softmax(attn, axis=-1)  # (N, N)

        if self.use_mask and mask is not None:
            # Retain gradient of g_{ij} for end-to-end training (MAGIC §4.3).
            alpha = alpha * mask

        out = jnp.matmul(alpha, Wh)  # (N, D')
        return out


class MultiHeadGATLayer(nn.Module):
    out_features: int
    num_heads: int = 1
    negative_slope: float = 0.2
    use_mask: bool = False

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        heads = []
        for h in range(self.num_heads):
            head = GATLayer(
                out_features=self.out_features,
                negative_slope=self.negative_slope,
                use_mask=self.use_mask,
                name=f"head_{h}",
            )(x, mask)
            heads.append(head)

        # Concatenate heads  (N, num_heads * D')
        concat = jnp.concatenate(heads, axis=-1)

        # Project back to out_features
        out = nn.Dense(self.out_features, name="proj")(concat)
        return out


class SubScheduler(nn.Module):
    """Single sub-scheduler: GAT encoder + MLP + Gumbel-Softmax → adjacency.

    Produces one binary adjacency matrix *G^{t(l)}* for one communication
    round (MAGIC §4.2, Eq. 4).
    """

    hidden_dim: int
    use_gat_encoder: bool = True
    temperature: float = 1.0

    @nn.compact
    def __call__(
        self,
        messages: jax.Array,
        rng: jax.Array | None = None,
        hard: bool = True,
        temperature_override: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Parameters
        ----------
        messages : shape ``(N, D)``
        rng : PRNGKey for Gumbel sampling.  If None, use deterministic argmax.
        hard : If True, produce hard (binary) adjacency via straight-through
               Gumbel-Softmax.
        temperature_override : optional JAX scalar to override ``self.temperature``.
            Enables dynamic annealing without triggering JIT recompilation.

        Returns
        -------
        adj : shape ``(N, N)``  — binary adjacency matrix.
        node_features : shape ``(N, D')`` — node features from GAT encoder.
        """
        n = messages.shape[0]

        if self.use_gat_encoder:
            node_feat = GATLayer(
                out_features=self.hidden_dim,
                use_mask=False,
                name="gat_enc",
            )(messages)
            node_feat = nn.elu(node_feat)  # MAGIC Eq. 6: ELU activation
        else:
            node_feat = messages

        # Pairwise feature concatenation  E_{i,j} = (e_i || e_j) ---
        e_i = jnp.repeat(node_feat[:, None, :], n, axis=1)  # (N, N, D')
        e_j = jnp.repeat(node_feat[None, :, :], n, axis=0)  # (N, N, D')
        pair_feat = jnp.concatenate([e_i, e_j], axis=-1)  # (N, N, 2D')

        # MLP → logits for Gumbel-Softmax ---
        h = nn.Dense(self.hidden_dim, name="mlp1")(pair_feat)
        h = nn.relu(h)
        logits = nn.Dense(2, name="mlp2")(h)  # (N, N, 2) — binary [no-edge, edge]

        # Gumbel-Softmax (Jang et al. 2017, used in MAGIC §4.2) ---
        temperature = (
            temperature_override
            if temperature_override is not None
            else self.temperature
        )
        adj = gumbel_softmax(logits, rng=rng, temperature=temperature, hard=hard)
        # Take the "edge present" channel  (index 1)
        adj = adj[..., 1]  # (N, N)

        # Ensure self-loops (agents can message themselves, MAGIC §4.3)
        adj = adj + jnp.eye(n)
        adj = jnp.clip(adj, 0.0, 1.0)

        return adj, node_feat


class Scheduler(nn.Module):
    """Full Scheduler with *L* sub-schedulers (MAGIC §4.2, Eq. 4).

    Produces ``num_rounds`` adjacency matrices.
    """

    hidden_dim: int
    num_rounds: int = 1
    temperature: float = 1.0

    @nn.compact
    def __call__(
        self,
        messages: jax.Array,
        rng: jax.Array | None = None,
        hard: bool = True,
        temperature_override: jax.Array | None = None,
    ) -> list[jax.Array]:
        adjs: list[jax.Array] = []
        for round in range(self.num_rounds):
            sub = SubScheduler(
                hidden_dim=self.hidden_dim,
                use_gat_encoder=(round == 0),  # GAT encoder only in first round
                temperature=self.temperature,
                name=f"sub_sched_{round}",
            )
            if rng is not None:
                rng, sub_rng = jax.random.split(rng)
            else:
                sub_rng = None
            adj, _ = sub(
                messages,
                rng=sub_rng,
                hard=hard,
                temperature_override=temperature_override,
            )
            adjs.append(adj)
        return adjs


class MessageProcessor(nn.Module):
    """Message Processor with *L* sub-processors (MAGIC §4.3, Eqs. 7–9).

    Each sub-processor is a multi-head GAT masked by the adjacency matrix
    from the corresponding sub-scheduler.
    """

    hidden_dim: int
    num_heads: int = 1
    num_rounds: int = 1

    @nn.compact
    def __call__(
        self,
        messages: jax.Array,
        adjs: jax.Array | Sequence[jax.Array],
    ) -> jax.Array:
        m = messages
        for round in range(self.num_rounds):
            # Sub-processor l: GAT with adjacency mask G^{t(l)}
            m_new = MultiHeadGATLayer(
                out_features=self.hidden_dim,
                num_heads=self.num_heads,
                use_mask=True,
                name=f"sub_proc_{round}",
            )(m, adjs[round])
            m_new = nn.elu(m_new)

            # Residual connection + bias (MAGIC §4.3)
            bias = self.param(
                f"bias_{round}",
                nn.initializers.zeros,
                (self.hidden_dim,),
            )
            m = m_new + m + bias  # residual
        return m  # (N, hidden_dim)
