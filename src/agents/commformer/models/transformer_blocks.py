"""CommFormer Transformer blocks: Encoder and Decoder with relation-
enhanced attention and adjacency masking.

References
----------
- Hu et al. 2024 "CommFormer" (ICLR 2024), §3.2.
- Encoder: Eqs. 1–4 (relation-enhanced attention + value projection).
- Decoder: Eq. 5 (auto-regressive action generation with PPO).
- Wen et al. 2022 "MAT": sequential update scheme for monotonic improvement.
"""

from __future__ import annotations


import flax.linen as nn
import jax
import jax.numpy as jnp


class EdgeEmbedding(nn.Module):
    """Produces edge embeddings r_{i→j} from the adjacency matrix α.

    CommFormer §3.2, Eq. 2: "r_{*→*} is obtained from an embedding layer
    that takes the adjacency matrix α as input."
    """

    embed_dim: int

    @nn.compact
    def __call__(self, alpha: jax.Array) -> jax.Array:
        """
        Parameters
        ----------
        alpha : shape ``(N, N)``

        Returns
        -------
        shape ``(N, N, embed_dim)``
        """
        # Treat each scalar α_{ij} as a 1-D feature, embed it
        alpha_expanded = alpha[..., None]  # (N, N, 1)
        return nn.Dense(self.embed_dim, name="edge_dense")(alpha_expanded)


class RelationEnhancedMHA(nn.Module):
    """Multi-head attention with edge embeddings and adjacency masking.

    Implements CommFormer §3.2, Eqs. 2–3:
        s_{ij} = (o_i + r_{i→j}) W_q^T W_k (o_j + r_{j→i})
    with masking:
        s_{ij} = -∞  if e_{j→i} = 0.
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
        adj: jax.Array,
        edge_emb: jax.Array,
    ) -> jax.Array:
        """
        Parameters
        ----------
        query, key, value : shape ``(N, D)``
        adj : shape ``(N, N)`` — binary adjacency.
        edge_emb : shape ``(N, N, D)`` — edge embeddings.

        Returns
        -------
        shape ``(N, D)``
        """
        n = query.shape[0]
        d = self.num_heads * self.head_dim

        # Linear projections
        Q = nn.Dense(d, use_bias=False, name="Wq")(query)  # (N, d)
        K = nn.Dense(d, use_bias=False, name="Wk")(key)  # (N, d)
        V = nn.Dense(d, use_bias=False, name="Wv")(value)  # (N, d)

        # Reshape to (N, num_heads, head_dim)
        Q = Q.reshape(n, self.num_heads, self.head_dim)
        K = K.reshape(n, self.num_heads, self.head_dim)
        V = V.reshape(n, self.num_heads, self.head_dim)

        # Edge embedding projection to match head_dim
        edge_proj = nn.Dense(self.head_dim, use_bias=False, name="We")(
            edge_emb
        )  # (N, N, head_dim)

        # Relation-enhanced attention (CommFormer Eq. 2):
        # s_{ij} = (Q_i + r_{i→j}) · (K_j + r_{j→i})
        # For each head:
        # Q_i: (num_heads, head_dim), edge_proj[i,j]: (head_dim,)
        # We broadcast over heads.
        # Q_i + r_{i→j} for all pairs
        Q_exp = Q[:, None, :, :]  # (N, 1, H, hd)
        K_exp = K[None, :, :, :]  # (1, N, H, hd)

        # r_{i→j}: edge from i to j
        r_ij = edge_proj[:, :, None, :]  # (N, N, 1, hd)
        # r_{j→i}: edge from j to i
        r_ji = jnp.swapaxes(edge_proj, 0, 1)[:, :, None, :]  # (N, N, 1, hd)

        Q_rel = Q_exp + r_ij  # (N, N, H, hd)
        K_rel = K_exp + r_ji  # (N, N, H, hd)

        # Dot product attention scores
        scores = jnp.sum(Q_rel * K_rel, axis=-1)  # (N, N, H)
        scores = scores / jnp.sqrt(self.head_dim)

        # Adjacency masking (CommFormer Eq. 3)
        adj_mask = adj[:, :, None]  # (N, N, 1)
        scores = jnp.where(adj_mask > 0, scores, jnp.finfo(jnp.float32).min)

        # Causal mask for decoder (CommFormer §3.2: j < i)
        if self.use_causal_mask:
            causal = jnp.tril(jnp.ones((n, n)))[:, :, None]
            scores = jnp.where(causal > 0, scores, jnp.finfo(jnp.float32).min)

        attn = jax.nn.softmax(scores, axis=1)  # softmax over key dim

        # Weighted sum of values
        # attn: (N, N, H), V: (N, H, hd) → out: (N, H, hd)
        out = jnp.einsum("ijh,jhd->ihd", attn, V)

        # Concatenate heads
        out = out.reshape(n, d)

        # Output projection
        out = nn.Dense(query.shape[-1], name="Wo")(out)
        return out


class EncoderBlock(nn.Module):
    """Single Transformer encoder block with relation-enhanced attention."""

    num_heads: int
    head_dim: int
    mlp_dim: int

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        adj: jax.Array,
        edge_emb: jax.Array,
    ) -> jax.Array:
        # Self-attention with edge embeddings
        residual = x
        x = nn.LayerNorm(name="ln1")(x)
        x = RelationEnhancedMHA(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            use_causal_mask=False,
            name="mha",
        )(x, x, x, adj, edge_emb)
        x = x + residual

        # MLP
        residual = x
        x = nn.LayerNorm(name="ln2")(x)
        x = nn.Dense(self.mlp_dim, name="mlp1")(x)
        x = nn.relu(x)
        x = nn.Dense(residual.shape[-1], name="mlp2")(x)
        x = x + residual

        return x


class DecoderBlock(nn.Module):
    """Single Transformer decoder block with causal + adjacency masking.

    CommFormer §3.2: auto-regressive action generation with the sequential
    update scheme ensuring monotonic improvement (Wen et al. 2022).
    """

    num_heads: int
    head_dim: int
    mlp_dim: int

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        enc_out: jax.Array,
        adj: jax.Array,
        edge_emb: jax.Array,
    ) -> jax.Array:
        # Masked self-attention (causal + adjacency)
        residual = x
        x = nn.LayerNorm(name="ln1")(x)
        x = RelationEnhancedMHA(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            use_causal_mask=True,
            name="self_attn",
        )(x, x, x, adj, edge_emb)
        x = x + residual

        # Cross-attention to encoder output
        residual = x
        x = nn.LayerNorm(name="ln2")(x)
        x = RelationEnhancedMHA(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            use_causal_mask=False,
            name="cross_attn",
        )(x, enc_out, enc_out, adj, edge_emb)
        x = x + residual

        # MLP
        residual = x
        x = nn.LayerNorm(name="ln3")(x)
        x = nn.Dense(self.mlp_dim, name="mlp1")(x)
        x = nn.relu(x)
        x = nn.Dense(residual.shape[-1], name="mlp2")(x)
        x = x + residual

        return x
