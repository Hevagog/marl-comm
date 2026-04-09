"""Task & Entity Hopfield Pooling for MAMHM.

Upstream Hopfield-style set-pooling modules that extract structured context
from the local observation BEFORE the BiMamba encoder.  This replaces the
post-decoder HopfieldMemoryBank as the primary memory mechanism.

Key insight (from DeepRC / immune-repertoire paper):
  Hopfield attention is most powerful as set-pooling over bags of items.
  The task queue summary and vision-patch entities are natural "bags".

Two modules:
  - TaskHopfieldPooling:   pools over task-queue features
  - EntityHopfieldPooling: pools over vision-patch cell features

Both use the same pattern:
  1. Project raw features → keys/values in d_model space
  2. Learned query vectors attend over the projected set
  3. Gated residual adds the pooled context to the observation embedding


References
----------
- Widrich et al. 2020 "Modern Hopfield Networks for ... Immune Repertoire"
- Ramsauer et al. 2021 "Hopfield Networks is All You Need"
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class TaskHopfieldPooling(nn.Module):
    """Hopfield-style set-pooling over task-queue features.

    The warehouse observation contains 3 aggregate task features at the
    tail of the observation vector: (pending_ratio, dx_urgent, dy_urgent).
    We project these into d_model space and attend with learned queries.

    Parameters
    ----------
    d_model : int
        Hidden dimension (must match encoder's n_embd).
    num_query_heads : int
        Number of learned query prototypes.  Each head attends independently
        and results are summed, giving a multi-perspective task summary.
    beta : float
        Inverse-temperature scaling for the attention logits.
    gate_init : float
        Initial bias for the gating projection.  sigmoid(gate_init) sets
        the initial contribution fraction.  Default -3.0 → ~4.7%.
    task_slice_start : int
        Start index of task features in the observation vector.
    task_slice_end : int
        End index (exclusive) of task features in the observation vector.
    task_feat_dim : int
        Dimensionality of the raw task features (default 3).
    """

    d_model: int = 128
    num_query_heads: int = 4
    beta: float = 2.0
    gate_init: float = -3.0
    # ASSUMPTION: verify against src/environments/warehouse_grid/utils/agent_utils.py
    # Observation layout for warehouse with batteries+heterogeneous+tasks:
    #   [0:7]     own state
    #   [7:15]    relative positions to key cells
    #   [15:165]  local grid view (5×5×6)
    #   [165:183] other agents (3 × 6)
    #   [183:186] heterogeneous properties
    #   [186]     battery
    #   [187:190] task queue context (pending_ratio, dx_urgent, dy_urgent)
    task_slice_start: int = 187
    task_slice_end: int = 190
    task_feat_dim: int = 3

    @nn.compact
    def __call__(
        self,
        obs: jax.Array,
        obs_emb: jax.Array,
    ) -> jax.Array:
        """Apply task Hopfield pooling and add to obs embedding.

        Parameters
        ----------
        obs : (batch, N_agents, obs_dim)
            Raw observations (needed to extract task slice).
        obs_emb : (batch, N_agents, d_model)
            Observation embeddings from the obs_encoder MLP.

        Returns
        -------
        obs_emb_augmented : (batch, N_agents, d_model)
            Embedding with task context added via gated residual.
        """
        # Extract task features from raw observation
        task_raw = obs[..., self.task_slice_start : self.task_slice_end]
        # task_raw: (batch, N_agents, task_feat_dim)

        # Validity mask: tasks are valid when pending_ratio > 0
        # pending_ratio is the first task feature
        valid = task_raw[..., 0:1] > 0  # (batch, N_agents, 1)

        # Project task features to d_model key/value space
        # Shape: (batch, N_agents, d_model)
        task_keys = nn.Dense(self.d_model, use_bias=False, name="task_key_proj")(
            task_raw
        )
        task_values = nn.Dense(self.d_model, use_bias=False, name="task_value_proj")(
            task_raw
        )

        # Normalize keys for stable attention
        task_keys = task_keys * jax.lax.rsqrt(
            jnp.sum(jnp.square(task_keys), axis=-1, keepdims=True) + 1e-6
        )

        # Learned query prototypes: (num_query_heads, d_model)
        queries = self.param(
            "task_queries",
            nn.initializers.lecun_normal(),
            (self.num_query_heads, self.d_model),
        )

        # Scaled attention: queries attend to per-agent task keys
        # queries: (H, d_model), task_keys: (batch, N, d_model)
        beta_scale = self.beta / jnp.sqrt(jnp.asarray(self.d_model, dtype=jnp.float32))
        # scores: (batch, N, H)
        scores = beta_scale * jnp.einsum("hd,bnd->bnh", queries, task_keys)

        # Mask invalid tasks (no pending tasks)
        mask = jnp.broadcast_to(valid, scores.shape)  # (batch, N, H)
        scores = jnp.where(mask, scores, -1e9)

        # Attention weights over query heads
        alpha = jax.nn.softmax(scores, axis=-1)  # (batch, N, H)

        # Weighted sum of query vectors (retrieval direction)
        # For this architecture, each head contributes its learned query
        # weighted by how relevant the task context is.
        # We sum over heads to get per-agent task context.
        # alpha @ queries: (batch, N, H) @ (H, d_model) → (batch, N, d_model)
        task_context = jnp.einsum("bnh,hd->bnd", alpha, queries)

        # Mix with projected task values for content-dependent signal
        task_context = task_context + task_values

        # Project to d_model
        task_context = nn.Dense(self.d_model, use_bias=False, name="task_out_proj")(
            task_context
        )

        # Input-dependent gate: sigmoid(W_gate @ obs_emb + b_gate)
        # Bias initialised to gate_init so initial contribution ≈ sigmoid(gate_init)
        gate = jax.nn.sigmoid(
            nn.Dense(
                1,
                kernel_init=nn.initializers.zeros,
                bias_init=lambda _k, shape, _d=None: jnp.full(shape, self.gate_init),
                name="task_gate",
            )(obs_emb)
        )  # (batch, N, 1)

        # Gated residual + LayerNorm
        return nn.LayerNorm(name="task_post_ln")(obs_emb + gate * task_context)


class EntityHopfieldPooling(nn.Module):
    """Hopfield-style set-pooling over vision-patch entity tokens.

    The 5×5 vision patch (6 features per cell = 25 spatial tokens) is the
    natural "bag of items" for entity-level Hopfield pooling.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    num_query_heads : int
        Number of learned entity-query prototypes.
    beta : float
        Inverse temperature for attention.
    gate_init : float
        Initial gating bias.
    vision_slice_start : int
        Start of vision features in obs.
    vision_slice_end : int
        End (exclusive) of vision features in obs.
    num_cells : int
        Number of cells in the vision patch (5×5 = 25).
    cell_feat_dim : int
        Features per cell (6: shelf, treatment, goal, charger, repair, walkable).
    """

    d_model: int = 128
    num_query_heads: int = 4
    beta: float = 2.0
    gate_init: float = -3.0
    # ASSUMPTION: verify against src/environments/warehouse_grid/utils/agent_utils.py
    vision_slice_start: int = 15
    vision_slice_end: int = 165
    num_cells: int = 25
    cell_feat_dim: int = 6

    @nn.compact
    def __call__(
        self,
        obs: jax.Array,
        obs_emb: jax.Array,
    ) -> jax.Array:
        """Apply entity Hopfield pooling over vision-patch tokens.

        Parameters
        ----------
        obs : (batch, N_agents, obs_dim)
            Raw observations.
        obs_emb : (batch, N_agents, d_model)
            Observation embeddings.

        Returns
        -------
        obs_emb_augmented : (batch, N_agents, d_model)
        """
        batch, n_agents = obs.shape[0], obs.shape[1]

        # Extract and reshape vision patch
        vision_flat = obs[..., self.vision_slice_start : self.vision_slice_end]
        # (batch, N, 150) → (batch, N, 25, 6)
        vision_tokens = vision_flat.reshape(
            batch, n_agents, self.num_cells, self.cell_feat_dim
        )

        # Validity: a cell is "interesting" if any feature is non-zero
        # (cells outside the grid boundary are all-zero)
        valid = jnp.any(vision_tokens != 0, axis=-1, keepdims=True)
        # valid: (batch, N, 25, 1)

        # Project to d_model
        entity_keys = nn.Dense(self.d_model, use_bias=False, name="entity_key_proj")(
            vision_tokens
        )  # (batch, N, 25, d_model)
        entity_values = nn.Dense(
            self.d_model, use_bias=False, name="entity_value_proj"
        )(vision_tokens)  # (batch, N, 25, d_model)

        # Normalize keys
        entity_keys = entity_keys * jax.lax.rsqrt(
            jnp.sum(jnp.square(entity_keys), axis=-1, keepdims=True) + 1e-6
        )

        # Learned queries: (H, d_model)
        queries = self.param(
            "entity_queries",
            nn.initializers.lecun_normal(),
            (self.num_query_heads, self.d_model),
        )

        beta_scale = self.beta / jnp.sqrt(jnp.asarray(self.d_model, dtype=jnp.float32))

        # scores: (batch, N, 25, H) — each cell scored against each query
        scores = beta_scale * jnp.einsum("hd,bncd->bnch", queries, entity_keys)

        # Mask invalid cells
        valid_broad = jnp.broadcast_to(valid, scores.shape)
        scores = jnp.where(valid_broad, scores, -1e9)

        # Attention over cells (axis=2, the 25 cells)
        alpha = jax.nn.softmax(scores, axis=2)  # (batch, N, 25, H)

        # Pool: weighted sum of entity values
        # (batch, N, 25, H) × (batch, N, 25, d_model) → (batch, N, H, d_model)
        pooled = jnp.einsum("bnch,bncd->bnhd", alpha, entity_values)

        # Mean over heads → (batch, N, d_model)
        entity_context = jnp.mean(pooled, axis=2)

        # Output projection
        entity_context = nn.Dense(self.d_model, use_bias=False, name="entity_out_proj")(
            entity_context
        )

        # Gated residual
        gate = jax.nn.sigmoid(
            nn.Dense(
                1,
                kernel_init=nn.initializers.zeros,
                bias_init=lambda _k, shape, _d=None: jnp.full(shape, self.gate_init),
                name="entity_gate",
            )(obs_emb)
        )

        return nn.LayerNorm(name="entity_post_ln")(obs_emb + gate * entity_context)
