"""Optional recurrent encoder for MAGIC (LSTM / GRU).

Implements the per-agent LSTM cell from MAGIC §4.1 Eq. 3:

    h_i^t, c_i^t = LSTM(e(o_i^t), h_i^{t-1}, c_i^{t-1})

skrl's JAX agents do not provide built-in recurrence (skrl-torch has
`ppo_rnn`, but no JAX equivalent as of v1.4.x). We therefore manage hidden
state externally in `MAGICMAPPO` and feed it into the policy through
`inputs["recurrent_state"]`.

Training caveat (TBPTT(0)):
The skrl `RandomMemory` does not store hidden state. During the PPO update
the policy is called without a `recurrent_state` entry, in which case the
encoder falls back to a zero initial state. The cell weights still receive
gradients from the single-step apply, so the layer is trainable, but
gradients do not propagate across time. This is documented in the
`MAGICPolicyNet` docstring; if recurrence shows empirical lift, the next
step is to plumb hidden state into `RandomMemory.create_tensor` and re-use
the stored state during the update (see TBPTT(1) in skrl-torch's
`ppo_rnn.py`).

References
----------
- Niu et al. 2021 "MAGIC" (AAMAS) §4.1 Eq. 3
- Hochreiter & Schmidhuber 1997 (LSTM)
- Cho et al. 2014 (GRU)
- Flax `nn.OptimizedLSTMCell`, `nn.GRUCell`
"""

from __future__ import annotations

from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp


RecurrentType = Literal["lstm", "gru", "hopfield"]


def zero_carry(
    recurrent_type: RecurrentType,
    batch_size: int,
    hidden_size: int,
    dtype=jnp.float32,
):
    """Zero initial carry matching `RecurrentEncoder` expectations.

    For ``recurrent_type == "hopfield"`` ``hidden_size`` is the *full carry
    width* T*H (rolling buffer flattened); the caller is responsible for
    multiplying T by H before invoking this helper. Keeping the signature
    flat lets the existing TBPTT(1) side-buffer plumbing in
    ``MAGICMAPPO`` work unchanged.
    """
    if recurrent_type == "lstm":
        c = jnp.zeros((batch_size, hidden_size), dtype=dtype)
        h = jnp.zeros((batch_size, hidden_size), dtype=dtype)
        return (c, h)
    if recurrent_type in ("gru", "hopfield"):
        return jnp.zeros((batch_size, hidden_size), dtype=dtype)
    raise ValueError(f"unknown recurrent_type: {recurrent_type!r}")


def carry_batch_size(carry) -> int:
    """Return the leading batch dimension of any LSTM/GRU carry."""
    leaf = carry[0] if isinstance(carry, tuple) else carry
    return int(leaf.shape[0])


def reset_carry_where(carry, mask: jax.Array):
    """Zero rows of `carry` where `mask` is True.

    Parameters
    ----------
    carry : LSTM tuple (c, h) or GRU array h, with leading axis batch.
    mask : (B,) bool array — True at rows to reset.
    """
    keep = (~mask).astype(jnp.float32)[:, None]
    if isinstance(carry, tuple):
        c, h = carry
        return (c * keep, h * keep)
    return carry * keep


class RecurrentEncoder(nn.Module):
    """Wraps `nn.OptimizedLSTMCell` or `nn.GRUCell` as a per-step encoder.

    Inputs
    ------
    carry : LSTM (c, h) or GRU h, both shape `(B, hidden_size)`.
    x     : `(B, in_dim)` per-step input.

    Outputs
    -------
    new_carry : same structure as `carry`.
    h_t       : `(B, hidden_size)` — output features. For LSTM this equals
                the hidden component of the new carry; for GRU it equals
                the new state.
    """

    hidden_size: int
    recurrent_type: RecurrentType = "lstm"
    buffer_size: int = 8
    beta_init: float = 1.0
    gate_init: float = 0.0

    @nn.compact
    def __call__(self, carry, x: jax.Array):
        if self.recurrent_type == "lstm":
            cell = nn.OptimizedLSTMCell(features=self.hidden_size, name="lstm_cell")
            new_carry, h_t = cell(carry, x)
            return new_carry, h_t
        if self.recurrent_type == "gru":
            cell = nn.GRUCell(features=self.hidden_size, name="gru_cell")
            new_carry, h_t = cell(carry, x)
            return new_carry, h_t
        if self.recurrent_type == "hopfield":
            from agents.magic.models.episodic_hopfield import EpisodicHopfieldEncoder

            cell = EpisodicHopfieldEncoder(
                hidden_size=self.hidden_size,
                buffer_size=self.buffer_size,
                beta_init=self.beta_init,
                gate_init=self.gate_init,
                name="episodic_hopfield",
            )
            new_carry, h_t = cell(carry, x)
            return new_carry, h_t
        raise ValueError(f"unknown recurrent_type: {self.recurrent_type!r}")
