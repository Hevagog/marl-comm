from __future__ import annotations

import functools

import flax
import jax
import optax

from skrl.models.jax import Model


@functools.partial(jax.jit, static_argnames=("transformation",))
def _step(transformation, grad, state, state_dict):
    params, optimizer_state = transformation.update(grad, state, state_dict.params)
    params = optax.apply_updates(state_dict.params, params)
    return optimizer_state, state_dict.replace(params=params)


@functools.partial(jax.jit, static_argnames=("transformation",))
def _step_with_scale(transformation, grad, state, state_dict, scale):
    params, optimizer_state = transformation.update(grad, state, state_dict.params)
    params = jax.tree_util.tree_map(lambda p: scale * p, params)
    params = optax.apply_updates(state_dict.params, params)
    return optimizer_state, state_dict.replace(params=params)


class AdamW:
    """AdamW (decoupled weight-decay) optimiser for skrl models.

    Parameters
    ----------
    model : skrl.models.jax.Model
        Model whose parameters will be optimised.
    lr : float
        Learning rate (default: ``1e-3``).
    weight_decay : float
        Decoupled weight-decay coefficient (default: ``1e-4``).
    grad_norm_clip : float
        If > 0, clip gradients by global norm (default: ``0``).
    scale : bool
        If ``True`` (default), use ``optax.adamw`` with full learning-rate
        scaling.  If ``False``, use ``optax.scale_by_adam`` so a custom LR
        can be applied per step (mirrors skrl ``Adam(scale=False)``).
    """

    def __new__(
        cls,
        model: Model,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_norm_clip: float = 0,
        scale: bool = True,
    ) -> AdamW:

        class Optimizer(flax.struct.PyTreeNode):
            transformation: optax.GradientTransformation = flax.struct.field(
                pytree_node=False
            )
            state: optax.OptState = flax.struct.field(pytree_node=True)

            @classmethod
            def _create(cls, *, transformation, state, **kwargs):
                return cls(transformation=transformation, state=state, **kwargs)

            def step(
                self,
                grad: jax.Array,
                model: Model,
                lr: float | None = None,
            ) -> Optimizer:
                if lr is None:
                    optimizer_state, model.state_dict = _step(
                        self.transformation, grad, self.state, model.state_dict
                    )
                else:
                    optimizer_state, model.state_dict = _step_with_scale(
                        self.transformation,
                        grad,
                        self.state,
                        model.state_dict,
                        -lr,
                    )
                return self.replace(state=optimizer_state)

        if scale:
            transformation = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
        else:
            transformation = optax.chain(
                optax.scale_by_adam(),
                optax.add_decayed_weights(weight_decay=weight_decay),
            )

        if grad_norm_clip > 0:
            transformation = optax.chain(
                optax.clip_by_global_norm(grad_norm_clip), transformation
            )

        return Optimizer._create(
            transformation=transformation,
            state=transformation.init(model.state_dict.params),
        )
