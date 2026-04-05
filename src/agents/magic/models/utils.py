from __future__ import annotations


import jax
import jax.numpy as jnp


def gumbel_softmax(
    logits: jax.Array,
    rng: jax.Array | None,
    temperature: float | jax.Array = 1.0,
    hard: bool = True,
) -> jax.Array:
    """Gumbel-Softmax with optional straight-through gradient estimator.

    Reference: Jang et al. 2017 "Categorical Reparameterization with
    Gumbel-Softmax" — used in MAGIC §4.2 for differentiable hard attention.

    ``temperature`` may be a plain Python float or a JAX-traced scalar array.
    ``stop_gradient`` is applied so temperature is treated as a hyperparameter
    (no gradient flows through it) while still being a traced value (enabling
    dynamic annealing without JIT recompilation).
    """
    temp = jax.lax.stop_gradient(jnp.asarray(temperature, dtype=jnp.float32))
    if rng is not None:
        gumbels = jax.random.gumbel(rng, shape=logits.shape)
        y = (logits + gumbels) / temp
    else:
        y = logits / temp

    y_soft = jax.nn.softmax(y, axis=-1)

    if hard:
        # Straight-through: hard sample in forward, soft gradients in backward
        index = jnp.argmax(y_soft, axis=-1)
        y_hard = jax.nn.one_hot(index, logits.shape[-1])
        y = y_hard - jax.lax.stop_gradient(y_soft) + y_soft
    else:
        y = y_soft
    return y
