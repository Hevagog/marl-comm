import jax
import jax.numpy as jnp


def gumbel_softmax(
    logits: jax.Array,
    rng: jax.Array | None,
    temperature: float = 1.0,
    hard: bool = True,
) -> jax.Array:
    """Gumbel-Softmax with optional straight-through gradient estimator.

    Reference: Jang et al. 2017 "Categorical Reparameterization with
    Gumbel-Softmax" — used in MAGIC §4.2 for differentiable hard attention.
    """
    if rng is not None:
        gumbels = jax.random.gumbel(rng, shape=logits.shape)
        y = (logits + gumbels) / temperature
    else:
        y = logits / temperature

    y_soft = jax.nn.softmax(y, axis=-1)

    if hard:
        # Straight-through: hard sample in forward, soft gradients in backward
        index = jnp.argmax(y_soft, axis=-1)
        y_hard = jax.nn.one_hot(index, logits.shape[-1])
        y = y_hard - jax.lax.stop_gradient(y_soft) + y_soft
    else:
        y = y_soft
    return y
