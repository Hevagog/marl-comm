from __future__ import annotations
import jax
import jax.numpy as jnp


def gumbel_softmax_scaled(
    logits: jax.Array,
    rng: jax.Array,
    gumbel_scale: jax.Array,  # traced JAX scalar: 1.0 = stochastic, 0.0 = deterministic
    temperature: float | jax.Array = 1.0,
    hard: bool = True,
) -> jax.Array:
    """Gumbel-Softmax with gumbel_scale controlling noise magnitude.

    gumbel_scale=1.0 → standard stochastic Gumbel-Softmax (rollout).
    gumbel_scale=0.0 → deterministic argmax of (logits/temp) (training).

    Straight-Through Estimator applies in both cases so gradients flow
    through the discrete decision even at gumbel_scale=0.

    References
    ----------
    - Jang et al. 2017 "Categorical Reparameterization with Gumbel-Softmax"
    """
    temp = jax.lax.stop_gradient(jnp.asarray(temperature, dtype=jnp.float32))
    scale = jax.lax.stop_gradient(jnp.asarray(gumbel_scale, dtype=jnp.float32))

    gumbels = jax.random.gumbel(rng, shape=logits.shape)
    y = (logits + scale * gumbels) / temp
    y_soft = jax.nn.softmax(y, axis=-1)

    if hard:
        index = jnp.argmax(y_soft, axis=-1)
        y_hard = jax.nn.one_hot(index, logits.shape[-1])
        y = y_hard - jax.lax.stop_gradient(y_soft) + y_soft
    else:
        y = y_soft
    return y
