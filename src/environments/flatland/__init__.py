def _neutralize_flatland_msgpack_patch() -> None:
    """Undo flatland's global ``msgpack_numpy.patch()``.

    ``flatland/envs/persistence.py`` unconditionally calls
    ``msgpack_numpy.patch()`` at import time, which rebinds ``msgpack.Packer``
    (and friends) to the msgpack_numpy variants.  The msgpack_numpy ``Packer``
    wraps any user-supplied ``default`` hook in ``functools.partial(encode,
    chain=default)``; combined with ``strict_types=True`` this breaks
    ``flax.serialization.to_bytes``, which relies on tuples flowing through
    its own ``_msgpack_ext_pack`` hook.  The net effect is that *any* Flax
    checkpoint write fails with ``TypeError: can not serialize 'tuple'
    object`` once flatland has been imported — even when the payload has
    nothing to do with flatland.

    We snapshot the relevant ``msgpack`` attributes, trigger flatland's
    persistence import so the patch runs exactly once, then restore the
    originals so the rest of the process sees an unpatched msgpack.
    """
    import msgpack  # local import so module import order is explicit

    attrs = (
        "Packer",
        "Unpacker",
        "load",
        "loads",
        "dump",
        "dumps",
        "pack",
        "packb",
        "unpack",
        "unpackb",
    )
    snapshot = {name: getattr(msgpack, name) for name in attrs}

    import importlib

    importlib.import_module("flatland.envs.persistence")  # side effect: runs patch

    for name, value in snapshot.items():
        setattr(msgpack, name, value)


_neutralize_flatland_msgpack_patch()
del _neutralize_flatland_msgpack_patch


from .config import (
    FLATLAND_STATUS_DIM,
    TREE_BRANCHING_FACTOR,
    TREE_FEATURE_DIM,
    FlatlandConfig,
    flatland_observation_dim,
    flatland_state_dim,
    flatland_tree_node_count,
)
from .flatland_env import FlatlandPettingZooEnv, make_flatland_env

__all__ = [
    "TREE_BRANCHING_FACTOR",
    "TREE_FEATURE_DIM",
    "FLATLAND_STATUS_DIM",
    "FlatlandConfig",
    "FlatlandPettingZooEnv",
    "flatland_tree_node_count",
    "flatland_observation_dim",
    "flatland_state_dim",
    "make_flatland_env",
]
