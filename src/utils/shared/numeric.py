"""Numeric helpers shared across analysis modules."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_np(v: Any) -> np.ndarray:
    """Convert a JAX / numpy array to a plain numpy ndarray."""
    if isinstance(v, np.ndarray):
        return v
    try:
        import jax

        return np.asarray(jax.device_get(v))
    except Exception:
        return np.asarray(v)


def denorm(val: float, grid_size: int) -> int:
    """Convert a [0, 1]-normalised coordinate to a grid cell index."""
    return int(round(val * (grid_size - 1)))


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Manhattan distance between two (row, col) positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
