from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a configuration file and return it as a plain dict.

    Supported formats
    -----------------
    ``.py``
        The module must define a top-level ``CONFIG`` dict.  This is the
        preferred format because it supports Python literals (``None``,
        ``True``/``False``, numeric underscores, comments, etc.) and allows
        the file to be edited without escaping JSON.
    ``.json``
        Legacy JSON format, kept for backward compatibility.

    Parameters
    ----------
    path:
        Path to the config file.

    Returns
    -------
    dict
        The loaded configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file extension is not ``.py`` or ``.json``, or if a ``.py``
        config does not define a ``CONFIG`` variable.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("_cfg_module", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load Python config from: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        if not hasattr(module, "CONFIG"):
            raise ValueError(
                f"Python config file '{path}' must define a top-level 'CONFIG' dict."
            )
        return dict(module.CONFIG)  # type: ignore[arg-type]

    if path.suffix == ".json":
        with path.open() as f:
            return json.load(f)

    raise ValueError(
        f"Unsupported config format '{path.suffix}'. Use '.py' or '.json'."
    )
