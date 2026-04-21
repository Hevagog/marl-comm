# fmt: off
"""Ablation A — disable entropy bonus entirely.

Hypothesis: entropy bonus gradient dominates weak surrogate gradient,
pushing policy to uniform. If true, disabling it should unfreeze
entropy collapse and let the reward move.
"""
import copy
import importlib.util
from pathlib import Path

_base_path = Path(__file__).parent / "mam_warehouse_smoke.py"
_spec = importlib.util.spec_from_file_location("_mam_smoke_base_a", _base_path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
_BASE = _mod.CONFIG

CONFIG = copy.deepcopy(_BASE)
CONFIG["experiment"]["name"] = "mam_warehouse_smoke_a_no_entropy"
CONFIG["experiment"]["wandb_kwargs"]["tags"] = ["mam", "warehouse", "smoke", "ablation-a", "no-entropy"]

CONFIG["mam"]["entropy_loss_scale"] = 0.0
CONFIG["mam"]["entropy_annealing"] = False
CONFIG["mam"]["entropy_loss_scale_start"] = 0.0
CONFIG["mam"]["entropy_loss_scale_end"] = 0.0
