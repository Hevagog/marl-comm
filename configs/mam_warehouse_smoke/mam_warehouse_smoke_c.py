# fmt: off
"""Ablation C — much lower entropy bonus.

Hypothesis: entropy bonus is too large but not *zero*. 0.05→0.01 anneal
still dominates surrogate. Try 0.01→0.001 — softer exploration pressure
while still preventing outright determinism collapse.
"""
import copy
import importlib.util
from pathlib import Path

_base_path = Path(__file__).parent / "mam_warehouse_smoke.py"
_spec = importlib.util.spec_from_file_location("_mam_smoke_base_c", _base_path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
_BASE = _mod.CONFIG

CONFIG = copy.deepcopy(_BASE)
CONFIG["experiment"]["name"] = "mam_warehouse_smoke_c_low_entropy"
CONFIG["experiment"]["wandb_kwargs"]["tags"] = ["mam", "warehouse", "smoke", "ablation-c", "low-entropy"]

CONFIG["mam"]["entropy_loss_scale"] = 0.005
CONFIG["mam"]["entropy_loss_scale_start"] = 0.01
CONFIG["mam"]["entropy_loss_scale_end"] = 0.001
