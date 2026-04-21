# fmt: off
"""Ablation E — init fix only (no MAPPO tricks).

Isolates the effect of the residual MLP init fix (gain=0.01 → 1.0)
from the MAPPO tricks. Compare against D to see how much KL stopping
and weight decay contribute independently.
"""
import copy
import importlib.util
from pathlib import Path

_base_path = Path(__file__).parent / "mam_warehouse_smoke.py"
_spec = importlib.util.spec_from_file_location("_mam_smoke_base_e", _base_path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
_BASE = _mod.CONFIG

CONFIG = copy.deepcopy(_BASE)
CONFIG["experiment"]["name"] = "mam_warehouse_smoke_e_init_fix_only"
CONFIG["experiment"]["wandb_kwargs"]["tags"] = ["mam", "warehouse", "smoke", "ablation-e", "init-fix"]

# No additional changes — the init fix is already in policy.py
# This run uses the same config as base but with the fixed code
