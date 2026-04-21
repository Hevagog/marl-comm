# fmt: off
"""Ablation D — init fix + MAPPO tricks (KL threshold + weight decay).

Tests the combined effect of:
1. Residual MLP init fix (gain=0.01 → 1.0) — already in policy.py
2. KL early stopping (kl_threshold=0.05, warmup 30%)
3. Weight decay (1e-4)

These are the three differences between working MAPPO and failing MAM
that are independent of the architecture itself.
"""
import copy
import importlib.util
from pathlib import Path

_base_path = Path(__file__).parent / "mam_warehouse_smoke.py"
_spec = importlib.util.spec_from_file_location("_mam_smoke_base_d", _base_path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
_BASE = _mod.CONFIG

CONFIG = copy.deepcopy(_BASE)
CONFIG["experiment"]["name"] = "mam_warehouse_smoke_d_init_fix_mappo_tricks"
CONFIG["experiment"]["wandb_kwargs"]["tags"] = ["mam", "warehouse", "smoke", "ablation-d", "init-fix", "mappo-tricks"]

# MAPPO tricks
CONFIG["mam"]["kl_threshold"] = 0.05
CONFIG["mam"]["kl_warmup_fraction"] = 0.3
CONFIG["mam"]["weight_decay"] = 1e-4
