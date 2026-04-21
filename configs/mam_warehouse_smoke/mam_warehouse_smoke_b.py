# fmt: off
"""Ablation B — relax grad_norm_clip 0.5 → 2.0.

Hypothesis: Mamba's deep residual stack produces diffused gradients that
get clipped at 0.5, starving the policy of effective updates. 2.0 is the
standard PPO default.
"""
import copy
import importlib.util
from pathlib import Path

_base_path = Path(__file__).parent / "mam_warehouse_smoke.py"
_spec = importlib.util.spec_from_file_location("_mam_smoke_base_b", _base_path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
_BASE = _mod.CONFIG

CONFIG = copy.deepcopy(_BASE)
CONFIG["experiment"]["name"] = "mam_warehouse_smoke_b_grad_clip_2"
CONFIG["experiment"]["wandb_kwargs"]["tags"] = ["mam", "warehouse", "smoke", "ablation-b", "grad-clip-2"]

CONFIG["mam"]["grad_norm_clip"] = 2.0
