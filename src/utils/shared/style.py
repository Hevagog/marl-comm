from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_BG = "#F8F9FA"
_PANEL_BG = "#FFFFFF"

# Up to 8 agents — Seaborn "colorblind"-friendly but distinct
_AGENT_COLORS: list[str] = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B2",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # grey
]

_PHASE_COLORS: list[str] = ["#95B9D4", "#F4A261", "#E76F51", "#2A9D8F"]

# Common colormaps
CMAP_COMM = "Blues"
CMAP_HEAT = "RdYlGn_r"
CMAP_ATTN = "YlOrRd"
CMAP_SIM = "RdBu_r"


_PUBLICATION_RC: dict = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.facecolor": _BG,
    "axes.facecolor": _PANEL_BG,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,  # embed fonts in PDF — required by most venues
    "ps.fonttype": 42,
}


def apply_publication_style() -> None:
    """Apply dissertation-grade rcParams. Call once at module import."""
    plt.rcParams.update(_PUBLICATION_RC)


# Apply immediately on import so any module that does `from shared.style import …`
# gets the style without an explicit call.
apply_publication_style()


# ── Helpers ────────────────────────────────────────────────────────────────────


def save_figure(fig: plt.Figure, out_dir: Path, name: str, dpi: int = 300) -> None:
    """Save *fig* as both PDF and PNG under *out_dir/name*.{pdf,png}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(str(out_dir / f"{name}.{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_dir / name}.(pdf|png)")


def agent_colors(n: int) -> list[str]:
    """Return a list of *n* agent colours, cycling if n > 8."""
    return [_AGENT_COLORS[i % len(_AGENT_COLORS)] for i in range(n)]
