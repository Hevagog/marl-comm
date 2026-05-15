"""
comm_graph_renderer.py
======================
Render communication graphs as RGB frames for split-screen video recording.

MAGIC  : dynamic soft-adjacency — edges change weight every step.
         Shows last comm round (the one with actual inter-agent routing).

CommFormer : static binary graph — topology fixed at inference (α is a
             learned parameter, k-hot argmax at inference, no Gumbel noise).
             Edges are drawn as a permanent structure; node brightness encodes
             the per-step encoder representation norm (the only thing that moves).

Battery ring : circular arc around each node, clockwise from 12 o'clock,
               green→yellow→red as battery depletes.  Disappears with the node
               when the agent is dead.

Dead agents  : greyed-out node + white ✕ overlay.
"""

from __future__ import annotations

import io
from typing import Any

import matplotlib
import matplotlib.patches as mpatches

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── Palette — matches warehouse_grid/utils/rendering.py exactly ──────────────
# rendering.py _AGENT_COLORS (RGB tuples) converted to hex:
#   (220, 60, 60)   →  #DC3C3C   red      A0
#   (60, 120, 220)  →  #3C78DC   blue     A1
#   (220, 170, 40)  →  #DCAA28   yellow   A2
#   (170, 60, 220)  →  #AA3CDC   purple   A3
#   (50, 210, 120)  →  #32D278   green    A4
#   (220, 120, 50)  →  #DC7832   orange   A5
#   (50, 200, 210)  →  #32C8D2   cyan     A6
#   (180, 180, 180) →  #B4B4B4   grey     A7

_AGENT_COLORS = [
    "#DC3C3C",  # red     — A0
    "#3C78DC",  # blue    — A1
    "#DCAA28",  # yellow  — A2
    "#AA3CDC",  # purple  — A3
    "#32D278",  # green   — A4
    "#DC7832",  # orange  — A5
    "#32C8D2",  # cyan    — A6
    "#B4B4B4",  # grey    — A7
    "#DC3C96",  # pink    — A8
    "#78DC3C",  # lime    — A9
    "#3CDCDC",  # teal    — A10
    "#DC783C",  # salmon  — A11
    "#3C3CDC",  # indigo  — A12
    "#DC3CDC",  # magenta — A13
    "#78DCAA",  # mint    — A14
    "#DCDC3C",  # chartreuse — A15
]

_DEAD_COLOR = "#444444"
_BACKGROUND_COLOR = "#1E1E2E"
_TEXT_COLOR = "#E0E0E0"

_MIN_NODE_RADIUS = 0.10  # data units
_MAX_NODE_RADIUS = 0.17
_BATTERY_RING_GAP = 0.025  # gap between node edge and battery ring
_BATTERY_RING_LW = 2.5  # linewidth
_BATTERY_RING_BG_LW = 1.2  # background ring linewidth
_BATTERY_RING_BG_ALPHA = 0.25

_MIN_EDGE_WIDTH = 0.6
_MAX_EDGE_WIDTH = 4.0
_EDGE_THRESHOLD = 0.05

# Battery colour thresholds (matching rendering.py _BAT_HIGH/_MED/_LOW)
_BAT_HIGH = "#3CC850"  # ≥ 60 %
_BAT_MED = "#E6BE28"  # 25–60 %
_BAT_LOW = "#E63C32"  # < 25 %


# ─── Public API ───────────────────────────────────────────────────────────────


def render_comm_graph_frame(
    adj: np.ndarray,
    title: str = "",
    step: int = 0,
    agent_labels: list[str] | None = None,
    node_features: np.ndarray | None = None,
    agent_colors: list[str] | None = None,
    extra_info: dict[str, Any] | None = None,
    battery_levels: list[float] | None = None,
    agent_dead: list[bool] | None = None,
    width_px: int = 480,
    height_px: int = 480,
    dpi: int = 100,
    threshold: float = _EDGE_THRESHOLD,
    is_dynamic: bool = True,
) -> np.ndarray:
    """Render a communication graph as a (height_px, width_px, 3) uint8 frame.

    Parameters
    ----------
    adj           : (N, N) float — soft weights (MAGIC last round) or binary
                    (CommFormer static α).
    title         : panel title.
    step          : current env step.
    agent_labels  : N label strings.
    node_features : (N, D) — modulates node brightness by representation norm.
    agent_colors  : N hex strings; falls back to warehouse palette.
    extra_info    : key→value pairs shown as text overlay.
    battery_levels: N floats in [0, 1] — fraction of remaining battery.
                    None disables the battery ring.
    agent_dead    : N bools — dead agents rendered greyed-out with ✕.
    width_px      : output frame width.
    height_px     : output frame height.
    dpi           : matplotlib DPI.
    threshold     : minimum adjacency weight to draw an edge.
    is_dynamic    : True = MAGIC (edge weight → colour/width per step),
                    False = CommFormer (static topology; only nodes animate).
    """
    adj = np.asarray(adj, dtype=np.float32)
    n = adj.shape[0]

    if agent_labels is None:
        agent_labels = [f"A{i}" for i in range(n)]
    if agent_colors is None:
        agent_colors = [_AGENT_COLORS[i % len(_AGENT_COLORS)] for i in range(n)]
    dead_flags: list[bool] = agent_dead if agent_dead is not None else [False] * n

    fig_w = width_px / dpi
    fig_h = height_px / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BACKGROUND_COLOR)
    ax = fig.add_axes((0.05, 0.10, 0.90, 0.80))
    ax.set_facecolor(_BACKGROUND_COLOR)
    ax.axis("off")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")

    # ── Circular node layout ───────────────────────────────────────────────────
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    pos = {i: (float(np.cos(angles[i])), float(np.sin(angles[i]))) for i in range(n)}

    # ── Node radius: proportional to total comm weight ─────────────────────────
    off = adj.copy()
    np.fill_diagonal(off, 0.0)
    total_weight = off.sum(axis=1) + off.sum(axis=0)
    max_w = max(float(total_weight.max()), 1e-6)
    node_radii = _MIN_NODE_RADIUS + (_MAX_NODE_RADIUS - _MIN_NODE_RADIUS) * (
        total_weight / max_w
    )

    # Node brightness from representation norm (makes CommFormer nodes animate)
    node_alpha = np.ones(n)
    if node_features is not None:
        norms = np.linalg.norm(node_features, axis=-1).astype(float)
        max_norm = max(float(norms.max()), 1e-6)
        node_alpha = 0.45 + 0.55 * (norms / max_norm)

    # ── Draw edges ─────────────────────────────────────────────────────────────
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = float(off[i, j])
            if w < threshold:
                continue
            xi, yi = pos[i]
            xj, yj = pos[j]
            ri, rj = node_radii[i], node_radii[j]

            if is_dynamic:
                # MAGIC: weight drives colour intensity and line width
                edge_color = agent_colors[i]
                edge_w = _MIN_EDGE_WIDTH + (_MAX_EDGE_WIDTH - _MIN_EDGE_WIDTH) * w
                edge_alpha = max(0.2, min(1.0, w * 1.2))
                dashed = False
            else:
                # CommFormer: static binary structure — uniform grey, thinner
                edge_color = "#888888"
                edge_w = 1.4
                edge_alpha = 0.75
                dashed = False

            _draw_arrow(
                ax, xi, yi, xj, yj, ri, rj, edge_color, edge_w, edge_alpha, dashed
            )

    # ── Draw nodes ─────────────────────────────────────────────────────────────
    for i in range(n):
        xi, yi = pos[i]
        r = float(node_radii[i])
        dead = dead_flags[i]
        color = _DEAD_COLOR if dead else agent_colors[i]
        alpha = 0.35 if dead else float(node_alpha[i])

        circle = mpatches.Circle(
            (xi, yi),
            radius=r,
            facecolor=color,
            alpha=alpha,
            edgecolor="white" if not dead else "#666666",
            linewidth=0.8,
            zorder=5,
        )
        ax.add_patch(circle)

        # Agent label
        fontsize = max(6, min(11, 130 // n))
        ax.text(
            xi,
            yi,
            agent_labels[i],
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#AAAAAA" if dead else "white",
            fontweight="bold",
            zorder=6,
        )

        # Dead marker: white ✕
        if dead:
            _draw_dead_x(ax, xi, yi, r * 0.55)

        # Battery ring (only for living agents with battery info)
        if not dead and battery_levels is not None and i < len(battery_levels):
            frac = float(np.clip(battery_levels[i], 0.0, 1.0))
            _draw_battery_ring(ax, xi, yi, r + _BATTERY_RING_GAP, frac)

    # ── Title / subtitle ───────────────────────────────────────────────────────
    fig.text(
        0.5,
        0.97,
        title,
        ha="center",
        va="top",
        fontsize=9,
        color=_TEXT_COLOR,
        fontweight="bold",
    )

    off_mask = ~np.eye(n, dtype=bool)
    density = float(off[off_mask].mean()) if n > 1 else 0.0
    n_active = int((off[off_mask] >= threshold).sum())
    total_possible = n * (n - 1)

    if is_dynamic:
        subtitle = f"step {step}  |  density {density:.2f}  |  edges {n_active}/{total_possible}"
    else:
        subtitle = f"step {step}  |  static graph  |  node brightness = repr. norm"
    fig.text(0.5, 0.92, subtitle, ha="center", va="top", fontsize=7, color="#AAAAAA")

    # ── Extra info overlay ────────────────────────────────────────────────────
    if extra_info:
        lines = [f"{k}: {v}" for k, v in extra_info.items() if k != "density"]
        if lines:
            fig.text(
                0.03,
                0.08,
                "\n".join(lines),
                ha="left",
                va="bottom",
                fontsize=6,
                color="#AAAAAA",
                family="monospace",
            )

    # ── Type badge ───────────────────────────────────────────────────────────
    badge = "MAGIC — dynamic" if is_dynamic else "CommFormer — static α"
    fig.text(
        0.97,
        0.03,
        badge,
        ha="right",
        va="bottom",
        fontsize=6,
        color="#888888",
        style="italic",
    )

    # ── Render to RGB array ────────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=_BACKGROUND_COLOR
    )
    plt.close(fig)
    buf.seek(0)

    import PIL.Image

    _lanczos = getattr(PIL.Image, "LANCZOS", getattr(PIL.Image, "ANTIALIAS", 1))
    img = PIL.Image.open(buf).convert("RGB")
    img = img.resize((width_px, height_px), _lanczos)
    return np.asarray(img, dtype=np.uint8)


# ─── Arrow helper ─────────────────────────────────────────────────────────────


def _draw_arrow(ax, x0, y0, x1, y1, r0, r1, color, width, alpha, dashed=False):
    """Curved directed arrow, shortened to avoid overlapping node circles."""
    dx, dy = x1 - x0, y1 - y0
    dist = float(np.sqrt(dx * dx + dy * dy))
    if dist < 1e-6:
        return
    # Shorten by actual node radii + a tiny gap
    gap = 0.02
    x0s = x0 + (r0 + gap) * dx / dist
    y0s = y0 + (r0 + gap) * dy / dist
    x1s = x1 - (r1 + gap) * dx / dist
    y1s = y1 - (r1 + gap) * dy / dist

    lstyle = (0, (4, 3)) if dashed else "solid"
    ax.annotate(
        "",
        xy=(x1s, y1s),
        xytext=(x0s, y0s),
        arrowprops=dict(
            arrowstyle=f"->, head_width={0.10 + 0.05 * width:.2f}, head_length=0.10",
            color=color,
            lw=width,
            alpha=alpha,
            linestyle=lstyle,
            connectionstyle="arc3,rad=0.15",
        ),
        zorder=3,
    )


# ─── Battery ring helper ──────────────────────────────────────────────────────


def _draw_battery_ring(ax, cx, cy, ring_r, fraction):
    """Arc from 12 o'clock clockwise, covering `fraction` of the circle.

    Background ghost ring shows full circumference so the user can see what
    a full battery would look like.
    """
    # Background ghost (full circle, dim)
    theta_bg = np.linspace(0, 2 * np.pi, 128)
    ax.plot(
        cx + ring_r * np.cos(theta_bg),
        cy + ring_r * np.sin(theta_bg),
        color="#FFFFFF",
        linewidth=_BATTERY_RING_BG_LW,
        alpha=_BATTERY_RING_BG_ALPHA,
        zorder=4,
        solid_capstyle="round",
    )

    if fraction < 0.01:
        return

    # Active arc: starts at top (π/2), goes clockwise → decreasing angle
    theta_start = np.pi / 2
    theta_end = theta_start - fraction * 2 * np.pi
    n_pts = max(3, int(128 * fraction))
    theta = np.linspace(theta_start, theta_end, n_pts)

    if fraction > 0.60:
        bat_color = _BAT_HIGH
    elif fraction > 0.25:
        bat_color = _BAT_MED
    else:
        bat_color = _BAT_LOW

    ax.plot(
        cx + ring_r * np.cos(theta),
        cy + ring_r * np.sin(theta),
        color=bat_color,
        linewidth=_BATTERY_RING_LW,
        alpha=0.95,
        zorder=4,
        solid_capstyle="round",
    )


# ─── Dead agent marker ────────────────────────────────────────────────────────


def _draw_dead_x(ax, cx, cy, size):
    """White ✕ over a dead agent node."""
    ax.plot(
        [cx - size, cx + size],
        [cy + size, cy - size],
        color="white",
        linewidth=1.8,
        alpha=0.85,
        zorder=7,
        solid_capstyle="round",
    )
    ax.plot(
        [cx - size, cx + size],
        [cy - size, cy + size],
        color="white",
        linewidth=1.8,
        alpha=0.85,
        zorder=7,
        solid_capstyle="round",
    )


# ─── Split-screen helper ──────────────────────────────────────────────────────


def make_split_frame(
    env_frame: np.ndarray,
    comm_frame: np.ndarray,
) -> np.ndarray:
    """Concatenate env_frame and comm_frame side-by-side.

    Resizes comm_frame height to match env_frame if they differ.
    """
    import PIL.Image

    h_env = env_frame.shape[0]
    h_comm = comm_frame.shape[0]

    if h_comm != h_env:
        w_comm = comm_frame.shape[1]
        new_w = int(w_comm * h_env / max(h_comm, 1))
        _lanczos = getattr(PIL.Image, "LANCZOS", getattr(PIL.Image, "ANTIALIAS", 1))
        img = PIL.Image.fromarray(comm_frame).resize((new_w, h_env), _lanczos)
        comm_frame = np.asarray(img, dtype=np.uint8)

    return np.concatenate([env_frame, comm_frame], axis=1)
