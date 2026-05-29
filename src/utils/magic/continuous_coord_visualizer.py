"""
continuous_coord_comm_visualizer.py
===================================
S2-style pre-/post-GAT MAGIC message PCA for the Continuous Coordination
blind scenario.  Consumes :class:`MAGICCCData` from
``utils.magic.continuous_coord_analysis`` and writes:

  1. ``<prefix>_msg_pca_deepdive``  – 2 rows (pre-GAT / post-GAT) × 2 columns
     (coloured by agent type / by current target visibility), with the per-row
     PC1/PC2 variance shares annotated.  This is the continuous-coord analogue
     of the warehouse ``fig:wh_s2_pca_deepdive``.
  2. ``<prefix>_msg_variance``      – stacked explained-variance bar (PC1, PC2,
     residual tail) for the pre- and post-GAT bases.

If the post-GAT (``agg_messages``) tensor was not collected, only the pre-GAT
row/bar is produced and a note is printed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.magic.continuous_coord_analysis import MAGICCCData
from utils.shared.style import save_figure

# Type / visibility colour keys (analysis-tool palette, independent of the
# universal architecture colours used in the thesis figures).
_TYPE_COLORS = ["#3B5BA5", "#C44E52", "#3F8F5B", "#D98C2B"]
_VIS_COLORS = {0: "#9AA0A6", 1: "#E8A33D"}  # blind (grey) vs sees-target (gold)


def _pca(x: np.ndarray, n: int = 3):
    """Fit PCA and return ``(projection, explained_variance_ratio, n_rows)``.

    Returns ``(None, None, 0)`` when sklearn is unavailable or the data is too
    small/degenerate to project onto two components.
    """
    finite = x[np.all(np.isfinite(x), axis=1)] if x.size else x
    if finite.shape[0] < 3 or finite.shape[1] < 2:
        return None, None, 0
    try:
        from sklearn.decomposition import PCA
    except Exception:  # noqa: BLE001
        return None, None, 0
    k = int(min(n, finite.shape[1], finite.shape[0]))
    model = PCA(n_components=k)
    proj = model.fit_transform(finite)
    var = model.explained_variance_ratio_
    if len(var) < 2:  # need at least two components for a 2-D scatter
        return None, None, 0
    return proj, var, finite.shape[0]


def _scatter_by_type(ax, proj, types, var, title_prefix):
    uniq = sorted(set(int(t) for t in types))
    for t in uniq:
        m = types == t
        ax.scatter(
            proj[m, 0], proj[m, 1],
            c=_TYPE_COLORS[t % len(_TYPE_COLORS)], s=6, alpha=0.45,
            label=f"type {t}", edgecolors="none",
        )
    ax.axvline(0, color="gray", lw=0.5, alpha=0.4)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax.set_title(
        f"{title_prefix} — by agent type\n"
        f"(PC1={var[0]:.1%}, PC2={var[1]:.1%})", fontsize=10
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=2.5, fontsize=8, loc="best")


def _scatter_by_visibility(ax, proj, vis, var, title_prefix):
    for v, lbl in ((0, "no target visible"), (1, "sees a target")):
        m = vis == v
        if m.any():
            ax.scatter(
                proj[m, 0], proj[m, 1],
                c=_VIS_COLORS[v], s=6, alpha=0.45, label=lbl, edgecolors="none",
            )
    ax.axvline(0, color="gray", lw=0.5, alpha=0.4)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax.set_title(
        f"{title_prefix} — by target visibility\n"
        f"(PC1={var[0]:.1%}, PC2={var[1]:.1%})", fontsize=10
    )
    ax.set_xlabel("PC1")
    ax.legend(markerscale=2.5, fontsize=8, loc="best")


def _fig_pca_deepdive(
    data: MAGICCCData, out_dir: Path, prefix: str, encoder_label: str = ""
) -> None:
    labels = data.labels()
    types = labels["agent_type"]
    vis = labels["visible"]

    pre = data.pre_array()
    pre_res = _pca(pre)
    have_post = data.has_agg
    post_res = _pca(data.post_array()) if have_post else (None, None, 0)

    if pre_res[0] is None:
        print("  [MAGIC-CC] not enough message data for PCA — skipping deepdive.")
        return

    n_rows = 2 if (have_post and post_res[0] is not None) else 1
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(12, 5.4 * n_rows), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    enc = f" — encoder: {encoder_label}" if encoder_label else ""
    fig.suptitle(
        f"MAGIC message embeddings — PCA (continuous-coord blind scenario{enc})\n"
        "each point = one agent's emitted message at one step",
        fontsize=12, fontweight="bold",
    )

    pre_proj, pre_var, pre_n = pre_res
    # Align label rows to the finite rows used by PCA.
    pre_finite_mask = np.all(np.isfinite(pre), axis=1)
    _scatter_by_type(axes[0][0], pre_proj, types[pre_finite_mask], pre_var,
                     f"pre-GAT (n={pre_n})")
    _scatter_by_visibility(axes[0][1], pre_proj, vis[pre_finite_mask], pre_var,
                           "pre-GAT")

    if n_rows == 2:
        post_proj, post_var, post_n = post_res
        post = data.post_array()
        post_finite_mask = np.all(np.isfinite(post), axis=1)
        _scatter_by_type(axes[1][0], post_proj, types[post_finite_mask], post_var,
                         f"post-GAT (n={post_n})")
        _scatter_by_visibility(axes[1][1], post_proj, vis[post_finite_mask], post_var,
                               "post-GAT")

    save_figure(fig, out_dir, f"{prefix}_msg_pca_deepdive")


def _fig_explained_variance(data: MAGICCCData, out_dir: Path, prefix: str) -> None:
    bases: list[tuple[str, tuple]] = []
    pre_res = _pca(data.pre_array())
    if pre_res[0] is not None:
        bases.append(("pre-GAT", pre_res))
    if data.has_agg:
        post_res = _pca(data.post_array())
        if post_res[0] is not None:
            bases.append(("post-GAT", post_res))
    if not bases:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.6), constrained_layout=True)
    names = [b[0] for b in bases]
    pc1 = [float(b[1][1][0]) * 100 for b in bases]
    pc2 = [float(b[1][1][1]) * 100 for b in bases]
    tail = [max(0.0, 100.0 - p1 - p2) for p1, p2 in zip(pc1, pc2)]
    x = np.arange(len(names))
    ax.bar(x, pc1, color="#3B5BA5", label="PC1")
    ax.bar(x, pc2, bottom=pc1, color="#7FA0D0", label="PC2")
    ax.bar(x, tail, bottom=[a + b for a, b in zip(pc1, pc2)],
           color="#B57EA6", label=r"residual tail (PC$_{\geq 3}$)")
    for xi, (p1, p2) in enumerate(zip(pc1, pc2)):
        ax.text(xi, 101, f"tail {100 - p1 - p2:.0f}%", ha="center",
                va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Message-embedding explained variance\n(pre- vs post-GAT)",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    save_figure(fig, out_dir, f"{prefix}_msg_variance")


def save_all_magic_cc_figures(
    data: MAGICCCData,
    output_dir: str | Path = "eval_plots/continuous_coord/magic",
    prefix: str = "",
    encoder_label: str = "",
) -> None:
    """Write the pre-/post-GAT message PCA figures for a MAGIC continuous-coord run."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pfx = prefix or "magic_cc"

    if not data.has_comm_data or not data.pre_messages:
        print(
            "  [MAGIC-CC] no communication messages collected — "
            "skipping message PCA figures."
        )
        return

    print(
        f"  [MAGIC-CC] {len(data.pre_messages)} message rows "
        f"({data.n_episodes} episodes); post-GAT available: {data.has_agg}"
    )
    _fig_pca_deepdive(data, out, pfx, encoder_label=encoder_label)
    _fig_explained_variance(data, out, pfx)
    print(f"  [MAGIC-CC] message PCA written → {out}/")
