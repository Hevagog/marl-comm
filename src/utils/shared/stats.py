"""
Implements the Agarwal et al. (2021) "rliable" evaluation protocol:
  - Interquartile Mean (IQM)
  - Optimality Gap
  - 95% stratified bootstrap confidence intervals
  - Mann-Whitney U two-sided pairwise tests
  - Multi-seed aggregate metrics table

Reference: Agarwal et al. "Deep Reinforcement Learning at the Edge of
the Statistical Precipice." NeurIPS 2021.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats as scipy_stats


def iqm(scores: np.ndarray) -> float:
    """Interquartile Mean — mean of the middle 50% of scores.

    Parameters
    ----------
    scores : (N,) array of per-episode or per-seed scalar scores.
    """
    scores = np.asarray(scores, dtype=float)
    q25, q75 = np.percentile(scores, [25, 75])
    mask = (scores >= q25) & (scores <= q75)
    trimmed = scores[mask]
    return float(np.mean(trimmed)) if len(trimmed) > 0 else float(np.mean(scores))


def optimality_gap(scores: np.ndarray, optimal: float) -> float:
    """Fraction of the optimal score that agents fail to achieve, clipped to [0,1].

    optimality_gap = 1 - IQM(scores) / optimal
    """
    if optimal == 0:
        return 0.0
    return float(np.clip(1.0 - iqm(scores) / optimal, 0.0, 1.0))


def bootstrap_ci(
    scores: np.ndarray,
    metric: Callable[[np.ndarray], float] = iqm,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for a scalar metric.

    Returns
    -------
    (point_estimate, ci_low, ci_high)
    """
    rng = rng or np.random.default_rng(0)
    scores = np.asarray(scores, dtype=float)
    point = metric(scores)
    n = len(scores)
    boot = np.array(
        [metric(rng.choice(scores, size=n, replace=True)) for _ in range(n_resamples)]
    )
    alpha = 1.0 - confidence
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def mann_whitney_u(x: np.ndarray, y: np.ndarray) -> dict:
    """Two-sided Mann-Whitney U test. Returns statistic, p-value, and effect size r."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    stat, p = scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
    # Effect size r = Z / sqrt(n)
    n = len(x) + len(y)
    z = scipy_stats.norm.ppf(1 - p / 2) * (1 if stat > len(x) * len(y) / 2 else -1)
    r = z / np.sqrt(n) if n > 0 else 0.0
    return {"statistic": float(stat), "p_value": float(p), "effect_r": float(r)}


def pairwise_tests(
    named_scores: dict[str, np.ndarray],
) -> dict[tuple[str, str], dict]:
    """All unique pairs of agents; each entry is a mann_whitney_u result dict."""
    names = list(named_scores.keys())
    results: dict[tuple[str, str], dict] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            results[(a, b)] = mann_whitney_u(named_scores[a], named_scores[b])
    return results


def aggregate_metrics(
    named_scores: dict[str, np.ndarray],
    optimal: float | None = None,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, dict]:
    """Compute IQM + 95% bootstrap CI (+ optimality gap) for each agent family.

    Parameters
    ----------
    named_scores : mapping from agent name to 1-D array of episode returns.
    optimal      : known optimal score for optimality gap; omit to skip.
    n_resamples  : bootstrap resamples.
    confidence   : CI level (default 0.95).

    Returns
    -------
    {agent_name: {"iqm": float, "ci_low": float, "ci_high": float,
                  "mean": float, "std": float, "n": int,
                  "optimality_gap": float | None}}
    """
    rng = np.random.default_rng(0)
    out: dict[str, dict] = {}
    for name, scores in named_scores.items():
        scores = np.asarray(scores, dtype=float)
        point, lo, hi = bootstrap_ci(
            scores, metric=iqm, n_resamples=n_resamples, confidence=confidence, rng=rng
        )
        entry: dict = {
            "iqm": point,
            "ci_low": lo,
            "ci_high": hi,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "n": len(scores),
            "optimality_gap": optimality_gap(scores, optimal) if optimal else None,
        }
        out[name] = entry
    return out


def print_aggregate_table(metrics: dict[str, dict]) -> None:
    """Print a formatted dissertation-style aggregate metrics table."""
    header = f"{'Agent':<25}  {'IQM':>8}  {'CI 95%':>17}  {'Mean±Std':>16}  {'N':>5}"
    print(header)
    print("─" * len(header))
    for name, m in metrics.items():
        ci = f"[{m['ci_low']:.2f}, {m['ci_high']:.2f}]"
        ms = f"{m['mean']:.2f}±{m['std']:.2f}"
        print(f"{name:<25}  {m['iqm']:>8.3f}  {ci:>17}  {ms:>16}  {m['n']:>5}")
