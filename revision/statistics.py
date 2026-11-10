"""Statistical utilities for revision tables and figures."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Interval:
    low: float
    high: float
    method: str

    def as_dict(self, prefix: str) -> dict[str, float | str]:
        return {
            f"{prefix}_low": self.low,
            f"{prefix}_high": self.high,
            f"{prefix}_method": self.method,
        }


def proportion_ci(k: int, n: int, confidence: float = 0.95) -> Interval:
    """Binomial proportion interval.

    Uses Clopper-Pearson if scipy is available, otherwise Wilson score.
    """
    if n <= 0:
        return Interval(float("nan"), float("nan"), "undefined")
    alpha = 1.0 - confidence
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from scipy.stats import beta

        low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
        high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
        return Interval(low, high, "clopper_pearson")
    except Exception:
        # 1.959963984540054 is norm.ppf(0.975), suitable for confidence=0.95.
        z = 1.959963984540054 if abs(confidence - 0.95) < 1e-12 else 1.96
        phat = k / n
        denom = 1 + z * z / n
        center = (phat + z * z / (2 * n)) / denom
        half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
        return Interval(max(0.0, center - half), min(1.0, center + half), "wilson")


def bootstrap_ci(
    values: Sequence[float],
    statistic: str = "mean",
    confidence: float = 0.95,
    n_boot: int = 5000,
    seed: int = 1234,
) -> Interval:
    """Percentile bootstrap interval for mean or median."""
    if not values:
        return Interval(float("nan"), float("nan"), "undefined")
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for bootstrap_ci.") from exc

    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    reducer = np.mean if statistic == "mean" else np.median
    samples = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        draw = rng.choice(arr, size=len(arr), replace=True)
        samples[idx] = reducer(draw)
    alpha = 1.0 - confidence
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return Interval(float(low), float(high), f"percentile_bootstrap_{statistic}")


def permutation_test(
    left: Sequence[float],
    right: Sequence[float],
    *,
    statistic: str = "mean",
    n_perm: int = 10000,
    seed: int = 1234,
    alternative: str = "two-sided",
) -> float:
    """Permutation test for the difference between two samples."""
    if not left or not right:
        return float("nan")
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for permutation_test.") from exc

    reducer = np.mean if statistic == "mean" else np.median
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    observed = float(reducer(a) - reducer(b))
    pooled = np.concatenate([a, b])
    n_left = len(a)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = float(reducer(pooled[:n_left]) - reducer(pooled[n_left:]))
        if alternative == "greater":
            extreme += diff >= observed
        elif alternative == "less":
            extreme += diff <= observed
        else:
            extreme += abs(diff) >= abs(observed)
    return (extreme + 1) / (n_perm + 1)


def holm_correction(p_values: Iterable[float]) -> list[float]:
    """Holm adjusted p-values, returned in original order."""
    values = list(p_values)
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    m = len(values)
    adjusted = [float("nan")] * m
    running = 0.0
    for rank, (idx, p_value) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p_value)
        running = max(running, adj)
        adjusted[idx] = running
    return adjusted


def benjamini_hochberg(p_values: Iterable[float]) -> list[float]:
    """Benjamini-Hochberg FDR adjusted p-values in original order."""
    values = list(p_values)
    indexed = sorted(enumerate(values), key=lambda pair: pair[1], reverse=True)
    m = len(values)
    adjusted = [float("nan")] * m
    running = 1.0
    for reverse_rank, (idx, p_value) in enumerate(indexed):
        rank = m - reverse_rank
        adj = min(running, p_value * m / rank)
        running = adj
        adjusted[idx] = min(1.0, adj)
    return adjusted
