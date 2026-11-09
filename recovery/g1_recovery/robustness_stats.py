"""Pure statistics for the M3 (terrain) and M4 (push/friction) evaluations. numpy only; unit-tested on CPU (tests/test_robustness_stats.py)."""
from __future__ import annotations
import math
import numpy as np


def wilson_ci(k, n, z=1.959964):
    """95% Wilson score interval for k successes out of n (returns (lo, hi); (0, 1) when n = 0)."""
    if n <= 0: return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def hardest_level_with_rate(levels, survived, rate=0.8):
    """levels: sorted difficulty indices; survived: dict level -> (k, n). Returns the highest level L such that the survival share is >= rate at L AND at every easier level
    (so one lucky hard level above a failing one does not count). None if even the easiest level is below rate."""
    best = None
    for L in levels:
        k, n = survived[L]
        if n == 0 or k / n < rate: break
        best = L
    return best


def logistic_fit(x, k, n, iters=60):
    """Maximum-likelihood logistic regression of the FALL probability on x (grouped counts k falls out of n). Returns (a, b) with p(x) = 1 / (1 + exp(-(a + b x))), or None when
    the data are separable or flat (no usable slope)."""
    x = np.asarray(x, float); k = np.asarray(k, float); n = np.asarray(n, float)
    if k.sum() == 0 or k.sum() == n.sum(): return None
    a, b = 0.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a + b * x))); w = n * p * (1 - p) + 1e-9
        g = np.array([np.sum(k - n * p), np.sum((k - n * p) * x)]); H = np.array([[np.sum(w), np.sum(w * x)], [np.sum(w * x), np.sum(w * x * x)]])
        try: step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        except np.linalg.LinAlgError: return None
        a, b = a + step[0], b + step[1]
        if np.abs(step).max() < 1e-9: break
    return (float(a), float(b)) if abs(b) > 1e-6 and abs(a) < 1e3 and abs(b) < 1e3 else None


def m50_interp(x, k, n):
    """Push magnitude where the fall share crosses 50%, by linear interpolation of the observed fall shares (None if it never crosses; the lowest x if it is already >= 50% there)."""
    x = np.asarray(x, float); p = np.asarray(k, float) / np.maximum(np.asarray(n, float), 1)
    o = np.argsort(x); x, p = x[o], p[o]
    if p[0] >= 0.5: return float(x[0])
    for i in range(1, len(x)):
        if p[i] >= 0.5 > p[i - 1]: return float(x[i - 1] + (0.5 - p[i - 1]) / (p[i] - p[i - 1]) * (x[i] - x[i - 1]))
    return None


def m50_logistic(x, k, n):
    f = logistic_fit(x, k, n)
    return None if f is None else float(-f[0] / f[1])


def m50_bootstrap_ci(x, k, n, draws=400, seed=0):
    """95% percentile bootstrap interval of the logistic M50, resampling the binomial counts per magnitude. Returns (lo, hi, n_valid)."""
    rng = np.random.default_rng(seed); x = np.asarray(x, float); k = np.asarray(k, float); n = np.asarray(n, int); p = k / np.maximum(n, 1); vals = []
    for _ in range(draws):
        kb = rng.binomial(n, p); m = m50_logistic(x, kb, n)
        if m is not None and np.isfinite(m): vals.append(m)
    if len(vals) < 20: return (None, None, len(vals))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals))
