"""CPU tests for the M3/M4 statistics helpers. Run: <lab python> -I tests/test_robustness_stats.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from g1_recovery import robustness_stats as R


def test_wilson_matches_known_values_and_edges():
    lo, hi = R.wilson_ci(8, 10); assert abs(lo - 0.4902) < 2e-3 and abs(hi - 0.9433) < 2e-3          # standard Wilson interval for 8/10
    lo, hi = R.wilson_ci(0, 20); assert lo == 0.0 and 0.15 < hi < 0.17
    lo, hi = R.wilson_ci(20, 20); assert hi == 1.0 and 0.83 < lo < 0.85
    assert R.wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = R.wilson_ci(50, 100); assert abs((lo + hi) / 2 - 0.5) < 1e-6 and hi - lo < 0.21


def test_wilson_interval_narrows_with_n_and_contains_the_estimate():
    for k, n in ((3, 10), (30, 100), (300, 1000)):
        lo, hi = R.wilson_ci(k, n); assert lo <= k / n <= hi
    assert (lambda a, b: b - a)(*R.wilson_ci(300, 1000)) < (lambda a, b: b - a)(*R.wilson_ci(3, 10))


def test_hardest_level_requires_every_easier_level_to_pass():
    s = {0: (10, 10), 1: (9, 10), 2: (7, 10), 3: (10, 10)}                                          # level 3 passes but level 2 fails
    assert R.hardest_level_with_rate([0, 1, 2, 3], s, 0.8) == 1
    assert R.hardest_level_with_rate([0, 1, 2, 3], {0: (5, 10), 1: (10, 10)}, 0.8) is None
    assert R.hardest_level_with_rate([0, 1], {0: (8, 10), 1: (8, 10)}, 0.8) == 1                    # exactly at the threshold counts


def test_m50_interpolation_and_edges():
    x = [0, 1, 2, 3]; n = [20] * 4
    assert abs(R.m50_interp(x, [0, 4, 16, 20], n) - 1.5) < 1e-9                                      # 20% at 1, 80% at 2 -> 50% at 1.5
    assert R.m50_interp(x, [0, 1, 2, 3], n) is None                                                  # never reaches 50%
    assert R.m50_interp(x, [12, 15, 18, 20], n) == 0.0                                               # already above 50% at the lowest magnitude


def test_logistic_fit_recovers_a_known_curve():
    a, b = -6.0, 2.0; x = np.linspace(0, 6, 13); n = np.full(13, 4000); k = np.round(n / (1 + np.exp(-(a + b * x))))
    f = R.logistic_fit(x, k, n); assert abs(f[0] - a) < 0.15 and abs(f[1] - b) < 0.1
    assert abs(R.m50_logistic(x, k, n) - 3.0) < 0.05
    assert R.logistic_fit([0, 1, 2], [0, 0, 0], [10, 10, 10]) is None and R.logistic_fit([0, 1, 2], [10, 10, 10], [10, 10, 10]) is None


def test_bootstrap_interval_contains_the_estimate_and_is_reproducible():
    x = np.linspace(0, 4, 9); n = np.full(9, 40); k = np.round(n / (1 + np.exp(-(-6 + 3 * x))))
    lo, hi, nv = R.m50_bootstrap_ci(x, k, n, draws=300, seed=1); m = R.m50_logistic(x, k, n)
    assert lo < m < hi and nv > 250 and (lo, hi, nv) == R.m50_bootstrap_ci(x, k, n, draws=300, seed=1)


if __name__ == "__main__":
    c = 0
    for k, v in list(globals().items()):
        if k.startswith("test_"): v(); print("ok", k); c += 1
    print(c, "passed")
