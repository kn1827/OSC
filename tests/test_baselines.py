"""Tests for the replayed stopping baselines (baselines/stopping.py).

Run with pytest, or directly:  python tests/test_baselines.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from baselines.stopping import (hull_line, rule_ac_beta, rule_consensus, rule_fixed_depth,
                                rule_sprt, rule_stability, upper_hull)


def rnd(a, b, c, conf=0.9):
    return dict(a=a, b=b, c=c, ca=conf, cb=conf, cc=conf)


UNANIMOUS = [rnd("18", "18", "18")] * 6
SPLIT = [rnd("18", "18", "20")] * 6


def test_fixed_depth_and_consensus():
    assert rule_fixed_depth(SPLIT, 2) == (1, "18")
    assert rule_consensus(UNANIMOUS, 1) == (0, "18")
    assert rule_consensus(UNANIMOUS, 2) == (1, "18")
    assert rule_consensus(SPLIT, 1) == (5, "18")          # never unanimous -> last round


def test_ac_beta_matches_closed_form():
    # 3 votes vs 0: P(q > 1/2) for Beta(4, 1) = 1 - 0.5^4 = 0.9375
    assert rule_ac_beta(UNANIMOUS, 0.93)[0] == 0
    assert rule_ac_beta(UNANIMOUS, 0.95)[0] == 1          # 6 vs 0: 1 - 0.5^7 = 0.992
    assert rule_ac_beta(SPLIT, 0.5)[0] == 0


def test_sprt_counts_votes_as_independent():
    # 3 agreeing votes: 3 log(1.6) = 1.41 < log(0.9/0.1) = 2.20; 6 votes: 2.82 >= 2.20
    assert 3 * math.log(1.6) < math.log(9) < 6 * math.log(1.6)
    assert rule_sprt(UNANIMOUS, 0.1) == (1, "18")
    # a lasting 2-1 split adds 2 log 1.6 + log 0.4 = +0.02 per round: never reaches the bound
    assert rule_sprt(SPLIT, 0.1) == (5, "18")


def test_stability_waits_for_an_unchanged_round():
    rounds = [rnd("18", "20", "20"), rnd("18", "18", "20"), rnd("18", "18", "20"), rnd("18", "18", "18")]
    assert rule_stability(rounds, 1) == (2, "18")
    assert rule_stability(rounds, 2) == (3, "18")         # never 2 unchanged in a row -> last


def test_upper_hull_and_matched_cost_line():
    costs, accs = np.array([0.2, 0.4, 0.6, 0.8]), np.array([0.7, 0.72, 0.9, 0.85])
    assert upper_hull(costs, accs) == [0, 2]              # 0.4 is under the chord, 0.8 is worse
    correct = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], bool)
    cost = np.tile(costs[:, None], (1, 5))
    accs = correct.mean(1)                                 # 0.6, 0.8, 1.0, 0.8
    assert upper_hull(cost.mean(1), accs) == [0, 2]      # (0.4, 0.8) lies on the chord: dropped
    line = hull_line(correct, cost, 0.3)                   # halfway between the first two hull points
    assert abs(line.mean() - np.interp(0.3, cost.mean(1)[:3], accs[:3])) < 1e-9
    assert np.allclose(hull_line(correct, cost, 0.1), correct[0])   # cheaper than any setting
    assert np.allclose(hull_line(correct, cost, 0.95), correct[2])  # beyond the best setting


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                failed += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if failed else 0)
