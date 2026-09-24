"""Stopping thresholds and the binomial guarantee.

Threshold at checkpoint (round r, phase):
    threshold = a - b * r            after the whole round
    threshold = a - b * r + extra    right after the solvers (mid-round, upgrade N2)
The last checkpoint always stops.  Stop at the first checkpoint where
    lead = %(top answer) - %(second answer)  >=  threshold.

Guarantee (upgrade: N1 shape chosen on one half, a certified on the other half):
    a debate is "harmed" if the full debate is right but the early answer is wrong
    (or, label-free, if the early answer differs from the full-debate answer).
    With k harmed debates out of n, accept a setting when
        P = sum_{i=0..k} C(n, i) * eps^i * (1 - eps)^(n - i)  <=  delta.
    Settings are tried from the safest to the most aggressive and the search stops at the
    first failure (fixed order, so trying many settings cannot produce a lucky pass).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .features import parse_checkpoint

A_STEP = 0.01


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed exactly in log space."""
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    base = math.lgamma(n + 1)
    total = 0.0
    for i in range(k + 1):
        total += math.exp(base - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    return min(total, 1.0)


def min_questions(eps: float, delta: float) -> int:
    """Fewest independent questions for which zero harmed debates passes the test:
    (1 - eps)^n <= delta  <=>  n >= ln(delta) / ln(1 - eps)."""
    return math.ceil(math.log(delta) / math.log1p(-eps))


def smallest_eps(n: int, delta: float) -> float:
    """Smallest eps that n independent questions can certify at all (zero harmed debates)."""
    return 1.0 - delta ** (1.0 / max(n, 1))


@dataclass
class Shape:
    b: float = 0.0              # how much the threshold drops per round
    mid_extra: float | None = None  # extra lead required mid-round; None = no mid-round stop


def thresholds(checkpoints: list[str], a: float, shape: Shape) -> np.ndarray:
    thr = np.empty(len(checkpoints))
    for i, cp in enumerate(checkpoints):
        r, phase = parse_checkpoint(cp)
        t = a - shape.b * r
        if phase == "s":
            t = np.inf if shape.mid_extra is None else t + shape.mid_extra
        thr[i] = max(t, 0.0)
    thr[-1] = -np.inf  # last checkpoint always stops
    return thr


def a_grid(checkpoints: list[str], shape: Shape) -> np.ndarray:
    """From 'never stop early' down to 'stop at the first checkpoint'."""
    last_round = max(parse_checkpoint(cp)[0] for cp in checkpoints)
    extra = shape.mid_extra or 0.0
    start = 1.01 + shape.b * last_round + max(extra, 0.0)
    return np.round(np.arange(start, -A_STEP / 2, -A_STEP), 4)


@dataclass
class Arrays:
    """Per debate (rows) and per checkpoint (columns), computed from held-out observer output."""
    margin: np.ndarray     # lead of the top answer, 0..1
    correct: np.ndarray    # top answer is right (bool)
    top: np.ndarray        # top answer (object)
    cost: np.ndarray       # share of the full-debate cost spent up to this checkpoint
    group: np.ndarray      # question id (debates of the same question are not independent)

    def subset(self, m) -> "Arrays":
        return Arrays(self.margin[m], self.correct[m], self.top[m], self.cost[m], self.group[m])


def simulate(arr: Arrays, thr: np.ndarray) -> dict:
    hit = arr.margin >= thr[None, :]
    stop = hit.argmax(1)  # the last column always hits
    rows = np.arange(len(stop))
    last = arr.margin.shape[1] - 1
    early_right = arr.correct[rows, stop]
    return dict(
        stop=stop,
        correct=early_right,
        cost=arr.cost[rows, stop],
        harm=arr.correct[:, last] & ~early_right,
        disagree=arr.top[rows, stop] != arr.top[:, last],
    )


def loss_of(sim: dict, risk: str) -> np.ndarray:
    return (sim["harm"] if risk == "harm" else sim["disagree"]).astype(float)


def default_shapes(mid_round: bool) -> list[Shape]:
    extras = [0.0, 0.05, 0.10, None] if mid_round else [None]
    return [Shape(b, e) for b in (0.0, 0.02, 0.05, 0.10) for e in extras]


def choose_shape(arr: Arrays, checkpoints: list[str], eps: float, risk: str,
                 shapes: list[Shape]) -> tuple[Shape, list[dict]]:
    """On the first half: for each shape find the most aggressive a whose observed risk is
    still <= eps, then keep the shape with the lowest cost at that a."""
    report, best, best_key = [], None, None
    for sh in shapes:
        chosen = None
        for a in a_grid(checkpoints, sh):
            sim = simulate(arr, thresholds(checkpoints, a, sh))
            if loss_of(sim, risk).mean() <= eps:
                chosen = (a, sim)
            else:
                break
        if chosen is None:
            continue
        a, sim = chosen
        row = dict(b=sh.b, mid_extra=sh.mid_extra, a=float(a), cost=float(sim["cost"].mean()),
                   acc=float(sim["correct"].mean()))
        report.append(row)
        key = (row["cost"], -row["acc"])
        if best_key is None or key < best_key:
            best, best_key = sh, key
    return (best or Shape()), report


def one_per_question(groups: np.ndarray, seed: int = 0) -> np.ndarray:
    """Pick one debate per question so the certified units are independent."""
    rs = np.random.RandomState(seed)
    keep = np.zeros(len(groups), bool)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        keep[rs.choice(idx)] = True
    return keep


def certify(arr: Arrays, checkpoints: list[str], shape: Shape, eps: float, delta: float,
            risk: str, seed: int = 0) -> dict:
    """Fixed-order binomial test on independent debates. Returns the certified a (or None)."""
    unit = arr.subset(one_per_question(arr.group, seed))
    n = len(unit.group)
    certified, trace = None, []
    for a in a_grid(checkpoints, shape):
        sim = simulate(unit, thresholds(checkpoints, a, shape))
        k = int(loss_of(sim, risk).sum())
        p = binom_cdf(k, n, eps)
        trace.append(dict(a=float(a), k=k, n=n, p=p, cost=float(sim["cost"].mean())))
        if p <= delta:
            certified = float(a)
        else:
            break
    return dict(a=certified, n=n, trace=trace)
