"""Turn candidate features into a percentage per candidate answer.

    score(a) = sum_i w_i * x_i(a)                      (after centring/scaling each x_i)
    %(a)     = e^(score(a)/T) / sum_b e^(score(b)/T)

Weights w are learned by maximising  sum over past debates of ln(% given to the right answer)
minus 0.001 * sum_i w_i^2.  T (the softening factor) is chosen afterwards on held-out debates
so that "the model says 80%" means "right about 80% of the time".
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .features import NF

L2 = 1e-3


class Observer:
    def __init__(self, mu=None, sd=None, w=None, T: float = 1.0):
        self.mu = np.zeros(NF) if mu is None else np.asarray(mu, float)
        self.sd = np.ones(NF) if sd is None else np.asarray(sd, float)
        self.w = np.zeros(NF) if w is None else np.asarray(w, float)
        self.T = float(T)

    # -------------------------------------------------------------- learning
    def fit(self, items: list[tuple[np.ndarray, int]]) -> "Observer":
        """items: (feature matrix of one debate, index of the right answer or -1 if absent)."""
        rows = [F for F, _ in items if len(F)]
        if not rows:
            return self
        allX = np.vstack(rows)
        self.mu, self.sd = allX.mean(0), allX.std(0) + 1e-8
        usable = [(F, y) for F, y in items if y >= 0 and len(F)]
        if not usable:
            return self
        sizes = np.array([F.shape[0] for F, _ in usable])
        starts = np.r_[0, np.cumsum(sizes)[:-1]]
        X = (np.vstack([F for F, _ in usable]) - self.mu) / self.sd
        right_rows = starts + np.array([y for _, y in usable])
        group = np.repeat(np.arange(len(usable)), sizes)
        n = len(usable)
        X_right = X[right_rows].sum(0)

        def objective(w):
            z = X @ w
            zmax = np.maximum.reduceat(z, starts)
            e = np.exp(z - zmax[group])
            s = np.add.reduceat(e, starts)
            neg_log_lik = ((zmax + np.log(s)).sum() - z[right_rows].sum()) / n
            p = e / s[group]
            grad = (p @ X - X_right) / n + 2 * L2 * w
            return neg_log_lik + L2 * w @ w, grad

        res = minimize(objective, np.zeros(NF), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=500))
        self.w = res.x
        return self

    def raw_scores(self, F: np.ndarray) -> np.ndarray:
        return ((F - self.mu) / self.sd) @ self.w

    def probs(self, F: np.ndarray, T: float | None = None) -> np.ndarray:
        if len(F) == 0:
            return np.zeros(0)
        return softmax(self.raw_scores(F), self.T if T is None else T)

    def to_dict(self) -> dict:
        return dict(mu=self.mu.tolist(), sd=self.sd.tolist(), w=self.w.tolist(), T=self.T)

    @classmethod
    def from_dict(cls, d: dict) -> "Observer":
        return cls(d["mu"], d["sd"], d["w"], d.get("T", 1.0))


def softmax(scores: np.ndarray, T: float) -> np.ndarray:
    z = scores / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


T_GRID = np.round(np.exp(np.linspace(np.log(0.25), np.log(4.0), 41)), 4)


def calibration_gap(top_prob: np.ndarray, top_right: np.ndarray, n_bins: int = 10) -> float:
    """Group debates by the top percentage, compare mean percentage with the share that was
    right in each group, and average the gaps weighted by group size."""
    bins = np.minimum((top_prob * n_bins).astype(int), n_bins - 1)
    gap = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            gap += m.mean() * abs(top_prob[m].mean() - top_right[m].mean())
    return gap


def fit_temperature(score_lists: list[np.ndarray], right_idx: list[int]) -> float:
    """Pick T from T_GRID with the smallest calibration gap on held-out debates."""
    keep = [(s, y) for s, y in zip(score_lists, right_idx) if len(s)]
    if not keep:
        return 1.0
    best_T, best_gap = 1.0, None
    for T in T_GRID:
        tops, rights = [], []
        for s, y in keep:
            p = softmax(s, T)
            j = int(np.argmax(p))
            tops.append(p[j])
            rights.append(1.0 if j == y else 0.0)
        g = calibration_gap(np.array(tops), np.array(rights))
        if best_gap is None or g < best_gap - 1e-12:
            best_T, best_gap = float(T), g
    return best_T
