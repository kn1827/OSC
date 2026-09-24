"""The VERIFY action: verifier model, Bayes update, three-action policy and its certificate.

Verifier model (PLAN_OSC_V.md §2.3). For the leading answer â at a checkpoint with state s,
    tau1(s) = P(pass | â right, s)        tau0(s) = P(pass | â wrong, s)
are two logistic regressions on s = (logit p_top, margin, share of agents that ever backed â,
share of agents backing â in the latest, possibly partial, round, progress through the debate).
tau0 may depend on s because the verifier can share the debate's blind spots (it is fooled more
often when everyone agrees).

Bayes update. A verdict v on â multiplies â's odds against every other candidate by
    LR(pass) = tau1/tau0        LR(fail) = (1-tau1)/(1-tau0)
i.e. adds log LR to â's logit. The evidence is kept for the rest of the debate.

Three actions (Proposition 1). With two candidates, verifying beats stopping iff
    p <= u(s) = (1 - tau0 - kappa) / (2 - tau0 - tau1)
where kappa is the price of one verification in accuracy units. At each checkpoint:
    VERIFY  if the answer has not been verified and ell <= p_top <= u(s)
    STOP    if margin (after any update) >= threshold of the checkpoint
    DEBATE  otherwise.            At most one verification per debate.

Certificate (§2.5, Pareto testing). On half 1: fit tau, evaluate every (a, kappa, ell) on a
grid (kappa = None means never verify, i.e. plain OSC), keep the Pareto front of (cost, risk)
and sort it from safest to cheapest. On half 2 (one debate per question): fixed-sequence
binomial test in that order, stop at the first failure. The cheapest passing setting is kept.
"""
from __future__ import annotations

import glob
import json
import math
from dataclasses import dataclass

import numpy as np

from .certify import Shape, a_grid, binom_cdf, one_per_question, thresholds
from .features import AG, FEATURE_NAMES, canon, prefix_at
from .observer import softmax

STATE_NAMES = ["logit_top", "margin", "share_backing", "agree_latest", "progress"]
_SHARE = FEATURE_NAMES.index("share_of_agents_backing")
TAU_CLIP = (0.02, 0.98)
KAPPAS = (None, 0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
ELLS = (0.0, 0.3, 0.5)
VERDICT_CODE = {"pass": 1, "fail": -1}


def agree_latest(prefix: list[dict], answer) -> float:
    """Share of the agents that spoke in the latest (possibly partial) round backing answer."""
    last = prefix[-1] if prefix else {}
    said = [last.get(k) for k in AG if last.get(k) is not None]
    return sum(a == answer for a in said) / len(said) if said else 0.0


def state_vector(p_top: float, margin: float, f_top: np.ndarray, agree: float, ci: int,
                 n_cp: int) -> np.ndarray:
    p = min(max(float(p_top), 1e-4), 1 - 1e-4)
    return np.array([math.log(p / (1 - p)), float(margin), float(f_top[_SHARE]), float(agree),
                     ci / max(n_cp - 1, 1)])


def log_lr(verdict: int, t1: float, t0: float) -> float:
    if verdict > 0:
        return math.log(t1 / t0)
    if verdict < 0:
        return math.log((1 - t1) / (1 - t0))
    return 0.0


def verify_band_upper(t1, t0, kappa):
    """u(s) of Proposition 1 (vectorised)."""
    return (1.0 - t0 - kappa) / np.maximum(2.0 - t0 - t1, 1e-9)


# ----------------------------------------------------------------------------- verifier logs
class VerifyTable:
    """Offline verdicts keyed by (question group, canonical answer); see osc/verify_offline.py."""

    def __init__(self, records: list[dict]):
        self.map = {}
        for r in records:
            self.map.setdefault((r["group"], canon(r["answer"])), r)

    @classmethod
    def load(cls, pattern: str, task: str | None = None, mode: str | None = None) -> "VerifyTable":
        recs = []
        for f in sorted(glob.glob(pattern, recursive=True)):
            for line in open(f, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if (task and r.get("task") != task) or (mode and r.get("mode") != mode):
                    continue
                recs.append(r)
        return cls(recs)

    def __len__(self):
        return len(self.map)

    def get(self, group: str, answer) -> dict | None:
        return self.map.get((group, canon(answer)))

    def code(self, group: str, answer) -> int:
        r = self.get(group, answer)
        return VERDICT_CODE.get(r["verdict"], 0) if r else 0

    def tokens(self, group: str, answer) -> float:
        r = self.get(group, answer)
        return float(r.get("tokens", 0) or 0) if r else 0.0


# ----------------------------------------------------------------------------- tau model
@dataclass
class TauModel:
    mu: np.ndarray
    sd: np.ndarray
    w1: np.ndarray
    b1: float
    w0: np.ndarray
    b0: float

    def taus(self, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Z = (np.atleast_2d(S) - self.mu) / self.sd
        t1 = 1 / (1 + np.exp(-(Z @ self.w1 + self.b1)))
        t0 = 1 / (1 + np.exp(-(Z @ self.w0 + self.b0)))
        return np.clip(t1, *TAU_CLIP), np.clip(t0, *TAU_CLIP)

    @classmethod
    def fit(cls, S: np.ndarray, right: np.ndarray, passed: np.ndarray, weight: np.ndarray,
            C: float = 1.0, min_rows: int = 20) -> "TauModel":
        from sklearn.linear_model import LogisticRegression
        mu, sd = S.mean(0), S.std(0) + 1e-8
        Z = (S - mu) / sd
        params = []
        for m in (right, ~right):
            y, w = passed[m], weight[m]
            if m.sum() < min_rows or y.all() or (~y).all():
                # too little data or one outcome only: constant rate with a Laplace prior
                rate = (float((y * w).sum()) + 1) / (float(w.sum()) + 2)
                params.append((np.zeros(S.shape[1]), math.log(rate / (1 - rate))))
                continue
            lr = LogisticRegression(C=C, max_iter=1000).fit(Z[m], y, sample_weight=w)
            params.append((lr.coef_[0], float(lr.intercept_[0])))
        (w1, b1), (w0, b0) = params
        return cls(mu, sd, w1, b1, w0, b0)

    def to_dict(self) -> dict:
        return dict(mu=self.mu.tolist(), sd=self.sd.tolist(), w1=self.w1.tolist(), b1=self.b1,
                    w0=self.w0.tolist(), b0=self.b0, state=STATE_NAMES)

    @classmethod
    def from_dict(cls, d: dict) -> "TauModel":
        return cls(np.array(d["mu"]), np.array(d["sd"]), np.array(d["w1"]), float(d["b1"]),
                   np.array(d["w0"]), float(d["b0"]))


# ----------------------------------------------------------------------------- tables
@dataclass
class VTables:
    """Everything the three-action simulation needs, per debate (rows) and checkpoint (cols)."""
    p_top: np.ndarray
    margin: np.ndarray
    correct: np.ndarray
    top: np.ndarray
    cost: np.ndarray
    group: np.ndarray
    S: np.ndarray            # [n, C, k] state vectors
    verdict: np.ndarray      # [n, C] verdict on the top answer: +1 pass, -1 fail, 0 unavailable
    vcost: np.ndarray        # [n] cost of one verification, as a share of the full debate
    t1: np.ndarray = None    # [n, C]
    t0: np.ndarray = None
    post_margin: np.ndarray = None   # [n, C0, C] margin at C after verifying at C0
    post_correct: np.ndarray = None
    post_top: np.ndarray = None

    def subset(self, m) -> "VTables":
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v[m] if isinstance(v, np.ndarray) else v
        return VTables(**out)


def _debate_tokens(d) -> float:
    return float(sum(sum(float(v or 0) for v in (t or {}).values()) for t in d.round_tokens)) or 1.0


def build_tables(cache, idx, scores: dict, T: dict, vtab: VerifyTable,
                 verify_cost: float | None = None) -> tuple[VTables, list]:
    """Observer probabilities (held-out scores + T) joined with the offline verdicts.
    verify_cost: share of the full debate one verification costs; None = from logged tokens."""
    cps = cache.checkpoints
    n, C = len(idx), len(cps)
    k = len(STATE_NAMES)
    tb = VTables(p_top=np.zeros((n, C)), margin=np.zeros((n, C)), correct=np.zeros((n, C), bool),
                 top=np.empty((n, C), dtype=object), cost=cache.cost[idx], group=cache.group[idx],
                 S=np.zeros((n, C, k)), verdict=np.zeros((n, C), int), vcost=np.zeros(n))
    probs = []
    for r, i in enumerate(idx):
        d = cache.debates[i]
        row = []
        vtok = []
        for c, cp in enumerate(cps):
            s = scores[i, cp]
            cands = cache.cands[i, cp]
            if len(s) == 0:
                row.append((cands, np.zeros(0), -1))
                continue
            p = softmax(s, T[cp])
            order = np.argsort(-p)
            j = order[0]
            y = cache.y[i, cp]
            tb.p_top[r, c] = p[j]
            tb.margin[r, c] = p[j] - (p[order[1]] if len(p) > 1 else 0.0)
            tb.correct[r, c] = j == y
            tb.top[r, c] = cands[j]
            tb.S[r, c] = state_vector(p[j], tb.margin[r, c], cache.F[i, cp][j],
                                      agree_latest(prefix_at(d.rounds, cp), cands[j]), c, C)
            tb.verdict[r, c] = vtab.code(d.group, cands[j])
            if tb.verdict[r, c]:
                vtok.append(vtab.tokens(d.group, cands[j]))
            row.append((cands, p, y))
        probs.append(row)
        if verify_cost is not None:
            tb.vcost[r] = verify_cost
        else:
            tb.vcost[r] = (np.mean(vtok) if vtok else 0.0) / _debate_tokens(d)
    return tb, probs


def attach_tau(tb: VTables, probs: list, tau: TauModel) -> VTables:
    """tau at every cell, and the post-verification margins for every verification point."""
    n, C = tb.margin.shape
    t1, t0 = tau.taus(tb.S.reshape(n * C, -1))
    tb.t1, tb.t0 = t1.reshape(n, C), t0.reshape(n, C)
    tb.post_margin = np.full((n, C, C), -np.inf)
    tb.post_correct = np.zeros((n, C, C), bool)
    tb.post_top = np.empty((n, C, C), dtype=object)
    for r in range(n):
        for c0 in range(C):
            v = tb.verdict[r, c0]
            if v == 0:
                continue
            a = tb.top[r, c0]
            boost = log_lr(v, tb.t1[r, c0], tb.t0[r, c0])
            for c in range(c0, C):
                cands, p, y = probs[r][c]
                if len(p) == 0:
                    continue
                w = p * np.exp(boost * np.array([x == a for x in cands], float))
                w = w / w.sum()
                order = np.argsort(-w)
                tb.post_margin[r, c0, c] = w[order[0]] - (w[order[1]] if len(w) > 1 else 0.0)
                tb.post_correct[r, c0, c] = order[0] == y
                tb.post_top[r, c0, c] = cands[order[0]]
    return tb


# ----------------------------------------------------------------------------- simulation
def simulate_v(tb: VTables, thr: np.ndarray, kappa: float | None, ell: float = 0.0,
               mode: str = "band") -> dict:
    """mode 'band' = three-action policy; 'never' = plain OSC; 'at_stop' = verify the answer at
    the checkpoint where plain OSC would stop, then continue if the update removes the lead."""
    n, C = tb.margin.shape
    rows = np.arange(n)
    hit = tb.margin >= thr[None, :]
    hit[:, -1] = True
    c_s = hit.argmax(1)
    if mode == "never" or (mode == "band" and kappa is None):
        use = np.zeros(n, bool)
        c_v = c_s
    elif mode == "at_stop":
        c_v = c_s
        use = tb.verdict[rows, c_s] != 0
    else:
        u = verify_band_upper(tb.t1, tb.t0, kappa)
        band = (tb.p_top >= ell) & (tb.p_top <= u) & (tb.verdict != 0)
        c_v = band.argmax(1)
        use = band.any(1) & (c_v <= c_s)
    stop = c_s.copy()
    correct = tb.correct[rows, c_s].copy()
    top = tb.top[rows, c_s].copy()
    if use.any():
        r = rows[use]
        pm = tb.post_margin[r, c_v[r], :]
        hit2 = pm >= thr[None, :]
        hit2[np.arange(C)[None, :] < c_v[r][:, None]] = False
        hit2[:, -1] = True
        s2 = hit2.argmax(1)
        stop[r] = s2
        correct[r] = tb.post_correct[r, c_v[r], s2]
        top[r] = tb.post_top[r, c_v[r], s2]
    cost = tb.cost[rows, stop] + use * tb.vcost
    return dict(stop=stop, verified=use, verify_at=np.where(use, c_v, -1), correct=correct,
                cost=cost, harm=tb.correct[:, -1] & ~correct, disagree=top != tb.top[:, -1])


def _loss(sim: dict, risk: str) -> np.ndarray:
    return (sim["harm"] if risk == "harm" else sim["disagree"]).astype(float)


# ----------------------------------------------------------------------------- certificate
def pareto_front(points: list[dict]) -> list[dict]:
    """Settings not beaten on both cost and risk, sorted from safest (lowest risk) to cheapest."""
    pts = sorted(points, key=lambda p: (p["risk"], p["cost"]))
    front, best_cost = [], np.inf
    for p in pts:
        if p["cost"] < best_cost - 1e-12:
            front.append(p)
            best_cost = p["cost"]
    return front


def certify_v(tb1: VTables, tb2: VTables, checkpoints: list[str], shape: Shape, eps: float,
              delta: float, risk: str, seed: int = 0, kappas=KAPPAS, ells=ELLS,
              max_verify_rate: float | None = None) -> dict:
    grid = a_grid(checkpoints, shape)
    points = []
    for kappa in kappas:
        for ell in (ells if kappa is not None else (0.0,)):
            for a in grid:
                sim = simulate_v(tb1, thresholds(checkpoints, a, shape), kappa, ell)
                rate = float(sim["verified"].mean())
                if max_verify_rate is not None and rate > max_verify_rate:
                    continue
                points.append(dict(a=float(a), kappa=kappa, ell=ell, risk=float(_loss(sim, risk).mean()),
                                   cost=float(sim["cost"].mean()), acc=float(sim["correct"].mean()),
                                   verify_rate=rate))
    front = [p for p in pareto_front(points) if p["risk"] <= 1.5 * eps]
    unit = tb2.subset(one_per_question(tb2.group, seed))
    n = len(unit.group)
    chosen, trace = None, []
    for p in front:
        sim = simulate_v(unit, thresholds(checkpoints, p["a"], shape), p["kappa"], p["ell"])
        k = int(_loss(sim, risk).sum())
        pv = binom_cdf(k, n, eps)
        trace.append(dict(p, k=k, n=n, p_value=pv, cost_h2=float(sim["cost"].mean())))
        if pv <= delta:
            chosen = p
        else:
            break
    return dict(setting=chosen, n=n, front_size=len(front), trace=trace)


def tau_rows(tb: VTables) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Training rows for TauModel: every cell whose top answer has a pass/fail verdict.
    Each debate gets total weight 1 so long debates do not dominate."""
    m = tb.verdict != 0
    S = tb.S[m]
    right = tb.correct[m]
    passed = tb.verdict[m] > 0
    per_debate = m.sum(1)
    w = np.repeat(1.0 / np.maximum(per_debate, 1), per_debate)
    return S, right, passed, w


def verifier_report(tb: VTables) -> dict:
    """How informative the verifier is, overall and when all agents agree (blind spot)."""
    S, right, passed, w = tau_rows(tb)
    out = {}
    for name, m in (("all", np.ones(len(right), bool)), ("unanimous", S[:, 3] >= 0.999),
                    ("split", S[:, 3] < 0.999)):
        r1, r0 = m & right, m & ~right
        t1 = float(passed[r1].mean()) if r1.any() else float("nan")
        t0 = float(passed[r0].mean()) if r0.any() else float("nan")
        out[name] = dict(n_right=int(r1.sum()), n_wrong=int(r0.sum()), tau1=t1, tau0=t0,
                         youden=t1 - t0 if r1.any() and r0.any() else float("nan"))
    return out
