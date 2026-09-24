"""Train and certify an OSC policy from full-depth debate logs.

    python -m osc.train --logs "results/logs_osc/**/debate_full_*.jsonl" --task gsm8k \
        --out results/osc/policy_gsm8k.json --eps 0.01 --delta 0.10 --risk harm

Steps
  1. features for every debate at every checkpoint
  2. held-out percentages: 5 folds split by question, each fold scored by weights learned on
     the other four (so no debate is scored by weights that saw it)
  3. softening factor T per checkpoint, chosen on those held-out percentages
  4. split questions in two halves: half 1 picks how thresholds fall per round (N1) and the
     mid-round margin (N2); half 2 certifies the level a with the binomial test
  5. final weights learned on all debates, saved together with T and the thresholds

With --verify_logs (offline verdicts from osc.verify_offline) the VERIFY action is trained too:
half 1 also fits the verifier model tau and the Pareto front over (a, kappa, ell); half 2
certifies along that front (osc/verify.py). kappa = never-verify stays on the grid, so the
result falls back to plain OSC when verifying does not pay.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.model_selection import GroupKFold

from .certify import (Arrays, certify, choose_shape, default_shapes, min_questions, simulate,
                      smallest_eps, thresholds)
from .data import Debate, all_checkpoints, checkpoint_costs, load_debates
from .features import candidate_features, prefix_at
from .observer import Observer, fit_temperature, softmax
from .policy import OSCPolicy

TASKS = ["gsm8k", "mmlu", "strategyqa", "commonsenseqa", "truthfulqa", "bbh"]


class FeatureCache:
    def __init__(self, debates: list[Debate], checkpoints: list[str]):
        self.debates = debates
        self.checkpoints = checkpoints
        self.cands, self.F, self.y = {}, {}, {}
        for i, d in enumerate(debates):
            for cp in checkpoints:
                c, F = candidate_features(prefix_at(d.rounds, cp))
                self.cands[i, cp], self.F[i, cp] = c, F
                self.y[i, cp] = c.index(d.gt) if d.gt in c else -1
        self.cost = np.array([checkpoint_costs(d, checkpoints) for d in debates])
        self.group = np.array([d.group for d in debates])

    def fit(self, idx, cp: str) -> Observer:
        return Observer().fit([(self.F[i, cp], self.y[i, cp]) for i in idx])


def heldout_scores(cache: FeatureCache, idx: np.ndarray, n_folds: int = 5) -> dict:
    """Raw scores for every (debate, checkpoint), each from weights that never saw that question."""
    out = {}
    folds = min(n_folds, len(np.unique(cache.group[idx])))
    for tr, te in GroupKFold(n_splits=folds).split(idx, groups=cache.group[idx]):
        for cp in cache.checkpoints:
            obs = cache.fit(idx[tr], cp)
            for i in idx[te]:
                out[i, cp] = obs.raw_scores(cache.F[i, cp]) if len(cache.F[i, cp]) else np.zeros(0)
    return out


def temperatures(cache: FeatureCache, idx, scores: dict) -> dict:
    return {cp: fit_temperature([scores[i, cp] for i in idx], [cache.y[i, cp] for i in idx])
            for cp in cache.checkpoints}


def arrays_from(cache: FeatureCache, idx, scores: dict, T: dict) -> Arrays:
    n, C = len(idx), len(cache.checkpoints)
    margin, correct = np.zeros((n, C)), np.zeros((n, C), bool)
    top = np.empty((n, C), dtype=object)
    for r, i in enumerate(idx):
        for c, cp in enumerate(cache.checkpoints):
            s = scores[i, cp]
            if len(s) == 0:
                continue
            p = softmax(s, T[cp])
            order = np.argsort(-p)
            margin[r, c] = p[order[0]] - (p[order[1]] if len(p) > 1 else 0.0)
            top[r, c] = cache.cands[i, cp][order[0]]
            correct[r, c] = order[0] == cache.y[i, cp]
    return Arrays(margin, correct, top, cache.cost[idx], cache.group[idx])


def split_halves(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    uq = np.unique(groups)
    rs = np.random.RandomState(seed)
    first = set(rs.choice(uq, len(uq) // 2, replace=False))
    h1 = np.array([g in first for g in groups])
    return h1, ~h1


def prepare(cache: FeatureCache, idx: np.ndarray) -> dict:
    """Everything that does not depend on eps/delta: held-out arrays, T, final weights."""
    idx = np.asarray(idx)
    scores = heldout_scores(cache, idx)
    T = temperatures(cache, idx, scores)
    observers = {}
    for cp in cache.checkpoints:
        o = cache.fit(idx, cp)
        o.T = T[cp]
        observers[cp] = o
    return dict(idx=idx, T=T, arr=arrays_from(cache, idx, scores, T), observers=observers,
                scores=scores)


def fit_policy(cache: FeatureCache, idx: np.ndarray, task: str, max_rounds: int, mid_round: bool,
               eps: float, delta: float, risk: str, seed: int = 0,
               prep: dict | None = None) -> tuple[OSCPolicy, dict]:
    prep = prep or prepare(cache, idx)
    idx, T, arr, observers = prep["idx"], prep["T"], prep["arr"], prep["observers"]
    h1, h2 = split_halves(arr.group, seed)
    shape, shape_report = choose_shape(arr.subset(h1), cache.checkpoints, eps, risk,
                                       default_shapes(mid_round))
    cert = certify(arr.subset(h2), cache.checkpoints, shape, eps, delta, risk, seed)
    a = cert["a"]
    report = dict(n_debates=int(len(idx)), n_questions=int(len(np.unique(arr.group))),
                  eps=eps, delta=delta, risk=risk, shape=dict(b=shape.b, mid_extra=shape.mid_extra),
                  shape_search=shape_report, certified_a=a, certify_n=cert["n"],
                  temperatures=T)
    if a is not None:
        sim = simulate(arr.subset(h2), thresholds(cache.checkpoints, a, shape))
        report["half2"] = dict(acc=float(sim["correct"].mean()), cost=float(sim["cost"].mean()),
                               harm=float(sim["harm"].mean()), disagree=float(sim["disagree"].mean()),
                               full_depth_acc=float(arr.subset(h2).correct[:, -1].mean()))
    policy = OSCPolicy(task, max_rounds, mid_round, observers, a, shape, meta=report)
    return policy, report


def fit_policy_v(cache: FeatureCache, idx: np.ndarray, task: str, max_rounds: int, mid_round: bool,
                 eps: float, delta: float, risk: str, vtab, seed: int = 0, prep: dict | None = None,
                 verify_cost: float | None = None, verify_mode: str = "reasoning",
                 max_verify_rate: float | None = None) -> tuple[OSCPolicy, dict]:
    """OSC with the VERIFY action (PLAN_OSC_V.md sections 2.3-2.5)."""
    from .verify import (TauModel, attach_tau, build_tables, certify_v, simulate_v, tau_rows,
                         verifier_report)
    prep = prep or prepare(cache, idx)
    idx, T, arr, observers = prep["idx"], prep["T"], prep["arr"], prep["observers"]
    cps = cache.checkpoints
    tb, probs = build_tables(cache, idx, prep["scores"], T, vtab, verify_cost)
    h1, h2 = split_halves(arr.group, seed)
    shape, shape_report = choose_shape(arr.subset(h1), cps, eps, risk, default_shapes(mid_round))
    # tau is part of the certified policy, so it is fitted on half 1 only
    tau = TauModel.fit(*tau_rows(tb.subset(h1)))
    tb = attach_tau(tb, probs, tau)
    cert = certify_v(tb.subset(h1), tb.subset(h2), cps, shape, eps, delta, risk, seed,
                     max_verify_rate=max_verify_rate)
    st = cert["setting"]
    a = st["a"] if st else None
    verify = None
    if st is not None and st["kappa"] is not None:
        verify = dict(tau=tau.to_dict(), kappa=st["kappa"], ell=st["ell"], mode=verify_mode)
    report = dict(n_debates=int(len(idx)), n_questions=int(len(np.unique(arr.group))),
                  eps=eps, delta=delta, risk=risk, shape=dict(b=shape.b, mid_extra=shape.mid_extra),
                  shape_search=shape_report, certified_a=a, certify_n=cert["n"],
                  certified_setting=st, front_size=cert["front_size"],
                  verifier_half1=verifier_report(tb.subset(h1)),
                  verdict_coverage=float((tb.verdict != 0).mean()), temperatures=T)
    if st is not None:
        th = thresholds(cps, a, shape)
        t2 = tb.subset(h2)
        sim = simulate_v(t2, th, st["kappa"], st["ell"])
        base = simulate_v(t2, th, None)
        report["half2"] = dict(acc=float(sim["correct"].mean()), cost=float(sim["cost"].mean()),
                               harm=float(sim["harm"].mean()),
                               verify_rate=float(sim["verified"].mean()),
                               no_verify_acc=float(base["correct"].mean()),
                               no_verify_cost=float(base["cost"].mean()),
                               full_depth_acc=float(t2.correct[:, -1].mean()))
    policy = OSCPolicy(task, max_rounds, mid_round, observers, a, shape, meta=report, verify=verify)
    return policy, report


def check_sample_size(n_questions: int, eps: float, delta: float) -> str | None:
    """Warn before training when half of the questions cannot certify eps at all."""
    need, have = min_questions(eps, delta), n_questions // 2
    if have >= need:
        return None
    return (f"only {have} questions in the certification half; eps={eps} needs at least {need} "
            f"(smallest eps these questions can certify: {smallest_eps(have, delta):.3f})")


def main():
    ap = argparse.ArgumentParser(description="Train + certify an OSC stopping policy")
    ap.add_argument("--logs", required=True, help='glob, e.g. "results/logs_osc/**/debate_full_*.jsonl"')
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_rounds", type=int, default=6)
    ap.add_argument("--no_mid", action="store_true", help="disable mid-round stopping (N2)")
    ap.add_argument("--eps", type=float, default=0.01, help="largest acceptable harm rate")
    ap.add_argument("--delta", type=float, default=0.10, help="1 - confidence of the guarantee")
    ap.add_argument("--risk", choices=["harm", "disagree"], default="harm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify_logs", default=None,
                    help="glob of osc.verify_offline outputs; trains the VERIFY action too")
    ap.add_argument("--verify_mode", default="reasoning", choices=["reasoning", "blind"])
    ap.add_argument("--verify_cost", type=float, default=None,
                    help="cost of one verification as a share of a full debate "
                         "(default: from logged tokens)")
    ap.add_argument("--max_verify_rate", type=float, default=None,
                    help="optional budget: largest share of debates that may be verified")
    args = ap.parse_args()

    debates = load_debates(args.logs, args.task, args.max_rounds)
    if not debates:
        raise SystemExit("no full-depth debates found — collect with --stop none (or osc audits)")
    warn = check_sample_size(len({d.group for d in debates}), args.eps, args.delta)
    if warn:
        print("WARNING:", warn)
    cps = all_checkpoints(args.max_rounds, not args.no_mid)
    cache = FeatureCache(debates, cps)
    if args.verify_logs:
        from .verify import VerifyTable
        vtab = VerifyTable.load(args.verify_logs, args.task, args.verify_mode)
        if not len(vtab):
            raise SystemExit(f"no {args.verify_mode} verdicts for {args.task} in {args.verify_logs}")
        policy, report = fit_policy_v(cache, np.arange(len(debates)), args.task, args.max_rounds,
                                      not args.no_mid, args.eps, args.delta, args.risk, vtab,
                                      args.seed, verify_cost=args.verify_cost,
                                      verify_mode=args.verify_mode,
                                      max_verify_rate=args.max_verify_rate)
    else:
        policy, report = fit_policy(cache, np.arange(len(debates)), args.task, args.max_rounds,
                                    not args.no_mid, args.eps, args.delta, args.risk, args.seed)
    policy.save(args.out)
    print(json.dumps({k: v for k, v in report.items() if k != "temperatures"}, indent=1))
    if report["certified_a"] is None:
        print("WARNING: nothing could be certified (too few debates for this eps/delta); "
              "the policy will never stop early.")
    print("saved", args.out)


if __name__ == "__main__":
    main()
