"""Nested evaluation of OSC against fixed-depth debate and consensus stopping.

    python -m osc.evaluate --logs "results/logs_osc/**/debate_full_*.jsonl" --task gsm8k \
        --out results/osc/eval_gsm8k.json

Outer loop: 5 folds split by question.  On the four training folds the whole training recipe
runs (held-out scores, T, shape, certification); the resulting policy is applied untouched to
the fifth fold.  Every number below is therefore measured on debates the policy never saw.

Main comparison: OSC vs "fixed depth at the same cost".  A fixed-depth system can reach any
average cost between two depths k and k+1 by running k rounds on some questions and k+1 on the
rest, chosen at random; its accuracy is then the straight line between the two points.

With --verify_logs (osc.verify_offline outputs) two more systems are evaluated the same way:
  OSC-V            the certified three-action policy (STOP / VERIFY / DEBATE, osc/verify.py)
  verify-at-stop   plain OSC, but the answer is always verified where OSC stops and the debate
                   continues if the verdict removes the lead (the "always verify" design)
Their token cost includes the verifier calls.

With --baselines the stopping rules of baselines/stopping.py (Adaptive-Consistency Beta rule,
Wald SPRT, answer-distribution stability, consensus, fixed depth) are replayed on the same debates
and every OSC / OSC-V result is compared with each rule at the same average cost.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.model_selection import GroupKFold

from .data import all_checkpoints, load_debates
from .features import majority
from .train import TASKS, FeatureCache, arrays_from, fit_policy, fit_policy_v, prepare


def cluster_bootstrap(diff: np.ndarray, groups: np.ndarray, B: int = 2000, seed: int = 0):
    rs = np.random.RandomState(seed)
    uq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    cnts = np.bincount(inv)
    stats = np.empty(B)
    for t in range(B):
        pick = rs.randint(0, len(uq), len(uq))
        stats[t] = sums[pick].sum() / cnts[pick].sum()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    p = 2 * min((stats <= 0).mean(), (stats >= 0).mean())
    return float(diff.mean()), float(lo), float(hi), float(max(p, 1.0 / B))


def fixed_depth_line(acc_by_depth: np.ndarray, cost_by_depth: np.ndarray, cost: float) -> np.ndarray:
    """Per-debate expected correctness of the random mix of depths k and k+1 costing `cost`."""
    mean_cost = cost_by_depth.mean(0)
    if cost <= mean_cost[0]:
        return acc_by_depth[:, 0].astype(float)
    if cost >= mean_cost[-1]:
        return acc_by_depth[:, -1].astype(float)
    k = int(np.searchsorted(mean_cost, cost) - 1)
    w = (cost - mean_cost[k]) / (mean_cost[k + 1] - mean_cost[k])
    return (1 - w) * acc_by_depth[:, k] + w * acc_by_depth[:, k + 1]


def _eval_verify(cache, tr, te, prep, scores_te, vtab, args, eps, fold, policy, o_v, o_s, ver_rows):
    """Train OSC-V on the training folds, apply it (and verify-at-stop) to the test fold."""
    from .verify import TauModel, attach_tau, build_tables, simulate_v, tau_rows
    policy_v, rep_v = fit_policy_v(cache, tr, args.task, args.max_rounds, not args.no_mid, eps,
                                   args.delta, args.risk, vtab, seed=fold, prep=prep,
                                   verify_cost=args.verify_cost, verify_mode=args.verify_mode)
    tb_te, probs_te = build_tables(cache, te, scores_te, prep["T"], vtab, args.verify_cost)
    if policy_v._tau is not None:
        tau = policy_v._tau
    else:  # the certified setting never verifies; the baseline still needs a verifier model
        tb_tr, _ = build_tables(cache, tr, prep["scores"], prep["T"], vtab, args.verify_cost)
        tau = TauModel.fit(*tau_rows(tb_tr))
    tb_te = attach_tau(tb_te, probs_te, tau)
    if ver_rows is not None:
        ver_rows.append(tau_rows(tb_te))
    cps = cache.checkpoints
    for out, pol, kappa, ell, mode in (
            (o_v, policy_v, (policy_v.verify or {}).get("kappa"), (policy_v.verify or {}).get("ell", 0.0), "band"),
            (o_s, policy, None, 0.0, "at_stop")):
        thr = np.array([pol.thr[cp] for cp in cps])
        sim = simulate_v(tb_te, thr, kappa, ell, mode=mode)
        out["correct"][te], out["cost"][te] = sim["correct"], sim["cost"]
        out["harm"][te], out["verified"][te] = sim["harm"], sim["verified"]
    o_v["settings"].append(rep_v["certified_setting"])


def _pooled_report(rows: list) -> dict:
    """tau1 / tau0 of the verifier on all test folds, overall and by latest-round agreement."""
    if not rows:
        return {}
    S = np.concatenate([r[0] for r in rows])
    right = np.concatenate([r[1] for r in rows])
    passed = np.concatenate([r[2] for r in rows])
    out = {}
    for name, m in (("all", np.ones(len(right), bool)), ("unanimous", S[:, 3] >= 0.999),
                    ("split", S[:, 3] < 0.999)):
        r1, r0 = m & right, m & ~right
        out[name] = dict(n_right=int(r1.sum()), n_wrong=int(r0.sum()),
                         tau1=float(passed[r1].mean()) if r1.any() else float("nan"),
                         tau0=float(passed[r0].mean()) if r0.any() else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser(description="Nested evaluation of OSC")
    ap.add_argument("--logs", required=True)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_rounds", type=int, default=6)
    ap.add_argument("--no_mid", action="store_true")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--risk", choices=["harm", "disagree"], default="harm")
    ap.add_argument("--verify_logs", default=None, help="osc.verify_offline outputs (glob)")
    ap.add_argument("--verify_mode", default="reasoning", choices=["reasoning", "blind"])
    ap.add_argument("--verify_cost", type=float, default=None,
                    help="cost of one verification as a share of a full debate (default: tokens)")
    ap.add_argument("--baselines", action="store_true",
                    help="also compare with the literature stopping rules in baselines/stopping.py")
    args = ap.parse_args()

    vtab = None
    if args.verify_logs:
        from .verify import VerifyTable
        vtab = VerifyTable.load(args.verify_logs, args.task, args.verify_mode)
        if not len(vtab):
            raise SystemExit(f"no {args.verify_mode} verdicts for {args.task}")

    debates = load_debates(args.logs, args.task, args.max_rounds)
    if not debates:
        raise SystemExit("no full-depth debates found")
    cps = all_checkpoints(args.max_rounds, not args.no_mid)
    full_cols = [c for c, cp in enumerate(cps) if cp.endswith("f")]
    cache = FeatureCache(debates, cps)
    n = len(debates)
    groups = cache.group

    # baselines that need no learning
    maj = np.array([[majority(r) == d.gt for r in d.rounds] for d in debates])
    cost_full = cache.cost[:, full_cols]
    cons_round = np.array([next((r for r, x in enumerate(d.rounds) if x["a"] == x["b"] == x["c"]
                                 and x["a"] is not None), args.max_rounds - 1) for d in debates])
    rows = np.arange(n)

    obs_depth = np.zeros((n, len(full_cols)), bool)
    osc = {e: dict(correct=np.zeros(n, bool), cost=np.zeros(n), harm=np.zeros(n, bool),
                   stop_round=np.zeros(n, int), a=[]) for e in args.eps}
    blank = lambda: dict(correct=np.zeros(n, bool), cost=np.zeros(n), harm=np.zeros(n, bool),
                         verified=np.zeros(n, bool), settings=[])
    osc_v = {e: blank() for e in args.eps}
    at_stop = {e: blank() for e in args.eps}
    ver_rows = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(rows, groups=groups)):
        prep = prepare(cache, tr)
        scores = {(i, cp): prep["observers"][cp].raw_scores(cache.F[i, cp])
                  if len(cache.F[i, cp]) else np.zeros(0) for i in te for cp in cps}
        arr = arrays_from(cache, te, scores, prep["T"])
        for j, eps in enumerate(args.eps):
            policy, rep = fit_policy(cache, tr, args.task, args.max_rounds, not args.no_mid,
                                     eps, args.delta, args.risk, seed=fold, prep=prep)
            thr = np.array([policy.thr[cp] for cp in cps])
            hit = arr.margin >= thr[None, :]
            stop = hit.argmax(1)
            r = np.arange(len(te))
            o = osc[eps]
            o["correct"][te] = arr.correct[r, stop]
            o["cost"][te] = arr.cost[r, stop]
            o["harm"][te] = arr.correct[:, -1] & ~arr.correct[r, stop]
            o["stop_round"][te] = [int(cps[s][1:-1]) for s in stop]
            o["a"].append(rep["certified_a"])
            if j == 0:
                obs_depth[te] = arr.correct[:, full_cols]
            if vtab is not None:
                _eval_verify(cache, tr, te, prep, scores, vtab, args, eps, fold, policy,
                             osc_v[eps], at_stop[eps], ver_rows if j == 0 else None)
        print(f"  fold {fold} done", flush=True)

    res = dict(task=args.task, n_debates=n, n_questions=int(len(np.unique(groups))),
               mid_round=not args.no_mid, risk=args.risk, delta=args.delta)
    res["fixed_depth"] = [dict(rounds=k + 1, cost=float(cost_full[:, k].mean()),
                               acc_majority=float(maj[:, k].mean()), acc_observer=float(obs_depth[:, k].mean()))
                          for k in range(len(full_cols))]
    res["consensus"] = dict(cost=float(cost_full[rows, cons_round].mean()),
                            acc_majority=float(maj[rows, cons_round].mean()),
                            acc_observer=float(obs_depth[rows, cons_round].mean()))
    res["osc"] = []
    for eps, o in osc.items():
        cost = float(o["cost"].mean())
        line_obs = fixed_depth_line(obs_depth, cost_full, cost)
        line_maj = fixed_depth_line(maj, cost_full, cost)
        d_obs = cluster_bootstrap(o["correct"] - line_obs, groups)
        d_maj = cluster_bootstrap(o["correct"] - line_maj, groups)
        res["osc"].append(dict(
            eps=eps, certified_a_per_fold=o["a"], acc=float(o["correct"].mean()), cost=cost,
            harm=float(o["harm"].mean()), mean_rounds=float(o["stop_round"].mean() + 1),
            vs_fixed_depth_same_cost_observer=dict(zip(["delta", "lo", "hi", "p"], d_obs)),
            vs_fixed_depth_same_cost_majority=dict(zip(["delta", "lo", "hi", "p"], d_maj))))

    if vtab is not None:
        res["verifier_test_folds"] = _pooled_report(ver_rows)
        for name, table in (("osc_verify", osc_v), ("verify_at_stop", at_stop)):
            res[name] = []
            for eps, o in table.items():
                cost = float(o["cost"].mean())
                d_obs = cluster_bootstrap(o["correct"] - fixed_depth_line(obs_depth, cost_full, cost), groups)
                d_osc = cluster_bootstrap(o["correct"].astype(float) - osc[eps]["correct"], groups)
                res[name].append(dict(
                    eps=eps, acc=float(o["correct"].mean()), cost=cost, harm=float(o["harm"].mean()),
                    verify_rate=float(o["verified"].mean()), settings_per_fold=o["settings"],
                    vs_fixed_depth_same_cost_observer=dict(zip(["delta", "lo", "hi", "p"], d_obs)),
                    vs_plain_osc_acc=dict(zip(["delta", "lo", "hi", "p"], d_osc))))

    if args.baselines:
        from baselines.stopping import curve, hull_line, replay, upper_hull
        systems = [("OSC", e, osc[e]) for e in args.eps]
        if vtab is not None:
            systems += [("OSC-V", e, osc_v[e]) for e in args.eps]
        res["baselines"] = {}
        for name, bl in replay(debates, cost_full).items():
            rows_ = []
            mc = bl["cost"].mean(1)
            h = upper_hull(mc, bl["correct"].mean(1))
            for label, eps, o in systems:
                c = float(o["cost"].mean())
                line = hull_line(bl["correct"], bl["cost"], c)
                d = cluster_bootstrap(o["correct"].astype(float) - line, groups)
                # below the rule's cheapest setting the rule is charged its cheapest cost, i.e. it
                # is allowed to spend MORE than the system it is compared with
                b_cost = float(min(max(c, mc[h[0]]), mc[h[-1]]))
                rows_.append(dict(system=label, eps=eps, cost=c, baseline_cost=b_cost,
                                  baseline_acc_same_cost=float(line.mean()),
                                  **dict(zip(["delta", "lo", "hi", "p"], d))))
            res["baselines"][name] = dict(curve=curve(bl), vs=rows_)

    print(f"\n{args.task}: {n} debates, {res['n_questions']} questions (all numbers out-of-fold)")
    print("  fixed depth   rounds:   " + "  ".join(f"{x['rounds']:>6d}" for x in res["fixed_depth"]))
    print("                cost:     " + "  ".join(f"{x['cost']:6.1%}" for x in res["fixed_depth"]))
    print("                majority: " + "  ".join(f"{x['acc_majority']:6.3f}" for x in res["fixed_depth"]))
    print("                observer: " + "  ".join(f"{x['acc_observer']:6.3f}" for x in res["fixed_depth"]))
    c = res["consensus"]
    print(f"  consensus stop: cost {c['cost']:.1%}  acc(majority) {c['acc_majority']:.3f}  acc(observer) {c['acc_observer']:.3f}")
    for o in res["osc"]:
        v, m = o["vs_fixed_depth_same_cost_observer"], o["vs_fixed_depth_same_cost_majority"]
        print(f"  OSC eps={o['eps']:<6} acc {o['acc']:.3f}  cost {o['cost']:.1%}  harm {o['harm']:.3%}"
              f" | vs fixed depth same cost: observer {v['delta']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}]"
              f"  majority {m['delta']:+.3f} [{m['lo']:+.3f},{m['hi']:+.3f}]")
    if vtab is not None:
        vr = res["verifier_test_folds"]
        print("  verifier (test folds): " + "  ".join(
            f"{k}: tau1 {v['tau1']:.2f} tau0 {v['tau0']:.2f} (n={v['n_right']}/{v['n_wrong']})"
            for k, v in vr.items()))
        for name in ("osc_verify", "verify_at_stop"):
            for o in res[name]:
                v, d = o["vs_fixed_depth_same_cost_observer"], o["vs_plain_osc_acc"]
                print(f"  {name:<14} eps={o['eps']:<6} acc {o['acc']:.3f}  cost {o['cost']:.1%}  "
                      f"harm {o['harm']:.3%}  verified {o['verify_rate']:.1%} | vs fixed depth "
                      f"{v['delta']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}]  vs OSC acc "
                      f"{d['delta']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]")
    if args.baselines:
        print("  baselines at the same cost (knob mix chosen on these debates, favours the baseline):")
        for name, b in res["baselines"].items():
            for r in b["vs"]:
                more = " (baseline spends more)" if r["baseline_cost"] > r["cost"] + 1e-9 else ""
                print(f"    {r['system']:<5} eps={r['eps']:<6} @ {r['cost']:.1%} vs {name:<11} "
                      f"acc {r['baseline_acc_same_cost']:.3f} @ {r['baseline_cost']:.1%} | delta "
                      f"{r['delta']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]{more}")
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("saved", args.out)


if __name__ == "__main__":
    main()
