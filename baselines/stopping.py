"""Stopping rules from the literature, replayed on full-depth debate logs (no model calls).

    python -m baselines.stopping --logs "results/logs_osc/**/debate_full_*.jsonl" --task gsm8k

Every rule looks only at what is visible after each full round (the three answers and their
confidences), decides whether to stop, and answers with a vote. Replay is exact for the same
reason as for OSC: stopping early does not change the rounds before the stop. Cost is OSC's
cost model (share of the full-debate tokens spent), so a rule and OSC are compared at the same
cost (osc.evaluate --baselines).

  fixed_depth(k)        run k rounds, answer with the majority of round k
  consensus(p)          stop once all three agents agree for p consecutive rounds
  ac_beta(C)            Adaptive-Consistency, Beta stopping (Aggarwal et al., 2023): every answer
                        given so far is a vote; with v1, v2 the votes of the two leading answers,
                        stop when P(q > 1/2) >= C for q ~ Beta(v1 + 1, v2 + 1)
  sprt(alpha)           Wald sequential probability ratio test: each vote is Bernoulli "backs the
                        current leader", H1: q = 0.8 against H0: q = 0.5, beta = 0.1; stop when
                        log LR >= log((1 - beta) / alpha)
  stability(k)          stop when the round's multiset of answers has stayed the same for k
                        consecutive rounds (distribution-stability rule in the spirit of Hu et
                        al., 2025, "stop when the answer distribution stops moving")

ac_beta and sprt count every answer of every round as a fresh vote, i.e. they assume votes are
independent; in debate most later votes repeat earlier ones, which is exactly what OSC's
observer corrects for.

Each rule has one knob; the whole knob grid is replayed. For a comparison at cost c the rule
gets the best random mix of two knob settings on the upper hull of its (cost, accuracy) points,
chosen on the evaluation data itself — an optimistic choice for the baseline, so the comparison
is conservative for OSC.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.stats import beta as beta_dist

AG = ("a", "b", "c")
CONF = {"a": "ca", "b": "cb", "c": "cc"}

GRIDS = {
    "fixed_depth": (1, 2, 3, 4, 5, 6),
    "consensus": (1, 2, 3),
    "ac_beta": (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
    "sprt": (0.3, 0.2, 0.1, 0.05, 0.01, 0.001),
    "stability": (1, 2, 3),
}


# ----------------------------------------------------------------------------- votes
def _vote(counts: dict, conf: dict):
    """Leading answer: most votes, ties broken by the highest confidence behind an answer."""
    if not counts:
        return None
    top = max(counts.values())
    tied = [a for a, v in counts.items() if v == top]
    return tied[0] if len(tied) == 1 else max(tied, key=lambda a: conf.get(a, 0.0))


def _round_counts(rnd: dict):
    counts, conf = {}, {}
    for k in AG:
        a = rnd.get(k)
        if a is None:
            continue
        counts[a] = counts.get(a, 0) + 1
        conf[a] = max(conf.get(a, 0.0), rnd.get(CONF[k], 0.0) or 0.0)
    return counts, conf


def _cumulative(rounds: list, t: int):
    counts, conf = {}, {}
    for rnd in rounds[: t + 1]:
        c, f = _round_counts(rnd)
        for a, v in c.items():
            counts[a] = counts.get(a, 0) + v
            conf[a] = max(conf.get(a, 0.0), f[a])
    return counts, conf


# ----------------------------------------------------------------------------- rules
def rule_fixed_depth(rounds: list, k: int):
    t = min(k, len(rounds)) - 1
    return t, _vote(*_round_counts(rounds[t]))


def rule_consensus(rounds: list, patience: int):
    run = 0
    for t, rnd in enumerate(rounds):
        said = [rnd.get(k) for k in AG]
        run = run + 1 if (None not in said and len(set(said)) == 1) else 0
        if run >= patience:
            return t, said[0]
    t = len(rounds) - 1
    return t, _vote(*_round_counts(rounds[t]))


def rule_ac_beta(rounds: list, c_level: float):
    for t in range(len(rounds)):
        counts, conf = _cumulative(rounds, t)
        v = sorted(counts.values(), reverse=True) + [0]
        if beta_dist.sf(0.5, v[0] + 1, v[1] + 1) >= c_level:
            return t, _vote(counts, conf)
    t = len(rounds) - 1
    return t, _vote(*_cumulative(rounds, t))


def rule_sprt(rounds: list, alpha: float, p0: float = 0.5, p1: float = 0.8, beta: float = 0.1):
    bound = math.log((1 - beta) / alpha)
    up, down = math.log(p1 / p0), math.log((1 - p1) / (1 - p0))
    for t in range(len(rounds)):
        counts, conf = _cumulative(rounds, t)
        n, lead = sum(counts.values()), max(counts.values())
        if lead * up + (n - lead) * down >= bound:
            return t, _vote(counts, conf)
    t = len(rounds) - 1
    return t, _vote(*_cumulative(rounds, t))


def rule_stability(rounds: list, k: int):
    run = 0
    for t in range(1, len(rounds)):
        same = sorted(map(str, _round_counts(rounds[t])[0].items())) == \
            sorted(map(str, _round_counts(rounds[t - 1])[0].items()))
        run = run + 1 if same else 0
        if run >= k:
            return t, _vote(*_round_counts(rounds[t]))
    t = len(rounds) - 1
    return t, _vote(*_round_counts(rounds[t]))


RULES = {
    "fixed_depth": rule_fixed_depth,
    "consensus": rule_consensus,
    "ac_beta": rule_ac_beta,
    "sprt": rule_sprt,
    "stability": rule_stability,
}


def replay(debates: list, cost_full: np.ndarray, rules=None) -> dict:
    """For every rule and knob: per-debate correctness and cost.
    cost_full[i, t] = share of debate i's full cost spent after round t."""
    out = {}
    for name in rules or RULES:
        fn, grid = RULES[name], GRIDS[name]
        correct = np.zeros((len(grid), len(debates)), bool)
        cost = np.zeros((len(grid), len(debates)))
        stop = np.zeros((len(grid), len(debates)), int)
        for g, knob in enumerate(grid):
            for i, d in enumerate(debates):
                t, ans = fn(d.rounds, knob)
                correct[g, i] = ans is not None and ans == d.gt
                cost[g, i] = cost_full[i, t]
                stop[g, i] = t
        out[name] = dict(knobs=list(grid), correct=correct, cost=cost, stop=stop)
    return out


# ----------------------------------------------------------------------------- matched cost
def upper_hull(costs: np.ndarray, accs: np.ndarray) -> list[int]:
    """Indices of the points on the upper concave hull of (cost, accuracy), by increasing cost.
    Any point on a segment between two of them is reachable by a random mix of the two."""
    order = sorted(range(len(costs)), key=lambda i: (costs[i], -accs[i]))
    hull = []
    for i in order:
        if hull and accs[i] <= accs[hull[-1]]:
            continue  # costs more, not more accurate
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # drop b if it lies on or below the chord a -> i
            if (accs[b] - accs[a]) * (costs[i] - costs[a]) <= (accs[i] - accs[a]) * (costs[b] - costs[a]):
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


def hull_line(correct: np.ndarray, cost: np.ndarray, target: float) -> np.ndarray:
    """Per-debate expected correctness of the rule's best random mix costing `target` on average."""
    mc, ma = cost.mean(1), correct.mean(1)
    h = upper_hull(mc, ma)
    if target <= mc[h[0]]:
        return correct[h[0]].astype(float)
    if target >= mc[h[-1]]:
        return correct[h[-1]].astype(float)
    j = next(j for j in range(1, len(h)) if mc[h[j]] >= target)
    lo, hi = h[j - 1], h[j]
    w = (target - mc[lo]) / (mc[hi] - mc[lo])
    return (1 - w) * correct[lo] + w * correct[hi]


def curve(res: dict) -> list[dict]:
    return [dict(knob=k, cost=float(res["cost"][g].mean()), acc=float(res["correct"][g].mean()),
                 mean_rounds=float(res["stop"][g].mean() + 1)) for g, k in enumerate(res["knobs"])]


def main():
    from osc.data import all_checkpoints, checkpoint_costs, load_debates
    ap = argparse.ArgumentParser(description="Replay literature stopping rules on full-depth logs")
    ap.add_argument("--logs", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--max_rounds", type=int, default=6)
    args = ap.parse_args()
    debates = load_debates(args.logs, args.task, args.max_rounds)
    if not debates:
        raise SystemExit("no full-depth debates found")
    cps = all_checkpoints(args.max_rounds, False)
    cost_full = np.array([checkpoint_costs(d, cps) for d in debates])
    print(f"{args.task}: {len(debates)} debates")
    for name, res in replay(debates, cost_full).items():
        print(f"  {name}")
        for p in curve(res):
            print(f"    knob {p['knob']!s:<6} cost {p['cost']:6.1%}  acc {p['acc']:.3f}  "
                  f"rounds {p['mean_rounds']:.2f}")


if __name__ == "__main__":
    main()
