"""Upgrade N4: re-check the guarantee on live traffic, no correct answers needed.

    python -m osc.monitor --logs "results/logs_osc_live/**/debate_full_*.jsonl" \
        --policy results/osc/policy_gsm8k.json --eps 0.02 --delta 0.10

A random share of questions (MAD.py --audit_rate) runs every round even under OSC, and the log
keeps the answer OSC would have stopped with.  Count k audited questions where that answer
differs from the full-debate answer, out of n audited questions, and compute
    P = sum_{i=0..k} C(n, i) * eps^i * (1 - eps)^(n - i).
P <= delta: the disagreement rate is still below eps.  Otherwise the thresholds are raised:
the audited debates are replayed with larger a until the test passes again.
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np

from .certify import A_STEP, binom_cdf
from .features import canon, prefix_at, slot_view
from .policy import OSCPolicy


def load_audits(pattern: str, task: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(pattern, recursive=True)):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("task") == task and rec.get("osc", {}).get("audit"):
                out.append(rec)
    return out


def replay_disagreements(policy: OSCPolicy, audits: list[dict], a: float) -> int:
    """How many audited debates would stop on a different answer than the full debate, at level a."""
    trial = OSCPolicy(policy.task, policy.max_rounds, policy.mid_round, policy.observers, a,
                      policy.shape)
    k = 0
    for rec in audits:
        rounds = [slot_view(r) for r in rec["rounds"] if not r.get("partial")]
        if len(rounds) < policy.max_rounds:
            continue
        full = trial.assess(prefix_at(rounds, trial.checkpoints[-1]), trial.checkpoints[-1], True)["top"]
        for cp in trial.checkpoints:
            dec = trial.assess(prefix_at(rounds, cp), cp, cp == trial.checkpoints[-1])
            if dec["stop"]:
                k += dec["top"] != full
                break
    return k


def main():
    ap = argparse.ArgumentParser(description="Re-check the OSC guarantee on audited live debates")
    ap.add_argument("--logs", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--eps", type=float, default=0.02, help="largest acceptable disagreement rate")
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--update", default=None, help="write a policy with raised thresholds here if needed")
    args = ap.parse_args()

    policy = OSCPolicy.load(args.policy)
    audits = load_audits(args.logs, policy.task)
    n = len(audits)
    if n == 0:
        raise SystemExit("no audited debates yet")
    k = sum(1 for r in audits
            if r["osc"]["would_stop_answer"] is not None
            and canon(r["osc"]["would_stop_answer"]) != canon(r["final_answer"]))
    p = binom_cdf(k, n, args.eps)
    print(f"{policy.task}: {n} audited questions, {k} would have stopped on a different answer "
          f"({k / n:.2%}); P = {p:.4f} (pass if <= {args.delta})")
    if n < np.log(args.delta) / np.log(1 - args.eps):
        print(f"note: with n={n} even k=0 cannot pass; collect at least "
              f"{int(np.ceil(np.log(args.delta) / np.log(1 - args.eps)))} audited questions")
    if p <= args.delta:
        print("OK: guarantee still holds")
        return
    if policy.a is None:
        print("policy never stops early already")
        return
    a = policy.a
    while a < 1.01 + policy.shape.b * policy.max_rounds + 1:
        a = round(a + A_STEP, 4)
        k_a = replay_disagreements(policy, audits, a)
        if binom_cdf(k_a, n, args.eps) <= args.delta:
            print(f"RAISE: a {policy.a} -> {a} brings disagreements to {k_a}/{n}")
            if args.update:
                new = OSCPolicy(policy.task, policy.max_rounds, policy.mid_round, policy.observers,
                                a, policy.shape, dict(policy.meta, raised_by_monitor=True))
                new.save(args.update)
                print("saved", args.update)
            return
    print("RAISE: no level passes; stop using early stopping for this task and re-train")


if __name__ == "__main__":
    main()
