"""Candidate-answer features computed from a debate prefix.

A debate prefix is a list of rounds in "slot view":
    {"a": answer of solver A, "b": answer of solver B, "c": answer of the critic,
     "ca"/"cb"/"cc": their confidences in [0, 1],
     "va"/"vb": critic verdict on solver A / B ("agree", "disagree" or None),
     "vca"/"vcb": confidence of those verdicts}
Any slot may be None: a mid-round checkpoint has the solvers' answers but no critic yet.

Every candidate answer gets 15 numbers (see FEATURE_NAMES). The same code is used when
training on logs and when the orchestrator decides online, so the two cannot drift apart.
"""
from __future__ import annotations

import numpy as np

AG = ("a", "b", "c")
CONF = {"a": "ca", "b": "cb", "c": "cc"}
VERDICT = {"a": ("va", "vca"), "b": ("vb", "vcb")}

FEATURE_NAMES = [
    "first_round_confidence",
    "hold_early", "hold_late",
    "copy_early", "copy_late",
    "novel_early", "novel_late",
    "mass_solver_a", "mass_solver_b", "mass_critic",
    "critic_verdict_mass",
    "share_of_agents_backing",
    "first_appearance",
    "held_in_latest_round",
    "rounds_as_majority",
]
NF = len(FEATURE_NAMES)

HOLD, COPY, NOVEL = 1, 2, 3


def canon(answer) -> str | None:
    """Lower-case, strip, and write numbers one way ("18.0" and "18" become "18")."""
    if answer is None:
        return None
    s = str(answer).strip().lower()
    if not s:
        return None
    try:
        x = float(s)
    except ValueError:
        return s
    if x != x or x in (float("inf"), float("-inf")):
        return s
    return str(int(x)) if x == int(x) else repr(round(x, 6))


def _conf(x) -> float:
    try:
        return min(max(float(x), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


def slot_view(round_record: dict) -> dict:
    """Convert one orchestrator round record (as written to debate_full_*.jsonl)."""
    ag = round_record.get("agents", {})
    A, B, C = ag.get("solver_a"), ag.get("solver_b"), ag.get("critic")
    return {
        "a": canon(A["normalized_answer"]) if A else None,
        "b": canon(B["normalized_answer"]) if B else None,
        "c": canon(C["normalized_answer"]) if C else None,
        "ca": _conf(A["confidence"]) if A else 0.0,
        "cb": _conf(B["confidence"]) if B else 0.0,
        "cc": _conf(C["confidence"]) if C else 0.0,
        "va": C.get("verdict_vs_solver_a") if C else None,
        "vca": _conf(C.get("verdict_vs_solver_a_confidence")) if C else 0.0,
        "vb": C.get("verdict_vs_solver_b") if C else None,
        "vcb": _conf(C.get("verdict_vs_solver_b_confidence")) if C else 0.0,
    }


def solvers_only(sv: dict) -> dict:
    """The same round as seen right after the two solvers spoke, before the critic."""
    out = dict(sv)
    out.update(c=None, cc=0.0, va=None, vca=0.0, vb=None, vcb=0.0)
    return out


def verdict_sign(v) -> float:
    if v is None:
        return 0.0
    v = str(v).lower()
    if "disagree" in v:
        return -1.0
    return 1.0 if "agree" in v else 0.0


def majority(rnd: dict) -> str | None:
    """Majority answer of one round; ties broken by the highest confidence."""
    votes, best = {}, {}
    for k in AG:
        a = rnd[k]
        if a is None:
            continue
        votes[a] = votes.get(a, 0) + 1
        best[a] = max(best.get(a, 0.0), rnd[CONF[k]])
    if not votes:
        return None
    top = max(votes.values())
    tied = [a for a, v in votes.items() if v == top]
    return tied[0] if len(tied) == 1 else max(tied, key=lambda a: best[a])


def _role(prefix: list[dict], t: int, k: str) -> int:
    """0 = first round, HOLD = kept own answer, COPY = took another agent's, NOVEL = new."""
    if t == 0:
        return 0
    prev = prefix[t - 1]
    a = prefix[t][k]
    if prev[k] is not None and a == prev[k]:
        return HOLD
    if a in {prev[o] for o in AG if o != k and prev[o] is not None}:
        return COPY
    return NOVEL


def candidate_features(prefix: list[dict]) -> tuple[list[str], np.ndarray]:
    """Return (candidate answers, matrix with one row of 15 numbers per candidate)."""
    cands = sorted({r[k] for r in prefix for k in AG if r[k] is not None})
    F = np.zeros((len(cands), NF))
    if not cands:
        return cands, F
    idx = {c: i for i, c in enumerate(cands)}
    n_rounds = len(prefix)
    half = max(n_rounds / 2.0, 1.0)
    first_seen: dict[str, int] = {}
    majority_rounds: dict[str, int] = {}
    backers: dict[str, set] = {c: set() for c in cands}
    for t, rnd in enumerate(prefix):
        m = majority(rnd)
        if m is not None:
            majority_rounds[m] = majority_rounds.get(m, 0) + 1
        late = 1 if t >= half else 0
        for ai, k in enumerate(AG):
            a = rnd[k]
            if a is None:
                continue
            j = idx[a]
            c = rnd[CONF[k]]
            role = _role(prefix, t, k)
            if role == 0:
                F[j, 0] += c
            else:
                F[j, 1 + (role - 1) * 2 + late] += c
            F[j, 7 + ai] += c
            if k in VERDICT:
                vk, vc = VERDICT[k]
                F[j, 10] += verdict_sign(rnd[vk]) * rnd[vc]
            else:
                F[j, 10] += c
            first_seen.setdefault(a, t)
            backers[a].add(k)
    latest = {prefix[-1][k] for k in AG if prefix[-1][k] is not None}
    for a, j in idx.items():
        F[j, 11] = len(backers[a]) / 3.0
        F[j, 12] = first_seen.get(a, 0) / n_rounds
        F[j, 13] = 1.0 if a in latest else 0.0
        F[j, 14] = majority_rounds.get(a, 0) / n_rounds
    return cands, F


# ----------------------------------------------------------------------------- checkpoints
def checkpoint_id(round_id: int, phase: str) -> str:
    """phase "s" = after both solvers of this round, "f" = after the whole round."""
    return f"r{round_id}{phase}"


def checkpoint_order(max_rounds: int, mid_round: bool) -> list[str]:
    out = []
    for r in range(max_rounds):
        if mid_round:
            out.append(checkpoint_id(r, "s"))
        out.append(checkpoint_id(r, "f"))
    return out


def parse_checkpoint(cp: str) -> tuple[int, str]:
    return int(cp[1:-1]), cp[-1]


def prefix_at(rounds: list[dict], cp: str) -> list[dict]:
    """Slot-view prefix visible at checkpoint cp, given the full list of slot-view rounds."""
    r, phase = parse_checkpoint(cp)
    if phase == "f":
        return rounds[: r + 1]
    return rounds[:r] + [solvers_only(rounds[r])]
