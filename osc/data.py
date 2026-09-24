"""Load full-depth debates from debate_full_*.jsonl and compute the cost of each checkpoint.

Only debates that ran every round can be used for training: stopping early is judged against
what the rest of the debate would have produced.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field

from .features import canon, checkpoint_order, parse_checkpoint, slot_view

SOLVER_CALLS = ("solver_a", "solver_b")
CRITIC_CALLS = ("critic_independent", "critic_vs_a", "critic_vs_b")


@dataclass
class Debate:
    task: str
    seed: int
    sid: int
    gt: str | None
    rounds: list[dict]                 # slot view, one per round
    round_tokens: list[dict] = field(default_factory=list)
    critic_mode: str = "frozen"

    @property
    def group(self) -> str:
        return f"{self.task}|{self.sid}"


def _unit_tokens(round_id: int, critic_mode: str) -> dict:
    """Fallback when a log has no per-call token counts: every call costs 1."""
    t = {"solver_a": 1, "solver_b": 1, "critic_vs_a": 1, "critic_vs_b": 1}
    t["critic_independent"] = 1 if (round_id == 0 or critic_mode == "live") else 0
    return t


def checkpoint_costs(d: Debate, checkpoints: list[str]) -> list[float]:
    """Share of the full-debate cost spent when each checkpoint is reached."""
    per_round = []
    for r, tok in enumerate(d.round_tokens):
        tok = tok or _unit_tokens(r, d.critic_mode)
        s = sum(float(tok.get(k, 0) or 0) for k in SOLVER_CALLS)
        c = sum(float(tok.get(k, 0) or 0) for k in CRITIC_CALLS)
        per_round.append((s, c))
    total = sum(s + c for s, c in per_round) or 1.0
    out = []
    for cp in checkpoints:
        r, phase = parse_checkpoint(cp)
        spent = sum(s + c for s, c in per_round[:r]) + per_round[r][0]
        if phase == "f":
            spent += per_round[r][1]
        out.append(spent / total)
    return out


def load_debates(pattern: str, task: str | None = None, max_rounds: int = 6) -> list[Debate]:
    out = []
    for f in sorted(glob.glob(pattern, recursive=True)):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if task and rec.get("task") != task:
                continue
            rounds = [r for r in rec["rounds"] if not r.get("partial")]
            if len(rounds) < max_rounds:
                continue  # stopped early: no record of what the full debate would say
            rounds = rounds[:max_rounds]
            mode = rec.get("critic_mode", "frozen")
            out.append(Debate(
                task=rec["task"], seed=int(rec.get("seed", 0)), sid=int(rec["sample_id"]),
                gt=canon(rec.get("ground_truth")),
                rounds=[slot_view(r) for r in rounds],
                round_tokens=[r.get("tokens") or _unit_tokens(i, mode) for i, r in enumerate(rounds)],
                critic_mode=mode,
            ))
    return out


def all_checkpoints(max_rounds: int, mid_round: bool) -> list[str]:
    return checkpoint_order(max_rounds, mid_round)
