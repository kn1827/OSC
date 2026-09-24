"""Tests for the OSC package and the orchestrator changes.

Run with pytest, or directly:  python tests/test_osc.py
The orchestrator tests replace the LLM with a scripted fake, so no model server is needed.
"""
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from osc.certify import Arrays, Shape, binom_cdf, certify, simulate, thresholds
from osc.features import (HOLD, COPY, NOVEL, _role, candidate_features, canon, checkpoint_order,
                          prefix_at, solvers_only)
from osc.observer import Observer
from osc.policy import OSCPolicy


def rnd(a, b, c, ca=0.9, cb=0.9, cc=0.9, va=None, vb=None):
    return dict(a=a, b=b, c=c, ca=ca, cb=cb, cc=cc, va=va, vca=0.8, vb=vb, vcb=0.8)


# ----------------------------------------------------------------------------- features
def test_canon_unifies_numbers():
    assert canon("18.0") == canon("18") == "18"
    assert canon(" B ") == "b"
    assert canon(None) is None and canon("") is None


def test_roles():
    prefix = [rnd("1", "2", "3"), rnd("1", "1", "4")]
    assert _role(prefix, 1, "a") == HOLD      # kept its own answer
    assert _role(prefix, 1, "b") == COPY      # took solver A's answer
    assert _role(prefix, 1, "c") == NOVEL     # brand new answer


def test_repeats_are_not_counted_as_fresh_votes():
    prefix = [rnd("1", "1", "2")] + [rnd("1", "1", "2")] * 5
    cands, F = candidate_features(prefix)
    one = cands.index("1")
    assert math.isclose(F[one, 0], 1.8)             # first-round confidence 0.9 + 0.9
    assert F[one, 1] + F[one, 2] > 0 and F[one, 3] == F[one, 5] == 0   # later slots are HOLD


def test_mid_round_prefix_has_no_critic():
    rounds = [rnd("1", "2", "1", va="agree", vb="disagree"), rnd("1", "1", "1")]
    p = prefix_at(rounds, "r1s")
    assert len(p) == 2 and p[-1]["c"] is None and p[-1]["va"] is None
    assert prefix_at(rounds, "r0f") == rounds[:1]
    assert solvers_only(rounds[0])["a"] == "1"


def test_checkpoint_order():
    assert checkpoint_order(2, True) == ["r0s", "r0f", "r1s", "r1f"]
    assert checkpoint_order(2, False) == ["r0f", "r1f"]


# ----------------------------------------------------------------------------- certify
def test_binomial_matches_worked_example():
    assert abs(binom_cdf(3, 750, 0.01) - 0.0583) < 5e-4
    assert abs(binom_cdf(4, 750, 0.01) - 0.1308) < 5e-4
    assert binom_cdf(10, 10, 0.3) == 1.0


def test_thresholds_fall_per_round_and_last_always_stops():
    cps = checkpoint_order(3, True)
    thr = thresholds(cps, 0.8, Shape(b=0.1, mid_extra=0.05))
    assert np.allclose(thr[:5], [0.85, 0.8, 0.75, 0.7, 0.65])
    assert thr[-1] == -np.inf
    assert np.isinf(thresholds(cps, 0.8, Shape(0.0, None))[0])   # mid-round disabled


def test_certify_refuses_when_too_few_debates():
    cps = ["r0f", "r1f"]
    n = 20  # 0.99^20 = 0.82 > 0.1: even "never stop early" cannot be certified at eps=1%
    arr = Arrays(np.ones((n, 2)), np.ones((n, 2), bool), np.full((n, 2), "x", object),
                 np.tile([0.5, 1.0], (n, 1)), np.arange(n).astype(str))
    assert certify(arr, cps, Shape(), 0.01, 0.1, "harm")["a"] is None


def test_certify_accepts_safe_early_stop():
    cps = ["r0f", "r1f"]
    n = 400
    arr = Arrays(np.ones((n, 2)), np.ones((n, 2), bool), np.full((n, 2), "x", object),
                 np.tile([0.5, 1.0], (n, 1)), np.arange(n).astype(str))
    res = certify(arr, cps, Shape(), 0.01, 0.1, "harm")
    assert res["a"] is not None and res["a"] <= 1.0
    sim = simulate(arr, thresholds(cps, res["a"], Shape()))
    assert sim["cost"].mean() == 0.5 and not sim["harm"].any()


# ----------------------------------------------------------------------------- observer/policy
def test_observer_learns_that_first_round_support_matters():
    rs = np.random.RandomState(0)
    items = []
    for _ in range(300):
        F = rs.rand(3, 15)
        y = int(np.argmax(F[:, 0]))
        items.append((F, y))
    obs = Observer().fit(items)
    hits = sum(int(np.argmax(obs.probs(F))) == y for F, y in items)
    assert hits / len(items) > 0.9


def make_policy(a, mid=True, max_rounds=2):
    cps = checkpoint_order(max_rounds, mid)
    return OSCPolicy("gsm8k", max_rounds, mid, {cp: Observer() for cp in cps}, a, Shape(0.0, 0.0 if mid else None))


def test_policy_roundtrip_and_decision():
    pol = make_policy(0.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "p.json")
        pol.save(path)
        back = OSCPolicy.load(path)
    dec = back.assess([rnd("1", "2", None)], "r0s")
    assert dec["stop"] and dec["top"] in ("1", "2")
    never = make_policy(None)
    assert not never.assess([rnd("1", "1", "1")], "r0f")["stop"]
    assert never.assess([rnd("1", "1", "1")] * 2, "r1f", force_stop=True)["stop"]


# ----------------------------------------------------------------------------- orchestrator
def _fake_llm():
    """Solvers answer 18, the critic's own answer is 18, verdicts agree; counts every call."""
    calls = []

    def call_llm(self, prompt):
        calls.append((self.role, "DEBATE SO FAR" in prompt))
        if self.role == "critic" and "SOLVER's solution" in prompt:
            body = {"intent": "math check", "reasoning": ["checked 9x2=18"], "action": "agree",
                    "confidence": 0.9, "content": "fine"}
        else:
            body = {"intent": "arithmetic", "reasoning": ["16-3-4=9", "9x2=18"], "action": "18",
                    "confidence": 0.9, "content": "18"}
        return json.dumps(body), 10

    return call_llm, calls


def _orchestrator(**kw):
    from src.agents.base_agent import BaseAgent
    from src.orchestrator import MADOrchestrator
    fake, calls = _fake_llm()
    BaseAgent.call_llm = fake
    cfg = "config/model_config_solver_qwen.yaml"
    orch = MADOrchestrator("gsm8k", cfg, cfg, [cfg], max_rounds=kw.pop("max_rounds", 2), **kw)
    return orch, calls


def _in_repo(fn):
    def wrapper():
        cwd = os.getcwd()
        os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        try:
            fn()
        finally:
            os.chdir(cwd)
    wrapper.__name__ = fn.__name__
    return wrapper


@_in_repo
def test_live_critic_resolves_every_round_and_tokens_are_logged():
    orch, calls = _orchestrator(stop_policy="none", critic_mode="live")
    rec = orch.run(1, "Janet has 16 eggs...", "18")
    assert rec["rounds_used"] == 2 and rec["critic_mode"] == "live"
    assert any(role == "critic" and saw for role, saw in calls)   # round 1 saw the debate
    for r in rec["rounds"]:
        assert set(r["tokens"]) == {"solver_a", "solver_b", "critic_independent", "critic_vs_a", "critic_vs_b"}
        assert r["tokens"]["critic_independent"] > 0
    assert rec["total_tokens"] == sum(sum(r["tokens"].values()) for r in rec["rounds"])


@_in_repo
def test_frozen_critic_matches_old_behaviour():
    orch, calls = _orchestrator(stop_policy="none", critic_mode="frozen")
    rec = orch.run(1, "q", "18")
    assert rec["rounds"][1]["tokens"]["critic_independent"] == 0
    assert not any(saw for _, saw in calls)


@_in_repo
def test_osc_stops_mid_round_and_audit_runs_full():
    pol = make_policy(0.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "p.json")
        pol.save(path)
        orch, _ = _orchestrator(stop_policy="osc", critic_mode="live", osc_policy_path=path)
        rec = orch.run(1, "q", "18")
        assert rec["osc"]["stopped_at"] == "r0s" and rec["rounds"][0]["partial"]
        assert rec["rounds_used"] == 0 and rec["final_answer"] == "18" and rec["final_correct"]
        orch, _ = _orchestrator(stop_policy="osc", critic_mode="live", osc_policy_path=path,
                                audit_rate=1.0)
        rec = orch.run(1, "q", "18")
        assert rec["osc"]["audit"] and rec["rounds_used"] == 2
        assert rec["osc"]["would_stop_at"] == "r0s" and rec["osc"]["stopped_at"] == "r1f"


@_in_repo
def test_consensus_policy_still_stops_on_agreement():
    orch, _ = _orchestrator(stop_policy="consensus", critic_mode="live", max_rounds=3)
    rec = orch.run(1, "q", "18")
    assert rec["rounds_used"] == 1 and rec["final_answer_source"] == "majority"


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if failed else 0)
