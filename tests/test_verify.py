"""Tests for the VERIFY action, the verifier agent, the rule checks and the new benchmarks.

Run with pytest, or directly:  python tests/test_verify.py
LLM calls are replaced by scripted fakes, so no model server is needed.
"""
import json
import math
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np

from osc.certify import Shape, min_questions, simulate, smallest_eps, thresholds, Arrays
from osc.verify import (TauModel, VTables, attach_tau, certify_v, log_lr, simulate_v,
                        verify_band_upper)


def _in_repo(fn):
    def wrapper():
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            fn()
        finally:
            os.chdir(cwd)
    wrapper.__name__ = fn.__name__
    return wrapper


# ----------------------------------------------------------------------------- benchmarks
def test_letter_extraction():
    from tasks.choice import extract_letter
    for raw, want in [("B", "B"), ("(b) blue", "B"), ("I think B", "B"), ("answer is C", "C"),
                      ("Option: D.", "D"), ("K", "K"), ("(R)", "R")]:
        assert extract_letter(raw) == want, (raw, extract_letter(raw))


def test_parser_reads_letters_beyond_e():
    from src.communication.protocol import parse_agent_response
    m = parse_agent_response('{"reasoning":["shape is an ellipse"],"action":"(K)","confidence":0.8,'
                             '"content":"ellipse"}', "solver_letter", 0)
    assert m.action == "K"
    m = parse_agent_response('{"reasoning":["so the answer is (C)"],"action":"","confidence":0.8,'
                             '"content":""}', "solver_letter", 0)
    assert m.action == "C"


@_in_repo
def test_new_benchmarks_registered_and_data_scores():
    from tasks import get_benchmark
    for task in ("commonsenseqa", "truthfulqa", "bbh"):
        b = get_benchmark(task)
        assert b.answer_format == "letter" and b.solver_role == "solver_letter"
        path = os.path.join("data", task, "test.json")
        if os.path.exists(path):
            data = json.load(open(path, encoding="utf-8"))
            assert len(data) >= 600, (task, len(data))
            assert all(b.score(d["answer"], d["answer"]) for d in data[:50])
        assert "PROPOSED ANSWER" in b.build_verifier_prompt("q?\n(A) x\n(B) y", "B")


# ----------------------------------------------------------------------------- rules
def test_arithmetic_rules_catch_errors_without_false_alarms():
    from src.verification.rules import check_arithmetic, safe_eval
    ok = ["16-3-4=9 eggs", "9×2=18 dollars", "3 miles at 6 mph = 3/6 = 0.5 hours = 30 minutes",
          "30×(2.5/12)=30×0.2083=6.25", "Let son=x, Tom=3x", "3x+10=2(x+10)", "1,000 + 2,500 = 3,500",
          "100/3 = 33.33", "Today: 2015-01-25", "total = 3 × 20 = 60 cups"]
    assert all(not check_arithmetic([s]) for s in ok)
    assert all(check_arithmetic([s]) for s in ["7×8=54", "12 - 2 = 11", "200×1.5=500"])
    assert safe_eval('__import__("os")') is None and safe_eval("9**999999") is None


def test_answer_support_rules():
    from src.verification.rules import check_answer_support
    assert check_answer_support("20", ["16-3-4=9", "9x2=18"], "number")
    assert not check_answer_support("18", ["16-3-4=9", "9x2=18"], "number")
    assert check_answer_support("yes", ["x", "Therefore, no"], "yes or no")
    assert check_answer_support("B", ["x", "So the answer is (C)"], "letter")
    assert not check_answer_support("C", ["x", "So the answer is (C)."], "letter")


# ----------------------------------------------------------------------------- verifier agent
def _fake_verifier(reply):
    from src.agents.base_agent import BaseAgent
    seen = []

    def call_llm(self, prompt):
        seen.append(prompt)
        return (reply(prompt) if callable(reply) else reply), 7
    BaseAgent.call_llm = call_llm
    return seen


@_in_repo
def test_verifier_parses_overrides_and_blind_mode():
    from src.agents.verifier import VerifierAgent
    from tasks import get_benchmark
    b = get_benchmark("gsm8k")
    cfg = "config/model_config_verifier.yaml"
    _fake_verifier('{"check":["9*2=18"],"own_answer":"18","verdict":"pass","confidence":0.9}')
    r = VerifierAgent(cfg).verify("q", "18", b, ["9x2=18"])
    assert r.verdict == "pass" and r.tokens == 7 and not r.overridden and r.rule_issues == []
    # says pass but its own answer differs -> fail
    _fake_verifier('{"own_answer":"20","verdict":"pass","confidence":0.9}')
    r = VerifierAgent(cfg).verify("q", "18", b, ["9x2=18"])
    assert r.verdict == "fail" and r.overridden
    # garbage -> unknown (carries no evidence)
    _fake_verifier("I cannot decide")
    assert VerifierAgent(cfg).verify("q", "18", b, ["9x2=18"]).verdict == "unknown"
    # blind mode never shows the reasoning, but rule checks still run on it
    seen = _fake_verifier('{"own_answer":"18","verdict":"pass","confidence":0.9}')
    r = VerifierAgent(cfg, mode="blind").verify("q", "18", b, ["7x8=54", "so 18"])
    assert "7x8=54" not in seen[-1] and r.rule_issues


# ----------------------------------------------------------------------------- math
def test_proposition1_band_matches_direct_gain():
    rs = np.random.RandomState(0)
    for _ in range(2000):
        t0, t1 = sorted(rs.uniform(0.02, 0.98, 2))
        kappa, p = rs.uniform(0, 0.2), rs.uniform(0.5, 1.0)
        gain = (1 - p) * (1 - t0) - p * (1 - t1)           # verify-then-act minus stop now
        u = verify_band_upper(t1, t0, kappa)
        assert (gain >= kappa) == (p <= u) or abs(gain - kappa) < 1e-9
    # blind spot: tau0 = tau1 leaves no band above 1/2; a perfect verifier gives u = 1 - kappa
    assert verify_band_upper(0.8, 0.8, 0.01) < 0.5
    assert abs(verify_band_upper(1.0, 0.0, 0.05) - 0.95) < 1e-12


def test_bayes_update_matches_two_candidate_posterior():
    p, t1, t0 = 0.6, 0.9, 0.3
    for verdict, like1, like0 in ((1, t1, t0), (-1, 1 - t1, 1 - t0)):
        post = p * like1 / (p * like1 + (1 - p) * like0)
        odds = p / (1 - p) * math.exp(log_lr(verdict, t1, t0))
        assert abs(post - odds / (1 + odds)) < 1e-12
    assert log_lr(0, t1, t0) == 0.0


def test_sample_size_floor():
    assert min_questions(0.01, 0.10) == 230 and min_questions(0.02, 0.10) == 114
    assert abs(smallest_eps(150, 0.10) - 0.0152) < 1e-3


# ----------------------------------------------------------------------------- simulation
def _synthetic(n=600, C=4, t1=0.9, t0=0.15, seed=0):
    """Two candidates x (right) and y; observer calibrated; verifier with rates (t1, t0)."""
    rs = np.random.RandomState(seed)
    cps = [f"r{c}f" for c in range(C)]
    p_x = np.clip(np.cumsum(rs.normal(0.08, 0.12, (n, C)), 1) + rs.uniform(0.35, 0.75, (n, 1)), 0.02, 0.98)
    right_is_x = rs.uniform(size=n) < 0.8
    probs, top = [], np.empty((n, C), dtype=object)
    p_top, margin, correct = np.zeros((n, C)), np.zeros((n, C)), np.zeros((n, C), bool)
    verdict = np.zeros((n, C), int)
    passes = {}
    for i in range(n):
        row = []
        y = 0 if right_is_x[i] else 1
        for c in range(C):
            p = np.array([p_x[i, c], 1 - p_x[i, c]])
            row.append((["x", "y"], p, y))
            j = int(np.argmax(p))
            top[i, c], p_top[i, c], margin[i, c], correct[i, c] = ["x", "y"][j], p[j], abs(p[0] - p[1]), j == y
            key = (i, top[i, c])
            if key not in passes:
                passes[key] = rs.uniform() < (t1 if j == y else t0)
            verdict[i, c] = 1 if passes[key] else -1
        probs.append(row)
    S = np.stack([p_top, margin, np.zeros((n, C)), np.zeros((n, C)), np.tile(np.linspace(0, 1, C), (n, 1))], -1)
    tb = VTables(p_top=p_top, margin=margin, correct=correct, top=top,
                 cost=np.tile(np.linspace(0.25, 1.0, C), (n, 1)), group=np.array([f"q|{i}" for i in range(n)]),
                 S=S, verdict=verdict, vcost=np.full(n, 0.03))
    rate = lambda r: math.log(r / (1 - r))
    tau = TauModel(np.zeros(5), np.ones(5), np.zeros(5), rate(t1), np.zeros(5), rate(t0))
    return attach_tau(tb, probs, tau), cps


def test_never_verify_equals_plain_osc():
    tb, cps = _synthetic()
    thr = thresholds(cps, 0.5, Shape())
    v = simulate_v(tb, thr, None)
    arr = Arrays(tb.margin, tb.correct, tb.top, tb.cost, tb.group)
    s = simulate(arr, thr)
    assert (v["correct"] == s["correct"]).all() and np.allclose(v["cost"], s["cost"])
    assert not v["verified"].any()


def test_informative_verifier_raises_accuracy_and_costs_tokens():
    tb, cps = _synthetic(t1=0.95, t0=0.05)
    thr = thresholds(cps, 0.3, Shape())
    base = simulate_v(tb, thr, None)
    band = simulate_v(tb, thr, kappa=0.0)
    assert band["correct"].mean() > base["correct"].mean() + 0.02
    assert band["verified"].any() and (band["cost"] >= base["cost"] - 1e-12).sum() > 0
    # a useless verifier (tau0 = tau1) is never called
    tb2, _ = _synthetic(t1=0.6, t0=0.6)
    assert not simulate_v(tb2, thr, kappa=0.01)["verified"].any()


def test_certify_v_finds_a_setting_and_respects_eps():
    tb, cps = _synthetic(n=1200, t1=0.95, t0=0.05)
    rs = np.random.RandomState(1)
    h1 = rs.uniform(size=len(tb.group)) < 0.5
    res = certify_v(tb.subset(h1), tb.subset(~h1), cps, Shape(), eps=0.05, delta=0.1, risk="harm")
    st = res["setting"]
    assert st is not None and st["kappa"] is not None          # an informative verifier is used
    passed = [t for t in res["trace"] if t["p_value"] <= 0.1]
    assert passed and passed[-1]["a"] == st["a"] and passed[-1]["kappa"] == st["kappa"]
    # the certified setting is the cheapest one that passed, and its half-2 harm is below eps
    assert all(t["cost"] >= passed[-1]["cost"] for t in passed)
    t2 = tb.subset(~h1)
    sim = simulate_v(t2, thresholds(cps, st["a"], Shape()), st["kappa"], st["ell"])
    assert sim["harm"].mean() <= 0.05 and sim["correct"].mean() > t2.correct[:, -1].mean()


def test_tau_model_recovers_rates():
    rs = np.random.RandomState(0)
    n = 4000
    S = rs.normal(size=(n, 5))
    right = rs.uniform(size=n) < 0.7
    passed = np.where(right, rs.uniform(size=n) < 0.9, rs.uniform(size=n) < 0.2)
    tau = TauModel.fit(S, right, passed, np.ones(n))
    t1, t0 = tau.taus(np.zeros((1, 5)))
    assert abs(t1[0] - 0.9) < 0.04 and abs(t0[0] - 0.2) < 0.04
    back = TauModel.from_dict(json.loads(json.dumps(tau.to_dict())))
    assert np.allclose(back.taus(S[:5])[0], tau.taus(S[:5])[0])


# ----------------------------------------------------------------------------- orchestrator
def _debate_llm(verifier_says):
    """Solver A (qwen) says 18, solver B (llama) says 20, critic agrees; verifier scripted."""
    from src.agents.base_agent import BaseAgent
    calls = []

    def call_llm(self, prompt):
        calls.append(self.role)
        if self.role == "verifier":
            ans = "18" if "PROPOSED ANSWER: 18" in prompt else "20"
            return json.dumps({"own_answer": "18", "verdict": verifier_says(ans), "confidence": 0.9}), 11
        if self.role == "critic" and "SOLVER" in prompt:
            return json.dumps({"reasoning": ["ok"], "action": "agree", "confidence": 0.9, "content": "ok"}), 5
        a = "18" if "qwen" in self.model else "20"
        return json.dumps({"reasoning": ["16-3-4=9", f"answer {a}"], "action": a,
                           "confidence": 0.9, "content": a}), 5
    BaseAgent.call_llm = call_llm
    return calls


@_in_repo
def test_orchestrator_verifies_once_and_counts_tokens():
    from osc.observer import Observer
    from osc.features import checkpoint_order
    from osc.policy import OSCPolicy
    from src.orchestrator import MADOrchestrator
    rate = lambda r: math.log(r / (1 - r))
    tau = TauModel(np.zeros(5), np.ones(5), np.zeros(5), rate(0.9), np.zeros(5), rate(0.1))
    cps = checkpoint_order(2, True)
    pol = OSCPolicy("gsm8k", 2, True, {cp: Observer() for cp in cps}, 0.5, Shape(0.0, 0.0),
                    verify=dict(tau=tau.to_dict(), kappa=0.0, ell=0.0, mode="reasoning"))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "p.json")
        pol.save(path)
        calls = _debate_llm(lambda ans: "pass" if ans == "18" else "fail")
        qwen, llama = "config/model_config_solver_qwen.yaml", "config/model_config_solver_llama.yaml"
        orch = MADOrchestrator("gsm8k", qwen, llama, [qwen], max_rounds=2, stop_policy="osc",
                               critic_mode="live", osc_policy_path=path,
                               verifier_config_path="config/model_config_verifier.yaml")
        rec = orch.run(1, "Janet has 16 eggs...", "18")
    assert calls.count("verifier") == 1
    v = rec["osc"]["verify"]
    assert v["checkpoint"] == "r0s" and v["verdict"] in ("pass", "fail")
    assert rec["final_answer"] == "18" and rec["final_correct"]
    assert rec["osc"]["verify_tokens"] == 11 and rec["osc"]["stopped_at"] == "r0s"
    # a verifying policy without a verifier config is refused
    try:
        MADOrchestrator("gsm8k", qwen, llama, [qwen], max_rounds=2, stop_policy="osc",
                        osc_policy_path=path)
    except (ValueError, FileNotFoundError):
        pass
    else:
        raise AssertionError("expected ValueError")


def test_offline_pairs_are_deduplicated_across_seeds():
    from osc.verify_offline import collect_pairs
    rec = lambda seed, ans: {"task": "gsm8k", "sample_id": 3, "question": "q", "ground_truth": "18",
                             "seed": seed, "rounds": [{"agents": {
                                 "solver_a": {"normalized_answer": ans, "reasoning": ["r"]},
                                 "solver_b": {"normalized_answer": "20", "reasoning": ["s"]},
                                 "critic": {"normalized_answer": "18.0", "reasoning": ["t"]}}}]}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "debate_full_x.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec(1, "18")) + "\n" + json.dumps(rec(2, "18")) + "\n")
        jobs = collect_pairs(os.path.join(d, "*.jsonl"), "gsm8k")
    assert sorted(j["answer"] for j in jobs) == ["18", "20"]   # 18 and 18.0 are one answer


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                failed += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if failed else 0)
