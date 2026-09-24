"""
src/agents/verifier.py
Verifier agent for the OSC VERIFY action: one call judges one proposed answer.

Output is deliberately small — verdict pass/fail/unknown, a confidence, the verifier's own
answer and the free rule checks — because OSC does not trust the verdict at face value: it
learns P(pass | answer right, state) and P(pass | answer wrong, state) from logs (osc/verify.py)
and turns the verdict into a likelihood ratio. "unknown" (unparseable reply) carries no
evidence and is skipped by that model.

Modes
  reasoning : the verifier sees the question, the answer and the steps offered for it
  blind     : question and answer only, so it cannot inherit the debate's argument
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .base_agent import BaseAgent
from ..verification.rules import rule_check

logger = logging.getLogger(__name__)

MODES = ("reasoning", "blind")
_PASS = ("pass", "correct", "valid", "true", "yes", "right")
_FAIL = ("fail", "incorrect", "invalid", "false", "no", "wrong")


@dataclass
class VerifyResult:
    verdict: str                      # "pass" | "fail" | "unknown"
    confidence: float
    own_answer: Optional[str]         # verifier's own normalized answer, if it gave one
    issue: str
    rule_issues: List[str] = field(default_factory=list)
    tokens: int = 0
    mode: str = "reasoning"
    overridden: bool = False          # verdict flipped because own_answer contradicted it

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_json(raw: str) -> Optional[dict]:
    raw = re.sub(r"```(?:json)?", "", raw or "")
    start = raw.find("{")
    if start < 0:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(raw[start:])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        m = re.search(r'"verdict"\s*:\s*"(\w+)"', raw)
        return {"verdict": m.group(1)} if m else None


def _verdict_word(v) -> str:
    v = str(v or "").strip().lower()
    if any(v.startswith(w) for w in _FAIL):   # "incorrect" before "correct"
        return "fail"
    if any(v.startswith(w) for w in _PASS):
        return "pass"
    return "unknown"


class VerifierAgent(BaseAgent):
    def __init__(self, config_path: str, mode: str = "reasoning"):
        if mode not in MODES:
            raise ValueError(f"verifier mode must be one of {MODES}")
        super().__init__(role="verifier", config_path=config_path)
        self.mode = mode

    def verify(self, question: str, answer: str, benchmark,
               reasoning: Optional[List[str]] = None) -> VerifyResult:
        shown = benchmark.normalize_answer(str(answer))
        rule_issues = rule_check(shown, reasoning or [], benchmark.answer_format) if reasoning else []
        prompt = benchmark.build_verifier_prompt(
            question, shown, reasoning if self.mode == "reasoning" else None)
        raw, tokens = self.call_llm(prompt)
        data = _parse_json(raw)
        if data is None:
            logger.warning("[Verifier] unparseable reply: %r", (raw or "")[:200])
            return VerifyResult("unknown", 0.0, None, "unparseable reply", rule_issues, tokens, self.mode)

        verdict = _verdict_word(data.get("verdict"))
        try:
            conf = min(max(float(data.get("confidence", 0.5)), 0.0), 1.0)
        except (TypeError, ValueError):
            conf = 0.5
        own = data.get("own_answer")
        own = benchmark.normalize_answer(str(own)) if own not in (None, "") else None

        # The verifier solved the question itself; a verdict that contradicts its own answer is
        # resolved in favour of the explicit comparison.
        overridden = False
        if own is not None and verdict != "unknown":
            agrees = benchmark.score(own, shown)
            if verdict == "pass" and not agrees:
                verdict, overridden = "fail", True
            elif verdict == "fail" and agrees:
                verdict, overridden = "pass", True
        return VerifyResult(verdict, conf, own, str(data.get("issue", "") or ""), rule_issues,
                            tokens, self.mode, overridden)
