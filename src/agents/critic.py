"""
agents/critic.py

Changes vs original:
- Early-agree now works for number format too (GSM8K fix)
- Force critic LLM call on FACTOID questions even if actions match
- Uncertainty-triggered verification before agree
- Negation-aware keyword matching (via protocol)
- Critic prompt includes calibration suffix

FIX PATCH (2026-04-25):
- FIX-C1: force_disagree_assumption — chỉ fire khi actions KHÁC NHAU
- FIX-C2: force_disagree_no_fact — chỉ fire khi CẢ HAI không có fact VÀ conf thấp
- FIX-C3: low-conf agree threshold hạ xuống 0.65 (từ 0.80) để tránh block rescue
- FIX-C4: force_disagree_different_answers giữ nguyên (đúng)
"""

import logging
import re
from typing import Optional

from .base_agent import BaseAgent
from ..communication.protocol import (
    parse_agent_response,
    classify_question,
    has_specific_fact,
    is_uncertain,
    looks_like_non_answer,
    build_compact_prompt,
)

logger = logging.getLogger(__name__)

_COPY_THRESHOLD = 0.85

_CALIBRATION_SUFFIX = """
Rate your confidence HONESTLY:
- 0.95+ : you verified a specific fact (name, number, date)
- 0.80  : reasoning from solid general knowledge
- 0.65  : inferring, not directly recalling
- 0.40  : guessing

If you cannot recall a specific supporting fact → confidence MUST be below 0.70.
Do NOT fake certainty.
"""

_UNCERTAINTY_THRESHOLD = 0.72


def _jaccard(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a.lower().split()), set(b.lower().split())
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _actions_match(a1: str, a2: str) -> bool:
    a1, a2 = a1.strip().lower(), a2.strip().lower()
    if a1 == a2:
        return True
    try:
        return abs(float(a1) - float(a2)) < 1e-6
    except ValueError:
        return False


def _has_error_signal(text: str) -> bool:
    kws = ("wrong", "incorrect", "error", "mistake", "should be", "should have")
    return any(kw in text for kw in kws)


def has_assumption(text: str):
    kws = ["likely", "probably", "suggests", "may", "could", "possibly"]
    return any(k in text.lower() for k in kws)


class CriticAgent(BaseAgent):
    def __init__(self, config_path: str = "config/model_config.yaml"):
        super().__init__(role="critic", config_path=config_path)
        self._independent_answer = None

    def reset(self):
        self._independent_answer = None

    def _validate_action(self, action: str, benchmark) -> bool:
        fmt = benchmark.answer_format
        if fmt == "yes or no":
            return action in ("yes", "no")
        elif fmt == "number":
            try:
                float(action)
                return True
            except Exception:
                return False
        return True

    def prepare_independent(self, question: str, round_id: int, benchmark, peers=None) -> int:
        """Produce the critic's own answer for this round and return its token cost.

        Round 0 (or peers=None): solve blind, exactly as before.
        Later rounds in live mode: re-solve after seeing its previous answer and the two
        solvers' current solutions, so the critic can change its mind (it was frozen before).
        """
        if peers is None or self._independent_answer is None:
            extra = ""
        else:
            prev = self._independent_answer
            lines = [
                "\n\nDEBATE SO FAR:",
                f"Your previous answer: {prev.action}",
                f"Your previous steps: {prev.get_reasoning_text()[:1500]}",
            ]
            for name, m in zip(("Solver A", "Solver B"), peers):
                lines.append(f"{name} now answers: {m.action}")
                lines.append(f"{name} steps: {m.get_reasoning_text()[:1500]}")
            lines.append(
                "Solve the problem again yourself. Keep your previous answer unless you can point "
                "to a concrete error in your own steps; if you change it, your new steps must "
                "support the new answer. Agreeing with the solvers is not a reason by itself."
            )
            extra = "\n".join(lines)
        ind_msg, tokens = self._generate_independent_answer(question, round_id, benchmark, extra)
        self._independent_answer = ind_msg
        return tokens

    def _generate_independent_answer(self, question: str, round_id: int, benchmark, extra: str = ""):
        prompt_ind = benchmark.build_independent_prompt(question)
        prompt_ind += extra
        prompt_ind += _CALIBRATION_SUFFIX
        raw_ind, tokens_ind = self.call_llm(prompt_ind)

        parse_role = benchmark.solver_role
        ind_msg = parse_agent_response(raw_ind, role=parse_role, round_id=round_id)

        na_retries = 0
        while looks_like_non_answer(ind_msg) and na_retries < 2:
            na_retries += 1
            logger.warning("[Critic] Non-answer/refusal — compact retry %d", na_retries)
            raw_ind, tokens_ind = self.call_llm(
                build_compact_prompt(question, benchmark.answer_format)
            )
            ind_msg = parse_agent_response(raw_ind, role=parse_role, round_id=round_id)

        if "no reasoning" in ind_msg.get_reasoning_text().lower():
            logger.warning("[Critic] Empty reasoning — retry")
            raw_ind, tokens_ind = self.call_llm(
                prompt_ind + "\nYou MUST provide step-by-step reasoning."
            )
            ind_msg = parse_agent_response(raw_ind, role=parse_role, round_id=round_id)

        if not self._validate_action(ind_msg.action, benchmark):
            logger.warning(f"[Critic] Invalid action '{ind_msg.action}' — retry")
            fmt = benchmark.answer_format
            if fmt == "number":
                suffix = "\nYou MUST output a JSON with 'action' set to the final numeric answer only (e.g. 42). No units, no text."
            else:
                suffix = f"\nYou MUST answer strictly with {fmt}."
            raw_ind, tokens_ind = self.call_llm(prompt_ind + suffix)
            ind_msg = parse_agent_response(raw_ind, role=parse_role, round_id=round_id)

            if not self._validate_action(ind_msg.action, benchmark):
                logger.error(f"[FIX FAIL] Still invalid '{ind_msg.action}'")
                if fmt == "number":
                    m = re.search(r"-?\d+\.?\d*", raw_ind)
                    ind_msg.action = m.group(0) if m else "0"
                else:
                    m = re.search(r"\b(yes|no)\b", raw_ind, re.IGNORECASE)
                    ind_msg.action = m.group(1).lower() if m else "yes"
                ind_msg.confidence = 0.3

        return ind_msg, tokens_ind

    def _can_early_agree(
        self,
        ind_msg,
        solver_message,
        sim: float,
        benchmark,
        question: str,
        combined_reasoning: str,
    ) -> bool:

        if not _actions_match(ind_msg.action, solver_message.action):
            return False

        if sim >= _COPY_THRESHOLD:
            return False

        if _has_error_signal(combined_reasoning):
            return False

        if has_assumption(combined_reasoning):
            return False

        if is_uncertain(ind_msg) or is_uncertain(solver_message):
            return False

        if not has_specific_fact(ind_msg) or not has_specific_fact(solver_message):
            return False

        q_type = classify_question(question)
        if q_type in ["REASONING", "HYBRID"]:
            return False

        if benchmark.answer_format == "number":
            return (
                ind_msg.confidence >= 0.95
                and solver_message.confidence >= 0.95
            )

        return (
            ind_msg.confidence >= 0.95
            and solver_message.confidence >= 0.95
        )

    def respond(self, question: str, solver_message, round_id: int = 0, benchmark=None):

        # ── Independent answer (locked after round 0) ─────────────────────
        if self._independent_answer is None:
            ind_msg, tokens_ind = self._generate_independent_answer(
                question, round_id, benchmark
            )
            self._independent_answer = ind_msg
            logger.info(
                f"[Critic ind] '{ind_msg.action}' (conf={ind_msg.confidence:.2f}) "
                f"— {ind_msg.get_reasoning_text()}"
            )
        else:
            ind_msg = self._independent_answer
            tokens_ind = 0
            logger.info(f"[Critic ind] '{ind_msg.action}' (locked)")

        sim = _jaccard(
            ind_msg.get_reasoning_text(),
            solver_message.get_reasoning_text(),
        )

        combined_reasoning = (
            ind_msg.get_reasoning_text().lower() + " "
            + solver_message.get_reasoning_text().lower()
        )

        if self._can_early_agree(
            ind_msg, solver_message, sim, benchmark, question, combined_reasoning
        ):
            from ..communication.message import StructuredMessage
            msg = StructuredMessage(
                role="critic",
                intent="answer match",
                reasoning=[
                    f"My independent answer '{ind_msg.action}' matches solver's '{solver_message.action}'",
                    "Agreement based on identical conclusions from independent reasoning",
                    "Both agents have sufficient confidence and specific supporting facts",
                    "No factual error detected",
                ],
                action="agree",
                confidence=max(ind_msg.confidence, solver_message.confidence),
                content=f"Both independently answered '{ind_msg.action}' — agree.",
                round_id=round_id,
            )
            msg.token_count = tokens_ind
            msg.independent_answer = ind_msg
            logger.info(
                f"[Critic R{round_id}] ind='{ind_msg.action}' "
                f"solver='{solver_message.action}' sim={sim:.2f} → agree (early, skip LLM)"
            )
            return msg

        prompt_cmp = benchmark.build_critic_prompt(question, solver_message, ind_msg)

        prompt_cmp += """
        STRICT MODE:
        - DISAGREE only if you find a clear factual or logical error
        - If both answers are plausible and no clear error → AGREE
        - Do NOT disagree based on uncertainty alone
        - Do NOT be overly skeptical
        Default: AGREE unless a clear error is identified.
        """

        prompt_cmp += _CALIBRATION_SUFFIX

        if sim >= _COPY_THRESHOLD:
            logger.warning(
                f"[Critic] High similarity (sim={sim:.2f}) → independence instruction added"
            )
            raw_cmp, tokens_cmp = self.call_llm(
                prompt_cmp + "\nDo NOT reuse the solver's reasoning. Reason fully independently."
            )
        else:
            raw_cmp, tokens_cmp = self.call_llm(prompt_cmp)

        msg = parse_agent_response(raw_cmp, role="critic", round_id=round_id)


        if (
            msg.action == "disagree"
            and _actions_match(ind_msg.action, solver_message.action)
            and not _has_error_signal(msg.get_reasoning_text())
            and not _has_error_signal(msg.content or "")
        ):
            logger.warning(
                "[Critic FIX-C0] LLM disagree but actions match + no error signal → force agree"
            )
            msg.action = "agree"
            msg.confidence = min(msg.confidence, 0.85)

        if (
            has_assumption(combined_reasoning)
            and msg.action == "agree"
            and not _actions_match(ind_msg.action, solver_message.action)
        ):
            logger.warning("[Critic FIX-C1] Assumption + different answers → force disagree")
            msg.action = "disagree"
            msg.confidence = min(msg.confidence, 0.75)

        if (
            not has_specific_fact(ind_msg)
            and not has_specific_fact(solver_message)
            and msg.action == "agree"
            and msg.confidence < 0.70          # chỉ enforce khi conf không cao
        ):
            logger.warning("[Critic FIX-C2] No fact + low conf → force disagree")
            msg.action = "disagree"
            msg.confidence = min(msg.confidence, 0.65)

        if msg.action == "agree" and msg.confidence < 0.65:
            logger.warning("[Critic FIX-C3] Very low confidence agree (< 0.65) → force disagree")
            msg.action = "disagree"


        if not _actions_match(ind_msg.action, solver_message.action):
            msg.confidence *= 0.8   # giảm nhẹ thôi

        msg.token_count = tokens_ind + tokens_cmp
        msg.independent_answer = ind_msg

        logger.info(
            f"[Critic R{round_id}] ind='{ind_msg.action}' (conf={ind_msg.confidence:.2f}) "
            f"solver='{solver_message.action}' (conf={solver_message.confidence:.2f}) "
            f"sim={sim:.2f} → {msg.action}"
        )
        logger.info(f"[Critic reasoning] {msg.get_reasoning_text()}")
        logger.info(f"[Critic verdict] {msg.content}")
        return msg