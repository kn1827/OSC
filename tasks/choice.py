"""
tasks/choice.py
Shared base for multiple-choice benchmarks whose answer is one option letter
(CommonsenseQA, TruthfulQA, BBH). A subclass only supplies the task note, the worked
examples and the critic checklist; prompts, parsing and scoring are the same for all.
"""

import re

from .base import BenchmarkConfig

# BBH options go up to (R); keep the whole range so no valid letter is rejected
LETTERS = "ABCDEFGHIJKLMNOPQR"

_LETTER_PATTERNS = [
    re.compile(r"^\(?([A-R])\)?$"),                        # "B", "(B)"
    re.compile(r"^\(?([A-R])[\)\.:]"),                     # "B) text", "B. text"
    re.compile(r"\b(?:ANSWER|OPTION|CHOICE)\s*(?:IS\s*)?:?\s*\(?([A-R])\)?(?![A-Z])"),
    re.compile(r"\(([A-R])\)"),
    re.compile(r"\b([A-R])\)"),
]


def extract_letter(raw: str) -> str:
    """Option letter in raw text ("(b) blue" -> "B"); raw upper-cased if none found.
    Without an explicit marker the LAST standalone letter wins, so "I think B" gives B."""
    s = (raw or "").strip().upper()
    for pat in _LETTER_PATTERNS:
        m = pat.search(s)
        if m:
            return m.group(1)
    found = re.findall(r"\b([A-R])\b", s)
    return found[-1] if found else s


class ChoiceBenchmark(BenchmarkConfig):
    """Subclasses set NAME, NOTE, SOLVER_EXAMPLES, CRITIC_GUIDE (and optionally FLIP_KEYWORDS)."""

    NAME = ""
    NOTE = ""
    SOLVER_EXAMPLES = ""
    CRITIC_GUIDE = ""
    EXTRA_RULES = ""
    FLIP_KEYWORDS = (
        "incorrect", "wrong", "actually", "in fact", "factually", "correct answer is",
        "should be", "is not", "does not", "contradicts", "misconception", "myth", "error",
    )

    @property
    def task_name(self) -> str:
        return self.NAME

    @property
    def answer_format(self) -> str:
        return "letter"

    @property
    def solver_role(self) -> str:
        return "solver_letter"

    # ── Prompts ───────────────────────────────────────────────────────────────

    def _answer_block(self, question: str, certain_rule: str) -> str:
        return (
            f"{self.NOTE}\n\n"
            f"{self.SOLVER_EXAMPLES}\n"
            f"Q: {question}\n\n"
            f"Rules:\n"
            f"{self.EXTRA_RULES}"
            f"- 'action' MUST be exactly one letter of the options listed in the question\n"
            f"- Provide reasoning steps that justify your chosen answer\n"
            f"- {certain_rule}\n\n"
            f"Output ONLY a JSON object:\n"
            f'{{"intent":"<topic>","reasoning":["step 1","step 2"],"action":"<letter>","confidence":0.95,"content":"one sentence"}}\n'
            f'The "action" field MUST be the letter of one listed option.'
        )

    def build_solver_prompt(self, question: str, round_id: int, critic_messages=None) -> str:
        critique = ""
        if critic_messages:
            last = critic_messages[-1]
            critique = (
                f"\n\nCritic found: {last.content}\n"
                f"Only change your answer if there is a clear error identified."
            )
        return (
            "Answer this multiple-choice question. Think step by step.\n\n"
            + self._answer_block(question, "Set confidence 0.99 only if fully certain; lower if uncertain")
            + critique
        )

    def build_independent_prompt(self, question: str) -> str:
        return (
            "Answer this multiple-choice question. Think step by step.\n\n"
            + self._answer_block(question, "Set confidence 0.99 only if fully certain")
        )

    def build_critic_prompt(self, question: str, solver_message, critic_independent) -> str:
        mine = self.normalize_answer(critic_independent.action)
        theirs = self.normalize_answer(solver_message.action)
        note = (
            "NOTE: Both chose the same option. Verify the reasoning is actually correct."
            if mine == theirs
            else f"NOTE: Answers differ — yours={mine}, solver={theirs}. Determine which is correct."
        )
        return (
            f"You are checking the answer to a multiple-choice question.\n\n"
            f"{self.CRITIC_GUIDE}\n"
            f"Question: {question}\n\n"
            f"YOUR answer: {mine}\n"
            f"Your reasoning: {critic_independent.get_reasoning_text()}\n\n"
            f"SOLVER's answer: {theirs}\n"
            f"Solver reasoning: {solver_message.get_reasoning_text()}\n\n"
            f"{note}\n\n"
            f"Output ONLY a JSON object:\n"
            f'{{"intent":"check","reasoning":["what I verified","verdict"],"action":"agree or disagree","confidence":0.0-1.0,"content":"one sentence — if disagree, state the correct option and why"}}\n'
            f'The "action" field MUST be exactly "agree" or "disagree".'
        )

    def build_judge_prompt(self, question: str, solver_message, critic_message,
                           verification_result=None, solver_initial_message=None) -> str:
        return (
            f"You are the final judge for a multiple-choice question. Answer it yourself first, "
            f"then compare with the debate.\n\n{self.NOTE}\n\n"
            f"Question: {question}\n\n"
            f"Debate reference:\n"
            f"  Solver: {self.normalize_answer(solver_message.action)} — {solver_message.get_reasoning_text()}\n"
            f"  Critic: {critic_message.content}\n\n"
            f"Output ONLY a JSON object:\n"
            f'{{"intent":"final judgment","reasoning":["my reasoning","cross-check"],"action":"<letter>","confidence":0.0-1.0,"content":"one sentence"}}'
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def normalize_answer(self, raw: str) -> str:
        return extract_letter(raw)

    def score(self, pred: str, gt: str) -> bool:
        return extract_letter(pred) == extract_letter(gt)

    # ── Behaviour overrides ───────────────────────────────────────────────────

    def allow_flip(self, critic_msg) -> bool:
        text = ((critic_msg.get_reasoning_text() or "") + " " + (critic_msg.content or "")).lower()
        return any(kw in text for kw in self.FLIP_KEYWORDS)

    def should_early_stop(self, solver_msg, critic_msg, logical, round_id=0) -> bool:
        return (critic_msg.action == "agree" and logical.get("valid", False)
                and solver_msg.confidence >= 0.99)

    def verifier_settings(self) -> dict:
        return {"check_arithmetic": False, "check_percentage": False}
