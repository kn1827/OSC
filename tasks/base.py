from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.communication.message import StructuredMessage


class BenchmarkConfig(ABC):

    @property
    @abstractmethod
    def task_name(self) -> str: ...

    @property
    def answer_format(self) -> str:
        return "unknown"

    @property
    def solver_role(self) -> str:
        return "solver"

    @property
    def critic_role(self) -> str:
        return "critic"

    @property
    def judge_role(self) -> str:
        return "judge"


    @abstractmethod
    def build_solver_prompt(
        self,
        question: str,
        round_id: int,
        critic_messages: Optional[List["StructuredMessage"]] = None,
    ) -> str: ...

    @abstractmethod
    def build_independent_prompt(self, question: str) -> str: ...

    @abstractmethod
    def build_critic_prompt(
        self,
        question: str,
        solver_message: "StructuredMessage",
        critic_independent: "StructuredMessage",
    ) -> str: ...

    @abstractmethod
    def build_judge_prompt(
        self,
        question: str,
        solver_message: "StructuredMessage",
        critic_message: "StructuredMessage",
        verification_result: Optional[dict] = None,
        solver_initial_message: Optional["StructuredMessage"] = None,
    ) -> str: ...

    @abstractmethod
    def score(self, pred: str, gt: str) -> bool: ...

    def normalize_answer(self, raw: str) -> str:
        return raw.strip().lower()

    def allow_flip(self, critic_msg: "StructuredMessage") -> bool:
        text = (critic_msg.get_reasoning_text() or "").lower()
        allow_keywords = [
            "incorrect", "not true", "wrong", "evidence",
            "calculation", "miscalculated", "error", "factual",
            "actually", "in fact", "contradicts", "misidentif",
            "impossible", "cannot", "does not", "is not",
        ]
        if any(kw in text for kw in allow_keywords):
            return True
        if getattr(critic_msg, "confidence", 0) > 0.85 and critic_msg.action == "disagree":
            return True
        return False

    def should_early_stop(
        self,
        solver_msg: "StructuredMessage",
        critic_msg: "StructuredMessage",
        logical: dict,
        round_id: int = 0,
    ) -> bool:
        return (
            critic_msg.action == "agree"
            and logical.get("valid", False)
            and solver_msg.confidence > 0.98
        )

    def select_final_answer(
        self,
        question: str,
        solver_messages: List["StructuredMessage"],
        critic_messages: List["StructuredMessage"],
        verif_final: dict,
        selector,
        critic_ind_r0: Optional["StructuredMessage"] = None,
    ) -> "StructuredMessage":
        last_solver = solver_messages[-1]
        last_critic = critic_messages[-1]

        if getattr(last_solver, "unfounded_flip", False):
            r0 = solver_messages[0]
            r1 = solver_messages[-2] if len(solver_messages) >= 2 else solver_messages[0]
            # Pick whichever has fewer issues (caller provides verif via selector context)
            # Default: prefer r0 (original answer more reliable)
            return r0

        if verif_final["valid"] and last_critic.action == "agree":
            return last_solver

        return selector.select(
            question=question,
            solver_initial=solver_messages[0],
            solver_final=last_solver,
            critic_independent=critic_ind_r0,
        )

    # ── Verifier (OSC VERIFY action) ─────────────────────────────────────────

    def answer_hint(self) -> str:
        """How a final answer is written for this task (shown to the verifier)."""
        return {
            "number": "a single number, no units",
            "yes or no": "yes or no",
            "letter": "the letter of one listed option",
        }.get(self.answer_format, "a short answer")

    def build_verifier_prompt(self, question: str, answer: str,
                              reasoning: Optional[List[str]] = None) -> str:
        """Ask for a pass/fail verdict on ONE proposed answer.

        reasoning=None is the blind mode: the verifier sees only the question and the answer,
        so it cannot inherit the debate's line of argument. With reasoning it also judges the
        steps offered for the answer. Either way it must solve the problem itself first."""
        shown = ""
        if reasoning:
            steps = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(reasoning))[:2500]
            shown = f"\nReasoning offered for it:\n{steps}\n"
        return (
            "You are an independent verifier. Decide whether the PROPOSED ANSWER to the question "
            "below is correct.\n\n"
            f"Question:\n{question}\n\n"
            f"PROPOSED ANSWER: {answer}\n{shown}\n"
            "Procedure:\n"
            "1. Solve the question yourself, briefly and step by step, BEFORE judging.\n"
            "2. Compare your result with the proposed answer"
            + (" and check each offered step.\n" if reasoning else ".\n")
            + "3. Say 'pass' only if the proposed answer is correct. Being proposed, popular or "
            "confidently stated is not evidence.\n\n"
            "Output ONLY a JSON object:\n"
            '{"check":["my step 1","my step 2"],"own_answer":"<your answer>",'
            '"verdict":"pass or fail","confidence":0.0-1.0,"issue":"one sentence, empty if pass"}\n'
            f'"own_answer" must be {self.answer_hint()}. "verdict" must be exactly "pass" or "fail".'
        )

    def verifier_settings(self) -> dict:
        return {
            "check_arithmetic": True,
            "check_percentage": True,
        }