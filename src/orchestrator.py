

import logging
from typing import Dict, List, Optional

from .agents.solver import SolverAgent
from .agents.critic import CriticAgent
from .communication.message import StructuredMessage
from tasks.base import BenchmarkConfig
from osc.features import canon, slot_view
from osc.policy import OSCPolicy

logger = logging.getLogger(__name__)


class MADOrchestrator:
    """
    2 solvers (independent models) + 1 critic, debating for up to
    `max_rounds` rounds on a single benchmark task.

    stop_policy
      "consensus" : stop when all three agents agree (the old incumbent)
      "none"      : always run max_rounds (use this to collect training data for OSC)
      "osc"       : stop when the OSC policy says the leading answer is safe (see osc/)
    critic_mode
      "frozen"    : the critic's own answer is fixed after round 0 (old behaviour)
      "live"      : the critic re-solves every round after reading the solvers
    audit_rate    : share of questions that run to max_rounds anyway under "osc", logging where
                    the policy would have stopped (checked later by osc.monitor, upgrade N4)
    verifier_config_path : model config of the verifier; needed when the OSC policy carries a
                    certified VERIFY setting (osc/verify.py). At most one verification per question.
    """

    def __init__(
        self,
        task: str,
        solver_a_config_path: str,
        solver_b_config_path: str,
        critic_config_paths: List[str],
        max_rounds: int = 6,
        early_stop_on_consensus: bool = True,
        stop_policy: Optional[str] = None,
        critic_mode: str = "frozen",
        osc_policy_path: Optional[str] = None,
        audit_rate: float = 0.0,
        audit_seed: int = 0,
        verifier_config_path: Optional[str] = None,
    ):
        from tasks import get_benchmark

        self.benchmark: BenchmarkConfig = get_benchmark(task)
        self.max_rounds = max_rounds
        if stop_policy is None:
            stop_policy = "consensus" if early_stop_on_consensus else "none"
        if stop_policy not in ("consensus", "none", "osc"):
            raise ValueError(f"unknown stop_policy {stop_policy!r}")
        if critic_mode not in ("frozen", "live"):
            raise ValueError(f"unknown critic_mode {critic_mode!r}")
        self.stop_policy = stop_policy
        self.early_stop_on_consensus = stop_policy == "consensus"
        self.critic_mode = critic_mode
        self.audit_rate = audit_rate
        self.audit_seed = audit_seed

        self.osc = None
        if stop_policy == "osc":
            if not osc_policy_path:
                raise ValueError("stop_policy='osc' needs osc_policy_path")
            self.osc = OSCPolicy.load(osc_policy_path)
            if self.osc.task != task:
                raise ValueError(f"OSC policy was trained for {self.osc.task}, not {task}")
            if self.osc.max_rounds != max_rounds:
                raise ValueError(f"OSC policy was trained with max_rounds={self.osc.max_rounds}, "
                                 f"not {max_rounds}")

        self.verifier = None
        if self.osc is not None and self.osc.verify is not None:
            if not verifier_config_path:
                raise ValueError("this OSC policy verifies answers: pass verifier_config_path")
            from .agents.verifier import VerifierAgent
            self.verifier = VerifierAgent(verifier_config_path, self.osc.verify.get("mode", "reasoning"))

        self.solver_a_config_path = solver_a_config_path
        self.solver_b_config_path = solver_b_config_path
        # Alternate critic model across samples (round-robin, deterministic).
        self.critic_config_paths = critic_config_paths

        self.solver_a_model = self._peek_model_name(solver_a_config_path)
        self.solver_b_model = self._peek_model_name(solver_b_config_path)
        self.critic_models = [self._peek_model_name(p) for p in critic_config_paths]

    @staticmethod
    def _peek_model_name(config_path: str) -> str:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("model", {}).get("name", "unknown")

    def _pick_critic_config(self, sample_id: int) -> str:
        idx = sample_id % len(self.critic_config_paths)
        return self.critic_config_paths[idx]


    def _is_audited(self, sample_id: int) -> bool:
        """Deterministic per question, so a resumed run audits the same questions."""
        if self.osc is None or self.audit_rate <= 0:
            return False
        import random
        return random.Random(f"{self.audit_seed}|{sample_id}").random() < self.audit_rate

    @staticmethod
    def _reasoning_for(rounds_log: List[dict], answer) -> List[str]:
        """Steps offered when `answer` (canonical form) first appeared: earliest round, then
        solver A, solver B, critic. Same rule as osc.verify_offline, so the verifier sees the
        same input live as in the offline pass that trained and evaluated the policy."""
        for rnd in rounds_log:
            for k in ("solver_a", "solver_b", "critic"):
                ag = rnd.get("agents", {}).get(k)
                if ag and canon(ag.get("normalized_answer")) == answer:
                    return list(ag.get("reasoning") or [])
        return []

    def _maybe_verify(self, question: str, rounds_log: List[dict], dec: dict, state: dict) -> bool:
        """VERIFY action: called once per question at most, when the policy's band says so."""
        if self.verifier is None or state["verify"] is not None:
            return False
        band = self.osc.verify_now(dec)
        if band is None:
            return False
        res = self.verifier.verify(question, dec["top"], self.benchmark,
                                   self._reasoning_for(rounds_log, dec["top"]))
        lr = self.osc.log_lr(res.verdict, band["t1"], band["t0"])
        state["boost"] = (dec["top"], lr)
        state["verify_tokens"] += res.tokens
        state["verify"] = dict(checkpoint=dec["checkpoint"], answer=dec["top"], log_lr=round(lr, 4),
                               **band, **res.to_dict())
        return True

    def _osc_check(self, rounds_log: List[dict], cp: str, state: dict,
                   question: str = "") -> Optional[dict]:
        """Ask the OSC policy at checkpoint cp. Returns its decision if the debate stops now."""
        if self.osc is None or not self.osc.has_checkpoint(cp):
            return None
        is_last = cp == self.osc.checkpoints[-1]
        prefix = [slot_view(r) for r in rounds_log]
        dec = self.osc.assess(prefix, cp, force_stop=is_last, boost=state["boost"])
        if self._maybe_verify(question, rounds_log, dec, state):
            dec = self.osc.assess(prefix, cp, force_stop=is_last, boost=state["boost"])
            dec["verified_here"] = True
        state["checkpoints"].append(dec)
        if not dec["stop"]:
            return None
        if state["audit"] and not is_last:
            # audited question: remember where OSC would have stopped, keep debating
            if state["would_stop_at"] is None:
                state["would_stop_at"] = cp
                state["would_stop_answer"] = dec["top"]
            return None
        return dec

    def run(self, sample_id: int, question: str, ground_truth: Optional[str] = None) -> dict:
        # Fresh agent instances every sample: agents hold per-question state
        # (SolverAgent._r0_action, CriticAgent._independent_answer) that must
        # not leak across questions.
        solver_a = SolverAgent(config_path=self.solver_a_config_path)
        solver_b = SolverAgent(config_path=self.solver_b_config_path)

        critic_config_path = self._pick_critic_config(sample_id)
        critic = CriticAgent(config_path=critic_config_path)
        critic_model = self._peek_model_name(critic_config_path)

        rounds_log: List[dict] = []
        total_tokens = 0

        critic_msgs_for_a: List[StructuredMessage] = []
        critic_msgs_for_b: List[StructuredMessage] = []

        final_round_msgs = None
        osc_state = dict(audit=self._is_audited(sample_id), checkpoints=[],
                         would_stop_at=None, would_stop_answer=None,
                         boost=None, verify=None, verify_tokens=0)
        osc_decision = None

        for round_id in range(self.max_rounds):
            msg_a = solver_a.respond(
                question=question,
                round_id=round_id,
                critic_messages=critic_msgs_for_a if round_id > 0 else None,
                benchmark=self.benchmark,
            )
            total_tokens += msg_a.token_count

            msg_b = solver_b.respond(
                question=question,
                round_id=round_id,
                critic_messages=critic_msgs_for_b if round_id > 0 else None,
                benchmark=self.benchmark,
            )
            total_tokens += msg_b.token_count

            norm_a = self.benchmark.normalize_answer(msg_a.action.strip())
            norm_b = self.benchmark.normalize_answer(msg_b.action.strip())
            agents = {
                "solver_a": {
                    "model": self.solver_a_model,
                    "answer": msg_a.action,
                    "normalized_answer": norm_a,
                    "confidence": msg_a.confidence,
                    "reasoning": list(msg_a.reasoning),
                    "reasoning_text": msg_a.get_reasoning_text(),
                },
                "solver_b": {
                    "model": self.solver_b_model,
                    "answer": msg_b.action,
                    "normalized_answer": norm_b,
                    "confidence": msg_b.confidence,
                    "reasoning": list(msg_b.reasoning),
                    "reasoning_text": msg_b.get_reasoning_text(),
                },
            }
            tokens = {"solver_a": msg_a.token_count, "solver_b": msg_b.token_count}

            # mid-round checkpoint (N2): the solvers have spoken, the critic has not
            if self.osc is not None and self.osc.mid_round:
                partial = {"round_id": round_id, "partial": True,
                           "agents": dict(agents), "tokens": dict(tokens)}
                osc_decision = self._osc_check(rounds_log + [partial], f"r{round_id}s", osc_state,
                                               question)
                if osc_decision is not None:
                    rounds_log.append(partial)
                    break

            # critic: its own answer (round 0, or every round in live mode), then one verdict
            # per solver. The own answer is generated here so its tokens are logged separately.
            tokens_ind = 0
            if round_id == 0 or self.critic_mode == "live":
                tokens_ind = critic.prepare_independent(
                    question, round_id, self.benchmark,
                    peers=(msg_a, msg_b) if round_id > 0 else None,
                )
                total_tokens += tokens_ind

            critic_vs_a = critic.respond(
                question=question, solver_message=msg_a, round_id=round_id, benchmark=self.benchmark
            )
            total_tokens += critic_vs_a.token_count
            critic_vs_b = critic.respond(
                question=question, solver_message=msg_b, round_id=round_id, benchmark=self.benchmark
            )
            total_tokens += critic_vs_b.token_count
            tokens.update(critic_independent=tokens_ind, critic_vs_a=critic_vs_a.token_count,
                          critic_vs_b=critic_vs_b.token_count)

            critic_msgs_for_a.append(critic_vs_a)
            critic_msgs_for_b.append(critic_vs_b)

            critic_ind: StructuredMessage = critic_vs_a.independent_answer  # same for both calls

            norm_c = self.benchmark.normalize_answer(critic_ind.action.strip())

            consensus = norm_a == norm_b == norm_c

            agents["critic"] = {
                "model": critic_model,
                "answer": critic_ind.action,
                "normalized_answer": norm_c,
                "confidence": critic_ind.confidence,
                "reasoning": list(critic_ind.reasoning),
                "reasoning_text": critic_ind.get_reasoning_text(),
                "verdict_vs_solver_a": critic_vs_a.action,
                "verdict_vs_solver_a_confidence": critic_vs_a.confidence,
                "verdict_vs_solver_b": critic_vs_b.action,
                "verdict_vs_solver_b_confidence": critic_vs_b.confidence,
            }
            round_record = {
                "round_id": round_id,
                "agents": agents,
                "consensus": consensus,
                "tokens": tokens,
            }
            rounds_log.append(round_record)

            final_round_msgs = (msg_a, msg_b, critic_ind)

            if consensus and self.stop_policy == "consensus":
                logger.info(f"[Sample {sample_id}] consensus reached at round {round_id} → stop")
                break

            if self.osc is not None:
                osc_decision = self._osc_check(rounds_log, f"r{round_id}f", osc_state, question)
                if osc_decision is not None:
                    logger.info(f"[Sample {sample_id}] OSC stop at {osc_decision['checkpoint']}")
                    break

        if osc_decision is not None and osc_decision["top"] is not None:
            final_answer = osc_decision["top"]
            answer_source = "osc"
        elif final_round_msgs is not None:
            final_answer = self._select_final_answer(final_round_msgs)
            answer_source = "majority"
        else:  # OSC stopped before any full round and had no usable answer
            final_answer = self.benchmark.normalize_answer(msg_a.action.strip())
            answer_source = "solver_a"

        total_tokens += osc_state["verify_tokens"]

        correct = None
        if ground_truth is not None:
            correct = self.benchmark.score(final_answer, ground_truth)

        record = {
            "sample_id": sample_id,
            "task": self.benchmark.task_name,
            "question": question,
            "ground_truth": ground_truth,
            "critic_model": critic_model,
            "solver_a_model": self.solver_a_model,
            "solver_b_model": self.solver_b_model,
            "stop_policy": self.stop_policy,
            "critic_mode": self.critic_mode,
            "rounds": rounds_log,
            "rounds_used": sum(1 for r in rounds_log if not r.get("partial")),
            "final_answer": final_answer,
            "final_answer_source": answer_source,
            "final_correct": correct,
            "total_tokens": total_tokens,
        }
        if self.osc is not None:
            record["osc"] = {
                "stopped_at": osc_decision["checkpoint"] if osc_decision else None,
                "audit": osc_state["audit"],
                "would_stop_at": osc_state["would_stop_at"],
                "would_stop_answer": osc_state["would_stop_answer"],
                "checkpoints": osc_state["checkpoints"],
                "verify": osc_state["verify"],
                "verify_tokens": osc_state["verify_tokens"],
            }
        return record

    def _select_final_answer(self, final_round_msgs) -> str:
        msg_a, msg_b, critic_ind = final_round_msgs
        candidates = [
            (self.benchmark.normalize_answer(msg_a.action.strip()), msg_a.confidence),
            (self.benchmark.normalize_answer(msg_b.action.strip()), msg_b.confidence),
            (self.benchmark.normalize_answer(critic_ind.action.strip()), critic_ind.confidence),
        ]
        votes: Dict[str, int] = {}
        best_conf: Dict[str, float] = {}
        for ans, conf in candidates:
            votes[ans] = votes.get(ans, 0) + 1
            best_conf[ans] = max(best_conf.get(ans, 0.0), conf)

        max_votes = max(votes.values())
        tied = [a for a, v in votes.items() if v == max_votes]
        if len(tied) == 1:
            return tied[0]
        return max(tied, key=lambda a: best_conf[a])
