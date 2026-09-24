"""baselines/no_debate.py — baseline KHONG debate, chay qua MAD.py (truoc day la mv_baseline.py).

Ca hai deu: tra loi DOC LAP (khong thay cau tra loi cua nhau) roi bo phieu da so;
hoa phieu thi dap an co confidence cao nhat thang — dung quy tac
MADOrchestrator._select_final_answer dung de chot debate.

  (A) ENSEMBLE VOTE     --majority_voting [--mv_votes K]
      3 model KHAC NHAU, K mau moi model (mac dinh K=1 -> 3 phieu).
      -> results/logs_mv/<task>/seed<seed>/majority_vote_*.jsonl   mode="majority_voting"
      Do gia tri cua DA DANG MODEL.

  (B) SELF-CONSISTENCY  --majority_voting --sc_model {a,b,c} --mv_votes 3
      MOT model, 3 mau tu chinh no. Day la self-consistency@3 THAT (Wang et al. 2022).
      -> results/logs_sc/agent_<x>/<task>/seed<seed>/self_consistency_*.jsonl
                                                                  mode="self_consistency"
      Do gia tri cua DA DANG MAU.

  (C) SINGLE AGENT      --majority_voting --sc_model {a,b,c} --mv_votes 1
      MOT model, MOT lan goi (chain-of-thought truc tiep, khong debate, khong bo phieu).
      -> results/logs_sc/agent_<x>/<task>/seed<seed>/single_agent_*.jsonl   mode="single_agent"
      Diem re nhat cua duong cost-accuracy.

Hai che do dung CHUNG mot duong chay (run_one) — khac biet duy nhat la tap voter,
nen chenh lech giua chung quy ve dung mot bien.

BA AGENT NGANG QUYEN O CHE DO (A). Khong con vai "critic": ca ba chay o vai
"solver" nen cung temperature 0.3 va cung max_tokens 1024, cung mot prompt, moi
agent mot phieu. Luu y hieu ung: Gemma3 trong debate chay o vai critic (0.2 / 768),
o day no chay 0.3 / 1024 nhu hai model kia — do la cai gia cua viec de ba agent
ngang quyen, va la dinh nghia dung cua majority voting.

Log baseline nam o results/logs_mv/ va results/logs_sc/, ten file khac debate_full_*,
nen osc.data.load_debates khong bao gio tron chung vao phan tich debate.

CANH BAO NHAN — fixed_k1 trong paper KHONG phai ca (A) lan (B).
fixed_k1 la "vong 0 cua debate", suy ra offline tu log debate, va no khac ca hai:
  - Bat doi xung: 2 solver dung build_solver_prompt @ 0.3/1024, con voter thu ba
    la critic dung build_independent_prompt @ 0.2/768.
  - Khong dong nhat giua cac benchmark: rieng GSM8K, solver.py bat self-consistency
    NOI BO n=3 o round 0 (src/agents/solver.py, use_consistency), nen fixed_k1 tren
    GSM8K la ~7 lan sinh chu khong phai 3; MMLU/StrategyQA la 3.
  - Chi phi khong do duoc: log debate chi co total_tokens/cau, tok_per_round=total/6
    la XAP XI chia deu va no sai nang o GSM8K vong 0.
Ca (A) va (B) deu do total_tokens truc tiep, nen dung chung de bao cao chi phi.
Trong paper phai goi fixed_k1 la "round-0 vote (2 solvers + critic, no cross-talk)",
KHONG duoc goi la self-consistency@3.

SEED: moi phieu dat agent.seed (khong doi bien moi truong MAD_SEED), nen chay song song
nhieu cau hoi (MAD.py --workers) van tai lap duoc va khong ghi de seed cua nhau.

XOA BASELINE NAY khi khong dung nua:
  1. xoa file nay
  2. trong MAD.py: xoa argument group "no-debate baselines" va nhanh
     `if args.majority_voting:` trong main()
  3. xoa thu muc results/logs_mv/ va results/logs_sc/ neu da chay
Khong co gi trong src/ hay orchestrator bi dong den.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional

from src.agents.base_agent import BaseAgent

logger = logging.getLogger("baselines.no_debate")

# Vai dung cho CA BA voter. Doi thanh "critic" thi ca ba cung chay 0.2/768 — van
# ngang quyen; cai khong duoc phep la moi agent mot vai khac nhau.
VOTER_ROLE = "solver"


# ----------------------------------------------------------------- vote helpers
def _peek_model_name(config_path: str) -> str:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("model", {}).get("name", "unknown")


def _valid_action(action: str, fmt: str) -> bool:
    """Ban sao cua CriticAgent._validate_action - PHAI giu dong bo voi ham do.

    Luu y: fmt "letter" (MMLU) roi vao `return True`, tuc KHONG kiem tra gi. Do la
    lo hong that (Section VI-C cua paper: 6.30% dap an MMLU nam ngoai A-D), nhung
    CriticAgent cung dung y het nhu vay. Neu sua o day ma khong sua ben kia thi
    nhanh baseline duoc mot duong sua loi ma nhanh debate khong co, va hai nhanh
    thoi khong con so sanh duoc voi nhau nua - do la ly do de nguyen.
    """
    if fmt == "yes or no":
        return action in ("yes", "no")
    if fmt == "number":
        try:
            float(action)
            return True
        except (TypeError, ValueError):
            return False
    return True


def _independent_answer(agent: BaseAgent, question: str, benchmark, round_id: int = 0):
    """Mot cau tra loi doc lap tu `agent`, khong co ngu canh debate nao.

    Lap lai dung chuoi kiem tra ma CriticAgent._generate_independent_answer dung
    (non-answer -> compact retry, sai dinh dang -> retry co rang buoc, van sai ->
    trich bang regex) de baseline khong bi thiet chi vi parse hong nhieu hon nhanh
    debate. Khong goi thang ham cua CriticAgent vi ham do gan chat voi trang thai
    va vai tro "critic"; o day 2 solver phai giu temperature cua chinh chung.
    """
    from src.agents.critic import _CALIBRATION_SUFFIX
    from src.communication.protocol import (build_compact_prompt, looks_like_non_answer,
                                            parse_agent_response)

    fmt = benchmark.answer_format
    parse_role = benchmark.solver_role
    prompt = benchmark.build_independent_prompt(question) + _CALIBRATION_SUFFIX

    raw, tokens = agent.call_llm(prompt)
    msg = parse_agent_response(raw, role=parse_role, round_id=round_id)

    retries = 0
    while looks_like_non_answer(msg) and retries < 2:
        retries += 1
        logger.warning("[vote %s] Non-answer/refusal — compact retry %d", agent.role, retries)
        raw, tokens = agent.call_llm(build_compact_prompt(question, fmt))
        msg = parse_agent_response(raw, role=parse_role, round_id=round_id)

    if not _valid_action(msg.action, fmt):
        logger.warning("[vote %s] Invalid action '%s' — retry", agent.role, msg.action)
        suffix = ("\nYou MUST output a JSON with 'action' set to the final numeric answer "
                  "only (e.g. 42). No units, no text." if fmt == "number"
                  else f"\nYou MUST answer strictly with {fmt}.")
        raw, tokens = agent.call_llm(prompt + suffix)
        msg = parse_agent_response(raw, role=parse_role, round_id=round_id)

        if not _valid_action(msg.action, fmt):
            logger.error("[vote %s] Still invalid '%s' — regex fallback", agent.role, msg.action)
            if fmt == "number":
                m = re.search(r"-?\d+\.?\d*", raw)
                msg.action = m.group(0) if m else "0"
            else:
                m = re.search(r"\b(yes|no)\b", raw, re.IGNORECASE)
                msg.action = m.group(1).lower() if m else "yes"
            msg.confidence = 0.3

    return msg, tokens


def _majority_vote(candidates: List[tuple]) -> tuple:
    """(dap an thang, {dap an: so phieu}, cach quyet dinh).

    Hoa phieu -> confidence cao nhat thang. Cung quy tac voi
    MADOrchestrator._select_final_answer.

    Canh bao ve "ngang quyen": neu hoa phieu MA confidence cung bang nhau (rat de
    xay ra vi model hay tu cham 0.9/0.8 tron), max() lay phan tu dau tien, tuc la
    agent_a duoc uu the vi tri. Truong hop do duoc danh dau "first_voter" trong
    log de dem duoc no chiem bao nhieu — dung ngam bo qua.
    """
    votes: dict = {}
    best_conf: dict = {}
    for ans, conf in candidates:
        votes[ans] = votes.get(ans, 0) + 1
        best_conf[ans] = max(best_conf.get(ans, 0.0), conf)

    max_votes = max(votes.values())
    tied = [a for a, v in votes.items() if v == max_votes]
    if len(tied) == 1:
        return tied[0], votes, "majority"

    top_conf = max(best_conf[a] for a in tied)
    still_tied = [a for a in tied if best_conf[a] == top_conf]
    winner = max(tied, key=lambda a: best_conf[a])
    return winner, votes, ("confidence" if len(still_tied) == 1 else "first_voter")


# ------------------------------------------------------------------- one sample
def run_one(task_benchmark, sample_id: int, question: str, voters: List[tuple],
            ground_truth: Optional[str] = None, n_votes: int = 1,
            mode: str = "majority_voting") -> dict:
    """Mot cau hoi: moi voter tra loi n_votes lan doc lap, roi bo phieu da so.

    `voters`: list (ten, duong_dan_config, ten_model). Khong co truong vai tro —
    ca ba deu chay o VOTER_ROLE, do la y nghia cua "ngang quyen".

    `mode` chi la NHAN ghi vao log, khong doi hanh vi:
      - "majority_voting"  : len(voters) == 3, n_votes mau moi model (ensemble vote)
      - "self_consistency" : len(voters) == 1, n_votes mau tu CUNG mot model
    Hai che do dung chung dung mot duong chay -> khac biet duy nhat la tap voter.
    """
    votes_log: List[dict] = []
    candidates: List[tuple] = []
    total_tokens = 0

    # Cung model + cung prompt + cung seed => cung output, nen neu moi phieu dung
    # cung seed thi n_votes phieu cua mot model se GIONG HET nhau va --mv_votes vo
    # nghia. Phieu k=0 giu dung seed goc (nen --mv_votes 1 tai lap y het mot lan chay
    # thuong); k>0 lay seed dan xuat, van xac dinh => van tai lap duoc. Seed dat tren
    # agent (agent.seed), khong dong vao bien moi truong -> an toan khi chay song song.
    base_seed = int(os.environ.get("MAD_SEED", "0"))
    for name, config_path, model_name in voters:
        # role="solver" cho CA BA: ba agent ngang quyen thi phai cung
        # temperature/max_tokens. Khong co voter nao o vai "critic" o day.
        agent = BaseAgent(role=VOTER_ROLE, config_path=config_path)
        for k in range(n_votes):
            vote_seed = base_seed if k == 0 else base_seed * 1000 + k
            agent.seed = vote_seed
            msg, tokens = _independent_answer(agent, question, task_benchmark, k)
            total_tokens += tokens
            norm = task_benchmark.normalize_answer(msg.action.strip())
            candidates.append((norm, msg.confidence))
            votes_log.append({
                "agent": name,
                "model": model_name,
                "vote_id": k,
                "vote_seed": vote_seed,
                "answer": msg.action,
                "normalized_answer": norm,
                "confidence": msg.confidence,
                "reasoning": list(msg.reasoning),
                "reasoning_text": msg.get_reasoning_text(),
                "tokens": tokens,
            })

    final_answer, vote_counts, decided_by = _majority_vote(candidates)

    correct = None
    if ground_truth is not None:
        correct = task_benchmark.score(final_answer, ground_truth)

    return {
        "sample_id": sample_id,
        "task": task_benchmark.task_name,
        "mode": mode,
        "question": question,
        "ground_truth": ground_truth,
        "n_votes_per_model": n_votes,
        "n_votes_total": len(candidates),
        "votes": votes_log,
        "vote_counts": vote_counts,
        "decided_by": decided_by,      # majority | confidence | first_voter
        "final_answer": final_answer,
        "final_correct": correct,
        "total_tokens": total_tokens,
    }


# --------------------------------------------------------------------- one task
def _completed_ids(out_dir: Path, prefix: str = "majority_vote") -> set:
    """sample_id da chay, doc moi <prefix>_*.jsonl trong dung seed nay."""
    done = set()
    if not out_dir.exists():
        return done
    for f in out_dir.glob(f"{prefix}_*.jsonl"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        done.add(json.loads(line)["sample_id"])
        except Exception as e:
            logger.warning(f"Cannot read {f}: {e}")
    return done


def run_majority_vote_task(task: str, n: int, seed: int, resume: bool, n_votes: int,
                           load_samples: Callable, configs: tuple,
                           sc_model: Optional[str] = None, workers: int = 1) -> None:
    """Chay baseline tren mot benchmark.

    `load_samples` va `configs` duoc MAD.py truyen vao, khong copy lai o day: voi
    cung (task, n, seed) baseline phai chay tren DUNG tap cau hoi cua debate thi
    so sanh moi la ghep cap theo sample_id.

    `sc_model` = None  -> ENSEMBLE VOTE: 3 model khac nhau, n_votes mau moi model.
                          Log vao results/logs_mv/, mode="majority_voting".
    `sc_model` in {a,b,c} -> SELF-CONSISTENCY THAT: CHI mot model, n_votes mau tu
                          chinh no. Log vao results/logs_sc/<model>/, mode=
                          "self_consistency". Day la baseline (iii) that su, tach
                          hoan toan khoi ensemble vote.
    Duong chay giong het nhau; khac biet duy nhat la tap voter, nen chenh lech do
    duoc quy ve dung mot bien: da dang model hay da dang mau.
    """
    from tasks import get_benchmark

    if n_votes < 1:
        raise SystemExit("--mv_votes must be >= 1")
    if sc_model is not None:
        if sc_model not in ("a", "b", "c"):
            raise SystemExit("--sc_model must be one of: a, b, c")

    solver_a_config, solver_b_config, critic_configs = configs
    benchmark = get_benchmark(task)

    # sc_model + 1 vote = single agent (one call, no vote): logged under its own name
    mode = ("single_agent" if sc_model and n_votes == 1
            else "self_consistency" if sc_model else "majority_voting")
    # QUAN TRONG cho SC-k sweep: n_votes (K) phai nam trong PREFIX, khong chi trong
    # noi dung record. Neu khong, chay --mv_votes 3 roi --mv_votes 7 tren CUNG
    # (task, seed, sc_model) se ghi vao CUNG thu muc va _completed_ids() se coi cac
    # sample_id da chay o K=3 la "da xong" khi ban dinh chay K=7 - mat het du lieu K=7
    # (hoac tron lan hai K trong cung file khi glob "self_consistency_*.jsonl" o cac
    # script phan tich khac). Encode K vao ten file tach hoan toan cac lan sweep.
    prefix = ("single_agent" if mode == "single_agent"
              else f"self_consistency_k{n_votes}" if sc_model else f"majority_vote_k{n_votes}")
    tag = (f"SINGLE-{sc_model}" if mode == "single_agent"
           else f"SC-{sc_model}-k{n_votes}" if sc_model else f"MV-k{n_votes}")

    os.environ["MAD_SEED"] = str(seed)
    out_dir = (Path(f"results/logs_sc/agent_{sc_model}/{task}/seed{seed}") if sc_model
               else Path(f"results/logs_mv/{task}/seed{seed}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"{prefix}_{timestamp}.jsonl"

    completed = _completed_ids(out_dir, prefix) if resume else set()
    if completed:
        logger.warning(f"[{task}] --resume: skipping {len(completed)} sample_id that have already been run")

    samples = [(i, s) for i, s in load_samples(task, n, seed) if i not in completed]

    n_correct, n_done = 0, 0

    # Ba voter ngang quyen -> ten trung tinh agent_a/b/c, khong con "solver"/"critic".
    # Model thu ba van chon theo sample_id giong _pick_critic_config cua debate, de
    # voi cung sample_id hai nhanh dung dung bo model nhu nhau.
    def voters_for(sample_id):
        third_config = critic_configs[sample_id % len(critic_configs)]
        all_voters = [
            ("agent_a", solver_a_config, _peek_model_name(solver_a_config)),
            ("agent_b", solver_b_config, _peek_model_name(solver_b_config)),
            ("agent_c", third_config, _peek_model_name(third_config)),
        ]
        # self-consistency / single agent: giu DUNG mot voter. agent_c van xoay theo
        # sample_id giong debate, de cung sample_id thi ca hai nhanh dung cung model.
        return [all_voters["abc".index(sc_model)]] if sc_model else all_voters

    def one(sample_id, sample):
        t0 = time.time()
        record = run_one(benchmark, sample_id, sample["question"], voters_for(sample_id),
                         ground_truth=sample.get("answer"), n_votes=n_votes, mode=mode)
        record["seed"] = seed
        return record, time.time() - t0

    # Ba voter ngang quyen -> ten trung tinh agent_a/b/c. Model thu ba van chon theo
    # sample_id giong _pick_critic_config cua debate.
    with open(jsonl_path, "a", encoding="utf-8") as jf, \
            ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(one, sid, smp): sid for sid, smp in samples}
        for i, fut in enumerate(as_completed(futures), 1):
            sample_id = futures[fut]
            try:
                record, dt = fut.result()
            except Exception as e:
                logger.error(f"[{task}] Sample {sample_id} lỗi: {e} → bỏ qua")
                continue
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            jf.flush()

            n_done += 1
            if record["final_correct"]:
                n_correct += 1

            acc = n_correct / n_done * 100
            top = max(record["vote_counts"].values())
            print(
                f"[{task}|{tag}] {i}/{len(samples)} (id={sample_id}) "
                f"votes={top}/{record['n_votes_total']} correct={record['final_correct']} "
                f"acc_so_far={acc:.1f}% ({dt:.1f}s)"
            )

    print(f"\n[{task}|{tag}] DONE — {n_done} câu, accuracy={n_correct}/{n_done} "
          f"({(n_correct/n_done*100 if n_done else 0):.1f}%)")
    print(f"  full log : {jsonl_path}")