"""
MAD.py

Chạy thí nghiệm debate THẬT: 2 solver (Qwen + Llama, local qua Ollama) +
1 critic (Gemma3, local qua Ollama), trên N câu hỏi của
1 hoặc nhiều benchmark (gsm8k, strategyqa, mmlu, commonsenseqa, truthfulqa, bbh),
tối đa n vòng mỗi câu.

Output cho mỗi benchmark, trong <log_root>/<task>/seed<seed>/:
  debate_full_<timestamp>.jsonl
      1 dòng JSON / câu hỏi — TOÀN BỘ debate: mọi vòng, mọi agent (answer,
      normalized_answer, confidence, reasoning, verdict của critic), consensus và
      token theo từng lượt gọi mỗi vòng, final_answer, final_correct, total_tokens,
      ground_truth, seed. osc/ đọc thẳng file này (osc.data.load_debates).

Usage:
    python MAD.py --task gsm8k --n 100 --max_rounds 6
    python MAD.py --task all --n 100 --max_rounds 6
    python MAD.py --task strategyqa --n 100 --resume

OSC workflow (see README.md):
    1. collect full debates:  python MAD.py --task all --n 300 --seed 1 --stop none --log_root results/logs_osc
    2. evaluate + train:      python -m osc.evaluate / python -m osc.train
    3. run with OSC:          python MAD.py --task gsm8k --stop osc --osc_policy results/osc/policy_{task}.json
    4. re-check guarantee:    python -m osc.monitor
    VERIFY action (PLAN_OSC_V.md): after step 1 run python -m osc.verify_offline, then train with
    python -m osc.train --verify_logs ...; step 3 then also calls --verifier_config when needed.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.orchestrator import MADOrchestrator

logging.basicConfig(
    level=logging.WARNING,  # đặt INFO nếu muốn xem chi tiết từng agent/round
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("run_multi_agent_debate")

SOLVER_A_CONFIG = "config/model_config_solver_qwen.yaml"   # Solver A = Qwen
SOLVER_B_CONFIG = "config/model_config_solver_llama.yaml"  # Solver B = Llama
CRITIC_CONFIG  = "config/model_config_critic_gemma.yaml"   # Critic = Gemma3
CRITIC_CONFIGS = [CRITIC_CONFIG]                            

ALL_TASKS = ["gsm8k", "strategyqa", "mmlu", "commonsenseqa", "truthfulqa", "bbh"]


def load_samples(task: str, n: int, seed: int) -> list:
    data_path = Path(f"data/{task}/test.json")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if n < len(data):
        rng = random.Random(seed)
        idxs = sorted(rng.sample(range(len(data)), n))
    else:
        if n > len(data):
            logger.warning(
                f"[{task}] request n={n} but only {len(data)} samples in data/{task}/test.json "
                f"→ running all {len(data)} samples."
            )
        idxs = list(range(len(data)))
    return [(i, data[i]) for i in idxs]


def load_completed_ids(jsonl_path: Path) -> set:
    """Đọc các file jsonl cùng task đã có (mọi timestamp) để hỗ trợ --resume."""
    done = set()
    if not jsonl_path.parent.exists():
        return done
    for f in jsonl_path.parent.glob("debate_full_*.jsonl"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    done.add(rec["sample_id"])
        except Exception as e:
            logger.warning(f"Cannot read {f}: {e}")
    return done


def run_task(task: str, n: int, max_rounds: int, seed: int, resume: bool, stop_policy: str,
             critic_mode: str = "frozen", osc_policy: str | None = None, audit_rate: float = 0.0,
             log_root: str = "results/logs", verifier_config: str | None = None,
             configs: tuple | None = None, workers: int = 1):
    # base_agent._call_ollama doc MAD_SEED de seed sampling => moi seed = 1 rollout doc lap, tai lap duoc.
    os.environ["MAD_SEED"] = str(seed)
    # Tach log theo seed: khong de 2 seed ghi de nhau, va tranh dung sample_id
    # (sample_id = index cau hoi) va cham nhau khi phan tich. --resume cung tu dong
    # gioi han trong dung seed nay (load_completed_ids doc theo jsonl_path.parent).
    out_dir = Path(f"{log_root}/{task}/seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"debate_full_{timestamp}.jsonl"

    completed = load_completed_ids(jsonl_path) if resume else set()
    if completed:
        logger.warning(f"[{task}] --resume: skipping {len(completed)} sample_id that have already been run")

    samples = load_samples(task, n, seed)
    samples = [(i, s) for i, s in samples if i not in completed]

    solver_a_cfg, solver_b_cfg, critic_cfgs = configs or (SOLVER_A_CONFIG, SOLVER_B_CONFIG, CRITIC_CONFIGS)
    orchestrator = MADOrchestrator(
        task=task,
        solver_a_config_path=solver_a_cfg,
        solver_b_config_path=solver_b_cfg,
        critic_config_paths=critic_cfgs,
        max_rounds=max_rounds,
        stop_policy=stop_policy,
        critic_mode=critic_mode,
        osc_policy_path=osc_policy.format(task=task) if osc_policy else None,
        audit_rate=audit_rate,
        audit_seed=seed,
        verifier_config_path=verifier_config,
    )

    n_correct, n_done = 0, 0

    def one(sample_id, sample):
        t0 = time.time()
        record = orchestrator.run(sample_id=sample_id, question=sample["question"],
                                  ground_truth=sample.get("answer"))
        record["seed"] = seed
        return record, time.time() - t0

    # --workers > 1: several questions in flight at once (an OpenAI-compatible server such as
    # vLLM batches them). Each question gets fresh agents, so questions never share state;
    # records are written as they finish, so the file order differs from the question order.
    with open(jsonl_path, "a", encoding="utf-8") as jf, \
            ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(one, sid, s): sid for sid, s in samples}
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
            print(
                f"[{task}] {i}/{len(samples)} (id={sample_id}) "
                f"rounds={record['rounds_used']} correct={record['final_correct']} "
                f"acc_so_far={acc:.1f}% ({dt:.1f}s)"
            )

    print(f"\n[{task}] DONE — {n_done} câu, accuracy={n_correct}/{n_done} "
          f"({(n_correct/n_done*100 if n_done else 0):.1f}%)")
    print(f"  full log : {jsonl_path}")
    if samples and n_done == 0:
        # every question failed (e.g. a model server died mid-run): stop instead of moving on
        # to the next task with the same broken setup
        raise RuntimeError(f"[{task}] all {len(samples)} questions failed — see the errors above")


def main():
    p = argparse.ArgumentParser(description="2-solver + 1-critic debate runner")
    p.add_argument("--task", choices=ALL_TASKS + ["all"], default="all")
    p.add_argument("--n", type=int, default=100, help="Number of questions per benchmark")
    p.add_argument("--max_rounds", type=int, default=6, help="Maximum debate rounds per question")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling n questions")
    p.add_argument("--resume", action="store_true", help="Skip sample_id that have already been run (read from existing jsonl)")
    p.add_argument("--no_early_stop", action="store_true",
                    help="Same as --stop none (kept for old commands)")
    p.add_argument("--stop", choices=["consensus", "none", "osc"], default=None,
                   help="consensus = stop when all 3 agree (old default); none = always run "
                        "max_rounds (collect OSC training data); osc = OSC policy (needs --osc_policy)")
    p.add_argument("--critic_mode", choices=["frozen", "live"], default="live",
                   help="live (default) = critic re-solves every round; frozen = old behaviour, "
                        "critic answer fixed after round 0")
    p.add_argument("--osc_policy", default=None,
                   help="policy json from `python -m osc.train`; '{task}' is replaced, e.g. "
                        "results/osc/policy_{task}.json")
    p.add_argument("--audit_rate", type=float, default=0.05,
                   help="with --stop osc: share of questions run to max_rounds to re-check the "
                        "guarantee (osc.monitor)")
    p.add_argument("--log_root", default="results/logs",
                   help="where debate logs go; use a new folder per model setup, e.g. results/logs_osc")
    p.add_argument("--verifier_config", default="config/model_config_verifier.yaml",
                   help="verifier model, used only when the OSC policy has a certified VERIFY "
                        "setting (python -m osc.train --verify_logs ...)")
    p.add_argument("--workers", type=int, default=1,
                   help="questions run in parallel (use >1 with a vLLM / API backend; Ollama "
                        "serves one request at a time unless OLLAMA_NUM_PARALLEL is set)")
    p.add_argument("--solver_a_config", default=SOLVER_A_CONFIG)
    p.add_argument("--solver_b_config", default=SOLVER_B_CONFIG)
    p.add_argument("--critic_config", nargs="+", default=CRITIC_CONFIGS,
                   help="one or more critic configs (rotated by sample_id)")
    p.add_argument("--verbose", action="store_true", help="Log detailed information for each agent/round")

    # --- no-debate baselines
    mv = p.add_argument_group("no-debate baselines")
    mv.add_argument("--majority_voting", action="store_true",
                    help="Run a no-debate baseline instead: models answer independently, "
                         "then a majority vote decides. Logs to results/logs_mv/ (or "
                         "results/logs_sc/ with --sc_model), never mixed into the debate logs")
    mv.add_argument("--mv_votes", type=int, default=1, metavar="K",
                    help="Independent samples per model (default 1). Without --sc_model this "
                         "gives 3K votes from 3 models; with --sc_model it gives K votes from one")
    mv.add_argument("--sc_model", choices=["a", "b", "c"], default=None,
                    help="Switch to TRUE self-consistency: use only this one voter and take "
                         "--mv_votes samples from it (needs --mv_votes >= 2). a=solver_a, "
                         "b=solver_b, c=critic-slot model (rotates by sample_id as in debate)")
    # --- end no-debate baselines

    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    tasks = ALL_TASKS if args.task == "all" else [args.task]
    configs = (args.solver_a_config, args.solver_b_config, list(args.critic_config))
    from src.agents.base_agent import check_vllm_server
    for cfg in [args.solver_a_config, args.solver_b_config, *args.critic_config]:
        check_vllm_server(cfg)
    for task in tasks:
        if args.majority_voting:
            from baselines.no_debate import run_majority_vote_task
            run_majority_vote_task(
                task=task, n=args.n, seed=args.seed, resume=args.resume,
                n_votes=args.mv_votes, load_samples=load_samples,
                configs=configs, sc_model=args.sc_model, workers=args.workers,
            )
            continue
        stop = args.stop or ("none" if args.no_early_stop else "consensus")
        if stop == "osc" and not args.osc_policy:
            p.error("--stop osc needs --osc_policy")
        run_task(
            task=task,
            n=args.n,
            max_rounds=args.max_rounds,
            seed=args.seed,
            resume=args.resume,
            stop_policy=stop,
            critic_mode=args.critic_mode,
            osc_policy=args.osc_policy,
            audit_rate=args.audit_rate if stop == "osc" else 0.0,
            log_root=args.log_root,
            verifier_config=args.verifier_config,
            configs=configs,
            workers=args.workers,
        )


if __name__ == "__main__":
    sys.exit(main())
