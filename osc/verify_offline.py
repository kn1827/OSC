"""Run the verifier offline on full-depth debate logs (PLAN_OSC_V.md, task #3).

    python -m osc.verify_offline --logs "results/logs_osc/**/debate_full_*.jsonl" --task gsm8k \
        --verifier_config config/model_config_verifier.yaml --mode reasoning \
        --out results/verify/gsm8k_reasoning.jsonl

Every distinct candidate answer of every question is verified once. The reasoning shown with
it (mode "reasoning") is the one offered when the answer first appeared: earliest round, then
solver A, solver B, critic. Seeds of the same question share the verdict, so the number of
calls is the number of distinct (question, answer) pairs, not debates x checkpoints.
The OSC simulation then looks up the verdict of whichever answer leads at a checkpoint.
Re-running with the same --out resumes: pairs already in the file are skipped.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .features import canon

AGENTS = ("solver_a", "solver_b", "critic")


def collect_pairs(pattern: str, task: str, max_questions: int | None = None) -> list[dict]:
    """One job per (question, canonical answer), with the reasoning of its first appearance."""
    jobs, seen, questions = [], set(), set()
    for f in sorted(glob.glob(pattern, recursive=True)):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("task") != task:
                continue
            sid = int(rec["sample_id"])
            if sid not in questions:
                if max_questions is not None and len(questions) >= max_questions:
                    continue
                questions.add(sid)
            group = f"{task}|{sid}"
            for rnd in rec["rounds"]:
                for k in AGENTS:
                    ag = rnd.get("agents", {}).get(k)
                    if not ag:
                        continue
                    key = (group, canon(ag.get("normalized_answer")))
                    if key[1] is None or key in seen:
                        continue
                    seen.add(key)
                    jobs.append(dict(task=task, group=group, sample_id=sid, answer=key[1],
                                     shown=ag.get("normalized_answer"), question=rec["question"],
                                     reasoning=list(ag.get("reasoning") or []),
                                     ground_truth=rec.get("ground_truth")))
    return jobs


def done_keys(path: str) -> set:
    keys = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                keys.add((r["group"], canon(r["answer"]), r.get("mode")))
    return keys


def run(jobs: list[dict], verifier, benchmark, out_path: str, workers: int = 1) -> int:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    lock = threading.Lock()
    n_done = 0

    def one(job):
        res = verifier.verify(job["question"], job["shown"], benchmark, job["reasoning"])
        rec = dict(task=job["task"], group=job["group"], sample_id=job["sample_id"],
                   answer=job["answer"], shown=job["shown"], **res.to_dict())
        if job["ground_truth"] is not None:
            rec["correct"] = bool(benchmark.score(str(job["shown"]), str(job["ground_truth"])))
        return rec

    with open(out_path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, j) for j in jobs]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001 — one failed call must not stop the run
                print(f"  verifier call failed: {e}")
                continue
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                n_done += 1
                if n_done % 50 == 0:
                    print(f"  {n_done}/{len(jobs)} verified", flush=True)
    return n_done


def main():
    from src.agents.verifier import MODES, VerifierAgent
    from tasks import get_benchmark

    ap = argparse.ArgumentParser(description="Offline verifier pass over full-depth debate logs")
    ap.add_argument("--logs", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--verifier_config", default="config/model_config_verifier.yaml")
    ap.add_argument("--mode", choices=MODES, default="reasoning")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_questions", type=int, default=None,
                    help="only the first N questions (pilot for the go/no-go gate G1)")
    ap.add_argument("--workers", type=int, default=1, help="parallel calls (API providers)")
    args = ap.parse_args()

    benchmark = get_benchmark(args.task)
    jobs = collect_pairs(args.logs, args.task, args.max_questions)
    done = done_keys(args.out)
    todo = [j for j in jobs if (j["group"], j["answer"], args.mode) not in done]
    print(f"{args.task}: {len(jobs)} (question, answer) pairs, {len(jobs) - len(todo)} already "
          f"verified, {len(todo)} to go (mode={args.mode})")
    n = run(todo, VerifierAgent(args.verifier_config, args.mode), benchmark, args.out, args.workers)
    print(f"done: {n} new verdicts -> {args.out}")


if __name__ == "__main__":
    main()
