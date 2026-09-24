"""
tasks/truthfulqa/gen_truthfulqa.py
Download TruthfulQA (multiple_choice, MC1 targets) and save data/truthfulqa/test.json.
Each question keeps the correct answer plus up to 3 wrong ones; the options are shuffled with
a per-question seed, so re-running the script gives the same file.

Usage (from the repo root):
    python -m tasks.truthfulqa.gen_truthfulqa
    python -m tasks.truthfulqa.gen_truthfulqa --n_wrong 4
"""

import argparse
import json
import os
import random
from collections import Counter

from tasks.hf_download import load_rows

LETTERS = "ABCDEFGHIJKLMNOP"


def build_sample(row: dict, idx: int, n_wrong: int, seed: int) -> dict:
    choices, labels = row["mc1_targets"]["choices"], row["mc1_targets"]["labels"]
    correct = choices[labels.index(1)]
    wrong = [c for c, l in zip(choices, labels) if l == 0][:n_wrong]
    opts = [correct] + wrong
    random.Random(f"{seed}|{idx}").shuffle(opts)
    letter = LETTERS[opts.index(correct)]
    body = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(opts))
    return {"question": f"{row['question'].strip()}\n{body}", "answer": letter,
            "category": row.get("category", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_wrong", type=int, default=3, help="wrong options kept per question")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/truthfulqa/test.json")
    args = ap.parse_args()

    rows = load_rows("truthfulqa/truthful_qa", "multiple_choice", "validation")
    samples = [build_sample(r, i, args.n_wrong, args.seed) for i, r in enumerate(rows)]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(samples)} samples to {args.output}")
    print("Answer distribution:", dict(sorted(Counter(s["answer"] for s in samples).items())))


if __name__ == "__main__":
    main()
