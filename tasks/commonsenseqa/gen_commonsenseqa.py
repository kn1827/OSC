"""
tasks/commonsenseqa/gen_commonsenseqa.py
Download CommonsenseQA and save data/commonsenseqa/test.json.
The official test split has no labels, so the validation split (1,221 questions) is used.

Usage (from the repo root):
    python -m tasks.commonsenseqa.gen_commonsenseqa
    python -m tasks.commonsenseqa.gen_commonsenseqa --n 600
"""

import argparse
import json
import os
from collections import Counter

from tasks.hf_download import load_rows


def build_sample(row: dict) -> dict:
    labels, texts = row["choices"]["label"], row["choices"]["text"]
    opts = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
    return {"question": f"{row['question'].strip()}\n{opts}",
            "answer": row["answerKey"].strip().upper(),
            "concept": row.get("question_concept", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["train", "validation"])
    ap.add_argument("--n", type=int, default=None, help="keep the first n questions")
    ap.add_argument("--output", default="data/commonsenseqa/test.json")
    args = ap.parse_args()

    rows = load_rows("tau/commonsense_qa", "default", args.split)
    samples = [build_sample(r) for r in rows if str(r.get("answerKey", "")).strip()]
    if args.n:
        samples = samples[: args.n]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(samples)} samples to {args.output}")
    print("Answer distribution:", dict(sorted(Counter(s["answer"] for s in samples).items())))


if __name__ == "__main__":
    main()
