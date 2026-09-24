"""
tasks/bbh/gen_bbh.py
Download BIG-Bench Hard and save data/bbh/test.json as option-letter questions.

23 sub-tasks: 17 already multiple-choice, 6 binary ones (Yes/No, True/False, Valid/Invalid)
turned into (A)/(B). Free-text sub-tasks (object_counting, word_sorting, dyck_languages,
multistep_arithmetic_two) are left out because they have no options.
Questions are sampled evenly across sub-tasks with a fixed seed.

Usage (from the repo root):
    python -m tasks.bbh.gen_bbh                   # 1,200 questions, ~52 per sub-task
    python -m tasks.bbh.gen_bbh --n_total 2300
"""

import argparse
import json
import os
import random
import re
from collections import Counter

from tasks.hf_download import load_rows

BINARY_TASKS = {
    "causal_judgement": ("Yes", "No"),
    "navigate": ("Yes", "No"),
    "sports_understanding": ("Yes", "No"),
    "web_of_lies": ("Yes", "No"),
    "boolean_expressions": ("True", "False"),
    "formal_fallacies": ("Valid", "Invalid"),
}

MC_TASKS = [
    "date_understanding", "disambiguation_qa", "geometric_shapes", "hyperbaton",
    "logical_deduction_three_objects", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "movie_recommendation", "penguins_in_a_table",
    "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection",
    "snarks", "temporal_sequences", "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
]
ALL_TASKS = MC_TASKS + list(BINARY_TASKS)


def build_sample(row: dict, task: str):
    text, target = row["input"].strip(), row["target"].strip()
    if task in BINARY_TASKS:
        yes, no = BINARY_TASKS[task]
        t = target.lower()
        letter = "A" if t == yes.lower() else "B" if t == no.lower() else None
        if letter is None:
            return None
        return {"question": f"[Task: {task}]\n{text}\n(A) {yes}\n(B) {no}", "answer": letter, "task": task}
    m = re.match(r"\(([A-R])\)", target)
    if m:
        letter = m.group(1)
    else:  # target given as option text
        opts = re.findall(r"\(([A-R])\)\s*([^\n(]*)", text)
        letter = next((l for l, o in opts if o.strip().lower() == target.lower()), None)
        if letter is None:
            return None
    return {"question": f"[Task: {task}]\n{text}", "answer": letter, "task": task}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_total", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="data/bbh/test.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pools = {}
    for task in ALL_TASKS:
        rows = [s for s in (build_sample(r, task) for r in load_rows("lukaemon/bbh", task, "test")) if s]
        rng.shuffle(rows)
        pools[task] = rows
        print(f"  {task:<45} {len(rows)} usable")

    # even quota; a sub-task that runs out hands its spare quota to the others
    quota = {t: 0 for t in ALL_TASKS}
    left = args.n_total
    while left > 0:
        open_tasks = [t for t in ALL_TASKS if quota[t] < len(pools[t])]
        if not open_tasks:
            break
        for t in open_tasks:
            if left == 0:
                break
            quota[t] += 1
            left -= 1
    samples = [s for t in ALL_TASKS for s in pools[t][: quota[t]]]
    rng.shuffle(samples)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(samples)} samples to {args.output} ({len(Counter(s['task'] for s in samples))} sub-tasks)")


if __name__ == "__main__":
    main()
