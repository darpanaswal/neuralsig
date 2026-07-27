#!/usr/bin/env python3
"""
compute_icc.py — inter-rater agreement (ICC) between human annotators
(annotate.py output) and/or the LLM judge (run_judge.py's judge_score).

Takes one or more annotation .jsonl files (one per human, produced by
annotate.py against a shared --sample_file) and, unless --no_judge is
passed, also pulls judge_score for the same entries straight from the
scored response files under outputs/responses/<model>/... Only entries
labeled by every included rater are used (the intersection).

Reports:
  - ICC(2,1): two-way random effects, absolute agreement, single rater
  - ICC(3,1): two-way mixed effects, consistency, single rater
  - pairwise Cohen's kappa for every rater pair, as a sanity cross-check

Usage:
    python compute_icc.py --model gemma \
        --annotations annotations/alice.jsonl annotations/bob.jsonl annotations/carol.jsonl

    # human-only agreement, no LLM judge column
    python compute_icc.py --model gemma --no_judge \
        --annotations annotations/alice.jsonl annotations/bob.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from run_judge import OUT_ROOT, ENVIRONMENTS, build_sets


# ── loaders ─────────────────────────────────────────────────────────
def load_annotation_file(path: Path) -> Dict[str, int]:
    labels = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        labels[rec["key"]] = rec["human_label"]
    return labels


def collect_judge_scores(model: str) -> Dict[str, int]:
    scores = {}
    for env in ENVIRONMENTS:
        env_root = OUT_ROOT / model / env
        for subdir, split in build_sets(env_root):
            path = env_root / subdir / f"{split}.json"
            if not path.exists():
                continue
            entries = json.loads(path.read_text())
            for i, e in enumerate(entries):
                score = e.get("judge_score")
                if score is None:
                    continue
                # file_index (position within the file), not orig_index: some
                # transforms (e.g. prefix_injection) emit multiple distinct
                # variant rows per orig_index, so orig_index alone collides.
                key = f"{env}|{subdir}|{split}|{i}"
                scores[key] = score
    return scores


# ── ICC ─────────────────────────────────────────────────────────────
def icc_2_1_and_3_1(data: np.ndarray) -> Dict[str, float]:
    """data: (n_subjects, k_raters). Returns ICC(2,1) and ICC(3,1)."""
    n, k = data.shape
    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    icc_2_1 = (ms_rows - ms_error) / (
        ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    )
    icc_3_1 = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)

    return {"ICC(2,1)": icc_2_1, "ICC(3,1)": icc_3_1}


def pairwise_kappa(columns: Dict[str, List[int]]) -> None:
    from sklearn.metrics import cohen_kappa_score
    names = list(columns)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            kappa = cohen_kappa_score(columns[names[i]], columns[names[j]])
            print(f"  Cohen's kappa [{names[i]} vs {names[j]}]: {kappa:.3f}")


# ── main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                         help="target model the annotations/judge_score were computed over")
    parser.add_argument("--annotations", required=True, nargs="+",
                         help="one or more annotate.py .jsonl output files")
    parser.add_argument("--no_judge", action="store_true",
                         help="exclude the LLM judge column; human-only agreement")
    args = parser.parse_args()

    raters: Dict[str, Dict[str, int]] = {}
    for path_str in args.annotations:
        path = Path(path_str)
        raters[path.stem] = load_annotation_file(path)

    if not args.no_judge:
        judge_scores = collect_judge_scores(args.model)
        if not judge_scores:
            print(f"No judge_score found for model={args.model} "
                  f"(run run_judge.py first, or pass --no_judge). Aborting.")
            return
        raters["llm_judge"] = judge_scores

    if len(raters) < 2:
        print("Need at least 2 raters (annotators and/or judge) to compute agreement.")
        return

    common_keys = set.intersection(*(set(r) for r in raters.values()))
    print(f"Raters: {list(raters)}")
    print(f"Common labeled entries across all raters: {len(common_keys)}")
    if len(common_keys) < 2:
        print("Not enough overlapping entries to compute ICC.")
        return

    keys_sorted = sorted(common_keys)
    names = list(raters)
    data = np.array([[raters[name][k] for name in names] for k in keys_sorted], dtype=float)

    icc = icc_2_1_and_3_1(data)
    print(f"\nn_subjects={data.shape[0]}  n_raters={data.shape[1]}")
    for name, val in icc.items():
        print(f"{name}: {val:.3f}")

    print("\nPairwise Cohen's kappa:")
    columns = {name: data[:, i].astype(int).tolist() for i, name in enumerate(names)}
    pairwise_kappa(columns)


if __name__ == "__main__":
    main()
