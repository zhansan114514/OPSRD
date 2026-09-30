#!/usr/bin/env python3
"""Promote an entropy-fraction screen only if it beats matched reference samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .summarize_results import load_run
except ImportError:  # Direct script execution adds analysis/ rather than repo root.
    from summarize_results import load_run


def score_first_n(summary: dict, n: int) -> float:
    problem_scores = []
    for result in summary["results"]:
        generations = result["generations"][:n]
        if len(generations) != n:
            raise ValueError(
                f"Problem {result['problem_id']} has {len(generations)} generations; expected at least {n}."
            )
        problem_scores.append(sum(bool(item["correct"]) for item in generations) / n)
    return 100 * sum(problem_scores) / len(problem_scores)


def screen(candidate: dict[str, dict], reference: dict[str, dict], threshold: float) -> dict:
    datasets = sorted(set(candidate) & set(reference))
    if not datasets:
        raise ValueError("Candidate and reference do not share datasets.")

    rows = {}
    for dataset in datasets:
        candidate_n = int(candidate[dataset]["val_n"])
        candidate_score = score_first_n(candidate[dataset], candidate_n)
        reference_score = score_first_n(reference[dataset], candidate_n)
        rows[dataset] = {
            "matched_n": candidate_n,
            "candidate_pct": candidate_score,
            "reference_pct": reference_score,
            "difference_pp": candidate_score - reference_score,
        }

    candidate_macro = sum(row["candidate_pct"] for row in rows.values()) / len(rows)
    reference_macro = sum(row["reference_pct"] for row in rows.values()) / len(rows)
    difference = candidate_macro - reference_macro
    return {
        "datasets": rows,
        "candidate_macro_pct": candidate_macro,
        "reference_macro_pct": reference_macro,
        "difference_pp": difference,
        "minimum_improvement_pp": threshold,
        "promote": difference >= threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--minimum-improvement-pp", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    report = screen(
        load_run(args.candidate),
        load_run(args.reference),
        args.minimum_improvement_pp,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
    raise SystemExit(0 if report["promote"] else 3)


if __name__ == "__main__":
    main()
