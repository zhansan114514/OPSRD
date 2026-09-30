#!/usr/bin/env python3
"""Gate multi-seed replication on a pre-registered single-seed role signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_results import (
    bootstrap_mean_interval,
    load_run,
    paired_differences,
)


def macro_average(run: dict[str, dict], datasets: list[str]) -> float:
    return sum(run[dataset]["average_at_n_pct"] for dataset in datasets) / len(datasets)


def evaluate_gate(
    base: dict[str, dict],
    candidate: dict[str, dict],
    neutral: dict[str, dict],
    min_macro_improvement_pct: float,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    datasets = sorted(set(base) & set(candidate) & set(neutral))
    if not datasets:
        raise ValueError("The three runs do not share any datasets.")

    base_macro = macro_average(base, datasets)
    candidate_macro = macro_average(candidate, datasets)
    neutral_macro = macro_average(neutral, datasets)
    macro_improvement = candidate_macro - base_macro
    improved_tasks = sum(
        candidate[dataset]["average_at_n_pct"] > base[dataset]["average_at_n_pct"]
        for dataset in datasets
    )

    pooled_differences = []
    for dataset in datasets:
        pooled_differences.extend(paired_differences(base[dataset], candidate[dataset]))
    paired_point, paired_lower, paired_upper = bootstrap_mean_interval(
        pooled_differences,
        samples=bootstrap_samples,
        seed=seed,
    )
    paired_ci_pct = [100 * paired_lower, 100 * paired_upper]

    macro_and_control = (
        macro_improvement >= min_macro_improvement_pct
        and candidate_macro > neutral_macro
    )
    taskwise_and_paired = improved_tasks >= 2 and paired_ci_pct[0] > 0
    passed = macro_and_control or taskwise_and_paired
    return {
        "passed": passed,
        "datasets": datasets,
        "base_macro_pct": base_macro,
        "candidate_macro_pct": candidate_macro,
        "neutral_macro_pct": neutral_macro,
        "macro_improvement_pct": macro_improvement,
        "improved_tasks": improved_tasks,
        "paired_difference_pct": 100 * paired_point,
        "paired_ci95_pct": paired_ci_pct,
        "criteria": {
            "macro_and_control": macro_and_control,
            "taskwise_and_paired": taskwise_and_paired,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--neutral", type=Path, required=True)
    parser.add_argument("--min-macro-improvement-pct", type=float, default=2.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    result = evaluate_gate(
        load_run(args.base),
        load_run(args.candidate),
        load_run(args.neutral),
        min_macro_improvement_pct=args.min_macro_improvement_pct,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
