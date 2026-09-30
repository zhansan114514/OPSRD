#!/usr/bin/env python3
"""Summarize OPSD evaluation JSON files and compute paired bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Runs must use NAME=RESULT_DIRECTORY syntax.")
    name, directory = value.split("=", 1)
    path = Path(directory)
    if not name or not path.is_dir():
        raise argparse.ArgumentTypeError(f"Invalid run specification: {value}")
    return name, path


def load_run(directory: Path) -> dict[str, dict]:
    summaries = {}
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        summaries[summary["dataset"]] = summary
    if not summaries:
        raise ValueError(f"No evaluation JSON files found in {directory}.")
    return summaries


def problem_scores(summary: dict) -> dict[str, float]:
    return {
        str(result["problem_id"]): result["num_correct"] / result["val_n"]
        for result in summary["results"]
    }


def paired_differences(reference: dict, candidate: dict) -> list[float]:
    reference_scores = problem_scores(reference)
    candidate_scores = problem_scores(candidate)
    shared_ids = sorted(reference_scores.keys() & candidate_scores.keys())
    if not shared_ids:
        raise ValueError("Runs do not share problem IDs and cannot be paired.")
    return [candidate_scores[item] - reference_scores[item] for item in shared_ids]


def bootstrap_mean_interval(values: list[float], samples: int, seed: int) -> tuple[float, float, float]:
    rng = random.Random(seed)
    count = len(values)
    point_estimate = sum(values) / count
    bootstrapped = []
    for _ in range(samples):
        bootstrapped.append(sum(values[rng.randrange(count)] for _ in range(count)) / count)
    bootstrapped.sort()
    lower_index = int(0.025 * (samples - 1))
    upper_index = int(0.975 * (samples - 1))
    return point_estimate, bootstrapped[lower_index], bootstrapped[upper_index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="Run name and result directory as NAME=PATH. The first run is the paired reference.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    runs = {name: load_run(directory) for name, directory in args.run}
    run_names = list(runs)
    reference_name = run_names[0]
    datasets = sorted(set.intersection(*(set(run) for run in runs.values())))

    print("| Run | " + " | ".join(datasets) + " | Macro avg |")
    print("|---|" + "---:|" * (len(datasets) + 1))
    report = {"reference": reference_name, "runs": {}}
    for run_name, summaries in runs.items():
        scores = [summaries[dataset]["average_at_n_pct"] for dataset in datasets]
        macro_average = sum(scores) / len(scores)
        print(
            f"| {run_name} | "
            + " | ".join(f"{score:.2f}" for score in scores)
            + f" | {macro_average:.2f} |"
        )
        report["runs"][run_name] = {
            "average_at_n_pct": dict(zip(datasets, scores)),
            "macro_average_pct": macro_average,
        }

    print(f"\nPaired differences versus `{reference_name}` (percentage points):")
    reference = runs[reference_name]
    for run_offset, run_name in enumerate(run_names[1:], start=1):
        candidate = runs[run_name]
        pooled_differences = []
        report["runs"][run_name]["paired_difference_pct"] = {}
        for dataset_offset, dataset in enumerate(datasets):
            differences = paired_differences(reference[dataset], candidate[dataset])
            pooled_differences.extend(differences)
            point, lower, upper = bootstrap_mean_interval(
                differences,
                samples=args.bootstrap_samples,
                seed=args.seed + 100 * run_offset + dataset_offset,
            )
            report["runs"][run_name]["paired_difference_pct"][dataset] = {
                "estimate": 100 * point,
                "ci95": [100 * lower, 100 * upper],
            }
            print(f"- {run_name} / {dataset}: {100 * point:+.2f} [{100 * lower:+.2f}, {100 * upper:+.2f}]")

        point, lower, upper = bootstrap_mean_interval(
            pooled_differences,
            samples=args.bootstrap_samples,
            seed=args.seed + 1000 * run_offset,
        )
        report["runs"][run_name]["paired_difference_pct"]["pooled"] = {
            "estimate": 100 * point,
            "ci95": [100 * lower, 100 * upper],
        }
        print(f"- {run_name} / pooled: {100 * point:+.2f} [{100 * lower:+.2f}, {100 * upper:+.2f}]")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
