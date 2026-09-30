#!/usr/bin/env python3
"""Summarize repeated training seeds with hierarchical seed/problem bootstrap."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, stdev

from summarize_results import load_run, paired_differences, parse_run


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def percentile_interval(values: list[float]) -> tuple[float, float]:
    values.sort()
    lower_index = int(0.025 * (len(values) - 1))
    upper_index = int(0.975 * (len(values) - 1))
    return values[lower_index], values[upper_index]


def hierarchical_bootstrap(
    differences_by_seed: list[dict[str, list[float]]],
    datasets: list[str],
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    if not differences_by_seed:
        raise ValueError("At least one candidate seed is required.")
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive.")

    point_values = [
        mean(seed_differences[dataset])
        for seed_differences in differences_by_seed
        for dataset in datasets
    ]
    point = mean(point_values)
    rng = random.Random(seed)
    bootstrapped = []
    for _ in range(samples):
        sampled_values = []
        for _ in differences_by_seed:
            sampled_seed = differences_by_seed[rng.randrange(len(differences_by_seed))]
            for dataset in datasets:
                values = sampled_seed[dataset]
                sampled_values.append(
                    mean(values[rng.randrange(len(values))] for _ in values)
                )
        bootstrapped.append(mean(sampled_values))
    lower, upper = percentile_interval(bootstrapped)
    return point, lower, upper


def summarize_multiseed(
    base: dict[str, dict],
    candidates: dict[str, dict[str, dict]],
    bootstrap_samples: int,
    seed: int,
) -> dict:
    if not candidates:
        raise ValueError("At least one candidate run is required.")
    datasets = sorted(set(base).intersection(*(set(run) for run in candidates.values())))
    if not datasets:
        raise ValueError("Base and candidate runs do not share any datasets.")

    base_scores = {dataset: base[dataset]["average_at_n_pct"] for dataset in datasets}
    base_macro = mean(base_scores.values())
    per_seed = {}
    differences_by_seed = []
    for run_name, run in candidates.items():
        scores = {dataset: run[dataset]["average_at_n_pct"] for dataset in datasets}
        macro = mean(scores.values())
        per_seed[run_name] = {
            "average_at_n_pct": scores,
            "macro_average_pct": macro,
            "macro_difference_pct": macro - base_macro,
        }
        differences_by_seed.append(
            {dataset: paired_differences(base[dataset], run[dataset]) for dataset in datasets}
        )

    aggregate_datasets = {}
    for dataset_offset, dataset in enumerate(datasets):
        candidate_scores = [run[dataset]["average_at_n_pct"] for run in candidates.values()]
        point, lower, upper = hierarchical_bootstrap(
            differences_by_seed,
            [dataset],
            samples=bootstrap_samples,
            seed=seed + dataset_offset,
        )
        aggregate_datasets[dataset] = {
            "base_pct": base_scores[dataset],
            "candidate_mean_pct": mean(candidate_scores),
            "candidate_std_pct": sample_std(candidate_scores),
            "difference_mean_pct": 100 * point,
            "difference_hierarchical_ci95_pct": [100 * lower, 100 * upper],
        }

    macro_scores = [item["macro_average_pct"] for item in per_seed.values()]
    point, lower, upper = hierarchical_bootstrap(
        differences_by_seed,
        datasets,
        samples=bootstrap_samples,
        seed=seed + 1000,
    )
    return {
        "base": {"average_at_n_pct": base_scores, "macro_average_pct": base_macro},
        "candidate_seeds": per_seed,
        "aggregate": {
            "num_seeds": len(candidates),
            "datasets": aggregate_datasets,
            "macro": {
                "base_pct": base_macro,
                "candidate_mean_pct": mean(macro_scores),
                "candidate_std_pct": sample_std(macro_scores),
                "positive_seed_count": sum(score > base_macro for score in macro_scores),
                "difference_mean_pct": 100 * point,
                "difference_hierarchical_ci95_pct": [100 * lower, 100 * upper],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    report = summarize_multiseed(
        load_run(args.base),
        {name: load_run(directory) for name, directory in args.run},
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    print("| Dataset | Base | Candidate mean | SD | Mean difference | Hierarchical 95% CI |")
    print("|---|---:|---:|---:|---:|---:|")
    for dataset, summary in report["aggregate"]["datasets"].items():
        lower, upper = summary["difference_hierarchical_ci95_pct"]
        print(
            f"| {dataset} | {summary['base_pct']:.2f} | {summary['candidate_mean_pct']:.2f} | "
            f"{summary['candidate_std_pct']:.2f} | {summary['difference_mean_pct']:+.2f} | "
            f"[{lower:+.2f}, {upper:+.2f}] |"
        )
    macro = report["aggregate"]["macro"]
    lower, upper = macro["difference_hierarchical_ci95_pct"]
    print(
        f"| macro | {macro['base_pct']:.2f} | {macro['candidate_mean_pct']:.2f} | "
        f"{macro['candidate_std_pct']:.2f} | {macro['difference_mean_pct']:+.2f} | "
        f"[{lower:+.2f}, {upper:+.2f}] |"
    )
    print(
        f"\nPositive macro seeds: {macro['positive_seed_count']}/{report['aggregate']['num_seeds']}"
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
