#!/usr/bin/env python3
"""Summarize generated-token lengths and near-cap rates for evaluation runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Runs must use NAME=RESULT_DIRECTORY syntax.")
    name, raw_directory = value.split("=", 1)
    directory = Path(raw_directory)
    if not name or not directory.is_dir():
        raise argparse.ArgumentTypeError(f"Invalid run specification: {value}")
    return name, directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--near-cap-margin",
        type=int,
        default=92,
        help="Count a generation as near-cap when its token length is within this margin of max_new_tokens.",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    report = {}
    print("| Run | Generations | Format % | Mean tokens | Median | P95 | Near cap |")
    print("|---|---:|---:|---:|---:|---:|---:|")

    for run_name, directory in args.run:
        lengths = []
        formatted = 0
        total = 0
        near_cap = 0
        datasets = {}
        for path in sorted(directory.glob("*.json")):
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            texts = [
                generation["full_generation"]
                for result in payload["results"]
                for generation in result["generations"]
            ]
            local_lengths = [
                len(token_ids)
                for token_ids in tokenizer(texts, add_special_tokens=False)["input_ids"]
            ]
            threshold = max(0, int(payload["max_new_tokens"]) - args.near_cap_margin)
            local_near_cap = sum(length >= threshold for length in local_lengths)
            local_formatted = int(payload["formatted_count"])
            lengths.extend(local_lengths)
            formatted += local_formatted
            total += int(payload["total_solutions"])
            near_cap += local_near_cap
            datasets[payload["dataset"]] = {
                "generations": len(local_lengths),
                "format_rate_pct": float(payload["format_rate"]),
                "mean_tokens": statistics.mean(local_lengths),
                "near_cap": local_near_cap,
                "near_cap_threshold": threshold,
            }

        if not lengths:
            raise ValueError(f"No evaluation results found in {directory}.")
        lengths.sort()
        summary = {
            "generations": len(lengths),
            "format_rate_pct": 100 * formatted / total,
            "mean_tokens": statistics.mean(lengths),
            "median_tokens": statistics.median(lengths),
            "p95_tokens": lengths[int(0.95 * (len(lengths) - 1))],
            "near_cap": near_cap,
            "datasets": datasets,
        }
        report[run_name] = summary
        print(
            f"| {run_name} | {summary['generations']} | {summary['format_rate_pct']:.2f} | "
            f"{summary['mean_tokens']:.1f} | {summary['median_tokens']:.1f} | "
            f"{summary['p95_tokens']} | {summary['near_cap']} |"
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
