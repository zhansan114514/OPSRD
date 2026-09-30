#!/usr/bin/env python3
"""Compare paired saved on-policy rollouts from two OPSD training runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_generations(directory: Path) -> dict[str, list[dict]]:
    files = sorted(directory.glob("generations_step_*.json"))
    if not files:
        raise ValueError(f"No saved generation files found in {directory}.")
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))["generations"]
        for path in files
    }


def user_message(prompt: str) -> str:
    marker = "<|im_start|>user\n"
    if marker not in prompt:
        return prompt
    return prompt.split(marker, 1)[1].split("<|im_end|>", 1)[0]


def completion_summary(rows: list[dict]) -> dict:
    completions = [row["completion"] for row in rows]
    lengths = [len(text) for text in completions]
    return {
        "outputs": len(completions),
        "mean_chars": statistics.mean(lengths),
        "median_chars": statistics.median(lengths),
        "boxed_rate": sum("\\boxed" in text for text in completions) / len(completions),
    }


def compare(reference_directory: Path, candidate_directory: Path) -> dict:
    reference = load_generations(reference_directory)
    candidate = load_generations(candidate_directory)
    common_files = sorted(reference.keys() & candidate.keys())
    if not common_files:
        raise ValueError("The runs have no saved generation steps in common.")

    reference_rows = []
    candidate_rows = []
    for name in common_files:
        if len(reference[name]) != len(candidate[name]):
            raise ValueError(f"Mismatched row count for {name}.")
        reference_rows.extend(reference[name])
        candidate_rows.extend(candidate[name])

    pairs = list(zip(reference_rows, candidate_rows))
    return {
        "common_files": common_files,
        "paired_outputs": len(pairs),
        "same_user_message": sum(
            user_message(reference_row["prompt"]) == user_message(candidate_row["prompt"])
            for reference_row, candidate_row in pairs
        ),
        "identical_completion": sum(
            reference_row["completion"] == candidate_row["completion"]
            for reference_row, candidate_row in pairs
        ),
        "identical_first_200_chars": sum(
            reference_row["completion"][:200] == candidate_row["completion"][:200]
            for reference_row, candidate_row in pairs
        ),
        "reference": completion_summary(reference_rows),
        "candidate": completion_summary(candidate_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    result = compare(args.reference, args.candidate)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
