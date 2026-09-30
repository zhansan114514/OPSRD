#!/usr/bin/env python3
"""Summarize trainer-state diagnostics from one or more OPSD runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Runs must use NAME=OUTPUT_DIRECTORY syntax.")
    name, raw_directory = value.split("=", 1)
    directory = Path(raw_directory)
    if not name or not directory.is_dir():
        raise argparse.ArgumentTypeError(f"Invalid run specification: {value}")
    return name, directory


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return -1


def load_run(directory: Path) -> dict:
    checkpoints = sorted(
        (path for path in directory.glob("checkpoint-*") if path.is_dir()),
        key=checkpoint_step,
    )
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {directory}.")

    latest = checkpoints[-1]
    state_path = latest / "trainer_state.json"
    with state_path.open(encoding="utf-8") as handle:
        state = json.load(handle)

    history = [entry for entry in state.get("log_history", []) if "loss" in entry]
    if not history:
        raise ValueError(f"No logged loss records found in {state_path}.")

    losses = [float(entry["loss"]) for entry in history]
    grad_norms = [float(entry["grad_norm"]) for entry in history if "grad_norm" in entry]
    latest_metrics = dict(history[-1])
    optional_metrics = (
        "student_entropy",
        "selected_student_entropy",
        "unselected_student_entropy",
        "selected_token_fraction",
        "raw_position_divergence",
    )

    summary = {
        "latest_checkpoint": latest.name,
        "global_step": state.get("global_step"),
        "saved_checkpoints": [path.name for path in checkpoints],
        "num_log_records": len(history),
        "loss": {
            "first": losses[0],
            "last": losses[-1],
            "mean": mean(losses),
            "min": min(losses),
            "max": max(losses),
        },
        "grad_norm": {
            "last": grad_norms[-1] if grad_norms else None,
            "max": max(grad_norms) if grad_norms else None,
        },
        "latest_metrics": latest_metrics,
        "history": history,
    }
    summary["diagnostics"] = {
        key: latest_metrics[key] for key in optional_metrics if key in latest_metrics
    }
    return summary


def format_optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="Run name and output directory as NAME=PATH.",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    report = {name: load_run(directory) for name, directory in args.run}

    print(
        "| Run | Step | Loss first | Loss last | Max grad norm | Raw divergence | "
        "Selected fraction | Student entropy |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, summary in report.items():
        diagnostics = summary["diagnostics"]
        print(
            f"| {name} | {summary['global_step']} | {summary['loss']['first']:.4f} | "
            f"{summary['loss']['last']:.4f} | {format_optional(summary['grad_norm']['max'])} | "
            f"{format_optional(diagnostics.get('raw_position_divergence'))} | "
            f"{format_optional(diagnostics.get('selected_token_fraction'))} | "
            f"{format_optional(diagnostics.get('student_entropy'))} |"
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
