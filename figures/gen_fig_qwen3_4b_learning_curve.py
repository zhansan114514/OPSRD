#!/usr/bin/env python3
"""Plot Qwen3-4B checkpoint learning curves from a consolidated JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("aime24", "aime25", "hmmt25", "macro")
PANEL_TITLES = {
    "aime24": "AIME 2024",
    "aime25": "AIME 2025",
    "hmmt25": "HMMT 2025",
    "macro": "Macro Average",
}
METHOD_STYLE = {
    "OPSD-solution": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "Role-entropy50": {"color": "#D55E00", "marker": "s", "linestyle": "--"},
}


def load_payload(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    steps = payload.get("steps")
    methods = payload.get("methods")
    if steps != [0, 25, 50, 75, 100]:
        raise ValueError(f"Expected steps [0, 25, 50, 75, 100], got {steps!r}.")
    if set(methods or {}) != set(METHOD_STYLE):
        raise ValueError(f"Expected methods {sorted(METHOD_STYLE)}, got {sorted(methods or {})}.")

    for method, scores in methods.items():
        for dataset in DATASETS:
            values = scores.get(dataset)
            if not isinstance(values, list) or len(values) != len(steps):
                raise ValueError(f"{method}/{dataset} must contain {len(steps)} values.")
            if not all(isinstance(value, (int, float)) for value in values):
                raise ValueError(f"{method}/{dataset} contains a non-numeric value.")
    return payload


def padded_limits(values: list[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    padding = max(1.2, (high - low) * 0.25)
    return np.floor(low - padding), np.ceil(high + padding)


def plot(payload: dict, output_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 1.9,
            "lines.markersize": 5,
        }
    )

    steps = payload["steps"]
    methods = payload["methods"]
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 4.75), sharex=True)

    for ax, dataset in zip(axes.flat, DATASETS):
        all_values = []
        for method, style in METHOD_STYLE.items():
            values = methods[method][dataset]
            all_values.extend(values)
            ax.plot(
                steps,
                values,
                label=method,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markerfacecolor="white",
                markeredgewidth=1.2,
                zorder=3,
            )

            best_index = int(np.argmax(values[1:])) + 1
            ax.scatter(
                [steps[best_index]],
                [values[best_index]],
                marker="*",
                s=52,
                color=style["color"],
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
            )

        baseline = methods["OPSD-solution"][dataset][0]
        ax.axhline(baseline, color="#7A7A7A", linewidth=0.9, linestyle=":", zorder=1)
        ax.set_title(PANEL_TITLES[dataset], pad=5)
        ax.set_xticks(steps)
        ax.set_xlim(-3, 103)
        ax.set_ylim(*padded_limits(all_values))
        ax.set_ylabel("Avg@12 (%)")

    for ax in axes[-1, :]:
        ax.set_xlabel("Optimizer Steps")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(top=0.88, hspace=0.38, wspace=0.28)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig_qwen3_4b_learning_curve.pdf"
    png_path = output_dir / "fig_qwen3_4b_learning_curve.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("qwen3_4b_learning_curve.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    payload = load_payload(args.input)
    for path in plot(payload, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
