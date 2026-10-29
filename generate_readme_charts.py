"""Generate portfolio charts from REAL JordanHeritageVision experiment outputs.

Requirements: matplotlib (no PyTorch needed).
Usage from project root:
    python generate_readme_charts.py
    python generate_readme_charts.py --project-root "E:/learning/My GitHub ML/JordanHeritageVision"

Reads histories from outputs/<experiment>/reports/history.json, and final test
classification reports from outputs/final_evaluation/<model>/classification_report.txt.
Writes assets/validation_comparison.png, assets/final_test_comparison.png,
assets/results_summary.csv and assets/results_summary.json.

No results are hard-coded. If an expected source file is missing or invalid,
the script reports its path and exits without fabricating numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save images without opening graphical windows.
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

# Change these only if you renamed your experiment or evaluation folders.
MODELS = (
    {"name": "EfficientNetB0", "checkpoint_model": "efficientnet_b0", "frozen": "experiment_01", "finetuned": "experiment_02", "test": "efficientnet_b0"},
    {"name": "ResNet101", "checkpoint_model": "resnet101", "frozen": "experiment_03_resnet101", "finetuned": "experiment_04_resnet101", "test": "resnet101"},
    {"name": "VGG16", "checkpoint_model": "vgg16", "frozen": "experiment_05_vgg16", "finetuned": "experiment_06_vgg16", "test": "vgg16"},
    {"name": "InceptionV3", "checkpoint_model": "inception_v3", "frozen": "experiment_07_inceptionv3", "finetuned": "experiment_08_inceptionv3", "test": "inception_v3"},
)

BACKGROUND = "#0A0E17"
PANEL = "#101826"
WHITE = "#ECF2FA"
MUTED = "#A7B5C9"
GRID = "#2A364A"
BLUE = "#4B88DB"
GREEN = "#37A273"


def read_history_best(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"Missing history file:\n  {path}")
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    scores = obj.get("val_accuracy")
    if not isinstance(scores, list) or not scores:
        raise ValueError(f"History has no nonempty 'val_accuracy' list: {path}")
    numbers = [float(x) for x in scores]
    if any(not np.isfinite(x) or not 0 <= x <= 1 for x in numbers):
        raise ValueError(f"Invalid validation accuracy (expected 0..1) in {path}")
    return max(numbers) * 100


def parse_test_report(path: Path, expected_model: str) -> tuple[float, float, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing test classification report:\n  {path}")
    raw = path.read_text(encoding="utf-8-sig")
    model_match = re.search(r"^Model:\s*(\S+)\s*$", raw, re.M | re.I)
    if not model_match or model_match.group(1).lower() != expected_model.lower():
        raise ValueError(f"Report model does not match expected '{expected_model}': {path}")
    acc_match = re.search(r"^Test accuracy:\s*([\d.]+)%\s+on\s+(\d+)\s+samples", raw, re.M | re.I)
    macro_match = re.search(r"^\s*macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", raw, re.M)
    if not acc_match or not macro_match:
        raise ValueError(f"Cannot parse test accuracy / macro avg F1 from {path}")
    accuracy = float(acc_match.group(1))
    n = int(acc_match.group(2))
    macro_f1 = float(macro_match.group(3)) * 100
    if not (0 <= accuracy <= 100 and 0 <= macro_f1 <= 100) or int(macro_match.group(4)) != n:
        raise ValueError(f"Inconsistent or out-of-range test report metrics: {path}")
    return accuracy, macro_f1, n


def load_results(project_root: Path) -> list[dict]:
    rows = []
    for model in MODELS:
        frozen_path = project_root / "outputs" / model["frozen"] / "reports" / "history.json"
        fine_path = project_root / "outputs" / model["finetuned"] / "reports" / "history.json"
        report_path = project_root / "outputs" / "final_evaluation" / model["test"] / "classification_report.txt"
        frozen_acc = read_history_best(frozen_path)
        fine_acc = read_history_best(fine_path)
        test_acc, macro_f1, n = parse_test_report(report_path, model["checkpoint_model"])
        rows.append({
            "model": model["name"],
            "frozen_val_accuracy_pct": frozen_acc,
            "finetuned_val_accuracy_pct": fine_acc,
            "finetuning_gain_pp": fine_acc - frozen_acc,
            "test_accuracy_pct": test_acc,
            "test_macro_f1_pct": macro_f1,
            "test_samples": n,
        })
    sample_counts = {r["test_samples"] for r in rows}
    if len(sample_counts) != 1:
        raise ValueError(f"Test sample counts differ between models: {sample_counts}")
    return rows


def style_figure(title: str, subtitle: str, figsize=(12.8, 7.1)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(PANEL)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, alpha=0.70, linestyle="--", linewidth=0.9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=11, length=0, pad=10)
    fig.text(0.08, 0.95, title, color=WHITE, fontsize=21, weight="bold", ha="left", va="top")
    fig.text(0.08, 0.901, subtitle, color=MUTED, fontsize=11, ha="left", va="top")
    return fig, ax


def label_bars(ax, bars):
    for rect in bars:
        val = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, val + 1.15,
                f"{val:.2f}%", ha="center", va="bottom", color=WHITE,
                fontsize=10.5, weight="semibold")


def render_validation(rows: list[dict], dest: Path):
    names = [r["model"] for r in rows]
    base = [r["frozen_val_accuracy_pct"] for r in rows]
    fine = [r["finetuned_val_accuracy_pct"] for r in rows]
    indices = np.arange(len(rows))
    fig, ax = style_figure(
        "Four CNN Architectures — Validation Comparison",
        "Best validation accuracy for each training stage | source: experiment history.json",
    )
    width = 0.34
    bars1 = ax.bar(indices-width/2, base, width, color=BLUE, label="Frozen Backbone", zorder=3)
    bars2 = ax.bar(indices+width/2, fine, width, color=GREEN, label="Fine-Tuning", zorder=3)
    label_bars(ax, bars1)
    label_bars(ax, bars2)
    ax.set_xticks(indices, names)
    ax.set_ylim(0, 105)  # Show the full 0-100% scale to avoid exaggerating differences.
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Validation accuracy", color=MUTED, labelpad=15)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE,
              fontsize=10.5, ncol=2, loc="upper left")
    fig.text(0.08, 0.055, "Note: Frozen and fine-tuned checkpoints were selected using validation, not test data.",
             color=MUTED, fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.83, bottom=0.18)
    fig.savefig(dest, dpi=190, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def render_test(rows: list[dict], dest: Path):
    ranked = sorted(rows, key=lambda r: r["test_accuracy_pct"], reverse=True)
    best = ranked[0]
    n = best["test_samples"]
    names = [r["model"] for r in ranked]
    scores = [r["test_accuracy_pct"] for r in ranked]
    fig, ax = style_figure(
        f"Final Model Comparison — {n:,} Held-out Test Images",
        "Final test performance of validation-selected checkpoints | source: classification_report.txt",
    )
    colors = [GREEN] + [BLUE]*(len(ranked)-1)
    bars = ax.bar(np.arange(len(ranked)), scores, width=0.58, color=colors, zorder=3)
    label_bars(ax, bars)
    ax.set_xticks(np.arange(len(ranked)), names)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_ylabel("Test accuracy", color=MUTED, labelpad=15)
    fig.text(0.77, 0.81, "TOP TEST ACCURACY", color=MUTED, fontsize=11, ha="center")
    fig.text(0.77, 0.76, f"{best['test_accuracy_pct']:.2f}%", color=GREEN,
             fontsize=24, weight="bold", ha="center")
    fig.text(0.77, 0.715, f"{best['model']}  •  Macro F1: {best['test_macro_f1_pct']:.2f}%",
             color=WHITE, fontsize=10.5, ha="center")
    fig.text(0.08, 0.055, "Test metrics are descriptive; do not tune model hyperparameters on these results.",
             color=MUTED, fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.68, bottom=0.18)
    fig.savefig(dest, dpi=190, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def export_metrics(rows, target: Path):
    csv_path = target / "results_summary.csv"
    json_path = target / "results_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
                          for row in rows])
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent,
                        help="Project directory containing outputs/ and assets/")
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve()
    try:
        rows = load_results(project_root)
        out = project_root / "assets"
        out.mkdir(parents=True, exist_ok=True)
        render_validation(rows, out / "validation_comparison.png")
        render_test(rows, out / "final_test_comparison.png")
        csv_path, json_path = export_metrics(rows, out)
        print("\nGenerated images from source experiment files:")
        for p in (out / "validation_comparison.png", out / "final_test_comparison.png", csv_path, json_path):
            print(" ", p)
        print("\nMODEL         Frozen Val   Fine Val   Test Acc   Macro F1")
        for r in rows:
            print(f"{r['model']:<14}{r['frozen_val_accuracy_pct']:>7.2f}%"
                  f"{r['finetuned_val_accuracy_pct']:>11.2f}%"
                  f"{r['test_accuracy_pct']:>10.2f}%"
                  f"{r['test_macro_f1_pct']:>10.2f}%")
        print("\nREADME.md has NOT been changed.")
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"\nERROR: {exc}\n\nThis script reads actual training and evaluation reports."
              " Check the paths in MODELS if you renamed your experiment folders.", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
