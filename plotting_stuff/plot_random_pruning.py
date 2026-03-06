#!/usr/bin/env python3
"""
Plot GSM8K performance (exact_match,strict-match) vs fraction of pruned neurons
for random neuron selection experiments.

Creates one figure per language (English, German, Hindi), with subplots for each
unique pruning threshold (calculate value). Each subplot shows one line per model,
averaged over random seeds with std-dev error bands.

Usage:
    python plot_random_pruning.py
    python plot_random_pruning.py --baseline 0.45 0.78 0.62
    python plot_random_pruning.py --baseline 0.45 0.78 0.62 --save
"""

import argparse
import json
import os
import re
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "prune")

LANGUAGE_DIRS = {
    "English": os.path.join(BASE_DIR, "results_gsm8k_race_random"),
    "German": os.path.join(BASE_DIR, "results_gsm8k_race_german_random"),
    "Hindi": os.path.join(BASE_DIR, "results_gsm8k_race_hindi_max300_random"),
}

# The JSON task key varies by language
LANGUAGE_TASK_KEYS = {
    "English": "gsm8k_cot",
    "German": "gsm8k_de_cot",
    "Hindi": "gsm8k_hi_cot_max300",
}

# Short display names for models
MODEL_SHORT_NAMES = {
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-1B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-8B",
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen3-4B",
}

MODEL_COLORS = {
    "Llama-1B": "#1f77b4",
    "Llama-8B": "#ff7f0e",
    "Qwen3-4B": "#2ca02c",
}

MODEL_MARKERS = {
    "Llama-1B": "o",
    "Llama-8B": "s",
    "Qwen3-4B": "^",
}

METRIC_KEY = "exact_match,strict-match"


# ── Data loading ─────────────────────────────────────────────────────────────

def parse_filename(filepath: str):
    """Extract calculate, rseed, frac from a *_train_task.json filename."""
    fname = os.path.basename(filepath)
    m = re.search(
        r"calculate([\d.e+-]+)_scalar[\d.e+-]+_random_rseed(\d+)_frac([\d.]+)_train_task\.json",
        fname,
    )
    if not m:
        return None
    return {
        "calculate": m.group(1),
        "rseed": int(m.group(2)),
        "frac": float(m.group(3)),
    }


def extract_model(filepath: str, base_dir: str) -> str:
    """Extract the model identifier from the file path."""
    rel = filepath.split("eval_results/")[-1]
    parts = rel.split("/")
    return "/".join(parts[:-1])  # e.g. "meta-llama/Llama-3.2-1B-Instruct"


def load_results(base_dir: str, task_key: str = "gsm8k_cot"):
    """
    Load all *_train_task.json results from a language directory.

    Args:
        base_dir: Root directory for this language's results.
        task_key: JSON key for the task (e.g. "gsm8k_cot", "gsm8k_de_cot").

    Returns:
        dict: {model: {calculate_val: {frac: [scores_across_seeds]}}}
    """
    pattern = os.path.join(base_dir, "eval_results", "**", "*_train_task.json")
    files = glob.glob(pattern, recursive=True)

    # Nested dict:  model -> calculate -> frac -> list of scores
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for fpath in files:
        parsed = parse_filename(fpath)
        if parsed is None:
            continue

        model_full = extract_model(fpath, base_dir)
        model_short = MODEL_SHORT_NAMES.get(model_full, model_full)

        with open(fpath) as f:
            result = json.load(f)

        score = result.get(task_key, {}).get(METRIC_KEY)
        if score is None:
            continue

        data[model_short][parsed["calculate"]][parsed["frac"]].append(score)

    return data


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_language(
    language: str,
    data: dict,
    baselines: dict | None = None,
    save: bool = False,
    output_dir: str = ".",
):
    """
    Create a figure for one language with one subplot per calculate value.

    Args:
        language:  Language name (for title).
        data:      {model: {calculate: {frac: [scores]}}}.
        baselines: Optional {model: baseline_score} for x=0 point.
        save:      Whether to save to file.
        output_dir: Where to save.
    """
    # Collect all unique calculate values across models
    all_calcs = set()
    for model_data in data.values():
        all_calcs.update(model_data.keys())
    all_calcs = sorted(all_calcs, key=lambda x: float(x))

    n_subplots = len(all_calcs)
    if n_subplots == 0:
        print(f"[{language}] No data found, skipping.")
        return

    fig, axes = plt.subplots(1, n_subplots, figsize=(6 * n_subplots, 5), sharey=True)
    if n_subplots == 1:
        axes = [axes]

    fig.suptitle(f"GSM8K Performance after Random Pruning — {language}", fontsize=14, y=1.02)

    # Sort models for consistent legend order
    models_sorted = sorted(data.keys())

    for ax, calc_val in zip(axes, all_calcs):
        ax.set_title(f"threshold = {calc_val}", fontsize=12)
        ax.set_xlabel("Fraction of pruned neurons", fontsize=11)
        ax.grid(True, alpha=0.3)

        for model in models_sorted:
            calc_data = data[model].get(calc_val, {})
            if not calc_data:
                continue

            fracs = sorted(calc_data.keys())
            means = []
            stds = []

            x_vals = list(fracs)
            for frac in fracs:
                scores = calc_data[frac]
                means.append(np.mean(scores))
                stds.append(np.std(scores))

            means = np.array(means)
            stds = np.array(stds)

            # Prepend baseline at x=0 if provided
            if baselines and model in baselines:
                x_vals = [0.0] + x_vals
                means = np.concatenate([[baselines[model]], means])
                stds = np.concatenate([[0.0], stds])

            color = MODEL_COLORS.get(model, None)
            marker = MODEL_MARKERS.get(model, "o")

            ax.plot(x_vals, means, marker=marker, label=model, color=color,
                    linewidth=2, markersize=6)
            ax.fill_between(x_vals, means - stds, means + stds, alpha=0.15, color=color)

    axes[0].set_ylabel("GSM8K exact_match (strict)", fontsize=11)

    # Single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models_sorted),
               bbox_to_anchor=(0.5, 1.0), fontsize=10)

    fig.tight_layout()

    if save:
        out_path = os.path.join(output_dir, f"random_pruning_{language.lower()}.pdf")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {out_path}")

    plt.show()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot GSM8K vs fraction of randomly pruned neurons."
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Baseline GSM8K scores (y-value at x=0) for each model, in the order: "
            "Llama-1B, Llama-8B, Qwen3-4B. "
            "Provide fewer values if not all models are present."
        ),
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figures as PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.dirname(__file__),
        help="Directory to save output figures (default: script directory).",
    )
    args = parser.parse_args()

    # Build baselines dict
    model_order = ["Llama-1B", "Llama-8B", "Qwen3-4B"]
    baselines = None
    if args.baseline:
        baselines = {}
        for i, val in enumerate(args.baseline):
            if i < len(model_order):
                baselines[model_order[i]] = val

    for language, lang_dir in LANGUAGE_DIRS.items():
        if not os.path.isdir(lang_dir):
            print(f"[{language}] Directory not found: {lang_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"  {language}")
        print(f"{'='*60}")

        data = load_results(lang_dir, task_key=LANGUAGE_TASK_KEYS[language])

        # Show summary
        for model, calc_dict in sorted(data.items()):
            for calc, frac_dict in sorted(calc_dict.items(), key=lambda x: float(x[0])):
                n_fracs = len(frac_dict)
                total_pts = sum(len(v) for v in frac_dict.values())
                print(f"  {model} | threshold={calc} | {n_fracs} fracs, {total_pts} data points")

        plot_language(
            language=language,
            data=data,
            baselines=baselines,
            save=args.save,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
