#!/usr/bin/env python3
"""
Plot GSM8K performance (exact_match,strict-match) vs fraction of pruned neurons
for random neuron selection experiments.

Creates one figure per model, with subplots for each unique pruning threshold
(calculate value). Each subplot shows one line per language, averaged over
random seeds with std-dev error bands.

Baselines and endpoints are hardcoded per model in MODEL_BASELINES / MODEL_ENDPOINTS.

Usage:
    python plot_random_pruning.py
    python plot_random_pruning.py --save
"""

import argparse
import json
import os
import re
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# ── Font / style (consistent with plots.ipynb) ──────────────────────────────
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 20,
    'axes.titlesize': 13,
    'xtick.labelsize': 20,
    'ytick.labelsize': 18,
    'legend.fontsize': 14,
    'lines.linewidth': 3,
    'lines.markersize': 10,
})


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

LANGUAGE_COLORS = {
    "English": "green",
    "German": "violet",
    "Hindi": "orange",
}

LANGUAGE_MARKERS = {
    "English": "o",
    "German": "s",
    "Hindi": "^",
}

METRIC_KEY = "exact_match,strict-match"

# ── Per-model baselines (x=0) and endpoints (x=1.0) per language ─────────────
# Format: {model_short_name: {language: score}}
# TODO: replace placeholder values with actual measured scores
MODEL_BASELINES = {
    "Llama-1B": {"English": 0.340, "German": 0.235, "Hindi": 0.145},
    "Llama-8B": {"English": 0.765, "German": 0.585, "Hindi": 0.415},
    "Qwen3-4B": {"English": 0.735, "German": 0.685, "Hindi": 0.385},
}

def _parse_prune_logs(language_dirs: dict) -> dict:
    """
    Parse random_prune_log*.txt files from each language directory to extract
    math_only_count per (model, threshold).  Returns averaged counts across
    languages.

    Returns:
        {model_short_name: {threshold_str: avg_math_only_count}}
    """
    # Collect per (model, threshold) → list of math_only_count values
    counts: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    for _lang, lang_dir in language_dirs.items():
        if not os.path.isdir(lang_dir):
            continue
        log_files = glob.glob(os.path.join(lang_dir, "random_prune_log*.txt"))
        for log_file in log_files:
            # Try to identify the model from the filename (per-model logs)
            fname = os.path.basename(log_file)
            model_from_fname = None
            for full_name, short_name in MODEL_SHORT_NAMES.items():
                # e.g. "random_prune_log_meta-llama_Llama-3.1-8B-Instruct.txt"
                sanitised = full_name.replace("/", "_")
                if sanitised in fname:
                    model_from_fname = short_name
                    break

            with open(log_file) as f:
                for line in f:
                    m = re.search(
                        r"proportion=([\d.]+).*math_only_count=(\d+)", line
                    )
                    if not m:
                        continue
                    prop = m.group(1)
                    moc = int(m.group(2))

                    if model_from_fname:
                        counts[model_from_fname][prop].append(moc)
                    else:
                        # Single log for all models — group by unique
                        # (proportion, math_only_count) pairs; model identity
                        # will be resolved below.
                        counts["_unknown_"][f"{prop}_{moc}"].append(moc)

    # Handle the "_unknown_" bucket: cluster by model size
    if "_unknown_" in counts:
        raw = counts.pop("_unknown_")
        # Group unique math_only_counts per proportion
        prop_counts: dict[str, set[int]] = defaultdict(set)
        for key, vals in raw.items():
            prop = key.rsplit("_", 1)[0]
            prop_counts[prop].update(vals)
        # For each proportion, sort the unique counts and assign to models
        # by size (smallest → Llama-1B, middle → Qwen3-4B, largest → Llama-8B)
        model_order = ["Llama-1B", "Qwen3-4B", "Llama-8B"]
        for prop, unique_vals in prop_counts.items():
            sorted_vals = sorted(unique_vals)
            for model_short, val in zip(model_order, sorted_vals):
                counts[model_short][prop].append(val)

    # Average across languages
    result: dict[str, dict[str, int]] = {}
    for model, prop_dict in counts.items():
        result[model] = {}
        for prop, vals in prop_dict.items():
            # Deduplicate identical values from same language/seed lines
            unique_vals = list(set(vals))
            result[model][prop] = int(np.mean(unique_vals))
    return result


def _fmt_param_count(n: float) -> str:
    """Format a parameter count with K / M / B suffix."""
    abs_n = abs(n)
    if abs_n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if abs_n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if abs_n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return f"{n:.0f}"


# Format: {model_short_name: {threshold: {language: score}}}
MODEL_ENDPOINTS = {
    "Llama-1B": {
        "0.001": {"English": 0.063, "German": 0.078, "Hindi": 0.035},
        "0.01": {"English": 0.013, "German": 0.018, "Hindi": 0.008},
        "0.1": {"English": 0.022, "German": 0.015, "Hindi": 0.010},
    },
    "Llama-8B": {
        "0.001": {"English": 0.258,  "German": 0.260,  "Hindi": 0.155},
        "0.01": {"English": 0.015, "German": 0.017, "Hindi": 0.017},
        "0.1": {"English": 0.003, "German": 0.020, "Hindi": 0.012},
    },
    "Qwen3-4B": {
        "0.001": {"English": 0.645, "German": 0.525, "Hindi": 0.230},
        "0.01": {"English": 0.020, "German": 0.035, "Hindi": 0.085},
        "0.1": {"English": 0.015, "German": 0.027, "Hindi": 0.005},
    },
}


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

def plot_model(
    model: str,
    lang_data: dict,
    baselines: dict | None = None,
    endpoints: dict | None = None,
    math_only_counts: dict | None = None,
    save: bool = False,
    output_dir: str = ".",
):
    """
    Create a figure for one model with one subplot per calculate value.
    Each subplot compares different languages.

    Args:
        model:     Model short name (for title).
        lang_data: {language: {calculate: {frac: [scores]}}}.
        baselines:        Optional {language: baseline_score} for x=0 point.
        endpoints:        Optional {threshold: {language: endpoint_score}} for x=1.0 point.
        math_only_counts: Optional {threshold: avg_math_only_count} for top axis.
        save:             Whether to save to file.
        output_dir: Where to save.
    """
    # Collect all unique calculate values across languages
    all_calcs = set()
    for language_data in lang_data.values():
        all_calcs.update(language_data.keys())
    all_calcs = sorted(all_calcs, key=lambda x: float(x))

    n_subplots = len(all_calcs)
    if n_subplots == 0:
        print(f"[{model}] No data found, skipping.")
        return

    fig, axes = plt.subplots(1, n_subplots, figsize=(6 * n_subplots, 5), sharey=True)
    if n_subplots == 1:
        axes = [axes]

    # fig.suptitle(f"GSM8K Performance after Random Pruning — {model}", fontsize=16, y=1.02)

    # Sort languages for consistent legend order
    languages_sorted = sorted(lang_data.keys())


    for ax, calc_val in zip(axes, all_calcs):
        ax.set_title(f"Top-$k$ = {calc_val}", fontsize=24, pad=15)
        ax.set_xlabel("Fraction of Pruned Parameter", fontsize=22)
        ax.grid(True, alpha=0.3)

        for language in languages_sorted:
            calc_data = lang_data[language].get(calc_val, {})
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
            if baselines and language in baselines:
                x_vals = [0.0] + x_vals
                means = np.concatenate([[baselines[language]], means])
                stds = np.concatenate([[0.0], stds])

            # Append endpoint at x=1.0 if provided (per threshold)
            calc_endpoints = endpoints.get(calc_val, {}) if endpoints else {}
            if calc_endpoints and language in calc_endpoints:
                x_vals = x_vals + [1.0]
                means = np.concatenate([means, [calc_endpoints[language]]])
                stds = np.concatenate([stds, [0.0]])

            # Delta mode: percentage difference relative to baseline
            if plot_model.delta_mode and baselines and language in baselines:
                baseline = baselines[language]
                # Avoid division by zero
                if baseline != 0:
                    means = 100 * (means - baseline) / baseline
                    stds = 100 * stds / baseline
                else:
                    means = means * 0
                    stds = stds * 0

            color = LANGUAGE_COLORS.get(language, None)
            marker = LANGUAGE_MARKERS.get(language, "o")

            ax.plot(x_vals, means, marker=marker, label=language, color=color)
            ax.fill_between(x_vals, means - stds, means + stds, alpha=0.15, color=color)

        # ── Secondary top x-axis: number of parameters pruned ───────────
        total_params = (math_only_counts or {}).get(calc_val)
        if total_params is not None:
            ax_top = ax.twiny()
            ax_top.set_xlim(ax.get_xlim())
            # Use the actual data-point fractions so x=1.0 is always present
            tick_fracs = sorted({0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
                                & set(np.arange(0, 1.01, 0.2).round(2)))
            ax_top.set_xticks(tick_fracs)
            ax_top.set_xticklabels(
                [_fmt_param_count(t * total_params) for t in tick_fracs],
            )
            ax_top.set_xlabel("Parameters Pruned", fontsize=22, labelpad=15)

    ylabel = "GSM8K Performance"
    if plot_model.delta_mode:
        ylabel = "Performance Change \nfrom Baseline (%)"
    axes[0].set_ylabel(ylabel, fontsize=22)

    # Single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(languages_sorted),
               bbox_to_anchor=(0.5, 1.09), fontsize=20, frameon=False)

    fig.tight_layout()

    if save:
        out_path = os.path.join(output_dir, f"random_pruning_{model.lower().replace(' ', '_')}.pdf")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {out_path}")

    plt.show()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot GSM8K vs fraction of randomly pruned neurons."
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
    parser.add_argument(
        "--delta",
        action="store_true",
        help="Plot delta to baseline (x=0) value.",
    )
    args = parser.parse_args()

    # Parse log files to build math_only_count per (model, threshold)
    model_math_only_counts = _parse_prune_logs(LANGUAGE_DIRS)
    for m_name, m_counts in sorted(model_math_only_counts.items()):
        for thr, cnt in sorted(m_counts.items(), key=lambda x: float(x[0])):
            print(f"  {m_name} | threshold={thr} | math_only_count={cnt:,}")

    # Load data for all languages: {model: {language: {calculate: {frac: [scores]}}}}
    all_data = defaultdict(dict)

    for language, lang_dir in LANGUAGE_DIRS.items():
        if not os.path.isdir(lang_dir):
            print(f"[{language}] Directory not found: {lang_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"  {language}")
        print(f"{'='*60}")

        data = load_results(lang_dir, task_key=LANGUAGE_TASK_KEYS[language])

        # Show summary and reorganize: model -> language -> calc -> frac -> scores
        for model, calc_dict in sorted(data.items()):
            for calc, frac_dict in sorted(calc_dict.items(), key=lambda x: float(x[0])):
                n_fracs = len(frac_dict)
                total_pts = sum(len(v) for v in frac_dict.values())
                print(f"  {model} | threshold={calc} | {n_fracs} fracs, {total_pts} data points")
            all_data[model][language] = calc_dict

    # One figure per model, comparing languages in each subplot
    # Set delta mode flag for plot_model
    plot_model.delta_mode = args.delta

    for model, lang_data in sorted(all_data.items()):
        print(f"\nPlotting: {model}")
        plot_model(
            model=model,
            lang_data=lang_data,
            baselines=MODEL_BASELINES.get(model),
            endpoints=MODEL_ENDPOINTS.get(model),
            math_only_counts=model_math_only_counts.get(model),
            save=args.save,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
