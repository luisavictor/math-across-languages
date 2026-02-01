#!/usr/bin/env python3
"""
Script to generate accuracy table data from evaluation results.
Generates mean ± std over 3 seeds for Llama-3.1-8B IT across different top-k% values
for GSM8K (English, German, Hindi) and RACE (English, German, Hindi) datasets.
"""

import json
import os
import re
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple


# Dataset configurations
# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_race/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'race',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_race_german/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'race_de',
#         'label': 'German'
#     },
#     'hindi': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/scale/results_gsm8k_race_hindi_max300/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'race_hi_max300',
#         'label': 'Hindi'
#     }
# }

# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_race/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'race',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_race_german/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'race_de',
#         'label': 'German'
#     },
#     'hindi': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/scale/results_gsm8k_race_hindi_max300/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'race_hi_max300',
#         'label': 'Hindi'
#     }
# }

# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_race/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'race',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_race_german/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'race_de',
#         'label': 'German'
#     },
#     'hindi': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/prune/results_gsm8k_race_hindi_max300/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'race_hi_max300',
#         'label': 'Hindi'
#     }
# }

# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_race/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'race',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_race_german/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'race_de',
#         'label': 'German'
#     },
#     'hindi': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/prune/results_gsm8k_race_hindi_max300/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'race_hi_max300',
#         'label': 'Hindi'
#     }
# }

###################### MMLU #########################
# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_mmlu/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'mmlu',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_mmlu_german/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'mmlu_de',
#         'label': 'German'
#     },
#     'hindi': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/scale/results_gsm8k_mmlu_hindi_max300/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'mmlu_hi_max300',
#         'label': 'Hindi'
#     }
# }

# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_mmlu/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'mmlu',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_mmlu_german/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'mmlu_de',
#         'label': 'German'
#     },
#     'hindi': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/scale/results_gsm8k_mmlu_hindi_max300/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'mmlu_hi_max300',
#         'label': 'Hindi'
#     }
# }

# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_mmlu/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'mmlu',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_mmlu_german/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'mmlu_de',
#         'label': 'German'
#     },
#     'hindi': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/prune/results_gsm8k_mmlu_hindi_max300/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'mmlu_hi_max300',
#         'label': 'Hindi'
#     }
# }

DATASETS = {
    'english': {
        'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_mmlu/eval_results/Qwen/Qwen3-4B-Instruct-2507',
        'gsm8k_key': 'gsm8k_cot',
        'task_key': 'mmlu',
        'label': 'English'
    },
    'german': {
        'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_mmlu_german/eval_results/Qwen/Qwen3-4B-Instruct-2507',
        'gsm8k_key': 'gsm8k_de_cot',
        'task_key': 'mmlu_de',
        'label': 'German'
    },
    'hindi': {
        'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/prune/results_gsm8k_mmlu_hindi_max300/eval_results/Qwen/Qwen3-4B-Instruct-2507',
        'gsm8k_key': 'gsm8k_hi_cot_max300',
        'task_key': 'mmlu_hi_max300',
        'label': 'Hindi'
    }
}


# Top-k percentages to evaluate
TOP_K_VALUES = [0.000001, 0.00001, 0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.15]
NUM_RUNS = 3 


def load_json(filepath: str) -> Dict:
    """Load and parse a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_pretrain_scores(base_path: str, gsm8k_key: str, task_key: str) -> Dict[str, float]:
    """Extract pre-training scores from pre_results files."""
    scores = {}
    
    # GSM8K scores (strict and flexible)
    gsm8k_file = os.path.join(base_path, 'pre_results_train_task.json')
    if os.path.exists(gsm8k_file):
        data = load_json(gsm8k_file)
        if gsm8k_key in data:
            scores['gsm8k_strict'] = data[gsm8k_key].get('exact_match,strict-match', None)
            scores['gsm8k_flex'] = data[gsm8k_key].get('exact_match,flexible-extract', None)
    
    # Second task scores (RACE or MMLU)
    race_file = os.path.join(base_path, 'pre_results.json')
    if os.path.exists(race_file):
        data = load_json(race_file)
        if task_key in data:
            scores['race'] = data[task_key].get('acc,none', None)
    
    return scores


def get_topk_scores(base_path: str, topk: float, gsm8k_key: str, task_key: str) -> Dict[str, List[float]]:
    """Extract scores for a specific top-k value across all runs."""
    scores = {
        'gsm8k_strict': [],
        'gsm8k_flex': [],
        'race': []
    }

    def _find_matching_file(run: int, suffix: str) -> str | None:
        # Try both Race and MMLU patterns
        for prefix in ["Race", "MMLU"]:
            pattern = f"{prefix}_calculate*_run{run}{suffix}.json"
            candidates = sorted(Path(base_path).glob(pattern))
            for candidate in candidates:
                match = re.search(rf"{prefix}_calculate([0-9.eE-]+)_scalar", candidate.name)
                if not match:
                    continue
                try:
                    candidate_topk = float(match.group(1))
                except ValueError:
                    continue
                if np.isclose(candidate_topk, topk, rtol=0.0, atol=1e-12):
                    return str(candidate)
        return None
    
    for run in range(NUM_RUNS):
        # GSM8K scores (from train_task files)
        gsm8k_file = _find_matching_file(run, "_train_task")
        if gsm8k_file and os.path.exists(gsm8k_file):
            try:
                data = load_json(gsm8k_file)
                if gsm8k_key in data:
                    strict = data[gsm8k_key].get('exact_match,strict-match')
                    flex = data[gsm8k_key].get('exact_match,flexible-extract')
                    if strict is not None:
                        scores['gsm8k_strict'].append(strict)
                    if flex is not None:
                        scores['gsm8k_flex'].append(flex)
            except (json.JSONDecodeError, ValueError):
                # Skip empty or corrupted files
                pass
        
        # Second task scores (RACE or MMLU)
        race_file = _find_matching_file(run, "")
        if race_file and os.path.exists(race_file):
            try:
                data = load_json(race_file)
                if task_key in data:
                    acc = data[task_key].get('acc,none')
                    if acc is not None:
                        scores['race'].append(acc)
            except (json.JSONDecodeError, ValueError):
                # Skip empty or corrupted files
                pass
    
    return scores


def format_score(values: List[float]) -> str:
    """Format a list of scores as 'mean ± std'."""
    if not values:
        return "-"
    
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) == 3 else "N/A"
    
    if std == "N/A":
        return f"${mean:.3f}_{{N/A}}$"
    return f"${mean:.3f}_{{{std:.2f}}}$"


def format_single_score(value: float) -> str:
    """Format a single score value."""
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def generate_table():
    """Generate the complete accuracy table."""
    
    print("=" * 120)
    print("Accuracy Table: mean ± std over 3 seeds for Llama-3.1-8B IT")
    print("=" * 120)
    print()
    
    # Header row
    header = [r"\multicolumn{1}{l}{\multirow{2}{*}{Top-k}}"]
    for lang_name, config in DATASETS.items():
        label = config['label']
        header.extend([f"GSM8K_{label}", f"GSM8K_{label}_flex", f"RACE_{label}"])
    
    print(" | ".join(f"{h:^15}" for h in header))
    print("-" * 120)
    
    # Pre-train row
    pretrain_row = ["Pre-train"]
    for lang_name, config in DATASETS.items():
        scores = get_pretrain_scores(config['path'], config['gsm8k_key'], config['task_key'])
        pretrain_row.append(format_single_score(scores.get('gsm8k_strict')))
        pretrain_row.append(format_single_score(scores.get('gsm8k_flex')))
        pretrain_row.append(format_single_score(scores.get('race')))
    
    print(" | ".join(f"{v:^15}" for v in pretrain_row))
    print("-" * 120)
    
    # Top-k rows
    for topk in TOP_K_VALUES:
        row = [f"${topk}$"]
        
        for lang_name, config in DATASETS.items():
            scores = get_topk_scores(config['path'], topk, config['gsm8k_key'], config['task_key'])
            row.append(format_score(scores['gsm8k_strict']))
            row.append(format_score(scores['gsm8k_flex']))
            row.append(format_score(scores['race']))
        
        print(" | ".join(f"{v:^15}" for v in row))
    
    print("=" * 120)


def generate_csv(output_file: str = "accuracy_table.csv"):
    """Generate CSV file with the table data."""
    import csv
    
    # Prepare data
    rows = []
    
    # Header
    header = ["Top-k%"]
    for lang_name, config in DATASETS.items():
        label = config['label']
        header.extend([f"GSM8K_{label}", f"GSM8K_{label}_flex", f"RACE_{label}"])
    rows.append(header)
    
    # Pre-train row
    pretrain_row = ["Pre-train"]
    for lang_name, config in DATASETS.items():
        scores = get_pretrain_scores(config['path'], config['gsm8k_key'], config['task_key'])
        pretrain_row.append(format_single_score(scores.get('gsm8k_strict')))
        pretrain_row.append(format_single_score(scores.get('gsm8k_flex')))
        pretrain_row.append(format_single_score(scores.get('race')))
    rows.append(pretrain_row)
    
    # Top-k rows
    for topk in TOP_K_VALUES:
        row = [str(topk)]
        
        for lang_name, config in DATASETS.items():
            scores = get_topk_scores(config['path'], topk, config['gsm8k_key'], config['task_key'])
            row.append(format_score(scores['gsm8k_strict']))
            row.append(format_score(scores['gsm8k_flex']))
            row.append(format_score(scores['race']))
        
        rows.append(row)
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"\nCSV file saved to: {output_file}")


def generate_latex_table(output_file: str = "accuracy_table.tex"):
    """Generate LaTeX table code."""
    
    lines = []
    lines.append("\\begin{table*}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Accuracy (mean $\\pm$ std over 3 seeds) for Llama-3.1-8B IT}")
    lines.append("\\label{tab:accuracy}")
    
    # Table structure
    num_cols = 1 + len(DATASETS) * 3
    col_spec = "l" + " ccc" * len(DATASETS)
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    # Header
    header_parts = [r"\multicolumn{1}{l}{\multirow{2}{*}{Top-k}}"]
    for lang_name, config in DATASETS.items():
        label = config['label']
        header_parts.append(f"\\multicolumn{{3}}{{c}}{{{label}}}")
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append(r"\cmidrule(rl){2-4} \cmidrule(rl){5-7} \cmidrule(rl){8-10}")
    
    # Sub-header
    subheader_parts = [""]
    for lang_name in DATASETS.keys():
        if "mmlu" in DATASETS["english"]["task_key"]:
            subheader_parts.extend(["GSM8K", "GSM8K flex", "MMLU"])
        else:
            subheader_parts.extend(["GSM8K", "GSM8K flex", "RACE"])
    lines.append(" & ".join(subheader_parts) + " \\\\")
    lines.append("\\hline")
    
    # Pre-train row
    pretrain_parts = ["Pre-train"]
    for lang_name, config in DATASETS.items():
        scores = get_pretrain_scores(config['path'], config['gsm8k_key'], config['task_key'])
        pretrain_parts.append(format_single_score(scores.get('gsm8k_strict')))
        pretrain_parts.append(format_single_score(scores.get('gsm8k_flex')))
        pretrain_parts.append(format_single_score(scores.get('race')))
    lines.append(" & ".join(pretrain_parts) + " \\\\")
    lines.append("\\hline")
    
    # Top-k rows
    for topk in TOP_K_VALUES:
        row_parts = [str(topk)]
        
        for lang_name, config in DATASETS.items():
            scores = get_topk_scores(config['path'], topk, config['gsm8k_key'], config['task_key'])
            row_parts.append(format_score(scores['gsm8k_strict']))
            row_parts.append(format_score(scores['gsm8k_flex']))
            row_parts.append(format_score(scores['race']))
        
        lines.append(" & ".join(row_parts) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"LaTeX table saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate accuracy table from evaluation results')
    parser.add_argument('--format', choices=['console', 'csv', 'latex', 'all'], default='console',
                        help='Output format (default: console)')
    parser.add_argument('--csv-output', default='accuracy_table.csv',
                        help='CSV output filename (default: accuracy_table.csv)')
    parser.add_argument('--latex-output', default='accuracy_table.tex',
                        help='LaTeX output filename (default: accuracy_table.tex)')
    
    args = parser.parse_args()
    
    if args.format in ['console', 'all']:
        generate_table()
    
    if args.format in ['csv', 'all']:
        generate_csv(args.csv_output)
    
    if args.format in ['latex', 'all']:
        generate_latex_table(args.latex_output)
