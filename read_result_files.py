import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

DEFAULT_INPUT_PATH = (
    "results/scale/results_gsm8k_race_german/eval_results/meta-llama/Llama-3.1-8B-Instruct"
)

DEFAULT_K_VALUES = [0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15]

FILENAME_RE = re.compile(
    r"calculate(?P<k>[0-9.]+)_scalar(?P<scalar>[0-9.]+)_run(?P<run>\d+)"
)


def parse_filename(name):
    match = FILENAME_RE.search(name)
    if not match:
        return None
    return (
        float(match.group("k")),
        float(match.group("scalar")),
        int(match.group("run")),
    )


def extract_tasks(data):
    if isinstance(data, dict) and isinstance(data.get("results"), dict):
        return data["results"]
    if isinstance(data, dict):
        return data
    return {}


def normalize_dataset(task_name):
    lower = task_name.lower()
    if "gsm8k" in lower:
        return "GSM8K"
    if "race" in lower:
        return "Race"
    return None


def is_stderr(key):
    return "stderr" in key.lower()


def find_metric(metrics, keys, contains=None):
    for key in keys:
        if key in metrics and not is_stderr(key):
            return metrics[key]
    if contains:
        for key, value in metrics.items():
            key_lower = key.lower()
            if "stderr" in key_lower:
                continue
            if all(token in key_lower for token in contains):
                return value
    return None


def safe_mean_std(values):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def format_float(value):
    if value is None:
        return ""
    return f"{value:.6f}"


def parse_k_values(text):
    if not text:
        return None
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return set(values)


def load_results(input_path, k_filter):
    groups = {}
    for path in input_path.glob("*.json"):
        parsed = parse_filename(path.name)
        if not parsed:
            continue
        k_value, scalar, run = parsed
        if k_filter and k_value not in k_filter:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Skipping {path}: {exc}", file=sys.stderr)
            continue

        tasks = extract_tasks(data)
        for task_name, metrics in tasks.items():
            dataset = normalize_dataset(task_name)
            if not dataset or not isinstance(metrics, dict):
                continue
            key = (dataset, k_value, scalar)
            entry = groups.setdefault(
                key, {"acc": [], "strict": [], "flexible": [], "runs": set()}
            )
            entry["runs"].add(run)

            if dataset == "GSM8K":
                strict = find_metric(
                    metrics,
                    ["exact_match,strict-match"],
                    contains=["exact_match", "strict"],
                )
                flexible = find_metric(
                    metrics,
                    ["exact_match,flexible-extract"],
                    contains=["exact_match", "flexible"],
                )
                if strict is not None:
                    entry["strict"].append(strict)
                if flexible is not None:
                    entry["flexible"].append(flexible)
            elif dataset == "Race":
                acc = find_metric(metrics, ["acc,none", "acc"])
                if acc is not None:
                    entry["acc"].append(acc)

    return groups


def write_csv(output_path, groups):
    fields = [
        "dataset",
        "k",
        "scalar",
        "n_runs",
        "acc_mean",
        "acc_stddev",
        "strict_mean",
        "strict_stddev",
        "flexible_mean",
        "flexible_stddev",
    ]
    rows = []
    for (dataset, k_value, scalar), entry in groups.items():
        acc_mean, acc_std = safe_mean_std(entry["acc"])
        strict_mean, strict_std = safe_mean_std(entry["strict"])
        flexible_mean, flexible_std = safe_mean_std(entry["flexible"])
        rows.append(
            {
                "dataset": dataset,
                "k": f"{k_value:g}",
                "scalar": f"{scalar:g}",
                "n_runs": str(len(entry["runs"])),
                "acc_mean": format_float(acc_mean),
                "acc_stddev": format_float(acc_std),
                "strict_mean": format_float(strict_mean),
                "strict_stddev": format_float(strict_std),
                "flexible_mean": format_float(flexible_mean),
                "flexible_stddev": format_float(flexible_std),
            }
        )

    rows.sort(key=lambda row: (row["dataset"], float(row["k"]), float(row["scalar"])))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize eval results by k value and dataset."
    )
    parser.add_argument("--input_path", default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--output", default=None, help="Output CSV path (default: <input_path>/summary.csv)"
    )
    parser.add_argument(
        "--k_values",
        default=None,
        help="Comma-separated list of k values to include.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.is_dir():
        print(f"Input path not found: {input_path}", file=sys.stderr)
        return 2

    k_filter = parse_k_values(args.k_values)
    if k_filter is None:
        k_filter = set(DEFAULT_K_VALUES)

    groups = load_results(input_path, k_filter)
    output_path = Path("summary_results.csv")
    write_csv(output_path, groups)

    print(f"Wrote {output_path} with {len(groups)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
