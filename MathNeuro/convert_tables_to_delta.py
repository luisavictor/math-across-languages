import re
import os
import argparse


def parse_pretrain_row(line):
    """Parse the pretrain row which has plain floats (no subscripts)."""
    parts = line.split("&", 1)[1]
    values = re.findall(r'\b(0\.\d+)\b', parts)
    return [float(v) for v in values]


def parse_data_row(line):
    """Parse a data row with $mean_{std}$ format. Returns (label, means, stds)"""
    parts = line.split("&")
    label = parts[0].strip()
    means, stds = [], []
    for cell in parts[1:]:
        m = re.search(r'\$([\d.]+)_\{([\d.]+)\}\$', cell)
        if m:
            means.append(float(m.group(1)))
            stds.append(float(m.group(2)))
        else:
            m2 = re.search(r'([\d]+\.[\d]+)', cell)
            if m2:
                means.append(float(m2.group(1)))
                stds.append(0.0)
    return label, means, stds


def _strip_leading_zero(s):
    """Remove leading zero from decimal strings: '0.002' -> '.002', '-0.002' -> '-.002'."""
    s = re.sub(r'(?<![\d])0+(\.[\d])', r'\1', s)
    return s


def _apply_color(raw, value, max_neg, max_pos):
    """
    Wrap a cell with cellcolor whose intensity is proportional to the
    magnitude of `value`, normalised across the table.
    Negative values → red, positive → green.  Intensity 5–40.
    """
    if value < 0 and max_neg != 0:
        intensity = int(5 + 35 * abs(value) / abs(max_neg))  # 5..40
        return f"\\cellcolor{{red!{intensity}}} {raw}"
    elif value > 0 and max_pos != 0:
        intensity = int(5 + 35 * abs(value) / abs(max_pos))  # 5..40
        return f"\\cellcolor{{green!{intensity}}} {raw}"
    else:
        return raw


def format_pct_change_cell(mean, pretrain, std, color=False, max_neg=0, max_pos=0):
    """
    Format percentage change: (mean - pretrain) / pretrain * 100
    Std is scaled by the same factor:  std / pretrain * 100
    """
    pct_change = (mean - pretrain) / pretrain * 100
    pct_std    = std / pretrain * 100
    sign = "+" if pct_change >= 0 else "-"
    raw = f"{sign}${abs(pct_change):.1f}_{{{pct_std:.1f}}}$\\%"
    # raw = _strip_leading_zero(raw)
    if color:
        raw = _apply_color(raw, pct_change, max_neg, max_pos)
    return raw


def format_delta_cell(mean, pretrain, std, color=False, max_neg=0, max_pos=0):
    """Format absolute delta: mean - pretrain, keeping std unchanged."""
    delta = mean - pretrain
    sign = "+" if delta >= 0 else "-"
    raw = f"{sign}${abs(delta):.3f}_{{{std:.2f}}}$"
    raw = _strip_leading_zero(raw)
    if color:
        raw = _apply_color(raw, delta, max_neg, max_pos)
    return raw


def comment_out_latex(latex_src):
    return "\n".join("% " + line for line in latex_src.split("\n"))


def convert_table(latex_src, mode="pct", color=False):
    """
    Convert a LaTeX table to show changes relative to the pretrain row.
    mode: 'pct' for percentage change, 'delta' for absolute difference.
    color: if True, add cellcolor with intensity-mapped red/green encoding.
    """
    compute_change = (
        (lambda m, p: (m - p) / p * 100) if mode == "pct"
        else (lambda m, p: m - p)
    )
    formatter = format_pct_change_cell if mode == "pct" else format_delta_cell
    lines = latex_src.strip().split("\n")

    # --- Pass 1: collect all change values for normalisation ---
    pretrain_values = None
    all_changes = []
    data_rows = []  # [(line_index, label, means, stds)]

    indexed_lines = list(enumerate(lines))
    for idx, line in indexed_lines:
        stripped = line.strip()
        if "Pre-train" in stripped:
            pretrain_values = parse_pretrain_row(stripped)
            continue
        if pretrain_values and re.match(r'^\s*[\de.+-]+\s*&', line):
            label, means, stds = parse_data_row(line)
            if len(means) == len(pretrain_values):
                changes = [compute_change(m, p) for m, p in zip(means, pretrain_values)]
                all_changes.extend(changes)
                data_rows.append((idx, label, means, stds))

    max_neg = min(all_changes) if all_changes and min(all_changes) < 0 else 0
    max_pos = max(all_changes) if all_changes and max(all_changes) > 0 else 0

    # --- Pass 2: build output with formatted cells ---
    pretrain_values = None
    data_row_map = {idx: (label, means, stds) for idx, label, means, stds in data_rows}
    output_lines = []

    for idx, line in indexed_lines:
        stripped = line.strip()

        if "Pre-train" in stripped:
            pretrain_values = parse_pretrain_row(stripped)
            output_lines.append(line)
            continue

        if idx in data_row_map and pretrain_values:
            label, means, stds = data_row_map[idx]
            cells = [
                formatter(m, p, s, color=color, max_neg=max_neg, max_pos=max_pos)
                for m, p, s in zip(means, pretrain_values, stds)
            ]
            new_row = f"{label} & " + " & ".join(cells) + r"\\"
            output_lines.append(new_row)
            continue

        # Non-data rows with column-count mismatch or structural lines
        if pretrain_values and re.match(r'^\s*[\de.+-]+\s*&', line) and idx not in data_row_map:
            label, means, stds = parse_data_row(line)
            print(f"  WARNING: column count mismatch on row '{label}' "
                  f"({len(means)} vs {len(pretrain_values)} pretrain cols). "
                  f"Row kept unchanged.")

        output_lines.append(line)

    return "\n".join(output_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LaTeX accuracy tables to delta or percentage-change tables."
    )
    parser.add_argument(
        "--mode", choices=["pct", "delta"], default="pct",
        help="'pct' for percentage change (default), 'delta' for absolute difference."
    )
    parser.add_argument(
        "--color", action="store_true",
        help="Add \\cellcolor{red/green} encoding to cells based on sign."
    )
    parser.add_argument(
        "files", nargs="*", default=None,
        help="Input .tex files. If omitted, processes all .tex files in the default tables/ directory."
    )
    args = parser.parse_args()

    if args.files:
        latex_paths = args.files
    else:
        prefix = "/raid/s3/opengptx/behzad_shomali/LabTest/tt_tables/"
        latex_paths = sorted(
            os.path.join(prefix, f)
            for f in os.listdir(prefix)
        )

    mode_label = "Percentage change" if args.mode == "pct" else "Delta"
    print(f"Mode: {mode_label}\n")

    for input_path in latex_paths:
        print(f"Processing: {input_path}")
        with open(input_path, "r") as f:
            latex_src = f.read()

        converted_table    = convert_table(latex_src, mode=args.mode, color=args.color)
        commented_original = comment_out_latex(latex_src)

        if "brief" in input_path:
            print("skipping changing brief table")
            output = latex_src
        else:
            output = (
                f"% === Original table (commented out) ===\n"
                f"{commented_original}\n\n"
                f"% === {mode_label} table ===\n"
                f"{converted_table}\n"
            )


        out_dir = os.path.dirname(input_path).replace("tt_tables", "tables")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(input_path))
        with open(out_path, "w") as f:
            f.write(output)
        print(f"  Saved to {out_path}\n")