"""
Filter rows in a RACE German CSV so that any entry whose `id` contains
the substring "middle" is removed.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def filter_rows(input_path: Path, output_path: Path, inplace: bool) -> None:
    if inplace:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        shutil.copy2(input_path, backup_path)
        output_path = input_path

    with input_path.open("r", encoding="utf-8", newline="") as fin, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError("Column 'id' not found in input CSV.")

        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        kept = 0
        dropped = 0
        for row in reader:
            if "middle" in row["id"].lower():
                dropped += 1
                continue
            writer.writerow(row)
            kept += 1

    print(f"Kept {kept} rows, dropped {dropped} rows, wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove rows whose id contains 'middle' from race_de CSV files."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="race_de_test.csv",
        help="Path to the input CSV (default: race_de_test.csv in the current directory).",
    )
    parser.add_argument(
        "--output",
        help="Optional path for the filtered CSV. Ignored when --inplace is set.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Rewrite the input file in place (a .bak backup is created first).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if args.inplace and args.output:
        raise ValueError("Provide either --output or --inplace, not both.")

    output_path = (
        input_path.with_name(input_path.stem + "_filtered" + input_path.suffix)
        if not args.inplace
        else input_path
    )
    if args.output:
        output_path = Path(args.output)

    filter_rows(input_path, output_path, args.inplace)


if __name__ == "__main__":
    main()
