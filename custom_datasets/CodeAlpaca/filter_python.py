#!/usr/bin/env python3
import argparse
import ast
import pandas as pd

def looks_like_python_code(text: str) -> bool:
    """
    Return True if `text` can be parsed as Python source code.
    This will treat things like `class`, `def`, lists, etc. as Python.
    HTML/JS/SQL/Java will almost always raise SyntaxError.
    """
    if not isinstance(text, str):
        return False
    # Strip leading/trailing whitespace to be a bit more forgiving
    src = text.strip()
    if not src:
        return False
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False

def filter_python_only(input_csv: str, output_csv: str,
                       code_column: str = "completion") -> None:
    # Load the dataset
    df = pd.read_csv(input_csv)

    if code_column not in df.columns:
        raise ValueError(f"Column '{code_column}' not found in {input_csv}. "
                         f"Available columns: {list(df.columns)}")

    # Build a boolean mask: True if row looks like Python
    mask = df[code_column].apply(looks_like_python_code)

    df_py = df[mask].reset_index(drop=True)

    print(f"Loaded {len(df)} rows from {input_csv}")
    print(f"Kept  {len(df_py)} rows that look like Python code")

    # Save filtered CSV
    df_py.to_csv(output_csv, index=False)
    print(f"Saved Python-only subset to {output_csv}")

def main():

    filter_python_only("codealpaca_test.csv", "codealpaca_test_filtered.csv", "completion")

if __name__ == "__main__":
    main()
