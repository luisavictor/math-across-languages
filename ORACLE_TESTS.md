Offline oracle unit tests workflow

- Purpose: Build stored oracle cases from CodeAlpaca CSVs, then test model-generated code against them without needing the reference at eval time.
- Files involved:
  - build_oracle_cases.py: reads a CodeAlpaca CSV (prompt/completion), classifies each completion (expr/function/class), fuzzes deterministic inputs, runs the reference via runner.py, and writes oracle_cases.jsonl (including class state checks).
  - runner.py: sandbox executor used by both the builder and pytest to evaluate code safely in a subprocess.
  - tests/test_oracle_cases.py: pytest suite that reads oracle_cases.jsonl and checks your model outputs (from candidate_generations.jsonl or CANDIDATE_PATH) via runner.py.
- Run order:
  1) Build the oracle once (adjust paths/limits as needed):
     py build_oracle_cases.py --csv custom_datasets/CodeAlpaca/codealpaca_test_filtered.csv --output oracle_cases.jsonl --n_cases 30 --seed 0
     (Quick smoke: py build_oracle_cases.py --max_rows 5 --n_cases 2 --output oracle_cases_smoke.jsonl)
  2) Prepare model outputs as JSONL (default candidate_generations.jsonl):
     {"sample_id": 0, "code": "<model completion here>"}
     {"sample_id": 1, "completion": "<model completion here>"}
  3) Run pytest:
     # optional: set CANDIDATE_PATH=path\\to\\your_outputs.jsonl
     py -m pytest -q tests/test_oracle_cases.py

What is n_cases?
- n_cases in build_oracle_cases.py is the number of fuzzed input cases generated per sample (for functions/classes). Higher values give broader coverage but make oracle generation slower and increase oracle_cases.jsonl size. Default is 30; use a smaller number (e.g., 2-5) for quick smoke runs.
- By default, the builder enforces exactly n_cases unique cases per sample when possible. If it cannot reach n_cases after retries, the sample is skipped. Use --no_enforce_k to allow variable counts per sample.
