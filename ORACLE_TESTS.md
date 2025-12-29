# Offline oracle unit tests workflow

## Purpose

Build a local "oracle" of expected outputs from trusted reference code, then
evaluate model-generated code against that oracle without shipping the
reference code into the test run.

## Files involved

- build_oracle_cases.py: reads a CodeAlpaca CSV (prompt/completion), parses each
  completion as Python, generates deterministic test inputs, runs the reference
  via runner.py, and writes oracle_cases.jsonl.
- runner.py: small subprocess runner used by both the builder and pytest to
  execute code and return a serialized result payload.
- tests/test_oracle_cases.py: pytest suite that loads oracle_cases.jsonl and
  checks model outputs (from candidate_generations.jsonl or CANDIDATE_PATH).

## How oracle cases are created

1) Read each row from the Python CodeAlpaca CSV.
2) Classify the completion by syntax:
   - expr: a single expression (e.g., "1 + 2")
   - function: a top-level function (prefers solve/main/solution if present)
   - class: a top-level class (captures __init__ args and zero-arg methods)
   - plain scripts: skipped because those are not straightforward to evaluate with these unit tests
     (in our case mostly functions and classes are captured)
3) Generate inputs:
   - For functions/classes, build deterministic "fuzzed" arguments based on
     parameter names (e.g., name -> string, age/n/count -> int). Seed is fixed
     per sample_id for reproducibility.
4) Execute the reference in a subprocess:
   - run_in_runner encodes code + args, executes in runner.py, and returns
     result_b64, stdout, or an exception type.
5) Store a JSONL case:
   - For deterministic tasks: save expected_b64 and expected_stdout.
   - For random-style prompts (detected by simple keyword heuristics): save a
     property descriptor instead of exact outputs (e.g., list length/min/max).
6) Enforce n_cases (optional):
   - By default, the builder tries to produce exactly n_cases unique cases per
     sample. If it cannot, that sample is skipped. Use --no_enforce_k to allow
     variable case counts.

## How tests are executed

1) Load oracle cases and candidate code:
   - oracle_cases.jsonl contains the expected outputs or properties.
   - candidate_generations.jsonl maps sample_id -> code (or set CANDIDATE_PATH).
2) For each case, run the candidate code in runner.py:
   - expr: eval expression and compare expected_b64 (or property check)
   - function: call the target function with saved args
   - class: construct object and check __dict__ or zero-arg method result
3) Compare:
   - exact result_b64 and stdout, or
   - exception type when the oracle captured an error, or
   - property checks for random-style prompts.

Data format overview (oracle_cases.jsonl)

Common fields:

- sample_id: index into the CSV row
- kind: expr, expr_prop, function, function_prop, class_state, class_method
- expected_b64: base64-pickled result (for deterministic cases)
- expected_stdout: captured stdout during reference execution
- exc_type: expected exception name for error cases

## Run order to create oracle_cases and tests them

1) Build the oracle:
   py build_oracle_cases.py --csv custom_datasets/CodeAlpaca/codealpaca_test_filtered.csv --output oracle_cases.jsonl --n_cases 2 --seed 0
2) Prepare model outputs as JSONL (default candidate_generations.jsonl):
   {"sample_id": 0, "code": "<model completion here>"}
   {"sample_id": 1, "completion": "<model completion here>"}
3) Run pytest:
   py -m pytest -q tests/test_oracle_cases.py

Key arguments

- n_cases: number of fuzzed input cases per sample
- --no_enforce_k: allow fewer than n_cases if unique inputs cannot be generated.
- --timeout_s: per-case runner timeout in seconds.
- ORACLE_FILTER_TO_CANDIDATES=0: test all oracle cases, not just those with
  candidate code provided.
- ORACLE_CASES_PER_ID=N: cap cases per sample_id for faster test runs.
