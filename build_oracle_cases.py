"""Build oracle test cases from the CodeAlpaca CSVs.

This script parses each completion as Python code, fuzzes deterministic inputs,
runs the reference in a sandboxed subprocess (`runner.py`), and writes
`oracle_cases.jsonl` containing expected outputs (or properties for random-style
prompts).
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import json
import pickle
import random
import re
import string
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "runner.py"
BAD_EXC_TYPES = {"RunnerFailure", "DecodeError", "TimeoutExpired"}


def obj_to_b64(obj: Any) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode("utf-8")


def b64_to_obj(data: str) -> Any:
    return pickle.loads(base64.b64decode(data.encode("utf-8")))


def is_random_prompt(prompt: str, completion: str) -> bool:
    text = f"{prompt} {completion}".lower()
    return any(k in text for k in ["random", "shuffle", "generate", "sample", "randint", "randrange"])


def property_from_prompt(prompt: str) -> Dict[str, Any]:
    """Very small heuristic for random-list prompts."""
    p = prompt.lower()
    nums = [int(n) for n in re.findall(r"\b\d+\b", p)]
    if "list" in p and nums:
        # Try to infer length/min/max if present.
        length = nums[0] if len(nums) >= 1 else None
        min_v = nums[1] if len(nums) >= 2 else None
        max_v = nums[2] if len(nums) >= 3 else None
        return {
            "type": "list_int",
            "len": length,
            "min": min_v,
            "max": max_v,
        }
    return {"type": "any"}


def should_skip(run_result: Dict[str, Any], skip_type_error: bool = True) -> bool:
    """Decide whether to drop a case based on the runner result."""
    if not run_result.get("ok"):
        return True
    return False


def attach_result(case: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
    """Attach expected payload or exception to the case."""
    case = dict(case)
    case["expected_stdout"] = run_result.get("stdout", "")
    if run_result.get("ok"):
        case["expected_b64"] = run_result.get("result_b64")
    else:
        case["exc_type"] = run_result.get("exc_type")
    return case


def classify_reference(code: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"kind": "invalid", "error": str(exc)}

    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]

    if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
        # Single expression
        return {"kind": "expr", "expr": code.strip()}

    if funcs:
        # Prefer common solution names, otherwise first function.
        pref = None
        for name in ("solve", "main", "solution"):
            if any(f.name == name for f in funcs):
                pref = name
                break
        if pref is None:
            pref = funcs[0].name
        arity, arg_names = guess_func_arity(funcs, pref)
        return {"kind": "function", "name": pref, "arity": arity, "arg_names": arg_names}

    if classes:
        cls_name = classes[0].name
        ctor_arity, ctor_arg_names = guess_ctor_arity(classes[0])
        method_info = zero_arg_methods(classes[0])
        return {
            "kind": "class",
            "name": cls_name,
            "ctor_arity": ctor_arity,
            "ctor_arg_names": ctor_arg_names,
            "methods": method_info,
        }

    return {"kind": "script"}


def guess_func_arity(func_nodes: Sequence[ast.FunctionDef], target: str) -> Tuple[int, List[str]]:
    for fn in func_nodes:
        if fn.name == target:
            total = len(fn.args.args)
            defaults = len(fn.args.defaults)
            req = max(0, total - defaults)
            arg_names = [a.arg for a in fn.args.args[:req]]
            return req, arg_names
    return 0, []


def guess_ctor_arity(cls_node: ast.ClassDef) -> Tuple[int, List[str]]:
    for item in cls_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            total = len(item.args.args) - 1  # drop self
            defaults = len(item.args.defaults)
            req = max(0, total - defaults)
            arg_names = [a.arg for a in item.args.args[1 : 1 + req]]
            return req, arg_names
    return 0, []


def zero_arg_methods(cls_node: ast.ClassDef) -> List[str]:
    names: List[str] = []
    for item in cls_node.body:
        if isinstance(item, ast.FunctionDef):
            if item.name.startswith("__"):
                continue
            total = len(item.args.args) - 1  # drop self
            defaults = len(item.args.defaults)
            req = max(0, total - defaults)
            if req == 0:
                names.append(item.name)
    return names


def gen_value_for_name(name: str, rng: random.Random) -> Any:
    lname = name.lower()
    if "age" in lname or lname in {"n", "num", "count", "size", "length"}:
        return rng.randint(0, 120)
    if "name" in lname:
        return "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(1, 12)))
    if "addr" in lname or "address" in lname:
        return f"{rng.randint(1, 999)} " + "".join(
            rng.choice(string.ascii_letters + " ") for _ in range(rng.randint(3, 18))
        )
    choice = rng.randint(0, 4)
    if choice == 0:
        return rng.randint(-50, 50)
    if choice == 1:
        return rng.random() * rng.randint(-10, 10)
    if choice == 2:
        return "".join(rng.choice(string.ascii_letters + " _-") for _ in range(rng.randint(0, 20)))
    if choice == 3:
        return [rng.randint(-20, 20) for _ in range(rng.randint(0, 10))]
    return rng.choice([True, False, None])


def gen_args(arg_names: Sequence[str], rng: random.Random) -> Tuple[Any, ...]:
    return tuple(gen_value_for_name(n, rng) for n in arg_names)


def case_key(case: Dict[str, Any]) -> Tuple[Any, ...]:
    prop = case.get("property")
    prop_key = json.dumps(prop, sort_keys=True) if prop is not None else ""
    return (
        case.get("kind"),
        case.get("expr"),
        case.get("func_name"),
        case.get("args_b64"),
        case.get("kwargs_b64"),
        case.get("class_name"),
        case.get("ctor_args_b64"),
        case.get("method"),
        prop_key,
    )


def add_case_unique(
    cases: List[Dict[str, Any]],
    seen: set[Tuple[Any, ...]],
    case: Dict[str, Any],
) -> bool:
    key = case_key(case)
    if key in seen:
        return False
    seen.add(key)
    cases.append(case)
    return True


def run_in_runner(
    code: str,
    mode: str,
    name: str = "",
    method_name: str = "",
    expr: str = "",
    args: Tuple[Any, ...] = (),
    kwargs: Dict[str, Any] | None = None,
    timeout_s: float = 2.0,
) -> Dict[str, Any]:
    kwargs = kwargs or {}
    cmd = [
        sys.executable,
        str(RUNNER),
        "--mode",
        mode,
        "--code_b64",
        obj_to_b64(code),
        "--name",
        name,
        "--method_name",
        method_name,
        "--args_b64",
        obj_to_b64(args),
        "--kwargs_b64",
        obj_to_b64(kwargs),
    ]
    if mode == "eval_expr":
        cmd.extend(["--expr_b64", obj_to_b64(expr)])
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"ok": False, "exc_type": "TimeoutExpired", "exc_msg": "runner timeout", "stdout": ""}
    if cp.returncode != 0:
        return {"ok": False, "exc_type": "RunnerFailure", "exc_msg": cp.stderr, "stdout": ""}
    try:
        payload = b64_to_obj(cp.stdout.strip())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "exc_type": "DecodeError", "exc_msg": str(exc), "stdout": cp.stdout}
    return payload


def build_cases_for_sample(
    sample_id: int,
    prompt: str,
    ref_code: str,
    n_cases: int,
    seed: int,
    enforce_k: bool,
    timeout_s: float,
) -> List[Dict[str, Any]]:
    info = classify_reference(ref_code)
    rng = random.Random(seed + sample_id)
    randomy = is_random_prompt(prompt, ref_code)
    prop = property_from_prompt(prompt) if randomy else None

    cases: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()

    if info["kind"] == "invalid":
        return cases

    if info["kind"] == "expr":
        r = run_in_runner(ref_code, "eval_expr", expr=info["expr"], timeout_s=timeout_s)
        if not randomy and should_skip(r):
            return cases
        if randomy and not r.get("ok"):
            return cases
        case = {
            "sample_id": sample_id,
            "kind": "expr_prop" if randomy else "expr",
            "expr": info["expr"],
        }
        if randomy:
            case["property"] = prop
            case["expected_stdout"] = r.get("stdout", "")
        else:
            case = attach_result(case, r)
        if enforce_k:
            if n_cases != 1:
                return []
            add_case_unique(cases, seen, case)
            return cases
        cases.append(case)
        return cases

    if info["kind"] == "function":
        name = info["name"]
        arg_names = info.get("arg_names", [])
        if enforce_k:
            max_attempts = max(1, n_cases * 10)
            attempts = 0
            while len(cases) < n_cases and attempts < max_attempts:
                attempts += 1
                args = gen_args(arg_names, rng)
                r = run_in_runner(ref_code, "call_func", name=name, args=args, timeout_s=timeout_s)
                if randomy:
                    if not r.get("ok"):
                        continue
                    case = {
                        "sample_id": sample_id,
                        "kind": "function_prop",
                        "func_name": name,
                        "args_b64": obj_to_b64(args),
                        "kwargs_b64": obj_to_b64({}),
                        "property": prop,
                        "expected_stdout": r.get("stdout", ""),
                    }
                else:
                    if should_skip(r):
                        continue
                    case = {
                        "sample_id": sample_id,
                        "kind": "function",
                        "func_name": name,
                        "args_b64": obj_to_b64(args),
                        "kwargs_b64": obj_to_b64({}),
                    }
                    case = attach_result(case, r)
                add_case_unique(cases, seen, case)
            if len(cases) < n_cases:
                return []
            return cases

        for _ in range(n_cases):
            args = gen_args(arg_names, rng)
            r = run_in_runner(ref_code, "call_func", name=name, args=args, timeout_s=timeout_s)
            if randomy:
                if not r.get("ok"):
                    continue
                case = {
                    "sample_id": sample_id,
                    "kind": "function_prop",
                    "func_name": name,
                    "args_b64": obj_to_b64(args),
                    "kwargs_b64": obj_to_b64({}),
                    "property": prop,
                    "expected_stdout": r.get("stdout", ""),
                }
            else:
                if should_skip(r):
                    continue
                case = {
                    "sample_id": sample_id,
                    "kind": "function",
                    "func_name": name,
                    "args_b64": obj_to_b64(args),
                    "kwargs_b64": obj_to_b64({}),
                }
                case = attach_result(case, r)
            cases.append(case)
        return cases

    if info["kind"] == "class":
        cls = info["name"]
        ctor_arg_names = info.get("ctor_arg_names", [])
        method_names = info.get("methods") or []
        if enforce_k:
            max_attempts = max(1, n_cases * 10)
            attempts = 0
            while len(cases) < n_cases and attempts < max_attempts:
                attempts += 1
                ctor_args = gen_args(ctor_arg_names, rng)
                state_res = run_in_runner(
                    ref_code,
                    "make_obj_state",
                    name=cls,
                    args=ctor_args,
                    timeout_s=timeout_s,
                )
                if state_res.get("ok"):
                    state_case = {
                        "sample_id": sample_id,
                        "kind": "class_state",
                        "class_name": cls,
                        "ctor_args_b64": obj_to_b64(ctor_args),
                        "expected_state_b64": state_res.get("result_b64"),
                        "expected_stdout": state_res.get("stdout", ""),
                    }
                    add_case_unique(cases, seen, state_case)
                for m in method_names:
                    if len(cases) >= n_cases:
                        break
                    mr = run_in_runner(
                        ref_code,
                        "make_and_call",
                        name=cls,
                        method_name=m,
                        args=ctor_args,
                        timeout_s=timeout_s,
                    )
                    if should_skip(mr):
                        continue
                    case = {
                        "sample_id": sample_id,
                        "kind": "class_method",
                        "class_name": cls,
                        "ctor_args_b64": obj_to_b64(ctor_args),
                        "method": m,
                    }
                    case = attach_result(case, mr)
                    add_case_unique(cases, seen, case)
            if len(cases) < n_cases:
                return []
            return cases

        ctor_args = gen_args(ctor_arg_names, rng)
        state_res = run_in_runner(
            ref_code,
            "make_obj_state",
            name=cls,
            args=ctor_args,
            timeout_s=timeout_s,
        )
        if state_res.get("ok"):
            cases.append(
                {
                    "sample_id": sample_id,
                    "kind": "class_state",
                    "class_name": cls,
                    "ctor_args_b64": obj_to_b64(ctor_args),
                    "expected_state_b64": state_res.get("result_b64"),
                    "expected_stdout": state_res.get("stdout", ""),
                }
            )
        for m in method_names:
            mr = run_in_runner(
                ref_code,
                "make_and_call",
                name=cls,
                method_name=m,
                args=ctor_args,
                timeout_s=timeout_s,
            )
            if should_skip(mr):
                continue
            case = {
                "sample_id": sample_id,
                "kind": "class_method",
                "class_name": cls,
                "ctor_args_b64": obj_to_b64(ctor_args),
                "method": m,
            }
            case = attach_result(case, mr)
            cases.append(case)
        return cases

    return cases


def iter_rows(csv_path: Path, max_rows: int | None = None) -> Iterable[Tuple[int, str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            prompt = row.get("prompt") or row.get("instruction") or ""
            completion = row.get("completion") or row.get("output") or ""
            yield idx, prompt, completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "custom_datasets" / "CodeAlpaca" / "codealpaca_test_filtered.csv",
        help="Path to the CodeAlpaca CSV (prompt/completion columns).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "oracle_cases.jsonl",
        help="Where to write the oracle cases.",
    )
    parser.add_argument("--n_cases", type=int, default=30, help="How many fuzzed inputs per sample.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed.")
    parser.add_argument("--max_rows", type=int, default=None, help="Limit rows for quick runs.")
    parser.add_argument(
        "--enforce_k",
        dest="enforce_k",
        action="store_true",
        default=True,
        help="Enforce exactly n_cases per sample when possible.",
    )
    parser.add_argument(
        "--no_enforce_k",
        dest="enforce_k",
        action="store_false",
        help="Allow variable number of cases per sample.",
    )
    parser.add_argument(
        "--timeout_s",
        type=float,
        default=2.0,
        help="Runner timeout in seconds for each generated case.",
    )
    args = parser.parse_args()

    args.output.unlink(missing_ok=True)
    total = 0
    for sample_id, prompt, completion in iter_rows(args.csv, max_rows=args.max_rows):
        cases = build_cases_for_sample(
            sample_id,
            prompt,
            completion,
            args.n_cases,
            args.seed,
            args.enforce_k,
            args.timeout_s,
        )
        if not cases:
            continue
        with args.output.open("a", encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(case) + "\n")
                total += 1
    print(f"wrote {total} cases to {args.output}")


if __name__ == "__main__":
    main()
