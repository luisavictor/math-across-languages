from __future__ import annotations

import base64
import json
import os
import pickle
import random
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Tuple


def _obj_to_b64(obj: Any) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode("utf-8")


def _b64_to_obj(data: str) -> Any:
    return pickle.loads(base64.b64decode(data.encode("utf-8")))


def clean_code_completion(text: str) -> str:
    if text is None:
        return ""
    match = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    for marker in ("### Instruction", "### Response", "### Explanation"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    for key in ("def ", "class "):
        idx = text.find(key)
        if idx != -1:
            return text[idx:].strip()
    return text.strip()


def write_codealpaca_candidates(
    samples: List[Dict[str, Any]],
    out_path: str,
    allowed_sample_ids: set[int] | None = None,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in samples:
            resp = ex.get("filtered_resps", [None])[0]
            if isinstance(resp, list):
                resp = resp[0] if resp else ""
            if resp is None:
                resp = ex.get("resps", [[None]])[0][0]
            code = clean_code_completion(resp or "")
            sample_id = ex.get("doc", {}).get("sample_id")
            if sample_id is None:
                sample_id = ex["doc_id"]
            try:
                sample_id = int(sample_id)
            except (TypeError, ValueError):
                continue
            if allowed_sample_ids is not None and sample_id not in allowed_sample_ids:
                continue
            f.write(json.dumps({"sample_id": sample_id, "code": code}, ensure_ascii=False) + "\n")


def _oracle_paths() -> Tuple[str, str]:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(root_dir, "oracle_cases.jsonl"), os.path.join(root_dir, "runner.py")


def oracle_paths() -> Tuple[str, str]:
    return _oracle_paths()


def _run_in_subprocess(
    code: str,
    runner_path: str,
    mode: str,
    name: str = "",
    method_name: str = "",
    expr: str = "",
    args: tuple = (),
    kwargs: dict | None = None,
    timeout_s: float = 4.0,
) -> Dict[str, Any]:
    kwargs = kwargs or {}
    cmd = [
        sys.executable,
        runner_path,
        "--mode",
        mode,
        "--code_b64",
        _obj_to_b64(code),
        "--name",
        name,
        "--method_name",
        method_name,
        "--args_b64",
        _obj_to_b64(args),
        "--kwargs_b64",
        _obj_to_b64(kwargs),
    ]
    if mode == "eval_expr":
        cmd.extend(["--expr_b64", _obj_to_b64(expr)])
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "exc_type": "TimeoutExpired",
            "exc_msg": str(exc),
            "stdout": "",
        }
    if cp.returncode != 0:
        return {"ok": False, "exc_type": "RunnerFailure", "exc_msg": cp.stderr, "stdout": ""}
    try:
        return _b64_to_obj(cp.stdout.strip())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "exc_type": "DecodeError", "exc_msg": str(exc), "stdout": cp.stdout}


def _load_oracle_cases(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def load_oracle_sample_ids(oracle_path: str | None = None) -> set[int]:
    if oracle_path is None:
        oracle_path, _ = _oracle_paths()
    cases = _load_oracle_cases(oracle_path)
    return _case_sample_ids(cases)


def select_oracle_sample_ids(
    oracle_path: str | None = None,
    eval_subset: int | None = None,
    min_cases_per_id: int = 2,
    seed: int | None = None,
) -> List[int]:
    if oracle_path is None:
        oracle_path, _ = _oracle_paths()
    cases = _load_oracle_cases(oracle_path)
    valid_cases = _filter_oracle_cases(cases, sample_ids=None)
    counts = _count_sample_ids(valid_cases)
    eligible = sorted(sid for sid, count in counts.items() if count >= min_cases_per_id)
    if eval_subset is not None:
        rng = random.Random(seed)
        rng.shuffle(eligible)
        eligible = eligible[:eval_subset]
    return eligible


SUPPORTED_KINDS = {
    "expr",
    "expr_prop",
    "function",
    "function_prop",
    "class_ctor",
    "class_state",
    "class_method",
}


def _filter_oracle_cases(
    cases: Iterable[Dict[str, Any]],
    sample_ids: set[int] | None,
    max_cases_per_id: int | None = None,
) -> List[Dict[str, Any]]:
    filtered = []
    counts: Dict[int, int] = {}
    for case in cases:
        if case.get("kind") not in SUPPORTED_KINDS:
            continue
        sid = case.get("sample_id")
        if sid is None:
            continue
        try:
            sid = int(sid)
        except (TypeError, ValueError):
            continue
        if sample_ids is not None and sid not in sample_ids:
            continue
        if max_cases_per_id is not None:
            if counts.get(sid, 0) >= max_cases_per_id:
                continue
            counts[sid] = counts.get(sid, 0) + 1
        filtered.append(case)
    return filtered


def _case_sample_ids(cases: Iterable[Dict[str, Any]]) -> set[int]:
    ids: set[int] = set()
    for case in cases:
        sid = case.get("sample_id")
        if sid is None:
            continue
        try:
            ids.add(int(sid))
        except (TypeError, ValueError):
            continue
    return ids


def _count_sample_ids(cases: Iterable[Dict[str, Any]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for case in cases:
        sid = case.get("sample_id")
        if sid is None:
            continue
        try:
            sid = int(sid)
        except (TypeError, ValueError):
            continue
        counts[sid] = counts.get(sid, 0) + 1
    return counts


def _load_candidates(path: str) -> Dict[int, str]:
    if not os.path.exists(path):
        return {}
    mapping: Dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get("sample_id")
            code = row.get("code") or row.get("completion") or row.get("output")
            if sid is None or code is None:
                continue
            mapping[int(sid)] = code
    return mapping


def _check_property(prop: Dict[str, Any], value: Any) -> bool:
    if prop.get("type") == "list_int":
        if not isinstance(value, list):
            return False
        if prop.get("len") is not None and len(value) != prop["len"]:
            return False
        if prop.get("min") is not None and not all(v >= prop["min"] for v in value):
            return False
        if prop.get("max") is not None and not all(v <= prop["max"] for v in value):
            return False
        if not all(isinstance(v, int) for v in value):
            return False
    return True


def _case_passed(case: Dict[str, Any], code: str, runner_path: str) -> bool:
    kind = case.get("kind")
    if kind in {"expr", "expr_prop"}:
        got = _run_in_subprocess(code, runner_path, mode="eval_expr", expr=case["expr"])
        if not got.get("ok"):
            return False
        if kind == "expr_prop":
            if not _check_property(case["property"], _b64_to_obj(got["result_b64"])):
                return False
        else:
            if got.get("result_b64") != case.get("expected_b64"):
                return False
        return got.get("stdout", "") == case.get("expected_stdout", "")

    if kind in {"function", "function_prop"}:
        args = _b64_to_obj(case["args_b64"])
        got = _run_in_subprocess(code, runner_path, mode="call_func", name=case["func_name"], args=args, kwargs={})
        if kind == "function_prop":
            if not got.get("ok"):
                return False
            return _check_property(case["property"], _b64_to_obj(got["result_b64"]))
        if "exc_type" in case:
            return (not got.get("ok")) and got.get("exc_type") == case["exc_type"]
        if not got.get("ok"):
            return False
        if got.get("result_b64") != case.get("expected_b64"):
            return False
        return got.get("stdout", "") == case.get("expected_stdout", "")

    if kind == "class_ctor":
        ctor_args = _b64_to_obj(case["ctor_args_b64"])
        got = _run_in_subprocess(code, runner_path, mode="make_obj", name=case["class_name"], args=ctor_args)
        return bool(got.get("ok"))

    if kind == "class_state":
        ctor_args = _b64_to_obj(case["ctor_args_b64"])
        got = _run_in_subprocess(code, runner_path, mode="make_obj_state", name=case["class_name"], args=ctor_args)
        if not got.get("ok"):
            return False
        if got.get("result_b64") != case.get("expected_state_b64"):
            return False
        return got.get("stdout", "") == case.get("expected_stdout", "")

    if kind == "class_method":
        ctor_args = _b64_to_obj(case["ctor_args_b64"])
        got = _run_in_subprocess(
            code,
            runner_path,
            mode="make_and_call",
            name=case["class_name"],
            method_name=case["method"],
            args=ctor_args,
            kwargs={},
        )
        if "exc_type" in case:
            return (not got.get("ok")) and got.get("exc_type") == case["exc_type"]
        if not got.get("ok"):
            return False
        if got.get("result_b64") != case.get("expected_b64"):
            return False
        return got.get("stdout", "") == case.get("expected_stdout", "")

    return False


def compute_oracle_metrics(
    candidate_path: str,
    filter_to_candidates: bool = True,
    allowed_sample_ids: set[int] | None = None,
    max_cases_per_id: int | None = None,
) -> Dict[str, Any]:
    oracle_path, runner_path = _oracle_paths()
    cases = _load_oracle_cases(oracle_path)
    candidates = _load_candidates(candidate_path)
    candidate_ids = set(candidates)
    all_cases = _filter_oracle_cases(cases, sample_ids=None)
    selection_ids = set(allowed_sample_ids) if allowed_sample_ids is not None else None
    if selection_ids is None and filter_to_candidates:
        selection_ids = set(candidate_ids)
    eligible_ids = None
    if max_cases_per_id is not None:
        counts = _count_sample_ids(all_cases)
        eligible_ids = {sid for sid, count in counts.items() if count >= max_cases_per_id}
        selection_ids = eligible_ids if selection_ids is None else (selection_ids & eligible_ids)
    filtered_cases = _filter_oracle_cases(
        cases,
        sample_ids=selection_ids,
        max_cases_per_id=max_cases_per_id,
    )
    case_sample_ids = _case_sample_ids(filtered_cases)
    missing_oracle_samples = sorted((selection_ids or set()) - case_sample_ids)
    total = 0
    passed = 0
    missing = 0
    for case in filtered_cases:
        total += 1
        sid = int(case["sample_id"])
        code = candidates.get(sid)
        if code is None:
            missing += 1
            continue
        if _case_passed(case, code, runner_path):
            passed += 1
    evaluated = total - missing
    accuracy = passed / evaluated if evaluated else 0.0
    metrics = {
        "task": "codealpaca",
        "oracle_acc": accuracy,
        "passed": passed,
        "total": total,
        "oracle_total_cases": len(all_cases),
        "missing_candidates": missing,
        "missing_oracle_samples": len(missing_oracle_samples),
        "candidate_samples": len(candidate_ids),
        "oracle_samples": len(case_sample_ids),
        "filter_to_candidates": filter_to_candidates,
        "evaluated": evaluated,
        "candidate_path": candidate_path,
        "oracle_path": oracle_path,
    }
    if selection_ids is not None:
        metrics["requested_samples"] = len(selection_ids)
    if eligible_ids is not None:
        metrics["eligible_samples"] = len(eligible_ids)
    if max_cases_per_id is not None:
        metrics["oracle_cases_per_id"] = max_cases_per_id
    return metrics


def run_codealpaca_oracle_eval(
    candidate_path: str,
    metrics_path: str,
    run_pytest: bool = True,
    filter_to_candidates: bool = True,
    allowed_sample_ids: set[int] | None = None,
    max_cases_per_id: int | None = None,
) -> Dict[str, Any]:
    metrics = compute_oracle_metrics(
        candidate_path,
        filter_to_candidates=filter_to_candidates,
        allowed_sample_ids=allowed_sample_ids,
        max_cases_per_id=max_cases_per_id,
    )
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f)
    if run_pytest:
        env = os.environ.copy()
        env["CANDIDATE_PATH"] = candidate_path
        env["ORACLE_FILTER_TO_CANDIDATES"] = "1" if filter_to_candidates else "0"
        if max_cases_per_id is not None:
            env["ORACLE_CASES_PER_ID"] = str(max_cases_per_id)
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/test_oracle_cases.py"],
            env=env,
            timeout=8
        )
    return metrics
