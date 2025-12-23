import base64
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "runner.py"
DEFAULT_CASES = ROOT / "oracle_cases.jsonl"
DEFAULT_CANDIDATES = ROOT / "candidate_generations.jsonl"


def b64_to_obj(data: str) -> Any:
    return pickle.loads(base64.b64decode(data.encode("utf-8")))


def obj_to_b64(obj: Any) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode("utf-8")


def load_cases(
    path: Path = DEFAULT_CASES,
    allowed_sample_ids: set[int] | None = None,
    max_cases_per_id: int | None = None,
) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        pytest.skip(f"oracle cases not found at {path}")
    if max_cases_per_id is None:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    case = json.loads(line)
                    if allowed_sample_ids is not None:
                        sid = case.get("sample_id")
                        if sid is None:
                            continue
                        try:
                            sid = int(sid)
                        except (TypeError, ValueError):
                            continue
                        if sid not in allowed_sample_ids:
                            continue
                    yield case
        return

    cases = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            case = json.loads(line)
            sid = case.get("sample_id")
            if sid is None:
                continue
            try:
                sid = int(sid)
            except (TypeError, ValueError):
                continue
            if allowed_sample_ids is not None and sid not in allowed_sample_ids:
                continue
            cases.append(case)

    counts: Dict[int, int] = {}
    for case in cases:
        sid = int(case["sample_id"])
        counts[sid] = counts.get(sid, 0) + 1

    eligible_ids = {sid for sid, count in counts.items() if count >= max_cases_per_id}
    per_id_counts: Dict[int, int] = {}
    for case in cases:
        sid = int(case["sample_id"])
        if sid not in eligible_ids:
            continue
        if per_id_counts.get(sid, 0) >= max_cases_per_id:
            continue
        per_id_counts[sid] = per_id_counts.get(sid, 0) + 1
        yield case


def load_candidates(path: Path) -> Dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"candidate generations file not found: {path}")
    mapping: Dict[int, str] = {}
    with path.open(encoding="utf-8") as f:
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


@pytest.fixture(scope="session")
def candidate_code_by_id() -> Dict[int, str]:
    path = Path(os.environ.get("CANDIDATE_PATH", DEFAULT_CANDIDATES))
    return load_candidates(path)


def run_in_subprocess(
    code: str,
    mode: str,
    name: str = "",
    method_name: str = "",
    expr: str = "",
    args: tuple = (),
    kwargs: dict | None = None,
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
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if cp.returncode != 0:
        return {"ok": False, "exc_type": "RunnerFailure", "exc_msg": cp.stderr, "stdout": ""}
    try:
        return b64_to_obj(cp.stdout.strip())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "exc_type": "DecodeError", "exc_msg": str(exc), "stdout": cp.stdout}


def assert_property(prop: Dict[str, Any], value: Any) -> None:
    if prop.get("type") == "list_int":
        assert isinstance(value, list)
        if prop.get("len") is not None:
            assert len(value) == prop["len"]
        if prop.get("min") is not None:
            assert all(v >= prop["min"] for v in value)
        if prop.get("max") is not None:
            assert all(v <= prop["max"] for v in value)
        assert all(isinstance(v, int) for v in value)
    # "any" type means no additional checks


def test_candidate_against_oracle_cases(candidate_code_by_id: Dict[int, str]) -> None:
    filter_to_candidates = os.environ.get("ORACLE_FILTER_TO_CANDIDATES", "1") != "0"
    allowed_ids = set(candidate_code_by_id) if filter_to_candidates else None
    max_cases_per_id = None
    cases_per_id_raw = os.environ.get("ORACLE_CASES_PER_ID")
    if cases_per_id_raw:
        try:
            max_cases_per_id = int(cases_per_id_raw)
        except ValueError:
            max_cases_per_id = None
    for case in load_cases(allowed_sample_ids=allowed_ids, max_cases_per_id=max_cases_per_id):
        sid = int(case["sample_id"])
        assert sid in candidate_code_by_id, f"missing candidate code for sample {sid}"
        code = candidate_code_by_id[sid]

        if case["kind"] in {"expr", "expr_prop"}:
            got = run_in_subprocess(code, mode="eval_expr", expr=case["expr"])
            assert got.get("ok"), got
            if case["kind"] == "expr_prop":
                assert_property(case["property"], b64_to_obj(got["result_b64"]))
            else:
                assert got.get("result_b64") == case["expected_b64"]
            assert got.get("stdout", "") == case.get("expected_stdout", "")

        elif case["kind"] in {"function", "function_prop"}:
            args = b64_to_obj(case["args_b64"])
            got = run_in_subprocess(code, mode="call_func", name=case["func_name"], args=args, kwargs={})
            if case["kind"] == "function_prop":
                assert got.get("ok"), got
                assert_property(case["property"], b64_to_obj(got["result_b64"]))
            else:
                if "exc_type" in case:
                    assert not got.get("ok") and got.get("exc_type") == case["exc_type"]
                else:
                    assert got.get("ok"), got
                    assert got.get("result_b64") == case["expected_b64"]
                    assert got.get("stdout", "") == case.get("expected_stdout", "")

        elif case["kind"] == "class_ctor":
            ctor_args = b64_to_obj(case["ctor_args_b64"])
            got = run_in_subprocess(code, mode="make_obj", name=case["class_name"], args=ctor_args, kwargs={})
            assert got.get("ok"), got
        elif case["kind"] == "class_state":
            ctor_args = b64_to_obj(case["ctor_args_b64"])
            got = run_in_subprocess(
                code,
                mode="make_obj_state",
                name=case["class_name"],
                args=ctor_args,
                kwargs={},
            )
            assert got.get("ok"), got
            assert got.get("result_b64") == case["expected_state_b64"]
            assert got.get("stdout", "") == case.get("expected_stdout", "")

        elif case["kind"] == "class_method":
            ctor_args = b64_to_obj(case["ctor_args_b64"])
            got = run_in_subprocess(
                code,
                mode="make_and_call",
                name=case["class_name"],
                method_name=case["method"],
                args=ctor_args,
                kwargs={},
            )
            if "exc_type" in case:
                assert not got.get("ok") and got.get("exc_type") == case["exc_type"]
            else:
                assert got.get("ok"), got
                assert got.get("result_b64") == case["expected_b64"]
                assert got.get("stdout", "") == case.get("expected_stdout", "")

        else:
            pytest.skip(f"Unhandled case kind: {case['kind']}")
