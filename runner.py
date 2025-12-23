"""Lightweight runner executed in a subprocess to evaluate candidate code safely.

The runner accepts base64-pickled inputs, executes the requested operation,
and prints a base64-pickled result dict to stdout. This keeps the outer harness
simple and avoids importing untrusted code in-process.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import pickle
import sys
from typing import Any, Dict


def obj_to_b64(obj: Any) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode("utf-8")


def b64_to_obj(data: str) -> Any:
    return pickle.loads(base64.b64decode(data.encode("utf-8")))


def load_module(code_str: str) -> Dict[str, Any]:
    """Executes code_str in a fresh module namespace and returns that namespace."""
    mod: Dict[str, Any] = {"__name__": "__oracle__"}
    exec(code_str, mod, mod)
    return mod


def run(args: argparse.Namespace) -> Dict[str, Any]:
    code = b64_to_obj(args.code_b64)
    # If the decoded object is bytes, interpret as utf-8 text.
    if isinstance(code, (bytes, bytearray)):
        code = code.decode("utf-8")

    # Pre-decode common inputs.
    call_args = b64_to_obj(args.args_b64)
    call_kwargs = b64_to_obj(args.kwargs_b64)

    stdout_buffer = io.StringIO()
    payload: Dict[str, Any] = {"ok": True, "stdout": ""}
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            mod = load_module(code)
            if args.mode == "eval_expr":
                expr = b64_to_obj(args.expr_b64)
                payload["result_b64"] = obj_to_b64(eval(expr, mod, mod))
            elif args.mode == "call_func":
                fn = mod[args.name]
                res = fn(*call_args, **call_kwargs)
                payload["result_b64"] = obj_to_b64(res)
            elif args.mode == "make_obj":
                cls = mod[args.name]
                res = cls(*call_args, **call_kwargs)
                payload["result_b64"] = obj_to_b64(res)
            elif args.mode == "make_obj_state":
                cls = mod[args.name]
                obj = cls(*call_args, **call_kwargs)
                payload["result_b64"] = obj_to_b64(getattr(obj, "__dict__", {}))
            elif args.mode == "call_method":
                # expects first positional arg to be the object
                obj, *rest = call_args
                method = getattr(obj, args.name)
                res = method(*rest, **call_kwargs)
                payload["result_b64"] = obj_to_b64(res)
            elif args.mode == "make_and_call":
                cls = mod[args.name]
                obj = cls(*call_args, **call_kwargs)
                method = getattr(obj, args.method_name)
                res = method()
                payload["result_b64"] = obj_to_b64(res)
            elif args.mode == "get_state":
                obj = call_args[0] if isinstance(call_args, (list, tuple)) else call_args
                payload["result_b64"] = obj_to_b64(getattr(obj, "__dict__", {}))
            else:
                raise ValueError(f"Unsupported mode: {args.mode}")
    except Exception as exc:  # noqa: BLE001 - intentional broad capture for test harness
        payload = {
            "ok": False,
            "exc_type": type(exc).__name__,
            "exc_msg": str(exc),
        }
    payload["stdout"] = stdout_buffer.getvalue()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "eval_expr",
            "call_func",
            "make_obj",
            "make_obj_state",
            "call_method",
            "make_and_call",
            "get_state",
        ],
    )
    parser.add_argument("--code_b64", required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--method_name", default="")
    parser.add_argument("--expr_b64")
    parser.add_argument("--args_b64", default=obj_to_b64(()))
    parser.add_argument("--kwargs_b64", default=obj_to_b64({}))

    args = parser.parse_args()
    result = run(args)

    # Only emit the payload base64 to stdout.
    sys.stdout.write(obj_to_b64(result))


if __name__ == "__main__":
    main()
