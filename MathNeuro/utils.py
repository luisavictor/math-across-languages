import os, json

def save_only_resps(results: dict, out_dir: str):
    """
    Writes one JSONL per task:
      out_dir/<task>_resps.jsonl

    Each line: {"doc_id": <int>, "resps": <whatever lm-eval logged>}
    """
    samples_by_task = results.get("samples") or {}
    os.makedirs(out_dir, exist_ok=True)

    for task_name, samples in samples_by_task.items():
        out_path = os.path.join(out_dir, f"{task_name}_resps.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps({
                    "doc_id": s.get("doc_id"),
                    "resps": s.get("resps"),
                }, ensure_ascii=False) + "\n")

    return list(samples_by_task.keys())


def _first_text_resp(sample: dict):
    """
    Best-effort extraction of a human-readable response across different task types.
    - generate_until tasks often have `filtered_resps`
    - others may have `resps` (sometimes nested)
    """
    resp = None

    # Preferred: filtered_resps (often post-processed)
    fr = sample.get("filtered_resps", None)
    if fr is not None:
        # common shapes: [ "text" ] or [ ["text"] ] or [ ["text", ...] ]
        if isinstance(fr, list) and fr:
            x = fr[0]
            if isinstance(x, list):
                resp = x[0] if x else ""
            else:
                resp = x

    if resp is None:
        r = sample.get("resps", None)
        if isinstance(r, list) and r:
            x = r[0]
            if isinstance(x, list):
                resp = x[0] if x else ""
            else:
                resp = x

    # Some tasks store non-text (e.g., loglikelihoods); keep as-is
    return resp


def save_lm_eval_samples(results: dict, out_dir: str, dump_raw: bool = True):
    """
    Saves samples for every task present in results["samples"].

    Writes:
      out_dir/<task_name>_samples.jsonl
    """
    samples_by_task = results.get("samples") or {}
    os.makedirs(out_dir, exist_ok=True)

    for task_name, samples in samples_by_task.items():
        out_path = os.path.join(out_dir, f"{task_name}_samples.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                doc = s.get("doc") or {}
                record = {
                    "task": task_name,
                    "doc_id": s.get("doc_id"),
                    # keep a stable identifier if the task provides one
                    "sample_id": doc.get("sample_id", s.get("doc_id")),
                    "prediction": _first_text_resp(s),
                    "target": s.get("target"),
                }
                if dump_raw:
                    record["raw"] = s  # contains prompt/doc, filtered_resps, etc.
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return list(samples_by_task.keys())
 