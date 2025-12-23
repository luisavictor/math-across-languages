import json
import os
from pathlib import Path


def _load_oracle_sample_ids(path: str) -> set[int]:
    ids: set[int] = set()
    if not path:
        return ids
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = row.get("sample_id")
                if sid is None:
                    continue
                try:
                    ids.add(int(sid))
                except (TypeError, ValueError):
                    continue
    except FileNotFoundError:
        return set()
    return ids


def process_docs(dataset):
    oracle_ids = None
    oracle_path = os.environ.get("CODEALPACA_ORACLE_CASES_PATH")
    if oracle_path:
        if not os.path.isabs(oracle_path):
            root_dir = Path(__file__).resolve().parents[2]
            oracle_path = str(root_dir / oracle_path)
        loaded_ids = _load_oracle_sample_ids(oracle_path)
        if loaded_ids:
            oracle_ids = loaded_ids
    allowed_raw = os.environ.get("CODEALPACA_ALLOWED_SAMPLE_IDS")
    allowed_ids = None
    if allowed_raw is not None:
        ids: set[int] = set()
        for part in allowed_raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ids.add(int(part))
            except (TypeError, ValueError):
                continue
        allowed_ids = ids
    if oracle_ids is not None and allowed_ids is not None:
        oracle_ids = oracle_ids & allowed_ids
    elif allowed_ids is not None:
        oracle_ids = allowed_ids

    def _clean(doc, idx):
        prompt = str(doc.get("prompt", "")).strip()
        completion = str(doc.get("completion", "")).strip()
        return {
            "prompt": prompt,
            "completion": completion,
            "sample_id": idx,
        }

    dataset = dataset.map(_clean, with_indices=True)
    if oracle_ids is not None:
        dataset = dataset.filter(lambda ex: ex["sample_id"] in oracle_ids)
    return dataset

# Compute ROUGE-L F1 for a single prediction/target pair
def process_results(doc, results):
    from rouge_score import rouge_scorer

    prediction = str(results[0]).strip()
    target = str(doc.get("completion", "")).strip()

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(target, prediction)["rougeL"].fmeasure

    return {"rougeL": score}
