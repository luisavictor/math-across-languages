import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

import lm_eval
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _to_bool_metric(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value >= 0.5)
    return None


def _extract_correct_flag(sample: Dict[str, Any]) -> Optional[bool]:
    for key in ("exact_match", "acc", "accuracy", "em", "pass@1"):
        if key in sample:
            parsed = _to_bool_metric(sample[key])
            if parsed is not None:
                return parsed

    filtered_resps = sample.get("filtered_resps")
    if (
        isinstance(filtered_resps, list)
        and filtered_resps
        and isinstance(filtered_resps[0], (list, tuple))
        and len(filtered_resps[0]) >= 2
        and isinstance(filtered_resps[0][1], bool)
    ):
        return filtered_resps[0][1]

    return None


def _extract_doc_id(sample: Dict[str, Any]) -> Optional[int]:
    doc = sample.get("doc", {})
    sample_id = None
    if isinstance(doc, dict):
        sample_id = doc.get("sample_id")
    if sample_id is None:
        sample_id = sample.get("doc_id")
    try:
        return int(sample_id)
    except (TypeError, ValueError):
        return None


def _collect_groups(samples: Iterable[Dict[str, Any]], dataset_size: int) -> tuple[Set[int], Set[int]]:
    # lm_eval can log the same doc multiple times (e.g., once per filter).
    # Aggregate all flags per doc_id, then map each doc_id to exactly one group.
    per_doc_flags: Dict[int, list[bool]] = {}

    for sample in samples:
        doc_id = _extract_doc_id(sample)
        flag = _extract_correct_flag(sample)
        if doc_id is None or flag is None:
            continue
        if doc_id < 0 or doc_id >= dataset_size:
            continue
        per_doc_flags.setdefault(doc_id, []).append(flag)

    correct_ids: Set[int] = set()
    incorrect_ids: Set[int] = set()

    for doc_id, flags in per_doc_flags.items():
        # If any filter judged a sample as correct, keep it in correct bucket.
        # Otherwise, it is incorrect.
        if any(flags):
            correct_ids.add(doc_id)
        else:
            incorrect_ids.add(doc_id)

    return correct_ids, incorrect_ids


def _evaluate_training_subset(
    model,
    tokenizer,
    task_name: str,
    task_manager,
    batch_size: int,
    random_seed: int,
    limit: int,
) -> Dict[str, Any]:
    return lm_eval.simple_evaluate(
        model="hf",
        model_args={"pretrained": model, "dtype": "bfloat16", "tokenizer": tokenizer},
        tasks=task_name,
        task_manager=task_manager,
        log_samples=True,
        batch_size=batch_size,
        limit=limit,
        random_seed=random_seed,
    )


def _safe_model_tag(model: Any) -> str:
    if isinstance(model, str):
        name = model
    else:
        name = getattr(model, "name_or_path", None)
        if name is None and hasattr(model, "config"):
            name = getattr(model.config, "_name_or_path", None)
        if name is None:
            name = "model"
    return str(name).replace("/", "_").replace("\\", "_")


def prepare_filtered_training_data(
    train_df: Optional[pd.DataFrame] = None,
    model: Any = None,
    task_name: Optional[str] = None,
    output_dir: Any = "filtered_training_data",
    samples_per_group: Optional[int] = 500,
    batch_size: int = 1,
    random_seed: int = 1,
    tokenizer: Any = None,
    task_manager: Any = None,
    train_data: Optional[str] = None,
) -> Dict[str, Any]:
    if task_name is None:
        raise ValueError("task_name must be provided.")
    if model is None:
        raise ValueError("model must be provided.")

    if train_df is None:
        if train_data is None:
            raise ValueError("Provide either train_df or train_data.")
        train_df = pd.read_csv(train_data)
    train_df = train_df.reset_index(drop=True)

    if isinstance(model, str):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(
            model, device_map="auto", torch_dtype=torch.bfloat16
        )
        model.eval()
    else:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when model is a preloaded model object.")
        model.eval()

    if task_manager is None:
        task_manager = lm_eval.tasks.TaskManager(include_path="../lm_eval_tasks")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_tag = _safe_model_tag(model)
    correct_path = output_dir / f"train_correct_{model_tag}.csv"
    incorrect_path = output_dir / f"train_incorrect_{model_tag}.csv"
    summary_path = output_dir / "filter_summary.json"

    dataset_size = len(train_df)
    if dataset_size == 0:
        raise ValueError("Training dataset is empty.")

    if samples_per_group is not None and samples_per_group <= 0:
        raise ValueError("samples_per_group must be > 0 or None.")

    # If samples_per_group is None, evaluate full dataset once and keep all per group.
    if samples_per_group is None:
        initial_limit = dataset_size
        step = dataset_size
        target_per_group = None
    else:
        target_per_group = int(samples_per_group)
        initial_limit = min(dataset_size, target_per_group * 4)
        step = max(target_per_group, 250)

    limit = initial_limit

    best_correct: Set[int] = set()
    best_incorrect: Set[int] = set()

    while True:
        print(f"Running lm_eval to split training samples (limit={limit})...")
        results = _evaluate_training_subset(
            model=model,
            tokenizer=tokenizer,
            task_name=task_name,
            task_manager=task_manager,
            batch_size=batch_size,
            random_seed=random_seed,
            limit=limit,
        )
        task_samples = results.get("samples", {}).get(task_name, [])
        correct_ids, incorrect_ids = _collect_groups(task_samples, dataset_size=dataset_size)

        if len(correct_ids) > len(best_correct):
            best_correct = correct_ids
        if len(incorrect_ids) > len(best_incorrect):
            best_incorrect = incorrect_ids

        if (
            target_per_group is not None
            and len(best_correct) >= target_per_group
            and len(best_incorrect) >= target_per_group
        ):
            break
        if limit >= dataset_size:
            break

        limit = min(dataset_size, limit + step)

    # Safety: keep groups disjoint even if future scoring logic changes.
    overlap_ids = best_correct & best_incorrect
    if overlap_ids:
        best_incorrect = best_incorrect - overlap_ids

    # Keep all available per group if target is None. Otherwise cap by requested count.
    if target_per_group is None:
        selected_correct_ids = sorted(best_correct)
        selected_incorrect_ids = sorted(best_incorrect)
    else:
        selected_correct_ids = sorted(best_correct)[:target_per_group]
        selected_incorrect_ids = sorted(best_incorrect)[:target_per_group]

    train_reset = train_df.copy()
    train_reset["_doc_id"] = train_reset.index

    correct_df = train_reset.iloc[selected_correct_ids].copy()
    incorrect_df = train_reset.iloc[selected_incorrect_ids].copy()

    correct_df.to_csv(correct_path, index=False)
    incorrect_df.to_csv(incorrect_path, index=False)

    summary = {
        "task_name": task_name,
        "dataset_size": dataset_size,
        "limit_used": limit,
        "available_correct_count": len(best_correct),
        "available_incorrect_count": len(best_incorrect),
        "correct_count": len(correct_df),
        "incorrect_count": len(incorrect_df),
        "samples_per_group": samples_per_group,
        "random_seed": random_seed,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "correct_path": str(correct_path),
        "incorrect_path": str(incorrect_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="data/gsm8k.csv")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--task_name", type=str, default="gsm8k_cot_train")
    parser.add_argument("--output_dir", type=str, default="filtered_training_data")
    parser.add_argument("--samples_per_group", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--random_seed", type=int, default=1)
    args = parser.parse_args()

    prepare_filtered_training_data(
        train_data=args.train_data,
        model=args.model,
        task_name=args.task_name,
        output_dir=args.output_dir,
        samples_per_group=args.samples_per_group,
        batch_size=args.batch_size,
        random_seed=args.random_seed,
    )
