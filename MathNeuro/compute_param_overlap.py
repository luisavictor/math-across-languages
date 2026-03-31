'''
Given the .pt files, this script allows to compute the Jaccard similarity of two parameter sets, e.g., of English and German math specific params.
'''

import torch
from pathlib import Path
import json
import csv
import re
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os

# ── Score-file cache (avoids re-loading the same .pt for every good_percent) ──
_score_cache: dict[str, dict] = {}


def _load_scores_cached(path: str) -> dict:
    """Load a score .pt file, returning cached copy if already loaded."""
    if path not in _score_cache:
        _score_cache[path] = torch.load(path, map_location="cpu")
    return _score_cache[path]


def clear_score_cache():
    """Free memory occupied by cached score tensors."""
    _score_cache.clear()

def load_mask(path):
    print("Loading:", os.path.abspath(path))
    print("Exists:", os.path.exists(path))
    print("Size:", os.path.getsize(path) if os.path.exists(path) else "N/A")
    data = torch.load(path, map_location="cpu")
    return data["isolated_masks"], data


def _as_bool_mask(mask_dict: dict) -> dict:
    """
    Normalize masks to boolean tensors.
    Accepts dict[name -> tensor/bool/float/int], returns dict[name -> bool tensor].
    """
    return {k: (v == 0).to(torch.bool) if v.dtype != torch.bool else v for k, v in mask_dict.items()}


def compute_overlap(mask1, mask2):
    """
    mask1, mask2: dict[name -> tensor] with 1 = isolated, 0 = not
    Returns: dict with counts and ratios.
    """
    total1 = 0
    total2 = 0
    intersection = 0
    union = 0

    # ensure we only compare keys that exist in both
    keys = set(mask1.keys()) & set(mask2.keys())

    for k in keys:
        m1 = mask1[k].bool()
        m2 = mask2[k].bool()

        # basic sanity: shapes must match
        if m1.shape != m2.shape:
            print(f"Skipping {k}: shape mismatch {m1.shape} vs {m2.shape}")
            continue

        # count isolated positions per run
        total1 += m1.sum().item()
        total2 += m2.sum().item()

        # intersection & union
        inter = (m1 & m2).sum().item()
        uni   = (m1 | m2).sum().item()

        intersection += inter
        union += uni

    jaccard = intersection / union if union > 0 else 0.0

    return {
        "total_isolated_run1": total1,
        "total_isolated_run2": total2,
        "intersection": intersection,
        "union": union,
        "jaccard": jaccard,
    }


def find_good_params(keep_ratio, prune=True, largest=True, param_dict = "good_scores"):
    global chosen_params
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    # create dictionary to store mask
    mask_dict = {}

    for k, v in param_dict.items():
        # don't count classifier layer
        if "embed" in k:
            if prune == False:
                mask_dict[k] = torch.zeros_like(v).to(v.device)
            else:
                mask_dict[k] = torch.ones_like(v).to(v.device)

        else:
            if prune == False:
                sizes = v.shape
                num_params = v.numel()
                keep_num = int(num_params * keep_ratio)
                tensor = v.view(-1)
                top_pos = torch.topk(torch.abs(tensor), keep_num, largest=largest)[1]
                mask_dict[k] = torch.zeros_like(tensor, device=tensor.device)
                mask_dict[k][top_pos] = 1
                mask_dict[k] = mask_dict[k].reshape(v.shape).to(tensor.device)
            else:
                sizes = v.shape
                num_params = v.numel()
                keep_num = int(num_params * keep_ratio)
                tensor = v.view(-1)
                top_pos = torch.topk(torch.abs(tensor), keep_num, largest=largest)[1]
                mask_dict[k] = torch.ones_like(tensor, device='cpu')
                mask_dict[k][top_pos] = 0
                mask_dict[k] = mask_dict[k].reshape(v.shape).to('cpu')

    return mask_dict


def prune(bad_params, good_params, factor, return_good=False):
    prune_params = {}
    if return_good == False:
        for k, v in bad_params.items():
            prune_params[k] = bad_params[k] - good_params[k]
            indices = prune_params[k] != -1
            bad_indices = prune_params[k] == -1
            prune_params[k] = indices + (bad_indices * factor)

    else:
        for k, v in bad_params.items():
            prune_params[k] = good_params[k] - bad_params[k]
            indices = prune_params[k] != -1
            good_indices = prune_params[k] == -1
            prune_params[k] = indices + (good_indices * factor)
    return prune_params



def layer_group_name(name: str) -> str:
    """
    Map parameter names to groups.
    Adjust regex if your param names differ.
    Common HF Llama keys look like:
      model.embed_tokens.weight
      model.layers.0.self_attn.q_proj.weight
      model.layers.0.mlp.up_proj.weight
      model.norm.weight
      lm_head.weight
    """
    # Main transformer blocks
    m = re.search(r"(?:^|\.)(layers)\.(\d+)(?:\.|$)", name)
    if m:
        return f"layers.{int(m.group(2))}"

    # Embeddings
    if "embed_tokens" in name or name.startswith("model.embed_tokens"):
        return "embed_tokens"

    # Final norm
    if name.startswith("model.norm") or name.endswith(".norm.weight") or name.endswith(".norm.bias"):
        return "final_norm"

    # Output head
    if name.startswith("lm_head") or "lm_head" in name:
        return "lm_head"

    # Anything else (rotary, adapters, etc.)
    return "other"

def compute_overlap_per_group(mask1, mask2, group_fn=layer_group_name):
    """
    Compute Jaccard etc. per group (e.g., per transformer layer).
    Returns:
      per_group: dict[group -> stats dict]
      global_stats: stats dict across all compared keys
    """
    # Keys we can compare
    keys = set(mask1.keys()) & set(mask2.keys())

    # Accumulators per group
    acc = defaultdict(lambda: {
        "total_isolated_run1": 0,
        "total_isolated_run2": 0,
        "intersection": 0,
        "union": 0,
        "num_tensors": 0,
        "num_elements_compared": 0,
    })

    # Also global accumulator (same structure as your compute_overlap)
    global_total1 = global_total2 = global_inter = global_union = 0

    for k in keys:
        m1 = mask1[k].bool()
        m2 = mask2[k].bool()

        if m1.shape != m2.shape:
            print(f"Skipping {k}: shape mismatch {m1.shape} vs {m2.shape}")
            continue

        g = group_fn(k)

        t1 = m1.sum().item()
        t2 = m2.sum().item()
        inter = (m1 & m2).sum().item()
        uni   = (m1 | m2).sum().item()
        n_el  = m1.numel()

        acc[g]["total_isolated_run1"] += t1
        acc[g]["total_isolated_run2"] += t2
        acc[g]["intersection"] += inter
        acc[g]["union"] += uni
        acc[g]["num_tensors"] += 1
        acc[g]["num_elements_compared"] += n_el

        global_total1 += t1
        global_total2 += t2
        global_inter  += inter
        global_union  += uni

    # finalize jaccard per group
    per_group = {}
    for g, s in acc.items():
        j = s["intersection"] / s["union"] if s["union"] > 0 else 0.0
        per_group[g] = {**s, "jaccard": j}

    global_stats = {
        "total_isolated_run1": global_total1,
        "total_isolated_run2": global_total2,
        "intersection": global_inter,
        "union": global_union,
        "jaccard": (global_inter / global_union) if global_union > 0 else 0.0,
    }

    return per_group, global_stats

_layer_re = re.compile(r"^layers\.(\d+)$")

def group_sort_key(gname: str):
    m = _layer_re.match(gname)
    if m:
        return (0, int(m.group(1)))   # layers in numeric order
    return (1, gname)                 # everything else after, alphabetically

def sort_per_group(per_group: dict):
    return OrderedDict(
        (k, per_group[k]) for k in sorted(per_group.keys(), key=group_sort_key)
    )


def build_isolated_masks(mask_dir: str, good_percent: float, seed: int = 1):
    """
    Loads train/comparison score dicts and returns isolated_masks (dict[name->bool tensor]).
    Uses _score_cache to avoid redundant torch.load calls across good_percent values.
    """
    mask_dir = Path(mask_dir)

    train_path = str(mask_dir / f"train_scores_seed{seed}.pt")
    comp_path  = str(mask_dir / f"comparison_scores_seed{seed}.pt")

    param_good = _load_scores_cached(train_path)
    param_comp = _load_scores_cached(comp_path)

    good_params = find_good_params(
        keep_ratio=good_percent, prune=True, largest=True, param_dict=param_good
    )
    comp_params = find_good_params(
        keep_ratio=good_percent, prune=True, largest=True, param_dict=param_comp
    )

    # math-specific params
    isolated_zero_mask = prune(comp_params, good_params, factor=0.0, return_good=True)

    isolated_masks = {k: (m == 0).to(torch.bool) for k, m in isolated_zero_mask.items()}
    del isolated_zero_mask
    print("new isolated mask returned")

    return isolated_masks


def load_isolated_masks_auto(path_or_dir: str, good_percent: float = None, seed: int = 1):
    """
    Flexible loader:
      - If path points to a .pt with "isolated_masks", loads them directly.
      - If path points to a directory, uses train/comparison score checkpoints.
    """
    p = Path(path_or_dir)

    # If it's a file, try to read isolated_masks directly
    if p.is_file():
        data = torch.load(p, map_location="cpu")
        if isinstance(data, dict) and "isolated_masks" in data:
            return _as_bool_mask(data["isolated_masks"]), data
        raise ValueError(f"File {p} does not contain 'isolated_masks'.")

    # If it's a directory, fall back to train/comparison scores
    if p.is_dir():
        if good_percent is None:
            raise ValueError("good_percent must be provided when using score checkpoints.")
        masks = build_isolated_masks(str(p), good_percent, seed=seed)
        return masks, {"source": "score_checkpoints", "good_percent": good_percent, "seed": seed}

    raise FileNotFoundError(f"No such file or directory: {p}")


def _infer_good_percent(meta: dict, fallback: float = None) -> float:
    if isinstance(meta, dict):
        if "good_percent" in meta:
            return float(meta["good_percent"])
        if "good_ratio" in meta:
            return float(meta["good_ratio"])
        if "keep_ratio" in meta:
            return float(meta["keep_ratio"])
    return fallback


def _write_jaccard_outputs(all_results, per_layer_rows, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "jaccard_summary.json"
    csv_path = out_dir / "jaccard_per_layer.csv"

    # append to existing JSON if present
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        all_results = existing + all_results
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    if per_layer_rows:
        fieldnames = list(per_layer_rows[0].keys())
        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(per_layer_rows)


def run_jaccard_sweep(
    good_percents,
    mask_dir_lang1,
    mask_dir_lang2,
    out_dir,
    seed,
    group_fn=None,
):
    """
    Computes global + per-group Jaccard for each good_percent and stores results.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if group_fn is None:
        # default to your per-layer grouping function
        group_fn = layer_group_name

    json_path = out_dir / f"jaccard_summary_{seed}.json"
    csv_path = out_dir / f"jaccard_per_layer_{seed}.csv"

    all_results = []  # list of dicts (one per good_percent)
    per_layer_rows = []  # flattened rows for CSV

    for gp in good_percents:
        # build masks (score checkpoints)
        p1 = str(mask_dir_lang1)
        if "{gp}" in p1:
             p1 = p1.format(gp=gp)
        
        p2 = str(mask_dir_lang2)
        if "{gp}" in p2:
             p2 = p2.format(gp=gp)

        isolated_masks_lang1, _ = load_isolated_masks_auto(p1, gp, seed=seed)
        isolated_masks_lang2, _ = load_isolated_masks_auto(p2, gp, seed=seed)

        # global
        global_stats = compute_overlap(isolated_masks_lang1, isolated_masks_lang2)

        # per-layer / per-group
        per_group, global_from_groups = compute_overlap_per_group(
            isolated_masks_lang1, isolated_masks_lang2, group_fn=group_fn
        )

        per_group = sort_per_group(per_group)

        # package result for this gp
        record = {
            "good_percent": float(gp),
            "seed": int(seed),
            "mask_dir_lang1": str(mask_dir_lang1),
            "mask_dir_lang2": str(mask_dir_lang2),
            "global": global_stats,
            "per_group": per_group,
        }
        all_results.append(record)

        # flatten per-group for CSV
        for group_name, s in per_group.items():
            per_layer_rows.append({
                "good_percent": float(gp),
                "seed": int(seed),
                "group": group_name,
                "jaccard": float(s["jaccard"]),
                "intersection": int(s["intersection"]),
                "union": int(s["union"]),
                "total_isolated_run1": int(s["total_isolated_run1"]),
                "total_isolated_run2": int(s["total_isolated_run2"]),
                "num_tensors": int(s.get("num_tensors", 0)),
                "num_elements_compared": int(s.get("num_elements_compared", 0)),
            })

        # optional: quick console output per gp
        print(f"\n=== good_percent={gp} ===")
        print(f"Global Jaccard: {global_stats['jaccard']:.6f} "
              f"(union={global_stats['union']}, inter={global_stats['intersection']})")
        print(f"Sanity global-from-groups: {global_from_groups['jaccard']:.6f}")

    _write_jaccard_outputs(all_results, per_layer_rows, out_dir)

    return all_results, per_layer_rows


def run_jaccard_single(
    path_or_dir_lang1,
    path_or_dir_lang2,
    out_dir,
    good_percent: float = None,
    seed: int = 1,
    group_fn=None,
):
    """
    Single-run helper that writes the same output structure as the template.
    Works for mask-only .pt files or score checkpoint dirs.
    """
    if group_fn is None:
        group_fn = layer_group_name

    mask1, meta1 = load_isolated_masks_auto(path_or_dir_lang1, good_percent, seed=seed)
    mask2, meta2 = load_isolated_masks_auto(path_or_dir_lang2, good_percent, seed=seed)

    inferred_gp = _infer_good_percent(meta1, fallback=good_percent)
    inferred_gp = _infer_good_percent(meta2, fallback=inferred_gp)

    if inferred_gp is None:
        raise ValueError("good_percent is required for output when not present in metadata.")

    global_stats = compute_overlap(mask1, mask2)
    per_group, _ = compute_overlap_per_group(mask1, mask2, group_fn=group_fn)
    per_group = sort_per_group(per_group)

    record = {
        "good_percent": float(inferred_gp),
        "seed": int(seed),
        "mask_dir_lang1": str(path_or_dir_lang1),
        "mask_dir_lang2": str(path_or_dir_lang2),
        "global": global_stats,
        "per_group": per_group,
    }

    per_layer_rows = []
    for group_name, s in per_group.items():
        per_layer_rows.append({
            "good_percent": float(inferred_gp),
            "seed": int(seed),
            "group": group_name,
            "jaccard": float(s["jaccard"]),
            "intersection": int(s["intersection"]),
            "union": int(s["union"]),
            "total_isolated_run1": int(s["total_isolated_run1"]),
            "total_isolated_run2": int(s["total_isolated_run2"]),
            "num_tensors": int(s.get("num_tensors", 0)),
            "num_elements_compared": int(s.get("num_elements_compared", 0)),
        })

    _write_jaccard_outputs([record], per_layer_rows, out_dir)
    return record, per_layer_rows



def detect_model_name_and_language(path1: str, path2: str):
    model_name = None
    language1 = None
    language2 = None

    if "Llama-3.2-1B" in path1:
        model_name = "Llama-1B"
    elif "Llama-3.1-8B" in path1:
        model_name = "Llama-8B"
    elif "Qwen3-4B" in path1:
        model_name = "Qwen-4B"
    else:
        raise ValueError("Unknown model in path1")

    if "german" in path1.lower():
        language1 = "de"
    elif "hindi" in path1.lower():
        language1 = "hi"
    elif "french" in path1.lower():
        language1 = "fr"
    else:
        language1 = "en"

    if "german" in path2.lower():
        language2 = "de"
    elif "hindi" in path2.lower():
        language2 = "hi"
    elif "french" in path2.lower():
        language2 = "fr"
    else:
        language2 = "en"

    return model_name, language1, language2

def path_pair_generator(path_list: list):
    """
    Generator that yields tuples of (path1, path2)
    """
    for i in range(len(path_list)):
        for j in range(i + 1, len(path_list)):
            yield path_list[i], path_list[j]


# ─── Parallel helpers ───────────────────────────────────────────────────────────

def _worker_jaccard(job: dict) -> dict:
    """
    Compute Jaccard for one (path1, path2, good_percent) combo.
    Returns the full record and per-layer rows WITHOUT writing to disk,
    so that we can collect all results and write them grouped by out_dir
    afterwards (avoids race conditions on the same JSON file).
    """
    path1       = job["path1"]
    path2       = job["path2"]
    gp          = job["good_percent"]
    seed        = job["seed"]
    out_dir     = job["out_dir"]
    group_fn    = layer_group_name

    mask1, meta1 = load_isolated_masks_auto(path1, gp, seed=seed)
    mask2, meta2 = load_isolated_masks_auto(path2, gp, seed=seed)

    inferred_gp = _infer_good_percent(meta1, fallback=gp)
    inferred_gp = _infer_good_percent(meta2, fallback=inferred_gp)

    global_stats = compute_overlap(mask1, mask2)
    per_group, _ = compute_overlap_per_group(mask1, mask2, group_fn=group_fn)
    per_group = sort_per_group(per_group)

    record = {
        "good_percent": float(inferred_gp),
        "seed": int(seed),
        "mask_dir_lang1": str(path1),
        "mask_dir_lang2": str(path2),
        "global": global_stats,
        "per_group": per_group,
    }

    per_layer_rows = []
    for group_name, s in per_group.items():
        per_layer_rows.append({
            "good_percent": float(inferred_gp),
            "seed": int(seed),
            "group": group_name,
            "jaccard": float(s["jaccard"]),
            "intersection": int(s["intersection"]),
            "union": int(s["union"]),
            "total_isolated_run1": int(s["total_isolated_run1"]),
            "total_isolated_run2": int(s["total_isolated_run2"]),
            "num_tensors": int(s.get("num_tensors", 0)),
            "num_elements_compared": int(s.get("num_elements_compared", 0)),
        })

    return {
        "out_dir": out_dir,
        "record": record,
        "per_layer_rows": per_layer_rows,
    }


def _preload_scores_for_paths(paths: list, seed: int = 1):
    """
    Eagerly load (and cache) all score .pt files for a list of directories
    so that subsequent build_isolated_masks calls hit the cache.
    """
    for p in paths:
        d = Path(p)
        if d.is_dir():
            train_path = str(d / f"train_scores_seed{seed}.pt")
            comp_path  = str(d / f"comparison_scores_seed{seed}.pt")
            _load_scores_cached(train_path)
            _load_scores_cached(comp_path)


def run_jaccard_batch_parallel(
    jobs: list[dict],
    max_workers: int = None,
    use_threads: bool = True,
):
    """
    Run many independent Jaccard jobs in parallel.

    Parameters
    ----------
    jobs : list of dicts, each with keys:
        path1, path2, good_percent, seed, out_dir
    max_workers : int or None
        Number of parallel workers.  Defaults to min(len(jobs), cpu_count).
    use_threads : bool
        True  -> ThreadPoolExecutor  (shared memory, benefits from cached .pt files)
        False -> ProcessPoolExecutor (true parallelism, but no shared cache)

    For the directory-based workflow, prefer use_threads=True so that
    _score_cache is shared across workers and each .pt is loaded only once.
    """
    from concurrent.futures import ThreadPoolExecutor

    if max_workers is None:
        max_workers = min(len(jobs), multiprocessing.cpu_count())

    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    results = []
    with Executor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker_jaccard, job): job for job in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Jaccard jobs"):
            res = fut.result()
            results.append(res)
    return results


def _write_collected_results(results: list):
    """
    Group worker results by out_dir and write one jaccard_summary.json + CSV
    per directory, producing exactly the same format as the original sequential code.
    """
    grouped = defaultdict(lambda: {"records": [], "rows": []})
    for r in results:
        grouped[r["out_dir"]]["records"].append(r["record"])
        grouped[r["out_dir"]]["rows"].extend(r["per_layer_rows"])

    # Sort records within each group by good_percent for deterministic output
    for out_dir, data in grouped.items():
        data["records"].sort(key=lambda x: x["good_percent"])
        data["rows"].sort(key=lambda x: (x["good_percent"], x["group"]))
        _write_jaccard_outputs(data["records"], data["rows"], out_dir)
        print(f"Wrote {len(data['records'])} records to {out_dir}/jaccard_summary.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute Jaccard overlap")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel workers (default: CPU count)")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallelism, run sequentially")
    args = parser.parse_args()

    repeat = 0
    MODE = "language"  # "correct_incorrect" or "language"
    MODELS = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen3-4B-Instruct-2507",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]
    good_percents = [0.0001, 0.001, 0.01, 0.025, 0.05, 0.1, 0.15]

    # ── Build job lists for both correct_only and false_only ──
    configs = [
        # {
        #     "label": "correct_only",
        #     "out_dir_template": "/raid/s3/opengptx/behzad_shomali/LabTest/results_jaccard/correct_only/gsm8k_race_{language1}_vs_{language2}_{model_name}/repeat{repeat}",
        #     "path_template": "/raid/s3/opengptx/behzad_shomali/LabTest/results_correct_vs_incorrect/prune/correct_only/{dataset}/isolated_masks/{model}/",
        # },
        # {
        #     "label": "false_only",
        #     "out_dir_template": "/raid/s3/opengptx/behzad_shomali/LabTest/results_jaccard/false_only/gsm8k_race_{language1}_vs_{language2}_{model_name}/repeat{repeat}",
        #     "path_template": "/raid/s3/opengptx/behzad_shomali/LabTest/results_correct_vs_incorrect/prune/false_only/{dataset}/isolated_masks/{model}/",
        # },
        {
            "label": "math_specific",
            "out_dir_template": "/raid/s3/opengptx/behzad_shomali/LabTest/results_jaccard/gsm8k_race_{language1}_vs_{language2}_{model_name}/repeat{repeat}",
            "path_template": "/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/{dataset}/isolated_masks/{model}/",
        }
    ]

    DATASETS = [
        "results_gsm8k_race",           # en
        "results_gsm8k_race_german",    # de
        "results_gsm8k_race_hindi_max300",  # hi
        "results_gsm8k_race_french"     # fr
    ]

    all_jobs = []
    for cfg in configs:
        for model in MODELS:
            paths = [cfg["path_template"].format(dataset=ds, model=model) for ds in DATASETS]

            # Pre-load score files once per model (fills the cache)
            _preload_scores_for_paths(paths, seed=1)
            print(f"[{cfg['label']}] Cached scores for {model}")

            for gp in good_percents:
                for path1, path2 in path_pair_generator(paths):
                    model_name, lang1, lang2 = detect_model_name_and_language(path1, path2)
                    out_dir = cfg["out_dir_template"].format(
                        language1=lang1, language2=lang2,
                        model_name=model_name, repeat=repeat,
                    )
                    all_jobs.append({
                        "path1": path1,
                        "path2": path2,
                        "good_percent": gp,
                        "seed": 1,
                        "out_dir": out_dir,
                    })

    print(f"\nTotal Jaccard jobs: {len(all_jobs)}")

    if args.no_parallel:
        # Sequential fallback — still collect results and write at the end
        results = []
        for job in tqdm(all_jobs, desc="Jaccard (sequential)"):
            results.append(_worker_jaccard(job))
    else:
        results = run_jaccard_batch_parallel(
            all_jobs,
            max_workers=args.workers,
            use_threads=True,   # share the score cache across threads
        )

    # Write all results grouped by out_dir (same JSON/CSV format as before)
    _write_collected_results(results)
    print(f"\nCompleted {len(results)} Jaccard computations.")