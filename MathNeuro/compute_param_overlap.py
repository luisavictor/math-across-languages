'''
Given the .pt files, this script allows to compute the Jaccard similarity of two parameter sets, e.g., of English and German math specific params.
'''

import torch
from pathlib import Path
import json
import csv
import re
from collections import defaultdict,OrderedDict

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
    """
    mask_dir = Path(mask_dir)

    param_good = torch.load(mask_dir / f"train_scores_seed{seed}.pt", map_location="cpu")
    param_comp = torch.load(mask_dir / f"comparison_scores_seed{seed}.pt", map_location="cpu")

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

    return isolated_masks


def run_jaccard_sweep(
    good_percents,
    mask_dir_lang1,
    mask_dir_lang2,
    out_dir,
    seed: int = 1,
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

    json_path = out_dir / "jaccard_summary.json"
    csv_path = out_dir / "jaccard_per_layer.csv"

    all_results = []  # list of dicts (one per good_percent)
    per_layer_rows = []  # flattened rows for CSV

    for gp in good_percents:
        # build masks
        isolated_masks_lang1 = build_isolated_masks(mask_dir_lang1, gp, seed=seed)
        isolated_masks_lang2 = build_isolated_masks(mask_dir_lang2, gp, seed=seed)

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

    # ---- save JSON (convert tensors are already numbers in our stats) ----
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # ---- save CSV ----
    if per_layer_rows:
        fieldnames = list(per_layer_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_layer_rows)

    return all_results, per_layer_rows


good_percents = [0.0001, 0.001, 0.01,0.025,0.05, 0.1, 0.15]

mask_dir_lang1 = "/home/iailab75/selbacht0/Test_Lab/MathNeuro/results_codealpaca_0.01/isolated_masks/meta-llama/Llama-3.2-1B-Instruct/"
mask_dir_lang2 = "/home/iailab75/selbacht0/Test_Lab/MathNeuro/results_codealpaca_mmlu_filtered/isolated_masks/meta-llama/Llama-3.2-1B-Instruct/"

out_dir = "results_jaccard/gsm8k_mmlu_en_filtered_vs_code"

all_results, per_layer_rows = run_jaccard_sweep(
    good_percents=good_percents,
    mask_dir_lang1=mask_dir_lang1,
    mask_dir_lang2=mask_dir_lang2,
    out_dir=out_dir,
    seed=42,
)
