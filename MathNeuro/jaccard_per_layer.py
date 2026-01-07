from collections import defaultdict
from typing import Dict, Tuple, Any, Callable, Optional
import torch
import os
import csv
from typing import Dict, Any, Optional
import re

_LAYER_RE = re.compile(r"^layer_(\d+)$")

def layer_sort_key(layer_name: str):
    special_order = {
        "embeddings": -4,
        "final_norm": -3,
        "lm_head": -2,
        "other": -1,
    }
    if layer_name in special_order:
        return (0, special_order[layer_name])

    m = _LAYER_RE.match(layer_name)
    if m:
        return (1, int(m.group(1)))   # numeric sort for layer_i

    return (2, layer_name)  # fallback for unexpected bucket names



def save_jaccard_results(
    out_path: str,
    jacc: Dict[str, float],
    stats: Dict[str, Dict[str, int]],
    fmt: str = "csv",
    sort_key=layer_sort_key,
) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if fmt.lower() == "csv":
        fieldnames = ["layer", "jaccard", "intersection", "union", "selected_a", "selected_b"]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for layer in sorted(jacc.keys(), key=sort_key):
                s = stats.get(layer, {})
                w.writerow(
                    {
                        "layer": layer,
                        "jaccard": jacc[layer],
                        "intersection": s.get("intersection", 0),
                        "union": s.get("union", 0),
                        "selected_a": s.get("selected_a", 0),
                        "selected_b": s.get("selected_b", 0),
                    }
                )
        return out_path




def llama_layer_bucket(param_name: str) -> str:
    """
    Bucket HF LLaMA parameter names into:
      - layer_{i} for model.layers.i.*
      - embeddings for model.embed_tokens.*
      - final_norm for model.norm.*
      - lm_head for lm_head.*
      - other otherwise
    """
    m = re.search(r"(?:^|\.)(?:model\.)?layers\.(\d+)(?:\.|$)", param_name)
    if m:
        return f"layer_{int(m.group(1))}"

    if re.search(r"(?:^|\.)(?:model\.)?embed_tokens(?:\.|$)", param_name):
        return "embeddings"

    if re.search(r"(?:^|\.)(?:model\.)?norm(?:\.|$)", param_name):
        return "final_norm"

    if re.search(r"(?:^|\.)(?:lm_head)(?:\.|$)", param_name):
        return "lm_head"
    return "other"


def jaccard_similarity_per_llama_layer(
    pt_path_a: str,
    pt_path_b: str,
    mask_key: str = "isolated_masks",
    layer_bucket_fn: Callable[[str], str] = llama_layer_bucket,
    union_zero_value: float = float("nan"),
) -> Tuple[Dict[str, float], Dict[str, Dict[str, int]]]:
    """
    Computes Jaccard similarity per LLaMA layer by aggregating elementwise
    intersections/unions across all params in that layer.
    Assumes masks are boolean tensors where True means "selected".
    """
    if not os.path.exists(pt_path_a):
        raise FileNotFoundError(pt_path_a)
    if not os.path.exists(pt_path_b):
        raise FileNotFoundError(pt_path_b)

    ckpt_a: Dict[str, Any] = torch.load(pt_path_a, map_location="cpu")
    ckpt_b: Dict[str, Any] = torch.load(pt_path_b, map_location="cpu")

    if mask_key not in ckpt_a:
        raise KeyError(f"{pt_path_a} missing '{mask_key}'. Keys: {list(ckpt_a.keys())}")
    if mask_key not in ckpt_b:
        raise KeyError(f"{pt_path_b} missing '{mask_key}'. Keys: {list(ckpt_b.keys())}")

    masks_a: Dict[str, torch.Tensor] = ckpt_a[mask_key]
    masks_b: Dict[str, torch.Tensor] = ckpt_b[mask_key]

    all_names = set(masks_a.keys()) | set(masks_b.keys())

    acc = defaultdict(lambda: {"intersection": 0, "union": 0, "selected_a": 0, "selected_b": 0})

    for name in sorted(all_names):
        a = masks_a.get(name)
        b = masks_b.get(name)

        a = a.to(dtype=torch.bool, device="cpu").reshape(-1)
        b = b.to(dtype=torch.bool, device="cpu").reshape(-1)

        inter = torch.logical_and(a, b).sum().item()
        uni = torch.logical_or(a, b).sum().item()
        sa = a.sum().item()
        sb = b.sum().item()

        bucket = layer_bucket_fn(name)
        acc[bucket]["intersection"] += int(inter)
        acc[bucket]["union"] += int(uni)
        acc[bucket]["selected_a"] += int(sa)
        acc[bucket]["selected_b"] += int(sb)

    jaccard_by_bucket: Dict[str, float] = {}
    stats_by_bucket: Dict[str, Dict[str, int]] = {}

    for bucket, s in acc.items():
        u = s["union"]
        j = union_zero_value if u == 0 else (s["intersection"] / u)
        jaccard_by_bucket[bucket] = float(j)
        stats_by_bucket[bucket] = dict(s)
    return jaccard_by_bucket, stats_by_bucket




out_dir = "/home/iailab76/victorl0/pycharm_sync/MathNeuro/jaccard_results/"
jacc, stats = jaccard_similarity_per_llama_layer("/home/iailab76/victorl0/pycharm_sync/MathNeuro/results_gsm8k_race/isolated_masks/meta-llama/Llama-3.2-1B-Instruct/gsm8k_Race_0.05_repeat0.pt", "/home/iailab76/victorl0/pycharm_sync/MathNeuro/results_gsm8k_de_race_de/isolated_masks/meta-llama/Llama-3.2-1B-Instruct/gsm8k_Race_0.05_repeat0.pt")
out_csv = os.path.join(out_dir, "jaccard_gsm8k_Race_0.05_repeat0_race_vs_de_race.csv")
save_jaccard_results(out_csv, jacc, stats, fmt="csv")
for k in sorted(jacc.keys(), key=layer_sort_key):
    print(k, f"J={jacc[k]:.4f}", stats[k])



jacc, stats = jaccard_similarity_per_llama_layer("/home/iailab76/victorl0/pycharm_sync/MathNeuro/results_gsm8k_race/isolated_masks/meta-llama/Llama-3.2-1B-Instruct/gsm8k_Race_0.1_repeat0.pt", "/home/iailab76/victorl0/pycharm_sync/MathNeuro/results_gsm8k_race/isolated_masks/meta-llama/Llama-3.2-1B-Instruct/gsm8k_Race_0.1_repeat0.pt")
out_csv = os.path.join(out_dir, "jaccard_test.csv")
save_jaccard_results(out_csv, jacc, stats, fmt="csv")
for k in sorted(jacc.keys(), key=layer_sort_key):
    print(k, f"J={jacc[k]:.4f}", stats[k])

