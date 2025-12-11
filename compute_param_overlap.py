import torch

def load_mask(path):
    data = torch.load(path, map_location="cpu")
    return data["isolated_masks"], data

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

if __name__ == "__main__":


    # EXAMPLE: two specific mask files
    path1 = "results_gsm8k_race_de/isolated_masks/meta-llama/Llama-3.2-1B-Instruct/gsm8k_Race_0.01_repeat0.pt"
    path2 = "results_gsm8k_race_en/isolated_masks/meta-llama/Llama-3.2-1B-Instruct/gsm8k_Race_0.01_repeat0.pt"

    mask1, meta1 = load_mask(path1)
    mask2, meta2 = load_mask(path2)

    print("Run 1 meta:", {k: v for k, v in meta1.items() if k != "isolated_masks"})
    print("Run 2 meta:", {k: v for k, v in meta2.items() if k != "isolated_masks"})

    stats = compute_overlap(mask1, mask2)

    print("\n=== Overlap stats ===")
    print(f"Run 1 isolated params: {stats['total_isolated_run1']}")
    print(f"Run 2 isolated params: {stats['total_isolated_run2']}")
    print(f"Intersection:          {stats['intersection']}")
    print(f"Union:                 {stats['union']}")
    print(f"Jaccard similarity:    {stats['jaccard']:.6f}")
