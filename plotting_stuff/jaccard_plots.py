import json
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # no GUI; save-only backend
import matplotlib.pyplot as plt

_LAYER_RE = re.compile(r"^layers\.(\d+)$")



def compute_chance_jaccard(top_k):
    return (top_k*(1-top_k))/(top_k**2-top_k+2)



def plot_jaccard_per_good_percent(
    json_path: str | Path,
    save_dir: str | Path | None = None,
    show: bool = True,
    show_chance_line: bool = False,
):
    """
    Reads a jaccard_summary.json (list of entries) and, for each entry (i.e., each good_percent),
    creates ONE plot:
      - x-axis: layers (sorted)
      - left y-axis: Jaccard similarity per layer
      - right y-axis: total_isolated_run1 and total_isolated_run2 per layer
                      labeled as "Math-specific English" and "Math-specific German"

    Args:
        json_path: path to jaccard_summary.json
        save_dir: if provided, saves one PNG per good_percent into this folder
        show: if True, calls plt.show() at the end
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    def layer_index(name: str) -> int | None:
        m = _LAYERSAFE(name)
        if m is None:
            return None
        return int(m.group(1))

    def _LAYERSAFE(name: str):
        return _LAYER_RE.match(name)

    for entry in data:
        gp = entry.get("good_percent", None)
        seed = entry.get("seed", None)
        per_group = entry.get("per_group", {})

        # keep only "layers.N"
        layer_items = []
        for gname, stats in per_group.items():
            m = _LAYER_RE.match(gname)
            if not m:
                continue
            idx = int(m.group(1))
            layer_items.append((idx, gname, stats))

        if not layer_items:
            print(f"[WARN] No layer entries found for good_percent={gp}. Skipping.")
            continue

        # sort by layer index
        layer_items.sort(key=lambda t: t[0])

        x = [idx for idx, _, _ in layer_items]
        xlabels = [f"{idx}" for idx in x]

        jacc = [float(stats["jaccard"]) for _, _, stats in layer_items]
        iso_en = [int(stats["total_isolated_run1"]) for _, _, stats in layer_items]
        iso_de = [int(stats["total_isolated_run2"]) for _, _, stats in layer_items]

        fig, ax_left = plt.subplots(figsize=(12, 5))
        ax_right = ax_left.twinx()

        chance_jac = [compute_chance_jaccard(gp) for item in layer_items]

        # Left axis: Jaccard
        l1 = ax_left.plot(x, jacc, marker="o", label="Jaccard")
        if show_chance_line:
              l1_chance = ax_left.plot(x, chance_jac, marker="o", label="Chance Jaccard")

        # Right axis: isolated counts
        l2 = ax_right.plot(x, iso_en, marker=".",alpha=0.5,color="green", label="Math-specific English")
        l3 = ax_right.plot(x, iso_de, marker=".",color="violet", label="Code-specific")

        ax_left.set_xlabel("Layer")
        ax_left.set_ylabel("Jaccard similarity")
        ax_right.set_ylabel("# isolated parameters")

        ax_left.set_xticks(x)
        ax_left.set_xticklabels(xlabels, rotation=0)

        title = f"Top-k={gp}, seed={seed}"
        ax_left.set_title(title)


        ax_left.set_ylim(0.0, 0.5)
        #ax_right.set_ylim([])

        # Combine legends from both axes
        if show_chance_line:
             lines = l1 + l1_chance + l2 + l3
        else:
            lines = l1 + l2 + l3
        labels = [ln.get_label() for ln in lines]

        ax_left.legend(lines, labels, loc="upper right")  # <-- add this

        if save_dir is not None:
            # safe filename
            gp_str = str(gp)
            if show_chance_line:
                out_path = save_dir / f"jaccard_layers_with_chance_{gp_str}_seed_{seed}.pdf"
            else:
                out_path = save_dir / f"jaccard_layers_{gp_str}_seed_{seed}.pdf"
            fig.savefig(out_path, dpi=200)
            print("Saved:", out_path)

        if show:
            plt.show()
        else:
            plt.close(fig)



# plot_jaccard_per_good_percent(
#     f"../MathNeuro/results_jaccard/gsm8k_mmlu_old_vs_code/jaccard_summary.json",
#     save_dir="jaccard_results/gsm8k_mmlu_old_vs_code",
#     show=True,
# )