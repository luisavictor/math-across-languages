from MathNeuro.jaccard_per_layer import layer_sort_key
import os
import csv
import math
import matplotlib
matplotlib.use("Agg")  # no GUI; save-only backend
import matplotlib.pyplot as plt


def _read_jaccard_csv(csv_path: str) -> dict[str, float]:
    """Read CSV produced by save_jaccard_results and return {layer: jaccard}."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    jacc = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"layer", "jaccard"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path} missing required columns {required}. Got: {reader.fieldnames}")
        for r in reader:
            layer = r["layer"]
            try:
                val = float(r["jaccard"])
            except (TypeError, ValueError):
                val = float("nan")
            jacc[layer] = val
    return jacc


def plot_jaccard_two_pairs_for_threshold(
    *,
    threshold: str,
    csv_en_de: str,
    csv_en_hi: str,
    sort_key=layer_sort_key,
    title_prefix: str = "Jaccard per layer",
    out_path: str | None = None,
    show: bool = False,  # <- Agg backend: should be False
    ylim: tuple[float, float] = (0.0, 0.45),
    figsize: tuple[float, float] = (12, 8),
):
    j_de = _read_jaccard_csv(csv_en_de)
    j_hi = _read_jaccard_csv(csv_en_hi)

    layers = sorted(set(j_de.keys()) | set(j_hi.keys()), key=sort_key)

    def get_vals(jmap: dict[str, float]):
        vals = []
        for k in layers:
            v = jmap.get(k, float("nan"))
            vals.append(v if not math.isnan(v) else float("nan"))
        return vals

    y_de = get_vals(j_de)
    y_hi = get_vals(j_hi)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(len(layers)), y_de, marker="o", linewidth=1.5, label="EN vs DE")
    ax.plot(range(len(layers)), y_hi, marker="o", linewidth=1.5, label="EN vs HI")

    ax.set_title(f"{title_prefix} — {threshold}")
    ax.set_ylabel("Jaccard similarity")
    ax.set_xlabel("Layer / bucket")
    ax.set_ylim(*ylim)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=60, ha="right", fontsize=9)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()

    if out_path:
        d = os.path.dirname(out_path)
        if d:  # <- only mkdir if directory component exists
            os.makedirs(d, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    # Agg backend: no interactive window; always close to free memory
    plt.close(fig)
    return fig, ax



out_dir = "/home/iailab76/victorl0/pycharm_sync/MathNeuro/jaccard_results/"

plot_dir = "/home/iailab76/victorl0/pycharm_sync/plotting_stuff/jaccard_results/"
thresholds = ["0.001", "0.01", "0.1", "0.15"]

for thr in thresholds:
    csv_en_de = os.path.join(out_dir, f"jaccard_{thr}_repeat0_mmlu_gsm8k_de_en.csv")
    csv_en_hi = os.path.join(out_dir, f"jaccard_{thr}_repeat0_mmlu_gsm8k_hi_en.csv")

    out_png = os.path.join(plot_dir, f"plot_jaccard_{thr}_repeat0_en_de_vs_en_hi.png")

    plot_jaccard_two_pairs_for_threshold(
        threshold=thr,
        csv_en_de=csv_en_de,
        csv_en_hi=csv_en_hi,
        out_path=out_png,
        show=False,
    )
