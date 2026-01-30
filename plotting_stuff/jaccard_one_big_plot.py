import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_LAYER_RE = re.compile(r"^layers\.(\d+)$")

def compute_chance_jaccard(top_k: float) -> float:
    return (top_k * (1 - top_k)) / (top_k**2 - top_k + 2)

def _load_entry_for_good_percent(json_path: str | Path, good_percent: float, seed: int | None = None):
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # robust float matching (handles 1e-6 etc.)
    def _match_gp(x):
        try:
            return abs(float(x) - float(good_percent)) <= 1e-18
        except Exception:
            return str(x) == str(good_percent)

    candidates = [e for e in data if _match_gp(e.get("good_percent", None))]
    if seed is not None:
        candidates = [e for e in candidates if e.get("seed", None) == seed]

    if not candidates:
        raise ValueError(f"No entry found in {json_path} for good_percent={good_percent} (seed={seed}).")

    if len(candidates) > 1:
        # pick the first deterministically; but warn
        print(f"[WARN] Multiple entries in {json_path} for good_percent={good_percent} (seed={seed}). Using the first.")

    return candidates[0]

def _extract_layer_series(entry):
    """
    Returns:
      layers: sorted list[int]
      jacc: dict[layer] -> float
      iso1: dict[layer] -> int   (run1)
      iso2: dict[layer] -> int   (run2)
    """
    per_group = entry.get("per_group", {})
    layer_items = []
    for gname, stats in per_group.items():
        m = _LAYER_RE.match(gname)
        if not m:
            continue
        idx = int(m.group(1))
        layer_items.append((idx, stats))

    if not layer_items:
        raise ValueError("No layer entries found (expected keys like 'layers.N').")

    layer_items.sort(key=lambda t: t[0])
    layers = [idx for idx, _ in layer_items]

    jacc = {idx: float(stats["jaccard"]) for idx, stats in layer_items}
    iso1 = {idx: int(stats["total_isolated_run1"]) for idx, stats in layer_items}
    iso2 = {idx: int(stats["total_isolated_run2"]) for idx, stats in layer_items}

    return layers, jacc, iso1, iso2

def _merge_isolated(preferred, alternative, name: str):
    """
    preferred/alternative: dict[layer] -> int
    If both exist and differ, warn and average.
    """
    out = {}
    for layer in sorted(set(preferred.keys()) | set(alternative.keys())):
        a = preferred.get(layer, None)
        b = alternative.get(layer, None)
        if a is None:
            out[layer] = b
        elif b is None:
            out[layer] = a
        else:
            if a != b:
                print(f"[WARN] {name} isolated mismatch at layer {layer}: {a} vs {b}. Averaging.")
            out[layer] = int(round((a + b) / 2))
    return out

def plot_jaccard_triplet_twofigs(
    good_percent: float,
    json_en_de: str | Path,
    json_en_hi: str | Path,
    json_de_hi: str | Path,
    save_dir: str | Path | None = None,
    show: bool = True,
    show_chance_line: bool = False,
    seed: int | None = None,
):

    base = matplotlib.rcParams.get("font.size", 10)
    font_scale = 1.9
    fs = base * font_scale
    with matplotlib.rc_context({
        "font.size": fs,
        "axes.labelsize": fs * 1.1,
        "axes.titlesize": fs * 1.2,
        "xtick.labelsize": fs * 0.95,
        "ytick.labelsize": fs * 0.95,
        "legend.fontsize": fs * 0.95,
        "figure.titlesize": fs * 1.2,
    }):
        # --- load entries ---
        e_en_de = _load_entry_for_good_percent(json_en_de, good_percent, seed=seed)
        e_en_hi = _load_entry_for_good_percent(json_en_hi, good_percent, seed=seed)
        e_de_hi = _load_entry_for_good_percent(json_de_hi, good_percent, seed=seed)

        seed_used = seed if seed is not None else e_en_de.get("seed", None)

        layers_a, j_en_de, iso_en_from_en_de, iso_de_from_en_de = _extract_layer_series(e_en_de)
        layers_b, j_en_hi, iso_en_from_en_hi, iso_hi_from_en_hi = _extract_layer_series(e_en_hi)
        layers_c, j_de_hi, iso_de_from_de_hi, iso_hi_from_de_hi = _extract_layer_series(e_de_hi)

        # unify layers
        layers = sorted(set(layers_a) | set(layers_b) | set(layers_c))
        x = np.array(layers, dtype=int)

        def _as_array(d, fill=np.nan):
            return np.array([d.get(L, fill) for L in layers], dtype=float)

        # Jaccard arrays
        y_j_en_de = _as_array(j_en_de)
        y_j_en_hi = _as_array(j_en_hi)
        y_j_de_hi = _as_array(j_de_hi)

        # Isolated arrays (merge duplicates)
        iso_en = _merge_isolated(iso_en_from_en_de, iso_en_from_en_hi, "English")
        iso_de = _merge_isolated(iso_de_from_en_de, iso_de_from_de_hi, "German")
        iso_hi = _merge_isolated(iso_hi_from_en_hi, iso_hi_from_de_hi, "Hindi")

        y_iso_en = _as_array(iso_en)
        y_iso_de = _as_array(iso_de)
        y_iso_hi = _as_array(iso_hi)

        # --- Figure 1: Jaccard ---
        fig_j, ax_j = plt.subplots(figsize=(14, 7))

        lj1 = ax_j.plot(x, y_j_en_de, label="Jaccard EN–DE")
        lj2 = ax_j.plot(x, y_j_en_hi, label="Jaccard EN–HI")
        lj3 = ax_j.plot(x, y_j_de_hi,  label="Jaccard DE–HI")

        if show_chance_line:
            chance_val = compute_chance_jaccard(float(good_percent))
            y_chance = np.full_like(y_j_en_de, chance_val, dtype=float)
            ljc = ax_j.plot(x, y_chance, linestyle="--", label="Chance Jaccard")
        else:
            ljc = []

        ax_j.set_xlabel("Layer")
        ax_j.set_ylabel("Jaccard similarity")
        ax_j.set_ylim(0.0, 0.5)
        ax_j.set_xticks(x)
        ax_j.set_xticklabels([str(i) for i in x], rotation=0)

        lines_j = lj1 + lj2 + lj3 + ljc
        labels_j = [ln.get_label() for ln in lines_j]

        fig_j.legend(
            handles=lines_j,
            labels=labels_j,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(4, len(labels_j)),
            frameon=False,
            handlelength=2.0,
            columnspacing=1.5,
        )

        fig_j.tight_layout(rect=[0, 0, 1, 0.92])

        # --- Figure 2: Isolated params ---
        fig_i, ax_i = plt.subplots(figsize=(14, 7))

        li1 = ax_i.plot(x, y_iso_en,  label="Math-related params: English")
        li2 = ax_i.plot(x, y_iso_de, label="Math-related params: German")
        li3 = ax_i.plot(x, y_iso_hi, label="Math-related params: Hindi")

        ax_i.set_xlabel("Layer")
        ax_i.set_ylabel("# math-specific parameters")
        ax_i.set_xticks(x)
        ax_i.set_xticklabels([str(i) for i in x], rotation=0)

        # --- one legend block above figure (use THIS figure's handles/labels) ---
        lines_i = li1 + li2 + li3
        labels_i = [ln.get_label() for ln in lines_i]

        fig_i.legend(
            handles=lines_i,
            labels=labels_i,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.5,
        )

        # leave room for the legend above
        fig_i.tight_layout(rect=[0, 0, 1, 0.92])


        # --- save / show / close ---
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            gp_str = str(good_percent)

            out_j = save_dir / f"jaccard_triplet_{gp_str}_seed_{seed_used}.pdf"
            out_i = save_dir / f"isolated_triplet_{gp_str}_seed_{seed_used}.pdf"

            fig_j.savefig(out_j, dpi=200)
            fig_i.savefig(out_i, dpi=200)
            print("Saved:", out_j)
            print("Saved:", out_i)

        if show:
            plt.show()
        else:
            plt.close(fig_j)
            plt.close(fig_i)



seed = 42
good_percent = 0.001

plot_jaccard_triplet_twofigs(
    good_percent=good_percent,
    json_en_de=f"../MathNeuro/jaccard_results/gsm8k_race_en_vs_de/jaccard_summary_{seed}.json",
    json_en_hi=f"../MathNeuro/jaccard_results/gsm8k_race_en_vs_hi/jaccard_summary_{seed}.json",
    json_de_hi=f"../MathNeuro/jaccard_results/gsm8k_race_de_vs_hi/jaccard_summary_{seed}.json",
    save_dir="jaccard_results/combinations",
    show=True,
    show_chance_line=True,
    seed=seed,
)







from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_jaccard_code(
    good_percent: float,
    json_en_code: str | Path,
    save_dir: str | Path | None = None,
    show: bool = True,
    show_chance_line: bool = False,
    seed: int | None = None,
):
    base = matplotlib.rcParams.get("font.size", 10)
    font_scale = 1.9
    fs = base * font_scale

    with matplotlib.rc_context({
        "font.size": fs,
        "axes.labelsize": fs * 1.1,
        "axes.titlesize": fs * 1.2,
        "xtick.labelsize": fs * 0.95,
        "ytick.labelsize": fs * 0.95,
        "legend.fontsize": fs * 0.95,
        "figure.titlesize": fs * 1.2,
    }):
        # --- load entry ---
        e = _load_entry_for_good_percent(json_en_code, good_percent, seed=seed)
        seed_used = seed if seed is not None else e.get("seed", None)

        # Expect: layers, jaccard, iso_left, iso_right
        layers_a, j_by_layer, iso_left, iso_right = _extract_layer_series(e)

        layers = sorted(set(layers_a))
        x = np.array(layers, dtype=int)

        def _as_array(d, fill=np.nan):
            return np.array([d.get(L, fill) for L in layers], dtype=float)

        # Jaccard
        y_j = _as_array(j_by_layer)

        # Isolated counts: interpret left/right as Math/Code
        # (No merge needed unless you truly have two sources to merge.)
        y_iso_math = _as_array(iso_left)
        y_iso_code = _as_array(iso_right)

        # ---------- Figure 1: Jaccard ----------
        fig_j, ax_j = plt.subplots(figsize=(14, 7))

        lj1 = ax_j.plot(x, y_j, label="Jaccard (Math–Code)")

        if show_chance_line:
            chance_val = compute_chance_jaccard(float(good_percent))
            y_chance = np.full_like(y_j, chance_val, dtype=float)
            ljc = ax_j.plot(x, y_chance, linestyle="--", label="Chance Jaccard")
        else:
            ljc = []

        ax_j.set_xlabel("Layer")
        ax_j.set_ylabel("Jaccard similarity")
        ax_j.set_ylim(0.0, 0.5)
        ax_j.set_xticks(x)
        ax_j.set_xticklabels([str(i) for i in x], rotation=0)

        lines_j = lj1 + ljc
        labels_j = [ln.get_label() for ln in lines_j]

        # Legend above; reserve top space
        fig_j.legend(
            handles=lines_j,
            labels=labels_j,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(2, len(labels_j)),
            frameon=False,
            handlelength=2.0,
            columnspacing=1.5,
        )
        fig_j.tight_layout(rect=[0, 0, 1, 0.92])

        # ---------- Figure 2: Isolated params ----------
        fig_i, ax_i = plt.subplots(figsize=(14, 7))

        li1 = ax_i.plot(x, y_iso_math, label="Code-related params")
        li2 = ax_i.plot(x, y_iso_code, label="Math-related params")

        ax_i.set_xlabel("Layer")
        ax_i.set_ylabel("# isolated parameters")
        ax_i.set_xticks(x)
        ax_i.set_xticklabels([str(i) for i in x], rotation=0)

        lines_i = li1 + li2
        labels_i = [ln.get_label() for ln in lines_i]

        # Put legend above; use 2 columns to prevent side clipping
        fig_i.legend(
            handles=lines_i,
            labels=labels_i,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.5,
        )
        fig_i.tight_layout(rect=[0, 0, 1, 0.92])

        # --- save / show / close ---
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            gp_str = str(good_percent)

            out_j = save_dir / f"jaccard_code_{gp_str}_seed_{seed_used}.pdf"
            out_i = save_dir / f"isolated_code_{gp_str}_seed_{seed_used}.pdf"

            # bbox_inches="tight" prevents legend cut-off in PDF
            fig_j.savefig(out_j, dpi=200, bbox_inches="tight")
            fig_i.savefig(out_i, dpi=200, bbox_inches="tight")
            print("Saved:", out_j)
            print("Saved:", out_i)

        if show:
            plt.show()
        else:
            plt.close(fig_j)
            plt.close(fig_i)

seed = 42
good_percent = 0.001

plot_jaccard_code(
    good_percent=good_percent,
    json_en_code = f"jaccard_results/combinations/jaccard_summary.json",
    save_dir="jaccard_results/combinations",
    show=True,
    show_chance_line=True,
    seed=seed,
)
