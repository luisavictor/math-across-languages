import json
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # no GUI; save-only backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

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



def plot_jaccard_heatmap_global(
    json_paths: dict[str, str | Path],
    save_path: str | Path | None = None,
    show: bool = True,
    title: str = "Global Jaccard Similarity",
    cmap: str = "YlOrRd",
    figsize: tuple[float, float] | None = None,
):
    """
    Create a heatmap of global Jaccard similarity.

    Rows   = language pairs  (one per JSON file)
    Columns = good_percent values (extracted from the JSON entries)

    Args:
        json_paths: mapping  { "EN vs DE": "path/to/jaccard_summary.json", ... }
                    The keys are used as row labels.
        save_path:  if provided, save the figure to this path (pdf/png/…)
        show:       whether to call plt.show()
        title:      plot title
        cmap:       matplotlib colour-map name
        figsize:    optional (width, height) in inches
    """
    lang_pairs: list[str] = []
    all_good_percents: set[float] = set()
    pair_data: dict[str, dict[float, float]] = {}  # label -> {gp: jaccard}

    for label, path in json_paths.items():
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        lang_pairs.append(label)
        gp_map: dict[float, float] = {}
        for entry in data:
            gp = float(entry["good_percent"])
            jaccard = float(entry["global"]["jaccard"])
            gp_map[gp] = jaccard
            all_good_percents.add(gp)
        pair_data[label] = gp_map

    # sorted column headers
    good_percents = sorted(all_good_percents)

    # build matrix  (rows=lang_pairs + chance row, cols=good_percents)
    lang_pairs.append("Chance")
    matrix = np.full((len(lang_pairs), len(good_percents)), np.nan)
    for i, label in enumerate(lang_pairs[:-1]):
        for j, gp in enumerate(good_percents):
            matrix[i, j] = pair_data[label].get(gp, np.nan)
    # last row: random chance Jaccard for each top-k
    for j, gp in enumerate(good_percents):
        matrix[-1, j] = compute_chance_jaccard(gp)

    # ---------- plot ----------
    if figsize is None:
        figsize = (max(6, 1.4 * len(good_percents)), max(4, 0.8 * len(lang_pairs)))
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, aspect="auto", cmap=cmap)

    # axis labels
    ax.set_xticks(range(len(good_percents)))
    ax.set_xticklabels([f"{gp}" for gp in good_percents], rotation=0, fontsize=12)
    ax.set_yticks(range(len(lang_pairs)))
    ax.set_yticklabels(lang_pairs, fontsize=12)

    ax.set_xlabel("Top-$k$", fontsize=14)
    ax.set_ylabel("Language Pair", fontsize=14)
    ax.set_title(title, fontsize=16)

    # annotate cells with the Jaccard value
    for i in range(len(lang_pairs)):
        for j in range(len(good_percents)):
            val = matrix[i, j]
            if not np.isnan(val):
                # choose text colour for readability
                text_color = "white" if val > (np.nanmax(matrix) + np.nanmin(matrix)) / 2 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=11, color=text_color)

    # fig.colorbar(im, ax=ax, pad=0.02)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print("Saved:", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_params_and_heatmap(
    iso_sources: dict[str, tuple[str | Path, str]],
    json_paths: dict[str, str | Path],
    good_percents_filter: list[float] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    title_heatmap: str = "Global Jaccard Similarity",
    cmap: str = "YlOrRd",
    fontsize: int = 15,
    legend_fontsize: int = 12,
    linewidth: float = 1.5,
    figsize: tuple[float, float] | None = None,
):
    """
    Two side-by-side subplots:
      Left  – # math-specific params vs Top k%  (one line per language)
      Right – Jaccard-similarity heatmap (rows = language pairs + chance,
              columns = good_percent values)

    Args:
        iso_sources:    dict mapping language label to (json_path, "run1"|"run2").
                        Tells the function which file and which run to read for
                        each language's isolated-param counts.
                        Example for 4 languages:
                          {"En": (en_de_path, "run1"),
                           "De": (en_de_path, "run2"),
                           "Fr": (en_fr_path, "run2"),
                           "Hi": (en_hi_path, "run2")}
        good_percents_filter: optional list of good_percent values to include
                        (as raw fractions, e.g. [0.001, 0.01, 0.05, 0.1, 0.15]).
                        If None, all values found in the JSON files are used.
        json_paths:     dict  { "En-De": "path/to/json", ... }
                        Each key becomes a row in the heatmap (e.g. 6 pairs).
        save_path:      optional path to save the figure
        show:           whether to call plt.show()
        title_heatmap:  title shown above the heatmap
        cmap:           colour-map for the heatmap
        fontsize:       axis-label font size
        legend_fontsize: legend font size
        linewidth:      line width for the left subplot
        figsize:        optional (width, height) in inches
    """
    # ── load data for the LEFT plot ──────────────────────────────────────
    _iso_cache: dict[str, list[dict]] = {}  # path → sorted list of entries

    def _load_sorted(p: str | Path) -> list[dict]:
        key = str(p)
        if key not in _iso_cache:
            p = Path(p)
            with p.open("r", encoding="utf-8") as f:
                entries = json.load(f)
            entries.sort(key=lambda x: x.get("good_percent", 0))
            _iso_cache[key] = entries
        return _iso_cache[key]

    _RUN_KEY = {"run1": "total_isolated_run1", "run2": "total_isolated_run2"}

    # colours / markers – cycle if more than 4 languages
    _COLORS  = ["green", "violet", "orange", "crimson", "steelblue", "brown"]
    _MARKERS = ["o", "s", "^", "v", "D", "P"]

    iso_series: dict[str, list[int]] = {}
    xs: list[float] | None = None

    _gp_set = set(good_percents_filter) if good_percents_filter is not None else None

    for lang_label, (path, which_run) in iso_sources.items():
        entries = _load_sorted(path)
        if _gp_set is not None:
            entries = [e for e in entries if e["good_percent"] in _gp_set]
        if xs is None:
            xs = [entry["good_percent"] for entry in entries]
        run_key = _RUN_KEY[which_run]
        iso_series[lang_label] = [entry["global"][run_key] for entry in entries]

    assert xs is not None, "iso_sources must not be empty"

    # ── load data for the RIGHT plot (heatmap) ──────────────────────────
    lang_pairs: list[str] = []
    all_good_percents: set[float] = set()
    pair_data: dict[str, dict[float, float]] = {}

    for label, path in json_paths.items():
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        lang_pairs.append(label)
        gp_map: dict[float, float] = {}
        for entry in data:
            gp = float(entry["good_percent"])
            if _gp_set is not None and gp not in _gp_set:
                continue
            jaccard = float(entry["global"]["jaccard"])
            gp_map[gp] = jaccard
            all_good_percents.add(gp)
        pair_data[label] = gp_map

    good_percents = sorted(all_good_percents)

    # matrix rows = lang_pairs + "Chance"
    lang_pairs.append("Chance")
    matrix = np.full((len(lang_pairs), len(good_percents)), np.nan)
    for i, lbl in enumerate(lang_pairs[:-1]):
        for j, gp in enumerate(good_percents):
            matrix[i, j] = pair_data[lbl].get(gp, np.nan)
    for j, gp in enumerate(good_percents):
        matrix[-1, j] = compute_chance_jaccard(gp)

    # ── plotting ────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 18,
        'axes.titlesize': 15,
        'xtick.labelsize': 16,
        'ytick.labelsize': 14,
        'legend.fontsize': legend_fontsize,
        'lines.linewidth': linewidth,
        'lines.markersize': 10,
    })

    if figsize is None:
        figsize = (12, max(4, 0.8 * len(lang_pairs)))
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [1, 1.7]},
    )

    # ── LEFT subplot: # math-specific params ────────────────────────────
    markevery = max(1, len(xs) // 6)
    markersize = 5 if len(xs) <= 20 else 3

    handles_left: list = []
    labels_left: list[str] = []
    for idx, (lang_label, vals) in enumerate(iso_series.items()):
        color = _COLORS[idx % len(_COLORS)]
        marker = _MARKERS[idx % len(_MARKERS)]
        h, = ax_left.plot(xs, vals, marker=marker, label=lang_label,
                          color=color, alpha=0.8, markevery=markevery,
                          markersize=markersize)
        handles_left.append(h)
        labels_left.append(lang_label)

    ax_left.set_xlabel("Top-$k$", fontsize=fontsize)
    ax_left.set_ylabel("# Math-specific Params", fontsize=fontsize)
    ax_left.grid(True, alpha=0.25, linewidth=0.5)

    formatter_left = ticker.ScalarFormatter(useMathText=True)
    formatter_left.set_scientific(True)
    formatter_left.set_powerlimits((0, 0))
    formatter_left.set_useOffset(True)
    ax_left.set_title("Parameter Count", fontsize=fontsize)
    ax_left.yaxis.set_major_formatter(formatter_left)
    ax_left.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    # ── RIGHT subplot: heatmap ──────────────────────────────────────────
    im = ax_right.imshow(matrix, aspect="auto", cmap=cmap)

    ax_right.set_xticks(range(len(good_percents)))
    ax_right.set_xticklabels([f"{gp:g}" for gp in good_percents], rotation=0, fontsize=12)
    ax_right.set_yticks(range(len(lang_pairs)))
    ax_right.set_yticklabels(lang_pairs, fontsize=12, rotation=45)

    ax_right.set_xlabel("Top-$k$", fontsize=fontsize)
    ax_right.set_ylabel("Language Pair", fontsize=fontsize, labelpad=-3)
    ax_right.set_title(title_heatmap, fontsize=fontsize)

    fig.colorbar(im, ax=ax_right, pad=0.02, aspect=25)

    # annotate cells
    for i in range(len(lang_pairs)):
        for j in range(len(good_percents)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = ("white"
                              if val > (np.nanmax(matrix) + np.nanmin(matrix)) / 2
                              else "black")
                ax_right.text(j, i, f"{val:.3f}", ha="center", va="center",
                              fontsize=13.5, color=text_color)

    # ── legends & layout ────────────────────────────────────────────────
    fig.legend(
        handles_left, labels_left,
        loc="upper center",
        bbox_to_anchor=(0.225, 1.03),
        ncol=len(handles_left), frameon=False,
        handlelength=2.0, columnspacing=1.5,
    )

    # plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print("Saved:", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _prepare_per_layer_data(
    iso_sources: dict[str, tuple[str | Path, str]],
    json_paths: dict[str, str | Path],
    show_chance_line: bool = False,
) -> list[dict]:
    """
    Internal helper that loads the JSON files and returns a list of dicts,
    one per good_percent entry, each containing:
        gp, seed, x, xlabels, iso_per_lang, jacc_per_pair, row_labels, matrix
    """
    _cache: dict[str, list[dict]] = {}

    def _load(p: str | Path) -> list[dict]:
        key = str(p)
        if key not in _cache:
            p = Path(p)
            with p.open("r", encoding="utf-8") as f:
                _cache[key] = json.load(f)
        return _cache[key]

    _RUN_KEY = {"run1": "total_isolated_run1", "run2": "total_isolated_run2"}

    def _extract_layers(per_group):
        items = []
        for gname, stats in per_group.items():
            m = _LAYER_RE.match(gname)
            if not m:
                continue
            items.append((int(m.group(1)), gname, stats))
        items.sort(key=lambda t: t[0])
        return items

    # Pre-load
    pair_data_lists: dict[str, list[dict]] = {}
    for label, path in json_paths.items():
        pair_data_lists[label] = _load(path)

    iso_data_lists: dict[str, tuple[list[dict], str]] = {}
    for lang_label, (path, which_run) in iso_sources.items():
        iso_data_lists[lang_label] = (_load(path), which_run)

    first_pair_label = next(iter(pair_data_lists))
    n_entries = len(pair_data_lists[first_pair_label])

    results = []
    for entry_idx in range(n_entries):
        ref_entry = pair_data_lists[first_pair_label][entry_idx]
        gp = ref_entry.get("good_percent", None)
        seed = ref_entry.get("seed", None)

        pair_labels_list = list(json_paths.keys())
        jacc_per_pair: dict[str, list[float]] = {}
        x: list[int] | None = None

        skip = False
        for label in pair_labels_list:
            entry = pair_data_lists[label][entry_idx]
            layer_items = _extract_layers(entry.get("per_group", {}))
            if not layer_items:
                print(f"[WARN] No layer entries for {label}, good_percent={gp}. Skipping.")
                skip = True
                break
            if x is None:
                x = [idx for idx, _, _ in layer_items]
            jacc_per_pair[label] = [float(s["jaccard"]) for _, _, s in layer_items]

        if skip or x is None:
            continue

        xlabels = [f"{idx}" for idx in x]

        iso_per_lang: dict[str, list[int]] = {}
        for lang_label, (data_list, which_run) in iso_data_lists.items():
            entry = data_list[entry_idx]
            layer_items = _extract_layers(entry.get("per_group", {}))
            run_key = _RUN_KEY[which_run]
            iso_per_lang[lang_label] = [int(s[run_key]) for _, _, s in layer_items]

        row_labels = list(pair_labels_list)
        rows = [jacc_per_pair[lbl] for lbl in pair_labels_list]
        if show_chance_line:
            chance_val = compute_chance_jaccard(gp)
            rows.append([chance_val] * len(x))
            row_labels.append("Chance")

        matrix = np.array(rows)

        results.append({
            "gp": gp,
            "seed": seed,
            "x": x,
            "xlabels": xlabels,
            "iso_per_lang": iso_per_lang,
            "jacc_per_pair": jacc_per_pair,
            "row_labels": row_labels,
            "matrix": matrix,
        })

    return results


_COLORS  = ["green", "violet", "orange", "crimson", "steelblue", "brown"]
_MARKERS = ["o", "s", "^", "v", "D", "P"]


def plot_isolated_params_per_layer(
    iso_sources: dict[str, tuple[str | Path, str]],
    json_paths: dict[str, str | Path],
    show_chance_line: bool = False,
    save_dir: str | Path | None = None,
    show: bool = True,
    figsize: tuple[float, float] | None = None,
    fontsize: int = 11,
    title: str | None = None,
    k_values_to_have_xlabels: list[float] | None = None,
    k_values_to_have_ylabels: list[float] | None = None,
):
    """
    Standalone line plot: # isolated params per layer, one line per language.

    Args:
        iso_sources: dict mapping language label → (json_path, "run1"|"run2")
        json_paths:  dict { pair_label: json_path } (needed for data extraction)
        show_chance_line: forwarded to data prep
        save_dir:    if provided, saves one PDF per good_percent
        show:        if True, plt.show() and stop after first entry
        figsize:     optional (width, height)
        fontsize:    axis-label font size
        title:       optional custom title (default: "Top-$k$=…")
    """
    entries = _prepare_per_layer_data(iso_sources, json_paths, show_chance_line)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    fig = None
    for d in entries:
        x, xlabels = d["x"], d["xlabels"]
        iso_per_lang = d["iso_per_lang"]
        gp, seed = d["gp"], d["seed"]

        if figsize is None:
            _figsize = (max(4, 0.25 * len(x)), 3.5)
        else:
            _figsize = figsize

        fig, ax = plt.subplots(figsize=_figsize)

        markersize = 4 if len(x) <= 20 else 2
        xticks_step = 2 if len(x) <= 20 else 3

        handles, labels_leg = [], []
        all_vals: list[int] = []
        for idx_lang, (lang_label, vals) in enumerate(iso_per_lang.items()):
            color = _COLORS[idx_lang % len(_COLORS)]
            marker = _MARKERS[idx_lang % len(_MARKERS)]
            h, = ax.plot(x, vals, marker=marker, color=color,
                         label=lang_label, linewidth=1.5,
                         markersize=markersize, alpha=0.8)
            handles.append(h)
            labels_leg.append(lang_label)
            all_vals.extend(vals)

        if k_values_to_have_xlabels is not None and gp in k_values_to_have_xlabels:
            ax.set_xlabel("Layer Index", fontsize=fontsize)
        else:
            ax.set_xlabel(" ", fontsize=fontsize)

        if k_values_to_have_ylabels is not None and gp in k_values_to_have_ylabels:
            ax.set_ylabel("# Isolated Params", fontsize=fontsize)
        else:            
            ax.set_ylabel(" ", fontsize=fontsize)
        ax.grid(True, alpha=0.25, linewidth=0.5)

        ax.set_xticks(x[::xticks_step])
        ax.set_xticklabels(
            [xlabels[i] for i in range(0, len(xlabels), xticks_step)],
        )

        min_val, max_val = min(all_vals), max(all_vals)
        ax.set_ylim(min_val * 0.75, max_val * 1.1)

        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((0, 0))
        fmt.set_useOffset(True)
        # Force consistent decimal places so y-axis position is uniform
        _orig_set_format = fmt._set_format
        def _fixed_format(_fmt=fmt):
            _orig_set_format()
            _fmt.format = '%1.2f'
        fmt._set_format = _fixed_format
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, min_n_ticks=3))

        ax.set_title(title or f"Top-$k$={gp}", pad=8)

        fig.legend(
            handles, labels_leg,
            loc="upper center", bbox_to_anchor=(0.5, 1.02),
            ncol=len(handles), frameon=False,
            handlelength=2.0, columnspacing=1.5,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        if save_dir is not None:
            out = save_dir / f"isolated_params_{gp}_seed_{seed}.pdf"
            fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
            print("Saved:", out)

        if show:
            plt.show()
            break
        else:
            plt.close(fig)

    return fig


def plot_jaccard_heatmap_only(
    iso_sources: dict[str, tuple[str | Path, str]],
    json_paths: dict[str, str | Path],
    show_chance_line: bool = False,
    save_dir: str | Path | None = None,
    show: bool = True,
    figsize: tuple[float, float] | None = None,
    fontsize: int = 11,
    cmap: str = "YlOrRd",
    annot_fontsize: int = 7,
    title: str | None = None,
    **kwargs,
):
    """
    Standalone Jaccard heatmap: rows = language pairs (+ Chance),
    columns = layer indices.

    Args:
        iso_sources: dict mapping language label → (json_path, "run1"|"run2")
        json_paths:  dict { pair_label: json_path }
        show_chance_line: if True, adds a Chance row
        save_dir:    if provided, saves one PDF per good_percent
        show:        if True, plt.show() and stop after first entry
        figsize:     optional (width, height)
        fontsize:    axis-label font size
        cmap:        colour-map
        annot_fontsize: font size for cell annotations
        title:       optional custom title
    """
    entries = _prepare_per_layer_data(iso_sources, json_paths, show_chance_line)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': kwargs.get("label_fontsize", 18),
        'axes.titlesize': kwargs.get("title_fontsize", 18),
        'xtick.labelsize': kwargs.get("xtick_fontsize", 16),
        'ytick.labelsize': kwargs.get("ytick_fontsize", 14),
        'legend.fontsize': 14,
        'lines.linewidth': 2,
        'lines.markersize': 10,
    })

    fig = None
    for d in entries:
        x, xlabels = d["x"], d["xlabels"]
        row_labels, matrix = d["row_labels"], d["matrix"]
        gp, seed = d["gp"], d["seed"]

        n_rows = len(row_labels)
        if figsize is None:
            _figsize = (max(4, 0.25 * len(x)), max(2, 0.45 * n_rows))
        else:
            _figsize = figsize

        fig, ax = plt.subplots(figsize=_figsize)

        xticks_step = 2 if len(x) <= 20 else 3

        im = ax.imshow(matrix, aspect="auto", cmap=cmap)

        ax.set_xticks(range(len(x)))
        if gp == 0.15:
            ax.set_xticklabels(
                [xlabels[i] if i % xticks_step == 0 else ""
                for i in range(len(xlabels))]
            )
        else:
            ax.set_xticklabels(["" for _ in x], rotation=0)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, rotation=45)

        if gp == 0.15:
            ax.set_xlabel("Layer Index")
        ax.set_ylabel("Language Pair")
        ax.set_title(title or f"Top-$k$={gp}", pad=8)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if not np.isnan(val):
                    thresh = (np.nanmax(matrix) + np.nanmin(matrix)) / 2
                    text_color = "white" if val > thresh else "black"
                    ax.text(j, i, f"{val:.3f}".replace("0.", "."), ha="center", va="center",
                            fontsize=annot_fontsize, color=text_color)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.005, aspect=25)
        fig.colorbar(im, cax=cax)

        plt.tight_layout()

        if save_dir is not None:
            suffix = "_with_chance" if show_chance_line else ""
            out = save_dir / f"jaccard_heatmap{suffix}_{gp}_seed_{seed}.pdf"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print("Saved:", out)

        if show:
            plt.show()
            break
        else:
            plt.close(fig)

    return fig


def plot_jaccard_heatmap_per_layer(
    iso_sources: dict[str, tuple[str | Path, str]],
    json_paths: dict[str, str | Path],
    save_dir: str | Path | None = None,
    show: bool = True,
    show_chance_line: bool = False,
    cmap: str = "YlOrRd",
    figsize: tuple[float, float] | None = None,
    layout: str = "vertical",
):
    """
    Two subplots (one per good_percent entry):
      - Subplot A: # isolated params per layer (line plot, one line per language)
      - Subplot B: Jaccard heatmap per layer
            rows = language pairs (+ optional chance row)
            columns = layer indices

    Supports an arbitrary number of languages / pairs (3, 4, 6, …).

    Args:
        iso_sources: dict mapping language label to (json_path, "run1"|"run2").
                     Tells which file and which run to read for each language's
                     isolated-param counts.
                     Example for 4 languages:
                       {"En": (en_de_path, "run1"),
                        "De": (en_de_path, "run2"),
                        "Fr": (en_fr_path, "run2"),
                        "Hi": (en_hi_path, "run2")}
        json_paths:  dict  { "En-De": "path/to/json", ... }
                     Each key becomes a heatmap row.
        save_dir:    if provided, saves one PDF per good_percent into this folder
        show:        if True, calls plt.show()
        show_chance_line: if True, adds a "Chance" row to the heatmap
        cmap:        colour-map for the heatmap
        figsize:     optional (width, height) override
        layout:      "vertical" (stacked) or "horizontal" (side-by-side)
    """
    assert layout in ("vertical", "horizontal"), \
        f"layout must be 'vertical' or 'horizontal', got {layout!r}"

    entries = _prepare_per_layer_data(iso_sources, json_paths, show_chance_line)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 18,
        'axes.titlesize': 15,
        'xtick.labelsize': 16,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'lines.linewidth': 2,
        'lines.markersize': 10,
    })

    fig = None
    for d in entries:
        x, xlabels = d["x"], d["xlabels"]
        iso_per_lang = d["iso_per_lang"]
        row_labels, matrix = d["row_labels"], d["matrix"]
        gp, seed = d["gp"], d["seed"]

        # ── figure setup ────────────────────────────────────────────────
        n_heatmap_rows = len(row_labels)
        is_vertical = (layout == "vertical")

        if figsize is None:
            if is_vertical:
                fig_w = max(4, 0.25 * len(x))
                fig_h = max(4, 1.5 + 0.45 * n_heatmap_rows)
            else:
                fig_w = max(8, 0.5 * len(x))
                fig_h = max(4, 0.45 * n_heatmap_rows)
            _figsize = (fig_w, fig_h)
        else:
            _figsize = figsize

        if is_vertical:
            height_ratios = [3, max(1.5, 0.5 * n_heatmap_rows)]
            fig, (ax_line, ax_heat) = plt.subplots(
                2, 1, figsize=_figsize,
                gridspec_kw={"height_ratios": height_ratios},
            )
        else:
            fig, (ax_line, ax_heat) = plt.subplots(
                1, 2, figsize=_figsize,
                gridspec_kw={"width_ratios": [1, 2.5]},
            )

        # ── Line plot: isolated params per layer ────────────────────────
        markersize = 4 if len(x) <= 20 else 2
        xticks_step = 2 if len(x) <= 20 else 3

        handles_line = []
        labels_line_legend = []
        all_iso_vals: list[int] = []
        for idx_lang, (lang_label, vals) in enumerate(iso_per_lang.items()):
            color = _COLORS[idx_lang % len(_COLORS)]
            marker = _MARKERS[idx_lang % len(_MARKERS)]
            h, = ax_line.plot(x, vals, marker=marker, color=color,
                              label=lang_label, linewidth=1.5,
                              markersize=markersize, alpha=0.8)
            handles_line.append(h)
            labels_line_legend.append(lang_label)
            all_iso_vals.extend(vals)

        ax_line.set_ylabel("# Isolated Params", fontsize=11)
        ax_line.grid(True, alpha=0.25, linewidth=0.5)

        ax_line.set_xticks(x[::xticks_step])
        ax_line.set_xticklabels(
            [xlabels[i] for i in range(0, len(xlabels), xticks_step)],
            fontsize=9,
        )

        min_val = min(all_iso_vals)
        max_val = max(all_iso_vals)
        ax_line.set_ylim(min_val * 0.75, max_val * 1.1)

        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((0, 0))
        fmt.set_useOffset(True)
        ax_line.yaxis.set_major_formatter(fmt)
        ax_line.yaxis.get_offset_text().set_size(9)
        ax_line.yaxis.set_major_locator(
            ticker.MaxNLocator(nbins=4, min_n_ticks=3),
        )

        title = f"Top-$k$={gp}"
        ax_line.set_title(title, pad=8)

        # ── Heatmap: Jaccard per layer ──────────────────────────────────
        im = ax_heat.imshow(matrix, aspect="auto", cmap=cmap)

        ax_heat.set_xticks(range(len(x)))
        ax_heat.set_xticklabels(
            [xlabels[i] if i % xticks_step == 0 else ""
             for i in range(len(xlabels))],
            fontsize=9,
        )
        ax_heat.set_yticks(range(len(row_labels)))
        ax_heat.set_yticklabels(row_labels, fontsize=10)

        ax_heat.set_xlabel("Layer Index", fontsize=11)
        ax_heat.set_ylabel("Jaccard Similarity", fontsize=11)

        # Annotate each cell with the Jaccard value
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if not np.isnan(val):
                    thresh = (np.nanmax(matrix) + np.nanmin(matrix)) / 2
                    text_color = "white" if val > thresh else "black"
                    ax_heat.text(
                        j, i, f"{val:.3f}".replace("0.", "."),
                        ha="center", va="center",
                        fontsize=11, color=text_color,
                    )

        divider = make_axes_locatable(ax_heat)
        cax = divider.append_axes("right", size="3%", pad=0.05, aspect=25)
        fig.colorbar(im, cax=cax)

        # ── legends & layout ────────────────────────────────────────────
        if is_vertical:
            legend_anchor = (0.53, 1.01)
        else:
            legend_anchor = (0.27, 1.01)

        fig.legend(
            handles_line, labels_line_legend,
            loc="upper center",
            bbox_to_anchor=legend_anchor,
            ncol=len(handles_line), frameon=False,
            handlelength=2.0, columnspacing=1.5,
        )

        if is_vertical:
            fig.align_ylabels([ax_line, ax_heat])
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_dir is not None:
            gp_str = str(gp)
            suffix = "_with_chance" if show_chance_line else ""
            out_path = (
                save_dir
                / f"jaccard_heatmap_layers{suffix}_{gp_str}_seed_{seed}.pdf"
            )
            fig.savefig(
                out_path, dpi=300, bbox_inches="tight", pad_inches=0.02,
            )
            print("Saved:", out_path)

        if show:
            plt.show()
            break
        else:
            plt.close(fig)

    return fig

