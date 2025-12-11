import re
import matplotlib
matplotlib.use("Agg")      # headless, no interactive window, but also no crash
# or, on a local machine with GUI:
# matplotlib.use("Qt5Agg") # or "TkAgg" depending on what you have
import matplotlib.pyplot as plt



def plot_math_only_vs_top(
    log_file_path,
    min_top=0.001,
    max_top=1.0,
    save_path="math_only_vs_top.png",
    show=True,  # set True only if your matplotlib backend supports interactive windows
):
    """
    Parse pruning log lines and plot math-only-after-removing-non-math
    as a function of top-k fraction.

    Expected line format, e.g.:
        [Race] repeat 1, top 10.0000% — math: 97307792, non-math: 97307792,
        math-only after removing non-math: 12286949
    """

    # Regex to capture: top %, math, non-math, math-only
    # The ".*" after the % makes it robust to different dashes / spacing
    pattern = re.compile(
        r"top\s+([0-9.]+)%.*math:\s+(\d+),\s+non-math:\s+(\d+),\s+"
        r"math-only after removing non-math:\s+(\d+)"
    )

    top_fracs = []
    math_vals = []
    nonmath_vals = []
    math_only_vals = []

    with open(log_file_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            top_percent = float(m.group(1))   # e.g. 10.0000
            math = int(m.group(2))
            nonmath = int(m.group(3))
            math_only = int(m.group(4))

            top_frac = top_percent / 100.0    # convert % → fraction
            print("parsed top_frac:", top_frac)

            # keep only within requested range
            if not (min_top <= top_frac <= max_top):
                continue

            top_fracs.append(top_frac)
            math_vals.append(math)
            nonmath_vals.append(nonmath)
            math_only_vals.append(math_only)
            print("current top_fracs list:", top_fracs)

    if not top_fracs:
        print("No matching lines found in the given top range.")
        return

    # Sort by top fraction so the curve is nice
    sorted_idx = sorted(range(len(top_fracs)), key=lambda i: top_fracs[i])
    top_fracs      = [top_fracs[i]      for i in sorted_idx]
    math_only_vals = [math_only_vals[i] for i in sorted_idx]
    math_vals      = [math_vals[i]      for i in sorted_idx]
    nonmath_vals   = [nonmath_vals[i]   for i in sorted_idx]

    fig, ax = plt.subplots()
    ax.plot(top_fracs, math_only_vals, marker="o", label="math-only")

    # If you later want the other curves, just uncomment:
    ax.plot(top_fracs, math_vals, marker="x", linestyle="--", label="math")
    ax.plot(top_fracs, nonmath_vals, marker="s", linestyle="--", label="non-math")

    ax.set_xscale("log")
    ax.set_xlabel("Top-k fraction (0–1)")
    ax.set_ylabel("Number of parameters")
    ax.set_title("Math-only parameters vs. top-k fraction")
    ax.set_xlim(min_top, max_top)
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    fig.tight_layout()

    # Always save so you can inspect even if show() is broken
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)



log_path = "../results_gsm8k_race_en/eval_results/meta-llama/Llama-3.2-1B-Instruct/parameter_statistics"
plot_math_only_vs_top(log_path, min_top=0.001, max_top=1.0)

import numpy as np
import matplotlib.pyplot as plt


def plot_vector_stats(vectors, categories=None, title="Vector Statistics"):
    """
    vectors: list of NumPy arrays or lists of equal length
    categories: labels for each dimension (same length as vectors[0])
    """

    # Convert to array of shape (num_vectors, vector_length)
    data = np.array(vectors)

    if categories is None:
        categories = np.arange(data.shape[1])

    # Compute mean and standard deviation per index
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(categories, means, marker='o', label="Mean")
    plt.fill_between(categories, means - stds, means + stds,
                     alpha=0.25, label="Standard Deviation")

    plt.title(title)
    plt.xlabel("Top-k")
    plt.ylabel("Jaccard Similarity")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("jaccard_plot.png")

    return means, stds



categories = [0.001, 0.01, 0.05, 0.1, 0.15]

vec1 = [0.238, 0.219, 0.212, 0.191, 0.211]
vec2 = [0.242, 0.219, 0.219, 0.218, 0.201]


means, stds = plot_vector_stats([vec1, vec2], categories,
                                title="Parameter Statistics")
