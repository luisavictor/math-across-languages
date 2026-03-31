from jaccard_plots import plot_params_and_heatmap, plot_isolated_params_per_layer, plot_jaccard_heatmap_only

en_de_path = "/raid/s3/opengptx/behzad_shomali/LabTest/results_jaccard/gsm8k_race_en_vs_de_Llama-1B/repeat0/jaccard_summary.json"
en_fr_path = "/raid/s3/opengptx/behzad_shomali/LabTest/results_jaccard/gsm8k_race_en_vs_fr_Llama-1B/repeat0/jaccard_summary.json"  # placeholder
en_hi_path = "/raid/s3/opengptx/behzad_shomali/LabTest/results_jaccard/gsm8k_race_en_vs_hi_Llama-1B/repeat0/jaccard_summary.json"  # placeholder
de_fr_path = "/raid/s3/opengptx/behzad_shomali/LabTest/results_jaccard/gsm8k_race_de_vs_fr_Llama-1B/repeat0/jaccard_summary.json"  # placeholder
de_hi_path = "/raid/s3/opengptx/behzad_shomali/LabTest/results_jaccard/gsm8k_race_de_vs_hi_Llama-1B/repeat0/jaccard_summary.json"  # placeholder
fr_hi_path = "/raid/s3/opengptx/behzad_shomali/LabTest/results_jaccard/gsm8k_race_hi_vs_fr_Llama-1B/repeat0/jaccard_summary.json"  # placeholder

if "Qwen" in en_de_path:
    model_tag = "Qwen-4B"
elif "Llama-8B" in en_de_path:
    model_tag = "Llama-8B"
elif "Llama-1B" in en_de_path:
    model_tag = "Llama-1B"
else:
    raise ValueError("Model tag not found in path")

# plot_params_and_heatmap(
#     iso_sources={
#         "En": (en_de_path, "run1"),   # En is run1 in En-De file
#         "De": (en_de_path, "run2"),   # De is run2 in En-De file
#         "Fr": (en_fr_path, "run2"),   # Fr is run2 in En-Fr file
#         "Hi": (en_hi_path, "run2"),   # Hi is run2 in En-Hi file
#     },
#     json_paths={
#         "En-De": en_de_path,
#         "En-Fr": en_fr_path,
#         "En-Hi": en_hi_path,
#         "De-Hi": de_hi_path,
#         "Fr-Hi": fr_hi_path,
#         "De-Fr": de_fr_path,
#     },
#     good_percents_filter=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.15],
#     save_path="combined_params_heatmap.pdf",
#     show=True,
#     figsize=(11, 4),
#     linewidth=2,
#     fontsize=17,
#     legend_fontsize=14
# )

common = dict(
    iso_sources={"En": (en_de_path, "run1"), "De": (en_de_path, "run2"),
                 "Fr": (en_fr_path, "run2"), "Hi": (en_hi_path, "run2")},
    json_paths={
        "En-De": en_de_path,
        "En-Fr": en_fr_path,
        "En-Hi": en_hi_path,
        "De-Hi": de_hi_path,
        "Fr-Hi": fr_hi_path,
        "De-Fr": de_fr_path,
    },
    show_chance_line=True,
)

# plot_isolated_params_per_layer(
#     **common, 
#     figsize=(6, 3), 
#     fontsize=14, 
#     save_dir=f"./isolated_params_per_layer_plots/{model_tag}/", 
#     k_values_to_have_xlabels=[0.1, 0.15],
#     k_values_to_have_ylabels=[0.0001, 0.01, 0.1],
#     show=False
# )


# plot_jaccard_heatmap_only(
#     **common, 
#     figsize=(18, 4), 
#     annot_fontsize=11, 
#     save_dir=f"./per_layer_jaccard_heatmap/{model_tag}/", 
#     show=False
# )

plot_jaccard_heatmap_only(
    **common, 
    figsize=(18, 4.5), 
    annot_fontsize=22, 
    fontsize=15, 
    save_dir=f"./per_layer_jaccard_heatmap/{model_tag}/", 
    show=False,
    title_fontsize=26,
    label_fontsize=24,
    xtick_fontsize=18,
    ytick_fontsize=21
)