# LLM Parameters for Math Across Languages: Shared or Separate?

This repository contains the codebase for the paper "LLM Parameters for Math Across Languages: Shared or Separate?" and additional exploratory analyses extending the same parameter-isolation framework to coding and math/code parameter overlap.

Below, we explain how to use the provided code for identifying, intervening on, and comparing task-associated parameters in language models.

The core experiments reproduce the cross-lingual math-parameter analysis from our paper, where math-associated parameters are extracted for GSM8K-style reasoning across English, German, French, and Hindi, then compared via global and layer-wise Jaccard overlap.

The main finding is that math-associated parameters are neither fully shared across languages nor fully language-specific: they show partial cross-lingual overlap, especially in intermediate layers. English tends to yield the largest set of math-associated parameters, while lower-resource languages such as Hindi show smaller sets and lower overlap with English. Intervention experiments further suggest that these parameters act collectively: pruning or scaling larger subsets changes math behavior more reliably than targeting individual parameters.

We also include post-paper exploratory analyses for coding tasks using CodeAlpaca and a local oracle-based evaluator. These analyses investigate whether coding-associated parameters overlap with math-reasoning parameters.

Additional results, visualizations, and explanations are available on our project webpage: https://math-across-languages.github.io/.

## Project layout

- `MathNeuro/MathNeuroFast.py`: main experiment driver (identify task-specific parameters, prune/scale/fine-tune, eval).
- `MathNeuro/compute_param_overlap.py`: compares two isolated parameter masks, e.g., English vs German math-specific parameters in terms of Jaccard similarity.
- `MathNeuro/codealpaca_oracle.py`: utilities for CodeAlpaca oracle evaluation.
- `runner.py`: subprocess executor used by oracle tooling.
- `build_oracle_cases.py`: builds oracle cases from CodeAlpaca CSVs.
- `ORACLE_TESTS.md`: offline oracle workflow notes.
- `custom_datasets/`: translated GSM8K, MMLU, and RACE datasets for German, French, and Hindi, plus CodeAlpaca data used in exploratory coding analyses.
- `lm_eval_tasks/`: custom task/yaml files for lm_eval (e.g., translated GSM8K CoT tasks are not predefined in the standard lm_eval package).

## Setup

Create an environment with Python 3.10+ and install the main dependencies:

```bash
pip install torch transformers pandas numpy accelerate datasets evaluate
pip install -e MathNeuro
python -m spacy download xx_sent_ud_sm
```

## Key arguments (`MathNeuroFast.py`)

- `--model`: HuggingFace model id, e.g., `meta-llama/Llama-3.2-1B-Instruct`.
- `--train_dataset`: CSV with the task-specific dataset.
- `--calibration_datasets`: one or more CSVs with non-task-related calibration content.
- `--calibration_dataset_names`: human-readable names for the calibration datasets, in the same order as `--calibration_datasets`.
- `--eval_datasets`: lm_eval task names used to measure post-intervention behavior, e.g., `race_de`.
- `--save_path`: directory where evaluation results and score/mask artifacts are written.
- `--text_file`: filename for text logs; this is still parsed by the script and used in output paths.
- `--num_samples`: number of samples used to identify task-associated parameters, default `500`.
- `--eval_dataset_subset`: evaluation subset size for faster runs. We use `200` in the example/paper-style runs below; if omitted, `MathNeuroFast.py` currently falls back to `100`.
- `--eval_dataset_size`: optional size for task-specific evaluation datasets before subsetting.
- `--proportion`: fraction of top parameters to isolate, i.e., top-k.
- `--scalar`: intervention factor for isolated parameters. `0` prunes isolated parameters; values above `0` scale them.
- `--fine_tune`: if set, the model is fine-tuned on the extracted parameter set instead of pruning/scaling.
- `--pre_train_eval`: if set, the model is evaluated on the train/eval tasks before parameter intervention.
- `--train_lm_eval_task`: lm_eval task corresponding to the task-specific train dataset, e.g., `gsm8k_de_cot`.
- `--run_codealpaca_eval`: runs the CodeAlpaca oracle evaluation after lm_eval.
- `--batch_size`: batch size for lm_eval evaluation, default `1`.
- `--random_state`: seed for dataset sampling and evaluation, default `1` in `MathNeuroFast.py`.
- `--num_repeats`: number of repeated random samples in the original driver, default `5` in `MathNeuro.py`.


## Stored parameter files

For each run, the scripts create a mask/score directory at:

```text
<save_path>/isolated_masks/<model>/
```

The main saved `.pt` files are:

- `train_scores_seed<seed>.pt`: parameter-importance scores computed on the task-specific training dataset, e.g., GSM8K or CodeAlpaca.
- `comparison_scores_seed<seed>.pt`: parameter-importance scores computed on the calibration/non-task dataset, e.g., RACE or MMLU.
- `gsm8k_<calibration_name>_<proportion>_repeat<repeat>.pt`: only written by `MathNeuro.py` when `--store_params` is set. This stores the final boolean isolated-parameter masks plus metadata (`model_name`, `dataset_name`, `good_percent`, and `repeat`).

The score files are dictionaries mapping model parameter names to tensors. `compute_param_overlap.py` loads the train/comparison score files, rebuilds top-k isolated masks for a chosen `--proportion`, and then computes global and layer-wise Jaccard overlap. This means overlap sweeps can be rerun for several top-k values without recomputing forward-pass scores.

## Core workflow

Example pruning/scaling configurations:

- Scaling on German GSM8K and RACE datasets:

```bash
python MathNeuro/MathNeuroFast.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --save_path results_gsm8k_race_german \
  --train_dataset ../custom_datasets/GSM8k_German/gsm8k_de_train.csv \
  --train_lm_eval_task gsm8k_de_cot \
  --eval_datasets race_de \
  --calibration_datasets ../custom_datasets/Race_German/race_de_train.csv \
  --calibration_dataset_names Race \
  --eval_dataset_subset 200 \
  --num_samples 500 \
  --proportion 0.001 \
  --random_state 1 \
  --scalar 1.01
```

- Pruning on CodeAlpaca and MMLU datasets including baseline evaluation at the beginning:

```bash
python MathNeuro/MathNeuroFast.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --save_path results_codealpaca_mmlu \
  --train_dataset ../custom_datasets/CodeAlpaca/codealpaca_train.csv \
  --train_lm_eval_task codealpaca \
  --eval_datasets mmlu \
  --calibration_datasets data/mmlu.csv \
  --calibration_dataset_names MMLU \
  --eval_dataset_subset 200 \
  --num_samples 500 \
  --proportion 0.001 \
  --random_state 1 \
  --run_codealpaca_eval \
  --pre_train_eval
```

## CodeAlpaca oracle evaluation

See `ORACLE_TESTS.md` for the full workflow. Minimal flow for creating and testing oracle cases:

```bash
py build_oracle_cases.py --csv custom_datasets/CodeAlpaca/codealpaca_test_filtered.csv --output oracle_cases.jsonl --n_cases 2 --seed 0
py -m pytest -q tests/test_oracle_cases.py
```

The included `oracle_cases.jsonl` contains two oracle test cases for each of 330 filtered CodeAlpaca samples. During `MathNeuroFast.py` runs with `--run_codealpaca_eval`, `--eval_dataset_subset` controls how many eligible sample IDs are selected for the oracle evaluation.

Model outputs should be JSONL (default `candidate_generations.jsonl`) with:

```json
{"sample_id": 0, "code": "<model completion here>"}
```

The oracle workflow executes model-generated code locally through `runner.py`, use it in an isolated environment.

## Outputs

- `<save_path>/eval_results/<model>/`: lm_eval metrics, generated samples, and oracle metrics.
- `<save_path>/isolated_masks/<model>/`: saved `.pt` train/comparison score tensors and, when requested by `MathNeuro.py --store_params`, finalized isolated mask files.
- `MathNeuro/jaccard_results/`: example overlap-analysis outputs.
- `oracle_cases.jsonl`: oracle cases for CodeAlpaca evaluation.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{shomali2026llm,
  title = {LLM Parameters for Math Across Languages: Shared or Separate?},
  author = {Shomali, Behzad and Victor, Luisa and Selbach, Tim and Bashir, Ali Hamza and Berghaus, David and Koehler, Joachim and Ali, Mehdi and Frey, Markus},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop)},
  year = {2026},
  pages = {1212--1235},
  url = {https://aclanthology.org/2026.acl-srw.107/}
}
```
