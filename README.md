# LLM Parameters for Math Across Languages: Shared or Separate?

This repository contains the codebase for the paper "LLM Parameters for Math Across Languages: Shared or Separate?" and additional exploratory analyses extending the same parameter-isolation framework to coding and math/code parameter overlap.


Below, we explain how to use the provided code for identifying, intervening on, and comparing task-associated parameters in language models.

The core experiments reproduce the cross-lingual math-parameter analysis from our paper, where math-associated parameters are extracted for GSM8K-style reasoning across English, German, French, and Hindi, then compared via global and layer-wise Jaccard overlap.

We also include post-paper exploratory analyses for coding tasks using CodeAlpaca and a local oracle-based evaluator. These analyses investigate whether coding-associated parameters overlap with math-reasoning parameters.




## Project layout

- `MathNeuro/MathNeuroFast.py`: main experiment driver (identify task-specific parameters, prune/scale, eval).
- `MathNeuro/compute_param_overlap.py`: compares two isolated parameter masks, e.g., English vs German math-specific parameters in terms of Jaccard similarity.
- `MathNeuro/codealpaca_oracle.py`: utilities for CodeAlpaca oracle evaluation.
- `MathNeuro/find_important_layers.py`: layer selection by bypassing layers and measuring accuracy drops.
- `runner.py`: sandboxed executor used by oracle tooling.
- `build_oracle_cases.py`: builds oracle cases from CodeAlpaca CSVs.
- `ORACLE_TESTS.md`: offline oracle workflow notes.
- `custom_datasets/`: german translations of GSM8K, MMLU and Race, and for coding the original CodeAlpaca, each with train/test split
- `lm_eval_tasks/`: custom tasks/yaml files for lm_eval (e.g., German gsm8k_cot is not a predefined task in the standard lm_eval package, thus, we defined it ourselves).

## Setup for conda(local)

```
conda env create -n math_neuro -f MathNeuro/requirements.yml
conda activate math_neuro
python -m spacy download xx_sent_ud_sm
```

If you already have a working environment, skip this section.

## Key arguments (MathNeuro.py)

- `--model`: HuggingFace model id (string).
- `--train_dataset`: CSV with task-specific dataset, e.g., gsm8k
- `--calibration_datasets`: one or more CSVs with non-task-related content
- `--calibration_dataset_names`: human-readable names (same order as calibration datasets).
- `--eval_datasets`: lm_eval task names (e.g., `race`).
- `--save_path`: where eval results and masks are written.
- `--text_file`: filename for pruning/scaling logs.
- `--num_samples`: number of samples to identify task-specific params, default is 500.
- `--eval_dataset_subset`: evaluation subset size for faster runs, default is 200.
- `--proportion`: fraction of top params to isolate (top-k).
- `--scalar`: 0 for pruning, >0 for scaling.
- `--fine_tune`: if set, the model is fine-tuned on the extracted parameter set, pruning or scaling are ignored (`scalar` value does not matter)
- `--pre_train_eval`: if set, the model is evaluated on the given tasks before anything has been modified (baseline performance)
- `--store_params` saves isolated masks to `.../isolated_masks/`.
- `--train_lm_eval_task` lets you evaluate a train task via lm_eval (e.g., `gsm8k_cot`).
- `--run_codealpaca_eval` runs the CodeAlpaca oracle evaluation after lm_eval.

## Core workflow

Pruning/Scaling/Finetuning Example hyperparameter configurations

- Scaling on German gsm8k and race datasets:

```
--model
meta-llama/Llama-3.2-1B-Instruct
--save_path
results_gsm8k_race_german
--train_dataset
../custom_datasets/GSM8k_German/gsm8k_de_train.csv
--train_lm_eval_task
gsm8k_de_cot
--eval_datasets
race_de
--calibration_datasets
../custom_datasets/Race_German/race_de_train.csv
--calibration_dataset_names
Race
--eval_dataset_subset
200
--num_samples
500
--proportion=0.001
--random_state=1
--scalar=1.01
```

- Finetuning on English gsm8k and mmlu datasets:

```
--model
meta-llama/Llama-3.2-1B-Instruct
--save_path
results_gsm8k_mmlu_en
--train_dataset
data/gsm8k.csv
--train_lm_eval_task
gsm8k_cot
--eval_datasets
mmlu
--calibration_datasets
data/mmlu.csv
--calibration_dataset_names
MMLU
--eval_dataset_subset
200
--num_samples
500
--proportion=0.001
--random_state=1
--fine_tune
```

- Pruning on CodeAlpaca and MMLU datasets including baseline evaluation at the very beginning:

```
--model
meta-llama/Llama-3.2-1B-Instruct
--save_path
results_codealpaca_mmlu
--train_dataset
../custom_datasets/CodeAlpaca/codealpaca_train.csv
--train_lm_eval_task
codealpaca
--eval_datasets
mmlu
--calibration_datasets
data/mmlu.csv
--calibration_dataset_names
MMLU
--eval_dataset_subset
200
--num_samples
500
--proportion=0.001
--random_state=1
--run_codealpaca_eval
--pre_train_eval
```

## CodeAlpaca oracle evaluation (offline)

See `ORACLE_TESTS.md` for the full workflow. Minimal flow for creation (already done in this repository):

```
py build_oracle_cases.py --csv custom_datasets/CodeAlpaca/codealpaca_test_filtered.csv --output oracle_cases.jsonl --n_cases 30 --seed 0
py -m pytest -q tests/test_oracle_cases.py
```

At the moment oracle_cases.jsonl contains 2 test cases each for 200 samples from the CodeAlpaca_test_filtered.csv.

Model outputs should be JSONL (default `candidate_generations.jsonl`) with:

```
{"sample_id": 0, "code": "<model completion here>"}
```

## Outputs

- `MathNeuro/results_*`: experiment results by dataset setting.
- `MathNeuro/results_*/eval_results/`: lm_eval metrics JSON.
- `MathNeuro/results_*/isolated_masks/`: saved parameter masks (`.pt`).
- `oracle_cases.jsonl`: oracle cases for CodeAlpaca evaluation.

## Open questions / TODO

- Decide whether CodeAlpaca parameter identification should use Python-only tasks for both train and test (currently, testset is Python-only).
- Confirm if English MMLU needs math-related question filtering like already done for German MMLU.
- Wire layer selection into `MathNeuro/MathNeuro.py` (currently standalone) to only modifiy parameters from specific layers.
