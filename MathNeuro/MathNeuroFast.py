import os
import sys
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import pandas as pd
import lm_eval
import json
import random, numpy as np, torch
from pathlib import Path
from utils import *


sys.path.append(os.path.dirname(__file__))
from fine_tune import fine_tune_on_isolated_params
import codealpaca_oracle
import time

def pre_train_eval_already_done(save_path, model, train_task):
    base_dir = f"{save_path}/eval_results/{model}"
    pre_eval_path = f"{base_dir}/pre_results.json"
    if not os.path.exists(pre_eval_path):
        return False
    if train_task is None:
        return True
    pre_train_task_path = f"{base_dir}/pre_results_train_task.json"
    return os.path.exists(pre_train_task_path)

beg = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--model', help="Huggingface model to train, entered as string", type=str)
parser.add_argument('--eval_datasets', nargs='+',
                    help="dataset(s) to evaluate models on post pruning to evalaute catastrophic forgetting, entered as strings; should be task names from Eleuther AI LM Evaluation Harness",
                    type=str)
parser.add_argument('--train_dataset',
                    help="path to math train dataset; should be a path to a CSV file with question/solution pairs in a columns titled 'question' and 'solution' along with ground-truth answers in a column called 'answer'",
                    type=str)
parser.add_argument('--calibration_datasets', nargs='+',
                    help="path to calibration datasets; should be paths to CSV files with instruction/response pairs in a column titled 'qa'",
                    type=str)
parser.add_argument('--save_path',
                    help="save path for eval results after running Eleuther AI LM Evaluation Harness post pruning",
                    type=str)
parser.add_argument('--text_file',
                    help="name of text file for saving pruning results during training if evaluating math reasoning using a non-Eleuther AI LM Evaluation Harness task in a PoT format",
                    type=str)
parser.add_argument('--num_repeats', help="number of repeats for pruning or scaling experiment", type=int, default=5)
parser.add_argument('--pre_train_eval',
                    help="bool to indicate if full evaluation on eval and train datasets should be conducted before training",
                    action="store_true")
parser.add_argument('--random_state',
                    help="random state for initial dataset shuffling and creating train/eval split for train dataset",
                    type=int, default=1)
parser.add_argument('--scalar', help="scale factor for top parameters; default is 0 to run pruning experiments",
                    type=float, default=0)
parser.add_argument('--eval_dataset_size', help="desired number of samples for task specific eval dataset", type=int,
                    default=None)
parser.add_argument('--eval_dataset_subset',
                    help="desired number of samples for task specific eval dataset if subsetting to reduce run time",
                    type=int, default=100)
parser.add_argument('--calibration_dataset_names', nargs='+',
                    help="desired name of calibration datasets; should be strings entered in same order as calibration_datasets",
                    type=str)
parser.add_argument('--num_samples', help="desired number of samples for calculating task specific parameters",
                    type=int, default=500)
parser.add_argument('--train_lm_eval_task',
                    help="if your training dataset is an Eleuther AI LM Evaluation Harness task, specify the associated task for the test set.",
                    type=str, default=None)
parser.add_argument('--run_codealpaca_eval',
                    help="run CodeAlpaca oracle evaluation after lm_eval",
                    action="store_true")
parser.add_argument('--proportion', help="desired proportion of top parameters to calculate", type=float, nargs='+', default=None)
parser.add_argument('--fine_tune',
                    help="freeze all non-task-specific parameters and fine-tune only isolated task-specific weights",
                    action="store_true")
parser.add_argument('--batch_size', help="batch size for evaluation", type=int, default=1)
parser.add_argument('--random_prune',
                    help="Run random pruning experiment to test group vs individual neuron effect. "
                         "Randomly prunes the same number of neurons as the original experiment, "
                         "sampling proportional to score from the top 15%% pool.",
                    action="store_true")
parser.add_argument('--random_prune_seeds', nargs='+',
                    help="Random seeds for the random pruning sampling (default: 42 123 456)",
                    type=int, default=[42, 123, 456])
parser.add_argument('--random_prune_fractions', nargs='+',
                    help="Fractions of math-only neurons to prune in random experiment "
                         "(default: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)",
                    type=float,
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])



args = parser.parse_args()
random.seed(args.random_state)
np.random.seed(args.random_state)
torch.manual_seed(args.random_state)
torch.cuda.manual_seed_all(args.random_state)
torch.backends.cudnn.benchmark = False

oracle_cases_path = None
oracle_eval_ids = None
if args.run_codealpaca_eval:
    oracle_cases_path, _ = codealpaca_oracle.oracle_paths()
    oracle_eval_ids = codealpaca_oracle.select_oracle_sample_ids(
        oracle_cases_path,
        eval_subset=args.eval_dataset_subset,
        min_cases_per_id=2,
        seed = args.random_state
    )

mask_dir = f"{args.save_path}/isolated_masks/{args.model}"
os.makedirs(mask_dir, exist_ok=True)

if os.path.exists("../lm_eval_tasks"):
    task_manager = lm_eval.tasks.TaskManager(include_path="../lm_eval_tasks")
elif os.path.exists("lm_eval_tasks"):
    task_manager = lm_eval.tasks.TaskManager(include_path="lm_eval_tasks")
else:
    raise ValueError("Could not find lm_eval_tasks directory.")


train = pd.read_csv(args.train_dataset)
train = train.sample(frac=1, random_state=args.random_state)

calibration_datasets = []
for dataset in args.calibration_datasets:
    if '/' in dataset:
        dataset_name = dataset.split('/')[-1]
        dataset_name = dataset_name.split('.csv')[0]
        calibration_datasets.append(dataset_name)
    else:
        dataset_name = dataset.split('.csv')[0]
        calibration_datasets.append(dataset_name)

dataset_list = []
for dataset, dataset_name, name in zip(args.calibration_datasets, calibration_datasets, args.calibration_dataset_names):
    # Load the dataset into a DataFrame
    globals()[dataset_name] = pd.read_csv(dataset).sample(frac=1,
                                                          random_state=args.random_state)  # Shuffle the DataFrame
    # Assign a name attribute to the DataFrame
    globals()[dataset_name].name = name
    # Append the actual DataFrame object to the list
    dataset_list.append(globals()[dataset_name])

output_file = f"{args.save_path}/eval_results/{args.model}/{args.text_file}"
results_path = f"{args.save_path}/eval_results/{args.model}/"
print("Results Path: ", results_path)
os.makedirs(os.path.dirname(results_path), exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)

model.eval()
print("datasets and models loaded")
if args.pre_train_eval:
    if pre_train_eval_already_done(args.save_path, args.model, args.train_lm_eval_task):
        print("Skipping pre-train eval: outputs already exist.")
    elif args.train_lm_eval_task is not None:
        # task_manager = lm_eval.tasks.TaskManager()
        # --log_samples --output_path results/phi_15_base --device cuda:0 --batch_size auto:4
        # Setting `task_manager` to the one above is optional and should generally be done
        # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
        # `simple_evaluate` will instantiate its own task_manager if it is set to None here.

        model.eval()
        codealpaca_limit = args.eval_dataset_subset
        oracle_env_set = False
        if args.run_codealpaca_eval and oracle_eval_ids is not None:
            os.environ["CODEALPACA_ORACLE_CASES_PATH"] = oracle_cases_path
            os.environ["CODEALPACA_ALLOWED_SAMPLE_IDS"] = ",".join(
                str(sid) for sid in sorted(oracle_eval_ids)
            )
            oracle_env_set = True
            codealpaca_limit = len(oracle_eval_ids)

        try:
            results = lm_eval.simple_evaluate(  # call simple_evaluate
                model='hf',
                model_args={'pretrained': model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
                tasks=args.train_lm_eval_task,
                task_manager=task_manager,
                log_samples=True,
                batch_size=args.batch_size,
                limit=args.eval_dataset_subset,
                random_seed=args.random_state
            )

            samples_dir = f"{args.save_path}/eval_results/{args.model}/samples_pre_train_task_{args.scalar}"
            saved = save_lm_eval_samples(results, f"{samples_dir}/pre_modification")
            print("Saved:", saved)
        finally:
            if oracle_env_set:
                os.environ.pop("CODEALPACA_ORACLE_CASES_PATH", None)
                os.environ.pop("CODEALPACA_ALLOWED_SAMPLE_IDS", None)
        results_path = f"{args.save_path}/eval_results/{args.model}/pre_results_train_task.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as outfile:
            json.dump(results['results'], outfile)

        if args.run_codealpaca_eval:
            samples = results["samples"]["codealpaca"]
            out_path = f"{args.save_path}/eval_results/{args.model}/codealpaca_samples.jsonl"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            with open(out_path, "w", encoding="utf-8") as f:
                for ex in samples:
                    resp = ex.get("filtered_resps", [None])[0]
                    if isinstance(resp, list):
                        resp = resp[0] if resp else ""
                    if resp is None:
                        resp = ex.get("resps", [[None]])[0][0]
                    sample_id = ex.get("doc", {}).get("sample_id")
                    if sample_id is None:
                        sample_id = ex["doc_id"]
                    f.write(json.dumps({"sample_id": int(sample_id), "code": resp}, ensure_ascii=False) + "\n")

            candidate_path = f"{args.save_path}/eval_results/{args.model}/candidate_generations.jsonl"
            print(candidate_path)
            metrics_path = f"{args.save_path}/eval_results/{args.model}/codealpaca_oracle_metrics.json"
            codealpaca_oracle.write_codealpaca_candidates(
                samples,
                candidate_path,
                allowed_sample_ids=set(oracle_eval_ids) if oracle_eval_ids is not None else None,
            )
            codealpaca_oracle.run_codealpaca_oracle_eval(
                candidate_path,
                metrics_path,
                allowed_sample_ids=set(oracle_eval_ids) if oracle_eval_ids is not None else None,
                max_cases_per_id=2,
            )

        model.eval()
        results = lm_eval.simple_evaluate(  # call simple_evaluate
            model='hf',
            model_args={'pretrained': model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
            tasks=args.eval_datasets,
            task_manager=task_manager,
            log_samples=False,
            batch_size=args.batch_size,
        )
        results_path = f"{args.save_path}/eval_results/{args.model}/pre_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as outfile:
            json.dump(results['results'], outfile)

magnitude = {}

def compute_score_mask(model, train, num_samples, save_dir, save_path):
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    param_dict = {}
    for name, param in model.named_parameters():
        param_dict[name] = torch.zeros_like(param, device=param.device)

    for i in range(0, num_samples):
        if 'qa' in train.columns.to_list():
            prompt = train.iloc[i]['qa']
        else:
            question = train['question'].iloc[i]
            answer = train['solution'].iloc[i]
            prompt = f"""Instruct: {question} Let's write a Python program.\nOutput:\n{answer}"""
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(inputs)
            for key, tensor in magnitude.items():
                try:
                    param_dict[f"{key}.weight"] += tensor
                except:
                    print(f'passed at {key}')
    keys_to_remove = [key for key in param_dict if key.split('.weight')[0] not in magnitude]

    for key in keys_to_remove:
        del param_dict[key]

    # now, store param_dict.items()
    if save_path is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        full_path = save_dir / save_path
        full_path.parent.mkdir(parents=True, exist_ok=True)  # if save_path includes subfolders

        cpu_param_dict = {k: v.detach().cpu() for k, v in param_dict.items()}
        torch.save(cpu_param_dict, str(full_path))

        return param_dict, str(full_path)

    return param_dict


def find_good_params(keep_ratio, prune=True, largest=True, param_dict = "good_scores"):
    global chosen_params
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    # create dictionary to store mask
    mask_dict = {}

    for k, v in param_dict.items():
        # don't count classifier layer
        if "embed" in k:
            if prune == False:
                mask_dict[k] = torch.zeros_like(v).to(v.device)
            else:
                mask_dict[k] = torch.ones_like(v).to(v.device)

        else:
            if prune == False:
                sizes = v.shape
                num_params = v.numel()
                keep_num = int(num_params * keep_ratio)
                tensor = v.view(-1)
                top_pos = torch.topk(torch.abs(tensor), keep_num, largest=largest)[1]
                mask_dict[k] = torch.zeros_like(tensor, device=tensor.device)
                mask_dict[k][top_pos] = 1
                mask_dict[k] = mask_dict[k].reshape(v.shape).to(tensor.device)
            else:
                sizes = v.shape
                num_params = v.numel()
                keep_num = int(num_params * keep_ratio)
                tensor = v.view(-1)
                top_pos = torch.topk(torch.abs(tensor), keep_num, largest=largest)[1]
                mask_dict[k] = torch.ones_like(tensor, device='cpu')
                mask_dict[k][top_pos] = 0
                mask_dict[k] = mask_dict[k].reshape(v.shape).to('cpu')

    return mask_dict


def prune(bad_params, good_params, factor, return_good=False):
    prune_params = {}
    if return_good == False:
        for k, v in bad_params.items():
            prune_params[k] = bad_params[k] - good_params[k]
            indices = prune_params[k] != -1
            bad_indices = prune_params[k] == -1
            prune_params[k] = indices + (bad_indices * factor)

    else:
        for k, v in bad_params.items():
            prune_params[k] = good_params[k] - bad_params[k]
            indices = prune_params[k] != -1
            good_indices = prune_params[k] == -1
            prune_params[k] = indices + (good_indices * factor)
    return prune_params


def count_top_parameters(mask_dict):
    """Count how many parameters are marked as top-k (encoded as zeros in the mask)."""
    total = 0
    for tensor in mask_dict.values():
        total += torch.count_nonzero(tensor == 0).item()
    return total

def count_math_only_parameters(math_mask, nonmath_mask):
    """
    Count math top-k parameters that remain after removing those also selected as non-math.
    """
    total = 0
    for key, math_tensor in math_mask.items():
        if key not in nonmath_mask:
            continue
        math_top = math_tensor == 0
        nonmath_top = nonmath_mask[key] == 0
        remaining_math = math_top & (~nonmath_top)
        total += torch.count_nonzero(remaining_math).item()
    return total


def get_results_paths(save_path, model, dataset_name, good_percent, scalar, repeat, train_task, run_codealpaca_eval=False):
    base_dir = f"{save_path}/eval_results/{model}"
    eval_path = f"{base_dir}/{dataset_name}_calculate{good_percent}_scalar{scalar}_run{repeat}.json"
    train_task_path = None
    if train_task is not None:
        train_task_path = (
            f"{base_dir}/{dataset_name}_calculate{good_percent}_scalar{scalar}_run{repeat}_train_task.json"
        )
    codealpaca_paths = []
    if run_codealpaca_eval and train_task is not None:
        codealpaca_paths = [
            f"{base_dir}/candidate_generations_{dataset_name}_calculate{good_percent}_scalar{scalar}_run{repeat}.jsonl",
            f"{base_dir}/codealpaca_oracle_metrics_{dataset_name}_calculate{good_percent}_scalar{scalar}_run{repeat}.json"
        ]
    return eval_path, train_task_path, codealpaca_paths


def experiment_already_done(save_path, model, dataset_name, good_percent, scalar, repeat, train_task, run_codealpaca_eval=False):
    eval_path, train_task_path, codealpaca_paths = get_results_paths(
        save_path, model, dataset_name, good_percent, scalar, repeat, train_task, run_codealpaca_eval
    )
    if not os.path.exists(eval_path):
        return False
    if train_task_path is not None and not os.path.exists(train_task_path):
        return False
    # Check CodeAlpaca files if applicable
    for path in codealpaca_paths:
        if not os.path.exists(path):
            return False
    return True






def getActivation(name):
    # The hook function
    def hook(module, input, output):
        activations = input[0]  # Get the input activations
        weights = module.weight.data  # Get the weights
        device = weights.device
        # Compute the norm of activations along dim=1
        activations_norm = activations.norm(p=2, dim=1).to(torch.bfloat16)
        # Multiply activations by the absolute value of weights
        modified_output = activations_norm.to(device) * torch.abs(weights)
        magnitude[name] = modified_output.detach()  # Store the modified output

    # Return the hook function
    return hook


def parse_none_file(none_file_path, dataset_name, repeat=1):
    """Parse the None file to extract math-only param counts for a given repeat.

    Returns a dict mapping proportion (float) to math-only count (int).
    """
    import re as _re
    counts = {}
    with open(none_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pattern = (
                rf'\[{_re.escape(dataset_name)}\] repeat {repeat}, '
                rf'top ([\d.]+)%.*math-only after removing non-math: (\d+)'
            )
            match = _re.search(pattern, line)
            if match:
                proportion = float(match.group(1)) / 100.0
                math_only_count = int(match.group(2))
                counts[proportion] = math_only_count
    return counts


def parse_math_top_counts(none_file_path, dataset_name, repeat=1):
    """Parse the None file to extract math top-k param counts for a given repeat.

    Returns a dict mapping proportion (float) to math top-k count (int).
    """
    import re as _re
    counts = {}
    with open(none_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pattern = (
                rf'\[{_re.escape(dataset_name)}\] repeat {repeat}, '
                rf'top ([\d.]+)%.* math: (\d+), non-math:'
            )
            match = _re.search(pattern, line)
            if match:
                proportion = float(match.group(1)) / 100.0
                math_count = int(match.group(2))
                counts[proportion] = math_count
    return counts


def create_random_prune_mask(train_scores, math_only_count,
                             good_percent, random_seed=42, pool_size=None):
    """Create a pruning mask by randomly sampling neurons proportional to their score.

    Only neurons in the top *good_percent* fraction (by absolute score) are
    eligible for sampling.  Among those, the probability of being selected is
    proportional to the absolute score.

    Args:
        train_scores: dict  {param_name: score_tensor} from compute_score_mask.
        math_only_count: number of neurons to prune (from the None file).
        good_percent: only sample from neurons in this top fraction (same as
            the proportion used in the real pruning experiment).
        random_seed: random seed for reproducibility.
        pool_size: if given, use this as the number of top neurons to form
            the sampling pool instead of computing it as
            int(total_params * good_percent).

    Returns:
        mask_dict: dict with 1 everywhere except 0 at the randomly pruned positions.
    """
    rng = np.random.RandomState(random_seed)

    # Flatten all non-embedding scores
    flat_scores = []
    param_keys = []
    param_shapes = []
    for k, v in train_scores.items():
        if "embed" in k:
            continue
        param_keys.append(k)
        param_shapes.append(v.shape)
        flat_scores.append(v.view(-1).float().abs())

    all_scores = torch.cat(flat_scores)
    total_params = all_scores.numel()

    # Pool: top good_percent neurons by absolute score
    if pool_size is None:
        pool_size = int(total_params * good_percent)
    top_values, top_positions = torch.topk(all_scores, pool_size, largest=True)

    # pool_scores = top_values.numpy().astype(np.float64)
    # pool_indices = top_positions.numpy()

    # # Normalize to probabilities
    # pool_probs = pool_scores / pool_scores.sum()

    # # Sample without replacement
    # sample_size = min(math_only_count, pool_size)
    # # chosen = rng.choice(pool_size, size=sample_size, replace=False, p=pool_probs)
    # chosen = rng.choice(pool_size, size=sample_size, replace=False)
    # chosen_flat = pool_indices[chosen]

    pool_scores = top_values.numpy().astype(np.float64)
    pool_indices = top_positions.numpy()

    # Take the top sample_size neurons (pool is already sorted descending by score)
    sample_size = min(math_only_count, pool_size)
    chosen = np.arange(sample_size)
    chosen_flat = pool_indices[chosen]

    # Build flat mask: 1 everywhere, 0 at chosen positions
    mask_flat = torch.ones(total_params)
    mask_flat[chosen_flat] = 0

    # Reshape back to per-parameter tensors
    mask_dict = {}
    offset = 0
    for k, v in train_scores.items():
        if "embed" in k:
            mask_dict[k] = torch.ones_like(v)
            continue
        n = v.numel()
        mask_dict[k] = mask_flat[offset:offset + n].reshape(v.shape)
        offset += n

    return mask_dict


num_samples = args.num_samples
num_repeats = args.num_repeats
if args.proportion is None:
    good_percents = [.0001, .001,  .01,  .05, .1, .15]
if args.proportion is not None:
    if len(args.proportion) > 1 or isinstance(args.proportion, list):
        good_percents = args.proportion
    else:
        good_percents = [args.proportion]
scalar = args.scalar



for dataset in dataset_list:
    if args.random_prune:
        break
    for repeat in range(0, num_repeats):
        pending_good_percents = [
            good_percent
            for good_percent in good_percents
            if not experiment_already_done(
                args.save_path,
                args.model,
                dataset.name,
                good_percent,
                scalar,
                repeat,
                args.train_lm_eval_task,
                args.run_codealpaca_eval,
            )
        ]
        if not pending_good_percents:
            print(
                f"Skipping {dataset.name} repeat {repeat + 1}: outputs already exist for all proportions."
            )
            continue

        random_state = args.random_state + repeat*100
        sampled_train = train.sample(n=num_samples, replace=True, random_state=random_state)
        sampled_comparison = dataset.sample(n=num_samples, replace=True, random_state=random_state)

        # load model again (only important for multi run case) and register the hook function
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)
        model.eval()
        for name, module in model.named_modules():
            if (isinstance(module, (nn.Linear))):
                hook_fn = getActivation(name)  # Get the hook function
                module.register_forward_hook(hook_fn)  # Register the hook function

        compute_score_mask(model, sampled_train, num_samples=num_samples, save_dir=mask_dir, save_path = f"train_scores_seed{random_state}.pt")
        compute_score_mask(model, sampled_comparison, num_samples=num_samples, save_dir=mask_dir, save_path=f"comparison_scores_seed{random_state}.pt")

        for good_percent in pending_good_percents:
            if experiment_already_done(
                args.save_path,
                args.model,
                dataset.name,
                good_percent,
                scalar,
                repeat,
                args.train_lm_eval_task,
                args.run_codealpaca_eval,
            ):
                print(
                    f"Skipping {dataset.name} repeat {repeat + 1}, proportion {good_percent}: outputs already exist."
                )
                continue
            # load fresh model for each new run since modified during each run
            model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)
            model.eval()
            torch.cuda.empty_cache()
            magnitude = {}
            # register hook function for new model
            for name, module in model.named_modules():
                if (isinstance(module, (nn.Linear))):
                    hook_fn = getActivation(name)  # Get the hook function
                    module.register_forward_hook(hook_fn)  # Register the hook function


            print("load good param score mask")
            param_dict_good = torch.load(Path(mask_dir) / f"train_scores_seed{random_state}.pt", map_location="cpu")
            print("extract top-k params")
            good_params = find_good_params(keep_ratio=good_percent, prune=True, largest=True, param_dict = param_dict_good)
            torch.cuda.empty_cache()

            print("load comparison param score mask")
            param_dict_comparison = torch.load(Path(mask_dir) / f"comparison_scores_seed{random_state}.pt", map_location="cpu")
            print("extract top-k params")
            comparison_params = find_good_params(keep_ratio=good_percent, prune=True,largest=True, param_dict = param_dict_comparison)

            # count selected parameters and difference set
            math_top_params = count_top_parameters(good_params)  # how many parameters are imprtant for math (good_percent of all parameters)
            nonmath_top_params = count_top_parameters(comparison_params)  # should be same number as math_top_params
            math_only_params = count_math_only_parameters(good_params,comparison_params)  # how many params are math-specific? will be modified in the next steps
            torch.cuda.empty_cache()
            print(math_top_params, nonmath_top_params, math_only_params)
            with open(output_file, "a") as f:
                f.write(
                    f"[{dataset.name}] repeat {repeat + 1}, top {good_percent * 100:.4f}% — math: {math_top_params}, non-math: {nonmath_top_params}, math-only after removing non-math: {math_only_params}\n"
                )

            if args.fine_tune:
                isolated_zero_mask = prune(comparison_params, good_params, factor=0.0, return_good=True)
                # isolated_zero_mask[k] == 0 → isolated task-specific
                # isolated_zero_mask[k] == 1 → everything else
                # free these, we only need isolated_masks now
                del good_params
                del comparison_params
                isolated_masks = {}
                for k, m in isolated_zero_mask.items():
                    # We want mask == 1 *on isolated positions*, 0 elsewhere, for grad masking
                    isolated_masks[k] = (m == 0).to(torch.float32)

                del isolated_zero_mask
                # clear magnitude buffers as they are no longer needed
                magnitude.clear()


                def remove_hooks(model):
                    for name, module in model.named_modules():
                        # Check if the module has any forward hooks
                        if hasattr(module, "_forward_hooks") and len(module._forward_hooks) > 0:
                            # Remove all forward hooks
                            module._forward_hooks.clear()
                remove_hooks(model)
                torch.cuda.empty_cache()

                fine_tune_on_isolated_params(
                    model=model,
                    tokenizer=tokenizer,
                    train_df=sampled_train,
                    # fine-tune on the same 500 samples for that the params have been identified
                    isolated_masks=isolated_masks,
                    seed=random_state
                )
                model.eval()
                del isolated_masks

            # prune or scale based on scalar value
            else:
                print("prune or scale params")
                prune_params = prune(comparison_params, good_params, scalar, return_good=True)
                del good_params
                del comparison_params
                print("applying changes to LLM params")
                param_lookup = dict(model.named_parameters())
                for key, tensor in prune_params.items():
                    if key in param_lookup:
                        p = param_lookup[key]
                        p.data.mul_(tensor.to(p.device))
                del prune_params

            def remove_hooks(model):
                # Function to remove all hooks
                for name, module in model.named_modules():
                    # Check if the module has any forward hooks
                    if hasattr(module, "_forward_hooks") and len(module._forward_hooks) > 0:
                        # Remove all forward hooks
                        module._forward_hooks.clear()

            remove_hooks(model)

            if args.train_lm_eval_task is not None:
                # task_manager = lm_eval.tasks.TaskManager()
                # --log_samples --output_path results/phi_15_base --device cuda:0 --batch_size auto:4
                # Setting `task_manager` to the one above is optional and should generally be done
                # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
                # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
                model.eval()
                results = lm_eval.simple_evaluate(  # call simple_evaluate
                    model='hf',
                    model_args={'pretrained': model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
                    tasks=args.train_lm_eval_task,
                    task_manager=task_manager,
                    log_samples=True,
                    batch_size=args.batch_size,
                    limit=args.eval_dataset_subset,
                    random_seed=random_state
                )

                samples_dir = f"{args.save_path}/eval_results/{args.model}/samples_pre_train_task_{args.scalar}"
                saved = save_lm_eval_samples(results, f"{samples_dir}/post_modification")
                print("Saved:", saved)

                results_path = f"{args.save_path}/eval_results/{args.model}/{dataset.name}_calculate{good_percent}_scalar{scalar}_run{repeat}_train_task.json"
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, "w") as outfile:
                    json.dump(results['results'], outfile)


                if args.run_codealpaca_eval:
                    model.eval()
                    codealpaca_limit = args.eval_dataset_subset
                    oracle_env_set = False
                    if oracle_eval_ids is not None:
                        os.environ["CODEALPACA_ORACLE_CASES_PATH"] = oracle_cases_path
                        os.environ["CODEALPACA_ALLOWED_SAMPLE_IDS"] = ",".join(
                            str(sid) for sid in sorted(oracle_eval_ids)
                        )
                        oracle_env_set = True
                        codealpaca_limit = len(oracle_eval_ids)
                    try:
                        codealpaca_results = lm_eval.simple_evaluate(  # call simple_evaluate
                            model='hf',
                            model_args={'pretrained': model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
                            tasks=args.train_lm_eval_task,
                            task_manager=task_manager,
                            log_samples=True,
                            batch_size=args.batch_size,
                            limit=codealpaca_limit,
                            random_seed=random_state
                        )
                    finally:
                        if oracle_env_set:
                            os.environ.pop("CODEALPACA_ORACLE_CASES_PATH", None)
                            os.environ.pop("CODEALPACA_ALLOWED_SAMPLE_IDS", None)
                    samples = codealpaca_results["samples"]["codealpaca"]
                    candidate_path = (
                        f"{args.save_path}/eval_results/{args.model}/"
                        f"candidate_generations_{dataset.name}_calculate{good_percent}_scalar{scalar}_run{repeat}.jsonl"
                    )
                    metrics_path = (
                        f"{args.save_path}/eval_results/{args.model}/"
                        f"codealpaca_oracle_metrics_{dataset.name}_calculate{good_percent}_scalar{scalar}_run{repeat}.json"
                    )
                    codealpaca_oracle.write_codealpaca_candidates(
                        samples,
                        candidate_path,
                        allowed_sample_ids=set(oracle_eval_ids) if oracle_eval_ids is not None else None,
                    )
                    codealpaca_oracle.run_codealpaca_oracle_eval(
                        candidate_path,
                        metrics_path,
                        allowed_sample_ids=set(oracle_eval_ids) if oracle_eval_ids is not None else None,
                        max_cases_per_id=2,
                    )

                model.eval()
                results = lm_eval.simple_evaluate(  # call simple_evaluate
                    model='hf',
                    model_args={'pretrained': model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
                    tasks=args.eval_datasets,
                    task_manager=task_manager,
                    log_samples=False,
                    batch_size=args.batch_size,
                )
                results_path = f"{args.save_path}/eval_results/{args.model}/{dataset.name}_calculate{good_percent}_scalar{scalar}_run{repeat}.json"
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, "w") as outfile:
                    json.dump(results['results'], outfile)
            del model


# ──────────────────────────────────────────────────────────────────────────────
# Random Prune Experiment: group effect vs individual effect
# ──────────────────────────────────────────────────────────────────────────────
if args.random_prune:
    # Seed whose masks we reuse (repeat 0 → "repeat 1" in the None file)
    mask_seed = args.random_state  # default 1

    # Load existing train scores (only need train scores for sampling weights)
    train_scores_path = Path(mask_dir) / f"train_scores_seed{mask_seed}.pt"
    print(f"Loading train scores from {train_scores_path}")
    train_scores = torch.load(train_scores_path, map_location="cpu")

    for dataset in dataset_list:
        # Parse math-only counts from None file for this dataset, repeat 1
        none_counts = parse_none_file(output_file, dataset_name=dataset.name, repeat=1)
        if not none_counts:
            print(
                f"Warning: No entries found in {output_file} for dataset "
                f"'{dataset.name}' repeat 1. Skipping."
            )
            continue
        print(f"Parsed math-only counts for {dataset.name}: {none_counts}")

        # Parse math top-k counts (pool sizes) from None file for repeat 1
        math_top_counts = parse_math_top_counts(output_file, dataset_name=dataset.name, repeat=1)
        print(f"Parsed math top-k counts for {dataset.name}: {math_top_counts}")

        for good_percent in good_percents:
            # Convert to the percentage key used in the None file
            math_only_count = none_counts.get(good_percent)
            if math_only_count is None:
                print(
                    f"Warning: No math-only count for proportion "
                    f"{good_percent} ({good_percent*100:.4f}%). Skipping."
                )
                continue

            for rseed in args.random_prune_seeds:
              for prune_frac in args.random_prune_fractions:
                prune_count = max(1, int(math_only_count * prune_frac))
                result_tag = f"random_rseed{rseed}_frac{prune_frac:.2f}"
                base_dir = f"{args.save_path}_random/eval_results/{args.model}"
                eval_path = (
                    f"{base_dir}/{dataset.name}_calculate{good_percent}"
                    f"_scalar{scalar}_{result_tag}.json"
                )
                train_task_path = None
                if args.train_lm_eval_task is not None:
                    train_task_path = (
                        f"{base_dir}/{dataset.name}_calculate{good_percent}"
                        f"_scalar{scalar}_{result_tag}_train_task.json"
                    )

                # Check if already done
                already_done = os.path.exists(eval_path)
                if train_task_path and not os.path.exists(train_task_path):
                    already_done = False
                if already_done:
                    print(
                        f"Skipping random prune: dataset={dataset.name}, "
                        f"proportion={good_percent}, seed={rseed}, "
                        f"frac={prune_frac:.0%} — outputs exist."
                    )
                    continue

                print(
                    f"\n=== Random Prune: dataset={dataset.name}, "
                    f"proportion={good_percent}, seed={rseed}, "
                    f"frac={prune_frac:.0%} ==="
                )
                print(
                    f"    Pruning {prune_count} / {math_only_count} "
                    f"({prune_frac:.0%}) randomly sampled neurons"
                )

                # Create random prune mask weighted by score
                math_pool_size = math_top_counts.get(good_percent)
                random_mask = create_random_prune_mask(
                    train_scores, prune_count,
                    good_percent=good_percent, random_seed=rseed,
                    pool_size=math_pool_size,
                )

                # Load fresh model
                model = AutoModelForCausalLM.from_pretrained(
                    args.model, device_map="auto", torch_dtype=torch.bfloat16
                )
                model.eval()
                torch.cuda.empty_cache()

                # Apply the mask (prune)
                print("    Applying random prune mask to model")
                param_lookup = dict(model.named_parameters())
                for key, mask_tensor in random_mask.items():
                    if key in param_lookup:
                        p = param_lookup[key]
                        p.data.mul_(mask_tensor.to(p.device))
                del random_mask

                # Evaluate on train task
                if args.train_lm_eval_task is not None:
                    model.eval()
                    codealpaca_limit = args.eval_dataset_subset
                    oracle_env_set = False
                    if args.run_codealpaca_eval and oracle_eval_ids is not None:
                        os.environ["CODEALPACA_ORACLE_CASES_PATH"] = oracle_cases_path
                        os.environ["CODEALPACA_ALLOWED_SAMPLE_IDS"] = ",".join(
                            str(sid) for sid in sorted(oracle_eval_ids)
                        )
                        oracle_env_set = True
                        codealpaca_limit = len(oracle_eval_ids)

                    try:
                        results = lm_eval.simple_evaluate(
                            model='hf',
                            model_args={
                                'pretrained': model,
                                'dtype': 'bfloat16',
                                'tokenizer': tokenizer,
                            },
                            tasks=args.train_lm_eval_task,
                            task_manager=task_manager,
                            log_samples=True,
                            batch_size=args.batch_size,
                            limit=args.eval_dataset_subset,
                            random_seed=mask_seed,
                        )
                    finally:
                        if oracle_env_set:
                            os.environ.pop("CODEALPACA_ORACLE_CASES_PATH", None)
                            os.environ.pop("CODEALPACA_ALLOWED_SAMPLE_IDS", None)

                    os.makedirs(os.path.dirname(train_task_path), exist_ok=True)
                    with open(train_task_path, "w") as outfile:
                        json.dump(results['results'], outfile)

                    if args.run_codealpaca_eval:
                        samples = results["samples"]["codealpaca"]
                        cand_path = (
                            f"{base_dir}/candidate_generations_{dataset.name}"
                            f"_calculate{good_percent}_scalar{scalar}_{result_tag}.jsonl"
                        )
                        metr_path = (
                            f"{base_dir}/codealpaca_oracle_metrics_{dataset.name}"
                            f"_calculate{good_percent}_scalar{scalar}_{result_tag}.json"
                        )
                        codealpaca_oracle.write_codealpaca_candidates(
                            samples, cand_path,
                            allowed_sample_ids=(
                                set(oracle_eval_ids) if oracle_eval_ids is not None else None
                            ),
                        )
                        codealpaca_oracle.run_codealpaca_oracle_eval(
                            cand_path, metr_path,
                            allowed_sample_ids=(
                                set(oracle_eval_ids) if oracle_eval_ids is not None else None
                            ),
                            max_cases_per_id=2,
                        )

                # Evaluate on general eval datasets
                model.eval()
                results = lm_eval.simple_evaluate(
                    model='hf',
                    model_args={
                        'pretrained': model,
                        'dtype': 'bfloat16',
                        'tokenizer': tokenizer,
                    },
                    tasks=args.eval_datasets,
                    task_manager=task_manager,
                    log_samples=False,
                    batch_size=args.batch_size,
                )
                os.makedirs(os.path.dirname(eval_path), exist_ok=True)
                with open(eval_path, "w") as outfile:
                    json.dump(results['results'], outfile)

                del model
                torch.cuda.empty_cache()
                print(f"    Results saved to {eval_path}")

    del train_scores
    print("\nRandom prune experiment complete.")