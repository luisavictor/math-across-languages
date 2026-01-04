import os
import argparse
import random
import pandas as pd
import numpy as np
import torch

def parse_args():
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
                        type=int, default=42)
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
    parser.add_argument('--batch_size', help="batch size for evaluation", type=str, default="1")
    parser.add_argument('--max_batch_size', help="maximum batch size for auto batch sizing", type=int, default=18)
    parser.add_argument('--proportion', help="desired proportion of top parameters to calculate", type=float, default=None)
    parser.add_argument('--fine_tune', help="freeze all non-task-specific parameters and fine-tune only isolated task-specific weights",action="store_true")
    parser.add_argument('--store_params', help="store task-specific isolated parameters",action="store_true")
    args = parser.parse_args()

    if args.proportion is None:
        args.proportion = [.0001, .001, .005, .01, .025, .05, .1, .15]
    elif not isinstance(args.proportion, list):
        args.proportion = [args.proportion]

    if args.batch_size.isdigit():
        args.batch_size = int(args.batch_size)
    # elif args.batch_size != "auto":
    #     raise ValueError("batch_size must be an integer or 'auto'")

    return args

def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def load_train_val_datasets(train_dataset, random_state):
    if 'sgsm' in train_dataset:
        df = pd.read_csv(train_dataset)  # Load SGSM dataset for few-shot prompting
        df = df[df['subset'] == "sgsm_train"]  # Subset SGSM to verified training subset
        df = df.sample(frac=1, random_state=random_state)
        for i in range(0, len(df)):
            try:
                answer = df.iloc[i]['answer']
                answer = float(answer)
                df.iloc[i]['answer'] = answer
            except:
                df = df.drop([i])

        train = df.iloc[0:1500]

        val = df.iloc[1500:]
        val = val.sample(frac=1, random_state=random_state)

    if 'sgsm' not in train_dataset:
        train = pd.read_csv(train_dataset)  # Load SGSM dataset for few-shot prompting
        train = train.sample(frac=1, random_state=random_state)
        val = None

    return train, val

def load_calibration_datasets(calibration_datasets, calibration_dataset_names, random_state):
    datasets = []
    for dataset in calibration_datasets:
        if '/' in dataset:
            dataset_name = dataset.split('/')[-1]
            dataset_name = dataset_name.split('.csv')[0]
            datasets.append(dataset_name)
        else:
            dataset_name = dataset.split('.csv')[0]
            datasets.append(dataset_name)

    dataset_list = []
    for dataset, dataset_name, name in zip(calibration_datasets, datasets, calibration_dataset_names):
        # Load the dataset into a DataFrame
        globals()[dataset_name] = pd.read_csv(dataset).sample(frac=1,
                                                            random_state=random_state)  # Shuffle the DataFrame

        # Assign a name attribute to the DataFrame
        globals()[dataset_name].name = name

        # Append the actual DataFrame object to the list
        dataset_list.append(globals()[dataset_name])

    return dataset_list

def create_output_dirs_and_files(save_path, model, text_file):
    mask_dir = f"{save_path}/isolated_masks/{model}"
    os.makedirs(mask_dir, exist_ok=True)

    output_file = f"{save_path}/eval_results/{model}/{text_file}"
    results_path = f"{save_path}/eval_results/{model}/"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    return mask_dir, output_file, results_path


    
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

def scale(good_params, factor):
    prune_params = {}
    for k, v in good_params.items():
        good_indices = good_params[k] != 1
        keep_indices = good_params[k] == 1
        prune_params[k] = keep_indices + (good_indices * factor)
    return prune_params

def store_identified_params(
    comparison_params,
    good_params,
    dataset,
    good_percent,
    repeat,
    random_state,
    mask_dir,
    model_name
):
    isolated_zero_mask = prune(comparison_params, good_params, factor=0.0, return_good=True)
    isolated_masks = {}
    for k, m in isolated_zero_mask.items():
        isolated_masks[k] = (m == 0).to(torch.bool)
    del isolated_zero_mask
    
    mask_filename = f"gsm8k_{dataset.name}_{good_percent}_repeat{repeat}.pt"
    mask_path = os.path.join(mask_dir, mask_filename)
    cpu_masks = {k: v.to("cpu") for k, v in isolated_masks.items()}
    tmp_path = mask_path + ".tmp"

    torch.save(
        {
            "model_name": model_name,
            "dataset_name": dataset.name,
            "good_percent": good_percent,
            "repeat": repeat,
            "random_state": random_state,
            "isolated_masks": cpu_masks,
        },
        tmp_path,
    )

    os.replace(tmp_path, mask_path)  # atomic rename

    del isolated_masks
    del cpu_masks
    
    print(f"Saved isolated mask to {mask_path}")