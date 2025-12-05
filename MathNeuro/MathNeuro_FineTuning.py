import os
import argparse

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
parser.add_argument('--proportion', help="desired proportion of top parameters to calculate", type=float, default=None)
parser.add_argument('--fine_tune',
                    help="freeze all non-task-specific parameters and fine-tune only isolated task-specific weights",
                    action="store_true")



args = parser.parse_args()
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
import lm_eval
import json


if 'sgsm' not in args.train_dataset:
    train = pd.read_csv(args.train_dataset)  # Load SGSM dataset for few-shot prompting
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
    globals()[dataset_name] = pd.read_csv(dataset).sample(frac=1,random_state=args.random_state)  # Shuffle the DataFrame
    # Assign a name attribute to the DataFrame
    globals()[dataset_name].name = name
    # Append the actual DataFrame object to the list
    dataset_list.append(globals()[dataset_name])

output_file = f"{args.save_path}/eval_results/{args.model}/{args.text_file}"
results_path = f"{args.save_path}/eval_results/{args.model}/"
os.makedirs(os.path.dirname(results_path), exist_ok=True)

'''
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    quantization_config=quant_config,
    device_map="auto",
)
'''
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)

#from lm_eval.models.huggingface import HFLM
#from monkey_patch_logging import patch_hf_generate_until

#patch_hf_generate_until(HFLM)

if args.pre_train_eval:
    if args.train_lm_eval_task is not None:
        task_manager = lm_eval.tasks.TaskManager(include_path="../lm_eval_tasks")
        # task_manager = lm_eval.tasks.TaskManager()
        # --log_samples --output_path results/phi_15_base --device cuda:0 --batch_size auto:4
        # Setting `task_manager` to the one above is optional and should generally be done
        # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
        # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
        results = lm_eval.simple_evaluate(  # call simple_evaluate
            model='hf',
            model_args={'pretrained': model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
            tasks=args.train_lm_eval_task,
            task_manager=task_manager,
            batch_size='auto:4',
            log_samples=True,
            limit=args.eval_dataset_subset,
            random_seed=args.random_state
        )
        results_path = f"{args.save_path}/eval_results/{args.model}/pre_results_train_task.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as outfile:
            json.dump(results['results'], outfile)

        results = lm_eval.simple_evaluate(  # call simple_evaluate
            model='hf',
            model_args={'pretrained': model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
            tasks=args.eval_datasets,
            task_manager=task_manager,
            log_samples=False,
            batch_size='auto:4'
        )
        results_path = f"{args.save_path}/eval_results/{args.model}/pre_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as outfile:
            json.dump(results['results'], outfile)

magnitude = {}

def getActivation(name):
    # The hook function
    def hook(module, input, output):
        activations = input[0]  # Get the input activations
        weights = module.weight.data  # Get the weights
        # Compute the norm of activations along dim=1
        activations_norm = activations.norm(p=2, dim=1).to(torch.bfloat16)
        # Multiply activations by the absolute value of weights
        modified_output = activations_norm * torch.abs(weights)
        magnitude[name] = modified_output.detach()  # Store the modified output

    # Return the hook function
    return hook

for name, module in model.named_modules():
    if (isinstance(module, (nn.Linear))):
        hook_fn = getActivation(name)  # Get the hook function
        module.register_forward_hook(hook_fn)  # Register the hook function


def find_good_params(model, train, keep_ratio, prune=True, largest=True, num_samples=len(train)):
    global chosen_params
    import random

    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    param_dict = {}
    for name, param in model.named_parameters():
        param_dict[name] = torch.zeros_like(param).to(param.device)

    for i in range(0, num_samples):
        if 'qa' in train.columns.to_list():
            prompt = train.iloc[i]['qa']
        else:
            question = train['question'].iloc[i]
            answer = train['solution'].iloc[i]
            prompt = f"""Instruct: {question} Let's write a Python program.\nOutput:\n{answer}"""
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        outputs = model(inputs)
        for key, tensor in magnitude.items():
            try:
                param_dict[f"{key}.weight"] += tensor
            except:
                print(f'passed at {key}')
    keys_to_remove = [key for key in param_dict if key.split('.weight')[0] not in magnitude]

    for key in keys_to_remove:
        del param_dict[key]

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
                mask_dict[k] = torch.ones_like(tensor, device=tensor.device)
                mask_dict[k][top_pos] = 0
                mask_dict[k] = mask_dict[k].reshape(v.shape).to(tensor.device)

    return mask_dict

def prune(bad_params, good_params, factor, return_good = False):
    prune_params = {}
    if return_good ==False:
        for k, v in bad_params.items():
            prune_params[k] = bad_params[k] - good_params[k]
            indices = prune_params[k]!=-1
            bad_indices = prune_params[k]==-1
            prune_params[k] = indices + (bad_indices*factor)

    else:
        for k, v in bad_params.items():
            prune_params[k] = good_params[k] - bad_params[k]
            indices = prune_params[k]!=-1
            good_indices = prune_params[k]==-1
            prune_params[k] = indices + (good_indices*factor)
    return prune_params

'''
def scale(good_params, factor):
    prune_params = {}
    for k, v in good_params.items():
        good_indices = good_params[k] != 1
        keep_indices = good_params[k] == 1
        prune_params[k] = keep_indices + (good_indices * factor)
    return prune_params
'''


def fine_tune_on_isolated_params(model, tokenizer, train_df, isolated_masks, num_steps, lr=1e-4):
    """
    Fine-tune the model on `train_df`, but only update weights corresponding
    to isolated task-specific parameters (isolated_masks == 1).

    isolated_masks: dict[name] -> tensor with 1 where we allow updates, 0 elsewhere
    """

    # 1) Determine which parameters are actually trainable
    trainable_params = []
    for name, param in model.named_parameters():
        if name in isolated_masks and isolated_masks[name].sum() > 0:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    if not trainable_params:
        print("No isolated task-specific parameters found. Skipping fine-tuning.")
        return

    # 2) Use a lightweight optimizer (no huge state like Adam)
    optimizer = torch.optim.SGD(trainable_params, lr=lr)

    model.train()
    step = 0
    while step < num_steps:
        idx = step % len(train_df)

        # ---- build prompt ----
        if 'qa' in train_df.columns:
            prompt = train_df.iloc[idx]['qa']
        else:
            q = train_df['question'].iloc[idx]
            ans = train_df['solution'].iloc[idx]
            prompt = f"Instruct: {q} Let's write a Python program.\nOutput:\n{ans}"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        labels = inputs["input_ids"]

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()

        # 3) Mask gradients so only isolated positions inside those tensors get updated
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                if name in isolated_masks:
                    mask = isolated_masks[name].to(param.grad.device)
                    param.grad *= mask.to(param.grad.dtype)
                else:
                    # Shouldn't happen because requires_grad=False, but just in case
                    param.grad.zero_()

        optimizer.step()

        # 4) DEBUG PRINTS
        if step < 3 or step % 50 == 0:
            print("\n================ DEBUG FINE-TUNE STEP =================")
            print(f"Step: {step+1}/{num_steps}")
            print(f"Train index: {idx}")
            print(f"Loss: {loss.item():.4f}")
            print("\nPrompt fed to model:\n")
            print(prompt)

            # Try to generate a short continuation for debugging
            try:
                with torch.no_grad():
                    gen_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        max_new_tokens=64,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                        if tokenizer.eos_token_id is not None
                        else tokenizer.pad_token_id,
                    )
                decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                print("\nModel output (prompt + continuation):\n")
                print(decoded)
            except Exception as e:
                print(f"\n[DEBUG] Generation failed: {e}")

            print("======================================================\n")

        step += 1

    print(f"Finished fine-tuning for {num_steps} steps.")


num_samples = args.num_samples
num_repeats = 1
if args.proportion is None:
    good_percents = [.0001, .001, .005, .01, .025, .05, .1, .15]
if args.proportion is not None:
    good_percents = [args.proportion]
scalar = args.scalar
for dataset in dataset_list:
    for repeat in range(0, num_repeats):
        sampled_train = train.sample(n=num_samples, replace=True)
        sampled_comparison = dataset.sample(n=num_samples, replace=True)
        for good_percent in good_percents:
            model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)
            torch.cuda.empty_cache()
            magnitude = {}
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


            for name, module in model.named_modules():
                if (isinstance(module, (nn.Linear))):
                    hook_fn = getActivation(name)  # Get the hook function
                    module.register_forward_hook(hook_fn)  # Register the hook function
            good_params = find_good_params(model, sampled_train, keep_ratio=good_percent, prune=True, largest=True,
                                           num_samples=num_samples)
            torch.cuda.empty_cache()
            if 'Bad' in dataset.name:
                comparison_params = find_params(model, sampled_comparison, keep_ratio=good_percent, prune=True,
                                                largest=True, num_samples=num_samples)
            else:
                comparison_params = find_good_params(model, sampled_comparison, keep_ratio=good_percent, prune=True,
                                                     largest=True, num_samples=num_samples)

            # good_params and comparison_params computed above

            if args.fine_tune:
                # -------------------------------------------
                # Fine-tuning mode: no pruning or scaling.
                # Use prune(..., factor=0) to identify isolated task-specific params.
                # -------------------------------------------
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
                torch.cuda.empty_cache()

                # e.g., one pass over num_samples examples
                fine_tune_on_isolated_params(
                    model=model,
                    tokenizer=tokenizer,
                    train_df=sampled_train,
                    isolated_masks=isolated_masks,
                    num_steps=num_samples,  # or something else you like
                    lr=1e-5
                )
                # after fine-tuning, you continue with evaluation etc.
                del isolated_masks

            else:
                prune_params = prune(comparison_params, good_params, scalar, return_good=True)
                del good_params
                del comparison_params

                for key, tensor in prune_params.items():
                    device = model.state_dict()[key].device
                    tensor = tensor.to(device)
                    model.state_dict()[key] *= tensor

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
                task_manager = lm_eval.tasks.TaskManager(include_path="../lm_eval_tasks")
                # task_manager = lm_eval.tasks.TaskManager()
                # --log_samples --output_path results/phi_15_base --device cuda:0 --batch_size auto:4
                # Setting `task_manager` to the one above is optional and should generally be done
                # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
                # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
                results = lm_eval.simple_evaluate(  # call simple_evaluate
                    model='hf',
                    model_args={'pretrained': model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
                    tasks=args.train_lm_eval_task,
                    task_manager=task_manager,
                    log_samples=False,
                    batch_size='auto:16',
                    limit=args.eval_dataset_subset,
                    random_seed=args.random_state
                )
                results_path = f"{args.save_path}/eval_results/{args.model}/{dataset.name}_calculate{good_percent}_scalar{scalar}_run{repeat}_train_task.json"
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, "w") as outfile:
                    json.dump(results['results'], outfile)

                results = lm_eval.simple_evaluate(  # call simple_evaluate
                    model='hf',
                    model_args={'pretrained': model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
                    tasks=args.eval_datasets,
                    task_manager=task_manager,
                    log_samples=False,
                    batch_size=1
                )
                results_path = f"{args.save_path}/eval_results/{args.model}/{dataset.name}_calculate{good_percent}_scalar{scalar}_run{repeat}.json"
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, "w") as outfile:
                    json.dump(results['results'], outfile)
            del model
            torch.cuda.empty_cache()