import os
import argparse
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
import lm_eval
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model', help="Huggingface model to train, entered as string", type = str)
parser.add_argument('--eval_datasets', nargs='+', help="dataset(s) to evaluate models on post pruning to evalaute catastrophic forgetting, entered as strings; should be task names from Eleuther AI LM Evaluation Harness", type = str)
parser.add_argument('--train_dataset', help="path to math train dataset; should be a path to a CSV file with question/solution pairs in a columns titled 'question' and 'solution' along with ground-truth answers in a column called 'answer'", type = str)
parser.add_argument('--calibration_datasets', nargs='+', help="path to calibration datasets; should be paths to CSV files with instruction/response pairs in a column titled 'qa'", type = str)
parser.add_argument('--save_path', help="save path for eval results after running Eleuther AI LM Evaluation Harness post pruning", type = str)
parser.add_argument('--text_file', help="name of text file for saving pruning results during training if evaluating math reasoning using a non-Eleuther AI LM Evaluation Harness task in a PoT format", type = str)
parser.add_argument('--num_repeats', help="number of repeats for pruning or scaling experiment", type = int, default = 5)
parser.add_argument('--pre_train_eval', help="bool to indicate if full evaluation on eval and train datasets should be conducted before training", action="store_true")
parser.add_argument('--random_state', help="random state for initial dataset shuffling and creating train/eval split for train dataset", type = int, default = 42)  #42
parser.add_argument('--scalar', help="scale factor for top parameters; default is 0 to run pruning experiments", type = float, default = 0)
parser.add_argument('--eval_dataset_size', help="desired number of samples for task specific eval dataset", type = int, default = None)
parser.add_argument('--eval_dataset_subset', help="desired number of samples for task specific eval dataset if subsetting to reduce run time", type = int, default = 100)
parser.add_argument('--calibration_dataset_names', nargs='+', help="desired name of calibration datasets; should be strings entered in same order as calibration_datasets", type = str)
parser.add_argument('--num_samples', help="desired number of samples for calculating task specific parameters", type = int, default = 500)
parser.add_argument('--train_lm_eval_task', help="if your training dataset is an Eleuther AI LM Evaluation Harness task, specify the associated task for the test set.", type = str, default = None)
parser.add_argument('--proportion', help="desired proportion of top parameters to calculate", type = float, default = None)
parser.add_argument('--fine_tune', help="freeze all non-task-specific parameters and fine-tune only isolated task-specific weights",action="store_true")
parser.add_argument('--store_params', help="store task-specific isolated parameters",action="store_true")
args = parser.parse_args()

# Build a nice filename that encodes run configuration
mask_dir = f"{args.save_path}/isolated_masks/{args.model}"
os.makedirs(mask_dir, exist_ok=True)


'''
if 'sgsm' in args.train_dataset:
    print("sgsm there")
    df = pd.read_csv(args.train_dataset) # Load SGSM dataset for few-shot prompting
    df = df[df['subset']=="sgsm_train"] # Subset SGSM to verified training subset
    df = df.sample(frac = 1, random_state = args.random_state)
    for i in range(0, len(df)):
        try:
            answer = df.iloc[i]['answer']
            answer = float(answer)
            df.iloc[i]['answer'] = answer
        except:
            df = df.drop([i])
    
    train = df.iloc[0:1500]
    
    val = df.iloc[1500:]
    val = val.sample(frac = 1, random_state = args.random_state)

if 'sgsm' not in args.train_dataset:
    train = pd.read_csv(args.train_dataset) # Load SGSM dataset for few-shot prompting
    train = train.sample(frac = 1, random_state = args.random_state)


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
    globals()[dataset_name] = pd.read_csv(dataset).sample(frac=1, random_state=args.random_state)  # Shuffle the DataFrame
    
    # Assign a name attribute to the DataFrame
    globals()[dataset_name].name = name
    
    # Append the actual DataFrame object to the list
    dataset_list.append(globals()[dataset_name])
    
output_file = f"{args.save_path}/eval_results/{args.model}/{args.text_file}"
results_path =  f"{args.save_path}/eval_results/{args.model}/"
os.makedirs(os.path.dirname(results_path), exist_ok=True)



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

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)



from lm_eval.models.huggingface import HFLM
from monkey_patch_logging import patch_hf_generate_until

patch_hf_generate_until(HFLM)



if args.pre_train_eval:
    if 'sgsm' in args.train_dataset:
        prune_solve = []
        prune_code = []
        prune_solutions = []
        for i in range(0, min(args.eval_dataset_subset, len(val))):
            # Format the prompt
            prompts = []
            questions = []
            final_question = val.iloc[i]['question']
            final_answer = val.iloc[i]['answer']
            final_prompt = f"""Instruct: {final_question} Let's write a Python program.\nOutput:"""
    
            for j in range(0, 8):
                question = train['question'].iloc[j]
                questions.append(question)
                answer = train['solution'].iloc[j]
                prompt = f"""Instruct: {question} Let's write a Python program.\nOutput:\n{answer}"""
                if prompt not in prompts:
                    prompts.append(prompt)
    
            prompts.append(final_prompt)
            formatted_prompt = "\n\n".join(prompts)
            #Query the model 
            inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
            model_answer = None
            output = model.generate(inputs, max_new_tokens = 150)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # Split the generated text by the prompt to extract the newly generated part
            generated_text_parts = generated_text.split(final_prompt)
            solution_text = generated_text_parts[-1].strip()
            prune_solutions.append(solution_text)
            if "Instruct:" in solution_text:
                solution_text = solution_text.split("Instruct:")[0] # Split up a generation that contains more than one question
            if "print" in solution_text:
                solution_text = solution_text.split("print")[0] # Split up a generation that contains a print statement
            if "Student:" in solution_text:
                solution_text = solution_text.split("Student:")[0] # Split up a generation that contains more than one question
            if "Output:" in solution_text:
                solution_text = solution_text.split("Output:")[0] # Split up a generation that contains more than one question
            if "#TODO" in solution_text:
                solution_text = solution_text.split("#TODO")[0] # Split up a generation that contains more than one question
            #solutions.append(solution_text)
            if 'return result' in solution_text:
                # Split the string on 'return result' but keep 'return result' in the result
                parts = re.split(r'(return result)', solution_text)
    
                # Rejoin the parts correctly
                solution_text = parts[0] + parts[1]
            try:
                exec(solution_text)
                model_answer = solution()
                prune_code.append(1)
                model_answer = float(model_answer)
                if model_answer != final_answer:
                    prune_solve.append(0)
    
                if model_answer == final_answer:
                    prune_solve.append(1)
    
            except:
                prune_code.append(0)
                prune_solve.append(0)
    
        with open(output_file, "a") as f:  # Open the file in append mode ("a")
                f.write(f"Average eval accuracy on {min(args.eval_dataset_subset, len(val))} questions before training with greedy decoding (few-shot): {np.mean(prune_solve)}\n") 
        task_manager = lm_eval.tasks.TaskManager()
        #--log_samples --output_path results/phi_15_base --device cuda:0 --batch_size auto:4
        # Setting `task_manager` to the one above is optional and should generally be done
        # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
        # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
        results = lm_eval.simple_evaluate( # call simple_evaluate
            model = 'hf',
            model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
            tasks=args.eval_datasets,
            task_manager=task_manager,
            batch_size = 2,
            limit = args.eval_dataset_subset
        )
        results_path = f"{args.save_path}/eval_results/{args.model}/pre_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as outfile: 
            json.dump(results['results'], outfile)

    if args.train_lm_eval_task is not None:
        task_manager = lm_eval.tasks.TaskManager(include_path="../lm_eval_tasks")
        #task_manager = lm_eval.tasks.TaskManager()
        #--log_samples --output_path results/phi_15_base --device cuda:0 --batch_size auto:4
        # Setting `task_manager` to the one above is optional and should generally be done
        # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
        # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
        results = lm_eval.simple_evaluate( # call simple_evaluate
            model = 'hf',
            model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
            tasks=args.train_lm_eval_task,
            task_manager=task_manager,
            batch_size = 'auto:16',
            log_samples=True,
            limit = args.eval_dataset_subset, 
            random_seed = args.random_state
        )
        results_path = f"{args.save_path}/eval_results/{args.model}/pre_results_train_task.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as outfile: 
            json.dump(results['results'], outfile)

        results = lm_eval.simple_evaluate( # call simple_evaluate
            model = 'hf',
            model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
            tasks=args.eval_datasets,
            task_manager=task_manager,
            log_samples = False, 
            batch_size = 1,
            limit = args.eval_dataset_subset
        )
        results_path = f"{args.save_path}/eval_results/{args.model}/pre_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as outfile: 
            json.dump(results['results'], outfile)
'''

if 'sgsm' in args.train_dataset:
    df = pd.read_csv(args.train_dataset)  # Load SGSM dataset for few-shot prompting
    df = df[df['subset'] == "sgsm_train"]  # Subset SGSM to verified training subset
    df = df.sample(frac=1, random_state=args.random_state)
    for i in range(0, len(df)):
        try:
            answer = df.iloc[i]['answer']
            answer = float(answer)
            df.iloc[i]['answer'] = answer
        except:
            df = df.drop([i])

    train = df.iloc[0:1500]

    val = df.iloc[1500:]
    val = val.sample(frac=1, random_state=args.random_state)

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
    globals()[dataset_name] = pd.read_csv(dataset).sample(frac=1,
                                                          random_state=args.random_state)  # Shuffle the DataFrame

    # Assign a name attribute to the DataFrame
    globals()[dataset_name].name = name

    # Append the actual DataFrame object to the list
    dataset_list.append(globals()[dataset_name])

output_file = f"{args.save_path}/eval_results/{args.model}/{args.text_file}"
results_path = f"{args.save_path}/eval_results/{args.model}/"
os.makedirs(os.path.dirname(results_path), exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)
if args.pre_train_eval:
    if 'sgsm' in args.train_dataset:
        prune_solve = []
        prune_code = []
        prune_solutions = []
        for i in range(0, min(args.eval_dataset_subset, len(val))):
            # Format the prompt
            prompts = []
            questions = []
            final_question = val.iloc[i]['question']
            final_answer = val.iloc[i]['answer']
            final_prompt = f"""Instruct: {final_question} Let's write a Python program.\nOutput:"""

            for j in range(0, 8):
                question = train['question'].iloc[j]
                questions.append(question)
                answer = train['solution'].iloc[j]
                prompt = f"""Instruct: {question} Let's write a Python program.\nOutput:\n{answer}"""
                if prompt not in prompts:
                    prompts.append(prompt)

            prompts.append(final_prompt)
            formatted_prompt = "\n\n".join(prompts)
            # Query the model
            inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
            model_answer = None
            output = model.generate(inputs, max_new_tokens=150)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # Split the generated text by the prompt to extract the newly generated part
            generated_text_parts = generated_text.split(final_prompt)
            solution_text = generated_text_parts[-1].strip()
            prune_solutions.append(solution_text)
            if "Instruct:" in solution_text:
                solution_text = solution_text.split("Instruct:")[
                    0]  # Split up a generation that contains more than one question
            if "print" in solution_text:
                solution_text = solution_text.split("print")[0]  # Split up a generation that contains a print statement
            if "Student:" in solution_text:
                solution_text = solution_text.split("Student:")[
                    0]  # Split up a generation that contains more than one question
            if "Output:" in solution_text:
                solution_text = solution_text.split("Output:")[
                    0]  # Split up a generation that contains more than one question
            if "#TODO" in solution_text:
                solution_text = solution_text.split("#TODO")[
                    0]  # Split up a generation that contains more than one question
            # solutions.append(solution_text)
            if 'return result' in solution_text:
                # Split the string on 'return result' but keep 'return result' in the result
                parts = re.split(r'(return result)', solution_text)

                # Rejoin the parts correctly
                solution_text = parts[0] + parts[1]
            try:
                exec(solution_text)
                model_answer = solution()
                prune_code.append(1)
                model_answer = float(model_answer)
                if model_answer != final_answer:
                    prune_solve.append(0)

                if model_answer == final_answer:
                    prune_solve.append(1)

            except:
                prune_code.append(0)
                prune_solve.append(0)

        with open(output_file, "a") as f:  # Open the file in append mode ("a")
            f.write(
                f"Average eval accuracy on {min(args.eval_dataset_subset, len(val))} questions before training with greedy decoding (few-shot): {np.mean(prune_solve)}\n")
        task_manager = lm_eval.tasks.TaskManager()
        # --log_samples --output_path results/phi_15_base --device cuda:0 --batch_size auto:4
        # Setting `task_manager` to the one above is optional and should generally be done
        # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
        # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
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

    if args.train_lm_eval_task is not None:
        task_manager = lm_eval.tasks.TaskManager()
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
            batch_size=1,
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
        magnitude[name] = modified_output.detach() # Store the modified output
    # Return the hook function
    return hook

for name, module in model.named_modules():
    if (isinstance(module, (nn.Linear))):
        hook_fn = getActivation(name)  # Get the hook function
        module.register_forward_hook(hook_fn)  # Register the hook function

if 'bad_gens_full.csv' in args.calibration_datasets:
    def find_params(model, gens, keep_ratio, prune = True, largest = True, num_samples = len(bad_gens_full)):
        global chosen_params
        cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

        param_dict = {}
        for name, param in model.named_parameters():
            param_dict[name] = torch.zeros_like(param).to(param.device)
        
        for i in range(0, num_samples):
            inputs = tokenizer.encode(gens.iloc[i]['0'], return_tensors="pt").to(model.device)
            outputs = model(inputs)
            for key, tensor in magnitude.items():
                try:
                    param_dict[f"{key}.weight"] += tensor
                except:
                    pass
        keys_to_remove = [key for key in param_dict if key.split('.weight')[0] not in magnitude]
    
        for key in keys_to_remove:
            del param_dict[key]
        
        mask_dict = {}
    
    
        for k, v in param_dict.items():
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
                    top_pos = torch.topk(torch.abs(tensor), keep_num, largest = largest)[1]
                    mask_dict[k] = torch.zeros_like(tensor, device=tensor.device)
                    mask_dict[k][top_pos] = 1
                    mask_dict[k] = mask_dict[k].reshape(v.shape).to(tensor.device)
                else:
                    sizes = v.shape
                    num_params = v.numel()
                    keep_num = int(num_params * keep_ratio)
                    tensor = v.view(-1)
                    top_pos = torch.topk(torch.abs(tensor), keep_num, largest = largest)[1]
                    mask_dict[k] = torch.ones_like(tensor, device=tensor.device)
                    mask_dict[k][top_pos] = 0
                    mask_dict[k] = mask_dict[k].reshape(v.shape).to(tensor.device)
    
        return mask_dict
        
    
def find_good_params(model, train, keep_ratio, prune = True, largest = True, num_samples = len(train)):
    global chosen_params
    import random

    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    param_dict = {}
    for name, param in model.named_parameters():
        param_dict[name] = torch.zeros_like(param, device = param.device)
            
    for i in range(0, num_samples):
        # gsm8k, race
        if 'qa' in train.columns.to_list():
            prompt = train.iloc[i]['qa']
        # CodeAlpaca
        elif "prompt" in train.columns and "completion" in train.columns:
            src = train.iloc[i]["prompt"]
            tgt = train.iloc[i]["completion"]
            # choose whatever formatting you like:
            prompt = f"{src}\n{tgt}"
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
                top_pos = torch.topk(torch.abs(tensor), keep_num, largest = largest)[1]
                mask_dict[k] = torch.zeros_like(tensor, device= tensor.device)
                mask_dict[k][top_pos] = 1
                mask_dict[k] = mask_dict[k].reshape(v.shape).to(tensor.device)
            else:
                sizes = v.shape
                num_params = v.numel()
                keep_num = int(num_params * keep_ratio)
                tensor = v.view(-1)
                top_pos = torch.topk(torch.abs(tensor), keep_num, largest = largest)[1]
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

def scale(good_params, factor):
    prune_params = {}
    for k, v in good_params.items():
        good_indices = good_params[k]!=1
        keep_indices = good_params[k]==1
        prune_params[k] = keep_indices + (good_indices*factor)
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


import torch
from contextlib import nullcontext

def fine_tune_on_isolated_params(
    model,
    tokenizer,
    train_df,
    isolated_masks,
    num_epochs=5,
    lr=1e-4,
    max_length=512,
    use_amp=True,
    amp_dtype=torch.bfloat16,   # you can switch to torch.float16 if you want
    grad_clip=1.0,
):
    device = next(model.parameters()).device

    # ---------------------------------------------------------------
    # 0) Memory-friendly model settings
    # ---------------------------------------------------------------
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    #train_df = train.sample(2000)
    steps_per_epoch = len(train_df)
    total_steps = num_epochs * steps_per_epoch

    # ---------------------------------------------------------------
    # 1) Trainable params
    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # 2) Optimizer + scheduler (foreach=False to be more VRAM-friendly)
    # ---------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        foreach=False,  # avoids _multi_tensor_adam big temp buffers
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
    )

    # GradScaler: only for fp16
    if use_amp and amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        scaler = None  # no scaling for bf16 or no-AMP

    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        epoch_df = train_df.sample(frac=1).reset_index(drop=True)
        print(f"\n===== Starting epoch {epoch+1}/{num_epochs} =====")

        for i, (_, row) in enumerate(epoch_df.iterrows()):
            qa_text = row["qa"]

            # ------------------------------
            # Split into prompt and target
            # ------------------------------
            question_part, sep, answer_part = qa_text.partition("A:")
            if sep == "":
                prompt_text = qa_text
                target_text = ""
            else:
                prompt_text = (
                     question_part
                    + " "
                )
                target_text = answer_part.lstrip()

            full_text = prompt_text + " " + target_text if target_text else prompt_text

            tok = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = tok["input_ids"].to(device)
            labels = input_ids.clone()

            prompt_ids = tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            prompt_len = prompt_ids.size(1)
            labels[:, :prompt_len] = -100
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype)
            else:
                autocast_ctx = nullcontext()

            # ----------------------------------------------------------
            # Forward + backward (AMP + optional GradScaler)
            # ----------------------------------------------------------
            with autocast_ctx:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

            if scaler is not None:
                # fp16 + scaler path
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # mask grads
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if not param.requires_grad or param.grad is None:
                            continue
                        if name in isolated_masks:
                            mask = isolated_masks[name].to(param.grad.device)
                            param.grad.mul_(mask.to(param.grad.dtype))
                        else:
                            param.grad.zero_()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

                scaler.step(optimizer)
                scaler.update()
            else:
                # bf16 or no-AMP path, no GradScaler
                loss.backward()

                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if not param.requires_grad or param.grad is None:
                            continue
                        if name in isolated_masks:
                            mask = isolated_masks[name].to(param.grad.device)
                            param.grad.mul_(mask.to(param.grad.dtype))
                        else:
                            param.grad.zero_()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

                optimizer.step()

            scheduler.step()
            global_step += 1

            del outputs, tok, input_ids, labels, prompt_ids
            if global_step % 50 == 0:
                torch.cuda.empty_cache()

            # ----------------------------------------------------------
            # Debug logging
            # ----------------------------------------------------------
            if (epoch == 0 and i < 3) or (global_step % 200 == 0):
                print("\n================ DEBUG FINE-TUNE STEP =================")
                print(f"Epoch: {epoch+1}/{num_epochs}")
                print(f"Step in epoch: {i+1}/{steps_per_epoch}")
                print(f"Global step: {global_step}/{total_steps}")
                print(f"Loss: {loss.item():.4f}")
                print(f"Current LR: {scheduler.get_last_lr()[0]:.6e}")
                '''
                print("\nPrompt (prompt_text):\n")
                print(prompt_text)
                try:
                    with torch.no_grad():
                        if use_amp:
                            gen_autocast = torch.amp.autocast("cuda", dtype=amp_dtype)
                        else:
                            gen_autocast = nullcontext()
                        with gen_autocast:
                            gen_ids = model.generate(
                                **tokenizer(
                                    prompt_text,
                                    return_tensors="pt",
                                    truncation=True,
                                    max_length=max_length,
                                ).to(device),
                                max_new_tokens=64,
                                do_sample=False,
                                pad_token_id=(
                                    tokenizer.eos_token_id
                                    if tokenizer.eos_token_id is not None
                                    else tokenizer.pad_token_id
                                ),
                            )
                    decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    print("\nModel output (prompt_text + continuation):\n")
                    print(decoded)
                    del gen_ids
                    
                except Exception as e:
                    print(f"[DEBUG] Generation failed: {e}")'''




import gc

num_samples = args.num_samples
num_repeats = 1
if args.proportion is None:
    good_percents = [.0001, .001, .005, .01, .025, .05, .1, .15]
if args.proportion is not None:
    good_percents = [args.proportion]
good_percents = [0.01, 0.05, 0.1]
scalar = args.scalar
for dataset in dataset_list:
    for repeat in range(0, num_repeats):
        sampled_train = train.sample(n = num_samples, replace = True)
        sampled_comparison = dataset.sample(n = num_samples, replace = True)
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
            good_params = find_good_params(model, sampled_train, keep_ratio=good_percent, prune = True, largest = True, num_samples = num_samples)
            torch.cuda.empty_cache()
            if 'Bad' in dataset.name:    
                comparison_params = find_params(model, sampled_comparison, keep_ratio=good_percent, prune = True, largest = True, num_samples = num_samples)
            else:
                comparison_params = find_good_params(model, sampled_comparison, keep_ratio=good_percent, prune = True, largest = True, num_samples = num_samples)


            math_top_params = count_top_parameters(good_params)
            nonmath_top_params = count_top_parameters(comparison_params)
            math_only_params = count_math_only_parameters(good_params, comparison_params)
            print(math_top_params, nonmath_top_params, math_only_params)
            with open(output_file, "a") as f:
                f.write(
                    f"[{dataset.name}] repeat {repeat+1}, top {good_percent*100:.4f}% — math: {math_top_params}, non-math: {nonmath_top_params}, math-only after removing non-math: {math_only_params}\n"
                )


            def remove_hooks(model):
                # Function to remove all hooks
                for name, module in model.named_modules():
                    # Check if the module has any forward hooks
                    if hasattr(module, "_forward_hooks") and len(module._forward_hooks) > 0:
                        # Remove all forward hooks
                        module._forward_hooks.clear()


            if args.store_params:
                isolated_zero_mask = prune(comparison_params, good_params, factor=0.0, return_good=True)
                isolated_masks = {}
                for k, m in isolated_zero_mask.items():
                    isolated_masks[k] = (m == 0).to(torch.bool)
                del isolated_zero_mask
                mask_filename = f"gsm8k_{dataset.name}_{good_percent}_repeat{repeat}.pt"
                mask_path = os.path.join(mask_dir, mask_filename)
                cpu_masks = {k: v.to("cpu") for k, v in isolated_masks.items()}
                torch.save(
                    {
                        "model_name": args.model,
                        "dataset_name": dataset.name,
                        "good_percent": good_percent,
                        "repeat": repeat,
                        "isolated_masks": cpu_masks,
                    },
                    mask_path,
                )
                del isolated_masks
                del cpu_masks
                print(f"Saved isolated mask to {mask_path}")

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
                remove_hooks(model)
                torch.cuda.empty_cache()

                # e.g., one pass over num_samples examples
                fine_tune_on_isolated_params(
                    model=model,
                    tokenizer=tokenizer,
                    train_df=sampled_train,
                    isolated_masks=isolated_masks
                )
                del isolated_masks

            else:
                good_params = {k: v.to("cpu") for k, v in good_params.items()}
                comparison_params = {k: v.to("cpu") for k, v in comparison_params.items()}

                prune_params = prune(comparison_params, good_params, scalar, return_good = True)
                del good_params
                del comparison_params
                for key, tensor in prune_params.items():
                    device = model.state_dict()[key].device
                    tensor = tensor.to(device)
                    model.state_dict()[key]*=tensor
                del prune_params
        
            remove_hooks(model)
            if 'sgsm' in args.train_dataset:
                prune_solve = []
                prune_code = []
                prune_solutions = []
                for i in range(0, min(args.eval_dataset_subset, len(val))):
                    # Format the prompt
                    prompts = []
                    questions = []
                    final_question = val.iloc[i]['question']
                    final_answer = val.iloc[i]['answer']
                    final_prompt = f"""Instruct: {final_question} Let's write a Python program.\nOutput:"""
    
                    for j in range(0, 8):
                        question = train['question'].iloc[j]
                        questions.append(question)
                        answer = train['solution'].iloc[j]
                        prompt = f"""Instruct: {question} Let's write a Python program.\nOutput:\n{answer}"""
                        if prompt not in prompts:
                            prompts.append(prompt)
    
                    prompts.append(final_prompt)
                    formatted_prompt = "\n\n".join(prompts)
                    #Query the model 
                    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
                    model_answer = None
                    #output = model.generate(inputs, max_new_tokens = 150, temperature = .7, do_sample = True)
                    output = model.generate(inputs, max_new_tokens = 150)
                    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                    # Split the generated text by the prompt to extract the newly generated part
                    generated_text_parts = generated_text.split(final_prompt)
                    solution_text = generated_text_parts[-1].strip()
                    prune_solutions.append(solution_text)
                    if "Instruct:" in solution_text:
                        solution_text = solution_text.split("Instruct:")[0] # Split up a generation that contains more than one question
                    if "print" in solution_text:
                        solution_text = solution_text.split("print")[0] # Split up a generation that contains a print statement
                    if "Student:" in solution_text:
                        solution_text = solution_text.split("Student:")[0] # Split up a generation that contains more than one question
                    if "Output:" in solution_text:
                        solution_text = solution_text.split("Output:")[0] # Split up a generation that contains more than one question
                    if "#TODO" in solution_text:
                        solution_text = solution_text.split("#TODO")[0] # Split up a generation that contains more than one question
                    #solutions.append(solution_text)
                    if 'return result' in solution_text:
                        # Split the string on 'return result' but keep 'return result' in the result
                        parts = re.split(r'(return result)', solution_text)
    
                        # Rejoin the parts correctly
                        solution_text = parts[0] + parts[1]
                    try:
                        exec(solution_text)
                        model_answer = solution()
                        prune_code.append(1)
                        model_answer = float(model_answer)
                        if model_answer != final_answer:
                            prune_solve.append(0)
    
                        if model_answer == final_answer:
                            prune_solve.append(1)
    
                    except:
                        prune_code.append(0)
                        prune_solve.append(0)
    
                with open(output_file, "a") as f:  # Open the file in append mode ("a")
                        f.write(f"Average eval accuracy on {min(args.eval_dataset_subset, len(val))} questions for pruning top {good_percent}% good parameters based on not being activated by {dataset.name} based on {num_samples} training samples and greedy decoding (few-shot): {np.mean(prune_solve)}\n")  
                torch.cuda.empty_cache()
                #task_manager = lm_eval.tasks.TaskManager()
                #--log_samples --output_path results/phi_15_base --device cuda:0 --batch_size auto:4
                # Setting `task_manager` to the one above is optional and should generally be done
                # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
                # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
                results = lm_eval.simple_evaluate( # call simple_evaluate
                    model = 'hf',
                    model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
                    tasks=args.eval_datasets,
                    task_manager=task_manager,
                    log_samples = False, 
                    batch_size = 2,
                    limit = args.eval_dataset_subset
                )
                results_path = f"{args.save_path}/eval_results/{args.model}/{dataset.name}_calculate{good_percent}_run{repeat}.json"
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, "w") as outfile: 
                    json.dump(results['results'], outfile)
            if args.train_lm_eval_task is not None:
                task_manager = lm_eval.tasks.TaskManager(include_path="../lm_eval_tasks")
                #task_manager = lm_eval.tasks.TaskManager()
                #--log_samples --output_path results/phi_15_base --device cuda:0 --batch_size auto:4
                # Setting `task_manager` to the one above is optional and should generally be done
                # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
                # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
                results = lm_eval.simple_evaluate( # call simple_evaluate
                    model = 'hf',
                    model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
                    tasks=args.train_lm_eval_task,
                    task_manager=task_manager,
                    log_samples = False, 
                    batch_size = 'auto:16',
                    limit = args.eval_dataset_subset, 
                    random_seed = args.random_state
                )

                results_path = f"{args.save_path}/eval_results/{args.model}/{dataset.name}_calculate{good_percent}_scalar{scalar}_finetune{args.fine_tune}_run{repeat}_train_task.json"
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, "w") as outfile: 
                    json.dump(results['results'], outfile)
                    
                results = lm_eval.simple_evaluate( # call simple_evaluate
                    model = 'hf',
                    model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
                    tasks=args.eval_datasets,
                    task_manager=task_manager,
                    log_samples = False,
                    batch_size = 1,
                    limit = args.eval_dataset_subset
                )


                results_path = f"{args.save_path}/eval_results/{args.model}/{dataset.name}_calculate{good_percent}_scalar{scalar}_finetune{args.fine_tune}_run{repeat}.json"
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, "w") as outfile: 
                    json.dump(results['results'], outfile)
            del model

            gc.collect()
            torch.cuda.empty_cache()