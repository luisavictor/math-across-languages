# run with
# python MathNeuro/MathNeuro.py --model meta-llama/Llama-3.2-1B-Instruct --save_path results --train_dataset MathNeuro/data/gsm8k.csv --eval_datasets race --calibration_datasets MathNeuro/data/race.csv --eval_dataset_subset 5 --calibration_dataset_names Race --train_lm_eval_task gsm8k_cot --pre_train_eval --proportion 0.01  --num_samples 10



import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', help="Huggingface model to train, entered as string", type = str)
parser.add_argument('--eval_datasets', nargs='+', help="dataset(s) to evaluate models on post pruning to evalaute catastrophic forgetting, entered as strings; should be task names from Eleuther AI LM Evaluation Harness", type = str)
parser.add_argument('--train_dataset', help="path to math train dataset; should be a path to a CSV file with question/solution pairs in a columns titled 'question' and 'solution' along with ground-truth answers in a column called 'answer'", type = str)
parser.add_argument('--calibration_datasets', nargs='+', help="path to calibration datasets; should be paths to CSV files with instruction/response pairs in a column titled 'qa'", type = str)
parser.add_argument('--save_path', help="save path for eval results after running Eleuther AI LM Evaluation Harness post pruning", type = str)
parser.add_argument('--text_file', help="name of text file for saving pruning results during training if evaluating math reasoning using a non-Eleuther AI LM Evaluation Harness task in a PoT format", type = str)
parser.add_argument('--num_repeats', help="number of repeats for pruning or scaling experiment", type = int, default = 5)
parser.add_argument('--pre_train_eval', help="bool to indicate if full evaluation on eval and train datasets should be conducted before training", action="store_true")
parser.add_argument('--random_state', help="random state for initial dataset shuffling and creating train/eval split for train dataset", type = int, default = 42)
parser.add_argument('--scalar', help="scale factor for top parameters; default is 0 to run pruning experiments", type = float, default = 0)
parser.add_argument('--eval_dataset_size', help="desired number of samples for task specific eval dataset", type = int, default = None)
parser.add_argument('--eval_dataset_subset', help="desired number of samples for task specific eval dataset if subsetting to reduce run time", type = int, default = 100)
parser.add_argument('--calibration_dataset_names', nargs='+', help="desired name of calibration datasets; should be strings entered in same order as calibration_datasets", type = str)
parser.add_argument('--num_samples', help="desired number of samples for calculating task specific parameters", type = int, default = 500)
parser.add_argument('--train_lm_eval_task', help="if your training dataset is an Eleuther AI LM Evaluation Harness task, specify the associated task for the test set.", type = str, default = None)
parser.add_argument('--proportion', help="desired proportion of top parameters to calculate", type = float, default = None)
args = parser.parse_args()
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
import json 
import lm_eval
'''
# Wir verwenden sgsm nicht, daher wahrscheinlich nicht notwendig für uns
if 'sgsm' in args.train_dataset:
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
'''

if 'sgsm' not in args.train_dataset:
    train = pd.read_csv(args.train_dataset) # Load SGSM dataset for few-shot prompting
    train = train.sample(frac = 1, random_state = args.random_state)
    

# for computing T_non_math: Race
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
  

print("dataset list is", dataset_list)



output_file = f"{args.save_path}/eval_results/{args.model}/{args.text_file}"
results_path =  f"{args.save_path}/eval_results/{args.model}/"
os.makedirs(os.path.dirname(results_path), exist_ok=True)
print("directory at: ", os.path.dirname(results_path))




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


if args.pre_train_eval:
    '''
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
            model_args = {'pretrained':args.model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
            tasks=args.eval_datasets,
            task_manager=task_manager,
            log_samples = False, 
            batch_size = 'auto:4'
        )
        results_path = f"{args.save_path}/eval_results/{args.model}/pre_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as outfile: 
            json.dump(results['results'], outfile)
            '''
    
    # Perform Eleuther AI Harness evaluation BEFORE modifications
    if args.train_lm_eval_task is not None:
        print("start pre-evaluation")

        task_manager = lm_eval.tasks.TaskManager()
        #--log_samples --output_path results/phi_15_base --device cuda:0 --batch_size auto:4
        # Setting `task_manager` to the one above is optional and should generally be done
        # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
        # `simple_evaluate` will instantiate its own task_manager if it is set to None here.

        # Evaluate the model on the training dataset's task, e.g., GSM8K
        results = lm_eval.simple_evaluate( # call simple_evaluate
            model = 'hf',
            model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
            tasks=args.train_lm_eval_task,
            task_manager=task_manager,
            log_samples = False, 
            batch_size = 1,
            limit = args.eval_dataset_subset, 
            random_seed = args.random_state
        )

        print("results", results)
        results_path = f"{args.save_path}/eval_results/{args.model}/pre_results_train_task.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as outfile: 
            json.dump(results['results'], outfile)
        

        # Evaluate the model on the general evaluation datasets, e.g., Race
        results = lm_eval.simple_evaluate( # call simple_evaluate
            model = 'hf',
            model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
            tasks=args.eval_datasets,
            task_manager=task_manager,
            log_samples = False, 
            batch_size = 1
        )
        print("results", results)
        results_path = f"{args.save_path}/eval_results/{args.model}/pre_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as outfile: 
            json.dump(results['results'], outfile)
            


# Start MathNeuro-Algo
magnitude = {}

print("start mathneuro algorithm")

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



# for further calibration? we do not use it, dont we?
'''
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
'''   
    
def find_good_params(model, train, keep_ratio, prune = True, largest = True, num_samples = len(train)):
    global chosen_params
    import random
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"
    param_dict = {} # param_dict accumulates per-parameter magnitude scores
    for name, param in model.named_parameters():
        #param_dict[name] = torch.zeros_like(param).to(param.device)  # This will accumulate importance values
        param_dict[name] = torch.zeros_like(param, device = cuda_device)
            
    for i in range(0, num_samples):
        if 'qa' in train.columns.to_list():
            prompt = train.iloc[i]['qa']
        else:
            question = train['question'].iloc[i]
            answer = train['solution'].iloc[i]
            prompt = f"""Instruct: {question} Let's write a Python program.\nOutput:\n{answer}"""
        with torch.inference_mode():   # das ist Änderung
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            outputs = model(inputs)
        for key, tensor in magnitude.items():
            try:
                param_dict[f"{key}.weight"] += tensor
            except:
                print(f'passed at {key}')
    keys_to_remove = [key for key in param_dict if key.split('.weight')[0] not in magnitude]
    # some parameters may not have corresponding magnitude entries
    for key in keys_to_remove:
        del param_dict[key]

    # create dictionary to store mask 
    mask_dict = {}
    # Each mask has 0s and 1s indicating which parameters survive
    for k, v in param_dict.items():
        # don't count classifier layer
        if "embed" in k:
            if prune == False:  # == keeping the “most important” parameters
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
                mask_dict[k] = torch.zeros_like(tensor, device=tensor.device) # 0 → drop it
                mask_dict[k][top_pos] = 1 # 1 → keep this parameter
                mask_dict[k] = mask_dict[k].reshape(v.shape).to(tensor.device)

            else:
                sizes = v.shape
                num_params = v.numel()
                keep_num = int(num_params * keep_ratio)
                tensor = v.view(-1)
                top_pos = torch.topk(torch.abs(tensor), keep_num, largest = largest)[1]   # largest magnitude weights, considered the most important
                #mask_dict[k] = torch.ones_like(tensor, device=tensor.device)
                mask_dict[k] = torch.ones_like(tensor, device='cpu')
                mask_dict[k][top_pos] = 0   # prune strong weights
                mask_dict[k] = mask_dict[k].reshape(v.shape).to('cpu')



    # we use prune = True
    # mask_dict is a dict
    # {
    #     "layer1.weight": tensor([...]),
    #     "layer2.weight": tensor([...]),
    #     ...
    # }
    # Every mask starts filled with 1 → meaning “keep everything”
    # Then the top keep_ratio of parameters (by magnitude) are set to 0
    # --> zeroing out the parameters that the magnitude metric thinks are most important
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



num_samples = args.num_samples   # default = 500, number of samples for finding task-specific parameters
num_repeats = 1   # originally 5
if args.proportion is None:
    good_percents = [.0001, .001, .005, .01, .025, .05, .1, .15]
if args.proportion is not None:
    good_percents = [args.proportion]
scalar = args.scalar


for dataset in dataset_list:
    for repeat in range(0, num_repeats):
        sampled_train = train.sample(n = num_samples, replace = True)
        sampled_comparison = dataset.sample(n = num_samples, replace = True)
        for good_percent in good_percents:
            print("good percent: ", good_percent)
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
                    print("call hook function for layer ", name)
                    hook_fn = getActivation(name)  # Get the hook function
                    module.register_forward_hook(hook_fn)  # Register the hook function
            print("try to find good params")
            good_params = find_good_params(model, sampled_train, keep_ratio=good_percent, prune = True, largest = True, num_samples = num_samples)
            import sys
            print("good params found")
            torch.cuda.empty_cache()
            if 'Bad' in dataset.name:    
                comparison_params = find_params(model, sampled_comparison, keep_ratio=good_percent, prune = True, largest = True, num_samples = num_samples)
            else:
                print("find comparison params")
                comparison_params = find_good_params(model, sampled_comparison, keep_ratio=good_percent, prune = True, largest = True, num_samples = num_samples)

            print("Good params size:", sys.getsizeof(good_params))
            print("Comparison params size:", sys.getsizeof(comparison_params))
            prune_params = prune(comparison_params, good_params, scalar, return_good = True)
            del good_params
            del comparison_params
            for key, tensor in prune_params.items():
                device = model.state_dict()[key].device
                tensor = tensor.to(device)
                model.state_dict()[key]*=tensor
                
            del prune_params
            def remove_hooks(model):
                # Function to remove all hooks
                for name, module in model.named_modules():
                    # Check if the module has any forward hooks
                    if hasattr(module, "_forward_hooks") and len(module._forward_hooks) > 0:
                        # Remove all forward hooks
                        module._forward_hooks.clear()
        
            remove_hooks(model)


            #At this point you are zeroing out (permanently disabling) all weights that:

            # were important for the good dataset but NOT for the comparison dataset

            print("important params found")


            '''if 'sgsm' in args.train_dataset:
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
                    log_samples = False, 
                    batch_size = 'auto:4'
                )
                results_path = f"{args.save_path}/eval_results/{args.model}/{dataset.name}_calculate{good_percent}_run{repeat}.json"
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, "w") as outfile: 
                    json.dump(results['results'], outfile)'''

            torch.cuda.empty_cache()
            if args.train_lm_eval_task is not None:
                task_manager = lm_eval.tasks.TaskManager()
                #--log_samples --output_path results/phi_15_base --device cuda:0 --batch_size auto:4
                # Setting `task_manager` to the one above is optional and should generally be done
                # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
                # `simple_evaluate` will instantiate its own task_manager if it is set to None here.

                print("start final evaluation")
                results = lm_eval.simple_evaluate( # call simple_evaluate
                    model = 'hf',
                    model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
                    tasks=args.train_lm_eval_task,
                    task_manager=task_manager,
                    log_samples = False, 
                    batch_size = 1,
                    limit = args.eval_dataset_subset, 
                    random_seed = args.random_state
                )
                results_path = f"{args.save_path}/eval_results/{args.model}/{dataset.name}_calculate{good_percent}_run{repeat}_train_task.json"
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, "w") as outfile: 
                    json.dump(results['results'], outfile)
                    
                results = lm_eval.simple_evaluate( # call simple_evaluate
                    model = 'hf',
                    model_args = {'pretrained':model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
                    tasks=args.eval_datasets,
                    task_manager=task_manager,
                    log_samples = False, 
                    batch_size = 1
                )
                print("results are", results)
                results_path = f"{args.save_path}/eval_results/{args.model}/{dataset.name}_calculate{good_percent}_run{repeat}.json"
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, "w") as outfile: 
                    json.dump(results['results'], outfile)

            del model