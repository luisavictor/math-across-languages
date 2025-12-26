# Test Repo for Lab


## Important Hyperparams
- --eval_datasets: evaluate catastrophic forgetting after pruning/scaling, should be task names from Eleuther AI LM Evaluation Harness, e.g., race
- --train_dataset: math or coding dataset as path to a .csv file, question, solution and answer columns?
- --calibration_dataset: 'qa', paths to .csv files, e.g., data/race.csv
- --calibration_dataset_names: names of calibration dataset as strings, e.g., Race
- --scalar: for scalar = 0, run pruning experiments, else scaling experiments
- --eval_dataset_subset: How many samples to use for evaluation on math/coding tasks? (for reducing run time)
- --num_samples: desired number of samples for calculating task specific parameters, e.g., Math or Coding task
- --train_lm_eval_task: if Math/Coding dataset is an Eleuther AI LM Evaluation Harness task, specify task for test set as string, e.g., gsm8k_cot
- --proportion: proportion of top params to calculate, like top-k from paper








# Train/Test-Split:

GSM8K:
 - use the same 200 random samples from the GSM8K test split for evaluation, 8-shot chain-of-thought prompting format for lm_eval, gsm8k_de_test.csv, GSM8K-CoT accuracy = strict-match accuracy
 - use train split for finding important parameters: gsm8k_de_train.csv



CodeAlpaca:
- also use 200 random samples from test split for evaluation, test split in: codealpaca_train.csv, 
- train split in: codealpaca_train.csv



Race:


MMLU:





All datasets:
- training five times with different random subsets of 500 samples from each dataset
