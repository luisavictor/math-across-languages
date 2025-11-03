import subprocess
import sys

print(sys.executable)

venv_python = r"C:\Users\thmsr\Desktop\LabTest\.venv\Scripts\python.exe"

# Command as a list of arguments



# --train_dataset --> math dataset used to compute parameter importance for math reasoning (GSM8K or MATH)
# --eval_datasets --> for evaluating catastrophic forgetting after pruning or scaling. (RACE or MMLU)
# --calibration_datasets --> datasets used to compute parameter importance for non-math tasks
# --train_lm_eval_task --> only relevant if the train dataset is an LM evaluation harness task (like GSM8K). It tells MathNeuro which built-in task evaluation format to use (CoT prompting, etc.).




cmd = [
    venv_python,
    r"C:\Users\thmsr\Desktop\LabTest\MathNeuro\MathNeuro.py",  # full path to script
    "--model", "meta-llama/Llama-3.2-1B-Instruct",  # which model to use?
    "--save_path", "results_path",  # where to store
    "--train_dataset", "data/gsm8k.csv",  #data to find task-important params
    "--eval_datasets", "race",   # data to evaluate catastrophic forgetting after pruning/scaling
    "--calibration_datasets", "data/race.csv",  #data to find important non-task params
    "--calibration_dataset_names", "Race",
    "--eval_dataset_subset", "2",
    "--train_lm_eval_task", "gsm8k_cot", # for LM evaluation harness tasks
    "--pre_train_eval",
    #"--scalar", "0.01",
    #"--eval_dataset_size", "1",
    #"--num_samples", "1",
    "--proportion", "0.01"
]

# Run the command
subprocess.run(cmd)
