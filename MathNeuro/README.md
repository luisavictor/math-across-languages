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
