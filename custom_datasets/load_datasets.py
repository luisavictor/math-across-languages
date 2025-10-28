from datasets import load_dataset

# Load the dataset
ds = load_dataset("Sara237/gsm8k-translated", "fr")

# Save it locally
ds.save_to_disk("gsm8k_french")
