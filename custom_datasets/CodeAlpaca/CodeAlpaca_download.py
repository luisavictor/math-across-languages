import pandas as pd
from datasets import load_dataset
import pandas as pd

# Load dataset directly from HuggingFace
ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K")

# Convert train split to pandas
df_train = ds["train"].to_pandas()

# Save to CSV
df_train.to_csv("codealpaca_train.csv", index=False)

print("Saved codealpaca_train.csv")

df_test = ds["test"].to_pandas()
df_test.to_csv("codealpaca_test.csv", index=False)


df = pd.read_csv("codealpaca_train.csv")
df["qa"] = df.apply(
    lambda r: f"Instruct: {r['prompt']}\nOutput:\n{r['completion']}",
    axis=1
)
df[["qa"]].to_csv("codealpaca_train_qa.csv", index=False)


df = pd.read_csv("codealpaca_test.csv")
# Ensure prompt and completion are strings
df["prompt"] = df_test["prompt"].astype(str)
df["completion"] = df_test["completion"].astype(str)

# Re-save with correct quoting
df.to_csv(
    "../custom_datasets/codealpaca_test.csv",
    index=False,
    quoting=1,  # csv.QUOTE_ALL
    escapechar='\\'
)
