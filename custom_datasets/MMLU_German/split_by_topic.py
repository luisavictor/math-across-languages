import pandas as pd
from pathlib import Path

df = pd.read_csv("../MMLU_Hindi/mmlu_hi_test.csv")

out_dir = Path("../MMLU_Hindi/mmlu_hi_by_subject")
out_dir.mkdir(exist_ok=True)

for subject, sub_df in df.groupby("subject"):
    sub_df.to_csv(out_dir / f"{subject}.csv", index=False)

print("Done. Subjects:", df["subject"].unique())
