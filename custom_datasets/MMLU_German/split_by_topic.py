import pandas as pd
from pathlib import Path

df = pd.read_csv("mmlu_de_test.csv")

out_dir = Path("mmlu_de_by_subject")
out_dir.mkdir(exist_ok=True)

for subject, sub_df in df.groupby("subject"):
    sub_df.to_csv(out_dir / f"{subject}.csv", index=False)

print("Done. Subjects:", df["subject"].unique())
