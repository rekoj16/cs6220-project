import pandas as pd
import numpy as np

csv_path = "/home/hice1/ymai8/scratch/cs6220-project/CheXpert-v1.0-small/valid.csv"
df = pd.read_csv(csv_path)

meta_cols = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"]
label_cols = [c for c in df.columns if c not in meta_cols]

total_samples = len(df)
df_frontal = df[df["Frontal/Lateral"] == "Frontal"]
total_frontal = len(df_frontal)


df_frontal_clean = df_frontal[label_cols].fillna(0.0).replace(-1.0, 0.0)
results = {}
for col in label_cols:
    positives = (df_frontal_clean[col] == 1.0).sum()
    negatives = (df_frontal_clean[col] == 0.0).sum()
    results[col] = {"positive": positives, "negative": negatives}


print(f"Total samples: {total_samples}")
print(f"Total frontal samples: {total_frontal}\n")
print("Positive / Negative counts for frontal images:")
for label, counts in results.items():
    print(f"{label:25s}  Positive: {counts['positive']:6d}   Negative: {counts['negative']:6d}")
