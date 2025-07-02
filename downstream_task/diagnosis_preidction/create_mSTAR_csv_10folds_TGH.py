import pandas as pd

# Load the TSV file
df = pd.read_csv("/home/jma/Documents/Beatrice/configs/TGH_k=10.tsv", sep="\t")

# Rename base columns
df = df.rename(columns={
    "case_id": "case",
    "slide_id": "slide",
    "BRCA_mutation": "label"
})

# Dynamically rename fold columns and replace "val" with "test"
for i in range(10):
    old_col = f"fold_{i}"
    new_col = f"fold{i+1}"
    df = df.rename(columns={old_col: new_col})
    df[new_col] = df[new_col].replace("val", "test")

# Reorder columns
fold_cols = [f"fold{i+1}" for i in range(10)]
df = df[["case", "slide", "label"] + fold_cols]

# Save to CSV
df.to_csv("/home/jma/Documents/Beatrice/mSTAR/downstream_task/diagnosis_preidction/dataset_csv/TGH_10folds.csv", index=False)
