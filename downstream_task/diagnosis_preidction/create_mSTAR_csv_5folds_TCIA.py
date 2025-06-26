import pandas as pd

# Load the TSV file
df = pd.read_csv("/mnt/pool/ovariancancer/CCRCC_results/config/k=all_val.tsv", sep="\t")

# take only the first 5 folds
df = df.iloc[:, :8]

# Rename columns
df = df.rename(columns={
    "case_id": "case",
    "slide_id": "slide",
    "BAP1_mutation": "label",
    "fold_0": "fold1",
    "fold_1": "fold2",
    "fold_2": "fold3",
    "fold_3": "fold4",
    "fold_4": "fold5"
})

df['fold1'] = df['fold1'].replace('val', 'test')
df['fold2'] = df['fold2'].replace('val', 'test')
df['fold3'] = df['fold3'].replace('val', 'test')
df['fold4'] = df['fold4'].replace('val', 'test')
df['fold5'] = df['fold5'].replace('val', 'test')
# Reorder columns
df = df[["case", "slide", "label", "fold1", "fold2", "fold3", "fold4", "fold5"]]

# Save to CSV
df.to_csv("/home/jma/Documents/Beatrice/mSTAR/downstream_task/diagnosis_preidction/dataset_csv/CCRCC_5folds.csv", index=False)
