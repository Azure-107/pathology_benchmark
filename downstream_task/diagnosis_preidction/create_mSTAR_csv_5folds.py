import pandas as pd

# Load the TSV file
df = pd.read_csv("/home/jma/Documents/Beatrice/configs/k=5.tsv", sep="\t")

# Rename columns
df = df.rename(columns={
    "case_id": "case",
    "slide_id": "slide",
    "BRCA_mutation": "label",
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
df.to_csv("/home/jma/Documents/Beatrice/mSTAR/downstream_task/diagnosis_preidction/dataset_csv/UHN_5folds.csv", index=False)
