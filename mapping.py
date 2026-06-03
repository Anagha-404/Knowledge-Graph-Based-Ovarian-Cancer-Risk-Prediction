import pandas as pd

file_path = "GSE9891_series_matrix.txt"

gsm_ids = []
titles = []

with open(file_path, "r") as f:
    for line in f:
        if line.startswith("!Sample_geo_accession"):
            gsm_ids = line.strip().split("\t")[1:]
        
        if line.startswith("!Sample_title"):
            titles = line.strip().split("\t")[1:]

# Create mapping
mapping = {k.strip('"'): v.strip('"') for k, v in zip(gsm_ids, titles)}
# Print sample mapping
for i, (k, v) in enumerate(mapping.items()):
    print(k, "→", v)
    if i == 10:
        break


# Load clinical data
clinical = pd.read_csv("GSE9891_clinical_anns.csv")

label_map = {}

for gsm, x_id in mapping.items():
    row = clinical[clinical["AOCSID"] == x_id]
    
    if not row.empty:
        stage = str(row["StageCode"].values[0])
        
        # Convert stage → label
        if stage.startswith("III") or stage.startswith("IV"):
            label = 1   # High risk
        else:
            label = 0   # Low risk
        
        label_map[gsm] = label

# Convert to dataframe
label_df = pd.DataFrame(list(label_map.items()), columns=["patient", "label"])

print(label_df.head())

label_df.to_csv("labels.csv", index=False)

def get_labels():
    return pd.read_csv("labels.csv")