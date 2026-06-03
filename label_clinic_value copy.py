import pandas as pd

# Load clinical data
clinical = pd.read_csv("GSE9891_clinical_anns.csv")

label_map = {}

for gsm, x_id in mapping.items():
    row = clinical[clinical["AOCSID"] == x_id]
    
    if not row.empty:
        stage = str(row["StageCode"].values[0])
        
        # Convert stage → label
        if stage.startswith("I") or stage.startswith("II"):
            label = 0   # Low risk
        else:
            label = 1   # High risk
        
        label_map[gsm] = label

# Convert to dataframe
label_df = pd.DataFrame(list(label_map.items()), columns=["patient", "label"])

print(label_df.head())