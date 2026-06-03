import pandas as pd

# Load your wide dataset
df = pd.read_csv("final_dataset.csv")

print("Original shape:", df.shape)

# Convert wide → long format
df_long = df.melt(
    id_vars=["Gene"],
    var_name="Patient",
    value_name="Expression"
)

# Reorder columns
df_long = df_long[["Patient", "Gene", "Expression"]]

print("Converted shape:", df_long.shape)
print(df_long.head())

# Save new file
df_long.to_csv("converted_dataset.csv", index=False)

print("\n✅ converted_dataset.csv created successfully!")